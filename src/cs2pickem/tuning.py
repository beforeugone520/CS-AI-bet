from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .calibration import ProbabilityCalibrator
from .cleaning import clean_matches
from .evaluation import accuracy, auc, brier_score, calibration_table, log_loss, profit_loss
from .features import FeatureBuilder
from .imbalance import rebalance_training_data
from .models import default_ensemble
from .odds import market_probability_from_row
from .reliability import PLAYER_STATUS_REQUIRED_FEATURES, UNSTABLE_IDENTITY_FEATURES, prepare_reliability_features
from .selection import FeatureSelector
from .splitting import time_series_folds, time_series_split


WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "fast_logistic": {"logistic": 1.0},
    "logistic": {"logistic": 1.0},
    "random_forest": {"random_forest": 1.0},
    "xgboost": {"xgboost": 1.0},
    "default": {"logistic": 0.20, "random_forest": 0.30, "xgboost": 0.35, "neural_network": 0.0},
    "no_nn": {"logistic": 0.20, "random_forest": 0.35, "xgboost": 0.45, "neural_network": 0.0},
    "tree_blend": {"random_forest": 0.45, "xgboost": 0.55},
}


def optimize_match_predictions(
    rows: Iterable[Mapping[str, Any]],
    reference_date: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    max_age_days: int = 400,
    candidate_configs: Sequence[Mapping[str, Any]] | None = None,
    top_k_values: Sequence[int] | None = None,
    epochs_values: Sequence[int] | None = None,
    candidate_names: Sequence[str] | None = None,
    seed: int = 29,
    calibrate: bool = True,
    rolling_folds: int = 3,
    market_weight: float = 0.30,
    probability_objective: str = "log_loss",
    elo_modes: Sequence[str] | None = None,
    rating_modes: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Replay a chronological holdout and compare candidate model configs.

    ``rating_modes`` (WF-2C skeleton) selects which leakage-free rating source feeds the
    candidate's rows: ``"elo"`` (the default, online pre-match Elo -- unchanged behaviour)
    or ``"glicko"`` (the period-batched Glicko-2 injection, which also writes the
    ``glicko_diff`` candidate column). It mirrors the existing ``elo_modes`` axis so the
    backtest can A/B the rating engines, but this stage only wires the switch and reports
    the per-candidate ``rating_mode``; the actual significance adjudication between Elo and
    Glicko is deferred to WF-2F. Defaulting to ``"elo"`` keeps ``inject_glicko=False`` so
    the existing Elo-only hot path is bit-for-bit unchanged.
    """
    raw_cleaned = sorted(clean_matches([dict(row) for row in rows], reference_date=reference_date, max_age_days=max_age_days), key=lambda row: row["date"])
    default_cleaned, _, feature_preparation = prepare_reliability_features(raw_cleaned)
    split = time_series_split(default_cleaned, train_ratio=train_ratio, validation_ratio=validation_ratio)
    if not split.train or not split.validation or not split.test:
        raise ValueError("match optimization requires non-empty train, validation, and test splits")

    configs = [
        dict(config)
        for config in (
            candidate_configs
            or _candidate_grid(top_k_values, epochs_values, candidate_names, elo_modes=elo_modes, rating_modes=rating_modes)
        )
    ]
    baseline_validation = _constant_metrics(split.validation, 0.5)
    baseline_test = _constant_metrics(split.test, 0.5)
    # Rows depend on BOTH which Elo gets injected and which rating engine is selected, so the
    # split / feature-prep / matrix caches are keyed on (inject_elo, rating_mode). The default
    # 'elo' mode keeps inject_glicko=False -> rows are bit-identical to the historic path.
    default_rating_mode = _DEFAULT_RATING_MODE
    split_cache: Dict[tuple[bool, str], object] = {(True, default_rating_mode): split}
    feature_preparation_cache: Dict[tuple[bool, str], Dict[str, object]] = {(True, default_rating_mode): feature_preparation}
    prepared_cache: Dict[tuple[int, bool, str], Dict[str, object]] = {}
    rolling_cache: Dict[tuple[int, bool, str], List[Dict[str, object]]] = {}
    candidate_results = []
    for index, config in enumerate(configs):
        top_k = int(config.get("top_k", 25))
        inject_elo = _config_inject_elo(config)
        rating_mode = _config_rating_mode(config)
        rating_key = (inject_elo, rating_mode)
        if rating_key not in split_cache:
            prepared_cleaned, _, candidate_feature_preparation = prepare_reliability_features(
                raw_cleaned,
                inject_elo=inject_elo,
                inject_glicko=(rating_mode == "glicko"),
            )
            split_cache[rating_key] = time_series_split(prepared_cleaned, train_ratio=train_ratio, validation_ratio=validation_ratio)
            feature_preparation_cache[rating_key] = candidate_feature_preparation
        candidate_split = split_cache[rating_key]
        cache_key = (top_k, inject_elo, rating_mode)
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = _prepare_split_matrices(
                candidate_split.train,
                candidate_split.validation,
                candidate_split.test,
                top_k=top_k,
                feature_preparation=feature_preparation_cache[rating_key],
            )
        if cache_key not in rolling_cache:
            rolling_cache[cache_key] = _prepare_rolling_fold_matrices(
                candidate_split.train + candidate_split.validation + candidate_split.test,
                top_k=top_k,
                rolling_folds=rolling_folds,
                feature_preparation=feature_preparation_cache[rating_key],
            )
        candidate_results.append(
            _evaluate_candidate(
                prepared_cache[cache_key],
                config,
                seed=seed + index,
                calibrate=calibrate,
                rolling_prepared=rolling_cache[cache_key],
                market_weight=market_weight,
                probability_objective=probability_objective,
            )
        )
    # Authoritative model selection uses ONLY the validation split. The test/holdout
    # split is reserved for reporting and must not drive which candidate is "best".
    best_accuracy = max(candidate_results, key=lambda item: (item["validation_metrics"]["accuracy"], -item["validation_metrics"]["log_loss"], item["name"]))
    best_log_loss = min(candidate_results, key=lambda item: (item["validation_metrics"]["log_loss"], -item["validation_metrics"]["accuracy"], item["name"]))
    # Test-oracle picks are kept only as an after-the-fact diagnostic ("how good
    # could we have done if we had cheated and peeked at the holdout"). They are
    # explicitly tagged so no caller mistakes them for a usable selection.
    best_test_accuracy = max(candidate_results, key=lambda item: (item["test_metrics"]["accuracy"], -item["test_metrics"]["log_loss"], item["name"]))
    best_test_log_loss = min(candidate_results, key=lambda item: (item["test_metrics"]["log_loss"], -item["test_metrics"]["accuracy"], item["name"]))
    _TEST_ORACLE_BASIS = "test_oracle_diagnostic_do_not_use_for_model_selection"
    authoritative_best = _best_summary(best_accuracy, baseline_test, selection_basis="validation")
    return {
        "matches": len(raw_cleaned),
        "reference_date": reference_date,
        "max_age_days": max_age_days,
        "feature_preparation": feature_preparation,
        "rolling_folds": rolling_folds,
        "market_weight": market_weight,
        "probability_objective": probability_objective,
        "split_counts": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "leakage_guard": {
            "max_train_date": split.train[-1]["date"],
            "min_validation_date": split.validation[0]["date"],
            "max_validation_date": split.validation[-1]["date"],
            "min_test_date": split.test[0]["date"],
        },
        "baseline": {"validation_metrics": baseline_validation, "test_metrics": baseline_test},
        "candidate_results": candidate_results,
        "selection_basis": "validation",
        "authoritative_best": authoritative_best,
        "best_by_validation_accuracy": _best_summary(best_accuracy, baseline_test, selection_basis="validation"),
        "best_by_validation_log_loss": _best_summary(best_log_loss, baseline_test, selection_basis="validation"),
        "best_by_test_accuracy": _best_summary(best_test_accuracy, baseline_test, selection_basis=_TEST_ORACLE_BASIS),
        "best_by_test_log_loss": _best_summary(best_test_log_loss, baseline_test, selection_basis=_TEST_ORACLE_BASIS),
        "test_predictions": _prediction_rows(split.test, best_accuracy["test_probabilities"]),
        "best_test_accuracy_predictions": _prediction_rows(split.test, best_test_accuracy["test_probabilities"]),
    }


def _candidate_grid(
    top_k_values: Sequence[int] | None,
    epochs_values: Sequence[int] | None,
    candidate_names: Sequence[str] | None,
    elo_modes: Sequence[str] | None = None,
    rating_modes: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    top_values = [int(value) for value in (top_k_values or [12, 18, 25])]
    epoch_values = [int(value) for value in (epochs_values or [8])]
    names = [str(name) for name in (candidate_names or ["fast_logistic", "random_forest", "no_nn"])]
    modes = [_normalize_elo_mode(mode) for mode in (elo_modes or ["with"])]
    rating_mode_values = [_normalize_rating_mode(mode) for mode in (rating_modes or [_DEFAULT_RATING_MODE])]
    configs: List[Dict[str, object]] = []
    for top_k in top_values:
        for epochs in epoch_values:
            for name in names:
                weights = WEIGHT_PRESETS.get(name)
                if weights is None:
                    raise ValueError(f"unknown tuning candidate: {name}")
                for mode in modes:
                    mode_suffix = f"_{mode}_elo" if len(modes) > 1 else ""
                    for rating_mode in rating_mode_values:
                        # Suffix the rating mode whenever it is non-default (e.g. a
                        # glicko-only single-mode run) so its candidates never collide
                        # with the byte-identical default elo names in reports. The
                        # default 'elo' single-mode path stays unsuffixed for back-compat.
                        rating_suffix = (
                            f"_{rating_mode}"
                            if len(rating_mode_values) > 1 or rating_mode != _DEFAULT_RATING_MODE
                            else ""
                        )
                        configs.append({
                            "name": f"{name}{mode_suffix}{rating_suffix}_k{top_k}_e{epochs}",
                            "preset": name,
                            "top_k": top_k,
                            "epochs": epochs,
                            "weights": dict(weights),
                            "inject_elo": mode == "with",
                            "rating_mode": rating_mode,
                        })
    return configs


def _normalize_elo_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"with", "with_elo", "elo", "on", "true", "1"}:
        return "with"
    if normalized in {"without", "without_elo", "no_elo", "off", "false", "0"}:
        return "without"
    raise ValueError(f"unknown elo mode: {mode}")


def _config_inject_elo(config: Mapping[str, Any]) -> bool:
    if "inject_elo" in config:
        return bool(config.get("inject_elo"))
    if "elo_mode" in config:
        return _normalize_elo_mode(str(config.get("elo_mode"))) == "with"
    return True


# WF-2C rating-source switch. 'elo' is the default and keeps the existing online pre-match
# Elo path (inject_glicko stays False -> rows unchanged). 'glicko' selects the period-batched
# Glicko-2 injection. The A/B significance verdict between them is deferred to WF-2F.
RATING_MODES = ("elo", "glicko")
_DEFAULT_RATING_MODE = "elo"


def _normalize_rating_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized in {"elo", "elo_only", "online_elo"}:
        return "elo"
    if normalized in {"glicko", "glicko2", "glicko-2", "glicko_2"}:
        return "glicko"
    raise ValueError(f"unknown rating mode: {mode}")


def _config_rating_mode(config: Mapping[str, Any]) -> str:
    if "rating_mode" in config and config.get("rating_mode") not in (None, ""):
        return _normalize_rating_mode(str(config.get("rating_mode")))
    return _DEFAULT_RATING_MODE


def _evaluate_candidate(
    prepared: Mapping[str, object],
    config: Mapping[str, Any],
    seed: int,
    calibrate: bool,
    rolling_prepared: Sequence[Mapping[str, object]],
    market_weight: float,
    probability_objective: str,
) -> Dict[str, object]:
    top_k = int(config.get("top_k", 25))
    epochs = int(config.get("epochs", 8))
    weights = dict(config.get("weights") or WEIGHT_PRESETS.get(str(config.get("preset", "no_nn")), WEIGHT_PRESETS["no_nn"]))
    prefer_accelerated = bool(config.get("prefer_accelerated", config.get("preset") != "fast_logistic"))
    rebalanced = prepared["rebalanced"]
    model = default_ensemble(seed=seed, epochs=epochs, weights=weights, prefer_accelerated=prefer_accelerated, n_jobs=1).fit(
        rebalanced.rows,
        rebalanced.labels,
        sample_weights=rebalanced.sample_weights,
    )
    selected_train = prepared["selected_train"]
    validation_rows = prepared["validation_rows"]
    test_rows = prepared["test_rows"]
    validation_x = prepared["validation_x"]
    test_x = prepared["test_x"]
    validation_labels = prepared["validation_labels"]
    test_labels = prepared["test_labels"]
    validation_probabilities = model.predict_proba(validation_x)
    raw_test_probabilities = model.predict_proba(test_x)
    calibration_report: Dict[str, object] = {"basis": "not_applied"}
    calibrated_test_probabilities = raw_test_probabilities
    calibrated_validation_probabilities = validation_probabilities
    calibrated_available = bool(calibrate and validation_probabilities and validation_labels)
    if calibrated_available:
        calibrator = ProbabilityCalibrator().fit(validation_probabilities, validation_labels)
        calibrated_test_probabilities = calibrator.transform(raw_test_probabilities)
        calibrated_validation_probabilities = calibrator.transform(validation_probabilities)
        calibration_report = calibrator.report()
        calibration_report["basis"] = "validation_platt_logistic"
    # The raw-vs-calibrated decision is made ONLY on the validation split so that
    # no test label ever participates in model/probability selection. The chosen
    # basis is then applied to the test probabilities; test metrics are reported
    # for both bases but never drive the choice.
    probability_selection = _probability_selection(
        validation_labels,
        validation_probabilities,
        calibrated_validation_probabilities,
        validation_rows,
        raw_test_probabilities,
        calibrated_test_probabilities,
        test_labels,
        test_rows,
        objective=probability_objective,
        calibrated_available=calibrated_available,
    )
    test_probabilities = probability_selection["selected_probabilities"]
    market_fusion = _market_fusion_report(test_labels, test_probabilities, test_rows, market_weight=market_weight)
    return {
        "name": str(config.get("name") or config.get("preset") or "candidate"),
        "preset": config.get("preset"),
        "top_k": top_k,
        "epochs": epochs,
        "weights": weights,
        "inject_elo": _config_inject_elo(config),
        "rating_mode": _config_rating_mode(config),
        "selected_feature_names": selected_train.feature_names,
        "excluded_feature_names": list(UNSTABLE_IDENTITY_FEATURES),
        "feature_preparation": prepared.get("feature_preparation", {}),
        "feature_selection": prepared.get("feature_selection", {}),
        "imbalance": rebalanced.report,
        "calibration": calibration_report,
        "probability_selection": {key: value for key, value in probability_selection.items() if key != "selected_probabilities"},
        "market_fusion": market_fusion,
        "rolling_validation": _rolling_validation_summary(rolling_prepared, config, seed=seed + 101),
        "validation_metrics": _metric_summary(validation_labels, validation_probabilities, validation_rows),
        "raw_test_metrics": _metric_summary(test_labels, raw_test_probabilities, test_rows),
        "test_metrics": _metric_summary(test_labels, test_probabilities, test_rows),
        "test_probabilities": test_probabilities,
    }


def _prepare_split_matrices(
    train_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    test_rows: Sequence[dict],
    top_k: int,
    feature_preparation: Dict[str, object] | None = None,
) -> Dict[str, object]:
    feature_preparation = feature_preparation or {
        "elo": {"basis": "not_applied", "rows": len(train_rows), "teams": 0},
        "excluded_feature_names": list(UNSTABLE_IDENTITY_FEATURES),
        "required_feature_names": list(PLAYER_STATUS_REQUIRED_FEATURES),
    }
    builder = FeatureBuilder()
    train_dataset = builder.fit_transform(train_rows)
    selector = FeatureSelector(
        top_k=top_k,
        excluded_feature_names=feature_preparation["excluded_feature_names"],
        required_feature_names=feature_preparation.get("required_feature_names", []),
    )
    selected_train = selector.fit_transform(train_dataset.rows, train_dataset.labels, train_dataset.feature_names)
    rebalanced = rebalance_training_data(selected_train.rows, train_dataset.labels)
    return {
        "selected_train": selected_train,
        "rebalanced": rebalanced,
        "feature_preparation": feature_preparation,
        "feature_selection": {
            "required_features": selector.required_feature_report,
        },
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "validation_x": selector.transform(builder.transform(validation_rows)).rows,
        "test_x": selector.transform(builder.transform(test_rows)).rows,
        "validation_labels": _labels(validation_rows),
        "test_labels": _labels(test_rows),
    }


def _prepare_rolling_fold_matrices(
    cleaned: Sequence[dict],
    top_k: int,
    rolling_folds: int,
    feature_preparation: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    prepared_folds: List[Dict[str, object]] = []
    for train_rows, validation_rows in time_series_folds(cleaned, folds=max(1, rolling_folds)):
        if not train_rows or not validation_rows:
            continue
        prepared_folds.append(
            _prepare_split_matrices(
                train_rows,
                validation_rows,
                [],
                top_k=top_k,
                feature_preparation=feature_preparation,
            )
        )
    return prepared_folds


def _rolling_validation_summary(
    rolling_prepared: Sequence[Mapping[str, object]],
    config: Mapping[str, Any],
    seed: int,
) -> Dict[str, object]:
    fold_rows = []
    for fold_number, prepared in enumerate(rolling_prepared, start=1):
        weights = dict(config.get("weights") or WEIGHT_PRESETS.get(str(config.get("preset", "no_nn")), WEIGHT_PRESETS["no_nn"]))
        prefer_accelerated = bool(config.get("prefer_accelerated", config.get("preset") != "fast_logistic"))
        rebalanced = prepared["rebalanced"]
        model = default_ensemble(
            seed=seed + fold_number,
            epochs=int(config.get("epochs", 8)),
            weights=weights,
            prefer_accelerated=prefer_accelerated,
            n_jobs=1,
        ).fit(rebalanced.rows, rebalanced.labels, sample_weights=rebalanced.sample_weights)
        probabilities = model.predict_proba(prepared["validation_x"])
        metrics = _metric_summary(prepared["validation_labels"], probabilities, prepared["validation_rows"])
        fold_rows.append(
            {
                "fold": fold_number,
                "train_count": len(rebalanced.rows),
                "validation_count": len(prepared["validation_rows"]),
                "metrics": metrics,
            }
        )
    return {
        "folds": len(fold_rows),
        "fold_metrics": fold_rows,
        "mean_metrics": _mean_metric_summary([row["metrics"] for row in fold_rows]),
        "worst_accuracy": min((float(row["metrics"]["accuracy"]) for row in fold_rows), default=0.0),
        "worst_log_loss": max((float(row["metrics"]["log_loss"]) for row in fold_rows), default=0.0),
    }


def _constant_metrics(rows: Sequence[dict], probability: float) -> Dict[str, float]:
    return _metric_summary(_labels(rows), [probability] * len(rows), rows)


def _metric_summary(labels: Sequence[int], probabilities: Sequence[float], rows: Sequence[dict]) -> Dict[str, float]:
    odds = [_picked_odds(row, probability) for row, probability in zip(rows, probabilities)]
    calibration = calibration_table(labels, probabilities) if labels else {"ece": 0.0}
    return {
        "accuracy": accuracy(labels, probabilities),
        "auc": auc(labels, probabilities),
        "log_loss": log_loss(labels, probabilities),
        "brier_score": brier_score(labels, probabilities),
        "ece": float(calibration["ece"]),
        "profit_loss": profit_loss(labels, probabilities, odds),
    }


def _mean_metric_summary(metric_rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {"accuracy": 0.0, "auc": 0.5, "log_loss": 0.0, "brier_score": 0.0, "ece": 0.0, "profit_loss": 0.0}
    keys = ("accuracy", "auc", "log_loss", "brier_score", "ece", "profit_loss")
    return {key: sum(float(row.get(key, 0.0)) for row in metric_rows) / len(metric_rows) for key in keys}


def _probability_selection(
    validation_labels: Sequence[int],
    validation_raw_probabilities: Sequence[float],
    validation_calibrated_probabilities: Sequence[float],
    validation_rows: Sequence[dict],
    test_raw_probabilities: Sequence[float],
    test_calibrated_probabilities: Sequence[float],
    test_labels: Sequence[int],
    test_rows: Sequence[dict],
    objective: str,
    calibrated_available: bool,
) -> Dict[str, object]:
    objective = objective if objective in {"accuracy", "log_loss", "brier_score", "ece"} else "log_loss"
    # Decide raw-vs-calibrated purely on the validation split: test labels must
    # never influence which probability basis is selected.
    validation_raw_metrics = _metric_summary(validation_labels, validation_raw_probabilities, validation_rows)
    validation_calibrated_metrics = _metric_summary(validation_labels, validation_calibrated_probabilities, validation_rows)
    if not calibrated_available:
        selected_basis = "raw_model"
    elif objective == "accuracy":
        selected_basis = (
            "calibrated_model"
            if validation_calibrated_metrics["accuracy"] > validation_raw_metrics["accuracy"]
            else "raw_model"
        )
    else:
        selected_basis = (
            "calibrated_model"
            if validation_calibrated_metrics[objective] < validation_raw_metrics[objective]
            else "raw_model"
        )
    use_calibrated = selected_basis == "calibrated_model"
    selected_probabilities = list(test_calibrated_probabilities if use_calibrated else test_raw_probabilities)
    # Test metrics are reported for transparency/audit but are NOT used to choose
    # the basis above.
    raw_test_metrics = _metric_summary(test_labels, test_raw_probabilities, test_rows)
    calibrated_test_metrics = _metric_summary(test_labels, test_calibrated_probabilities, test_rows)
    return {
        "objective": objective,
        "selected_basis": selected_basis,
        "selection_basis": "validation_only",
        "validation_raw_metrics": validation_raw_metrics,
        "validation_calibrated_metrics": validation_calibrated_metrics,
        "raw_test_metrics": raw_test_metrics,
        "calibrated_test_metrics": calibrated_test_metrics,
        "selected_probabilities": selected_probabilities,
    }


def _market_fusion_report(
    labels: Sequence[int],
    model_probabilities: Sequence[float],
    rows: Sequence[dict],
    market_weight: float,
) -> Dict[str, object]:
    market_weight = max(0.0, min(1.0, float(market_weight)))
    signals = [(index, market_probability_from_row(row)) for index, row in enumerate(rows)]
    market_signals = [(index, signal) for index, signal in signals if signal is not None]
    if not market_signals:
        return {
            "market_weight": market_weight,
            "test_rows_with_market": 0,
            "proxy_rows": 0,
            "signal_counts": {},
            "market_only_test_metrics": {},
            "fused_test_metrics": _metric_summary(labels, model_probabilities, rows),
        }
    market_indexes = [index for index, _ in market_signals]
    market_probabilities = [float(signal["probability_team1"]) for _, signal in market_signals]
    model_subset = [model_probabilities[index] for index in market_indexes]
    fused_subset = [
        _clip_probability((1.0 - market_weight) * model_probability + market_weight * market_probability)
        for model_probability, market_probability in zip(model_subset, market_probabilities)
    ]
    subset_labels = [labels[index] for index in market_indexes]
    subset_rows = [rows[index] for index in market_indexes]
    signal_counts: Dict[str, int] = {}
    proxy_rows = 0
    sources: Dict[str, int] = {}
    for _, signal in market_signals:
        basis = str(signal.get("basis", "unknown"))
        signal_counts[basis] = signal_counts.get(basis, 0) + 1
        if signal.get("proxy"):
            proxy_rows += 1
        source = str(signal.get("source", basis))
        sources[source] = sources.get(source, 0) + 1
    return {
        "market_weight": market_weight,
        "test_rows_with_market": len(market_indexes),
        "proxy_rows": proxy_rows,
        "signal_counts": signal_counts,
        "source_counts": sources,
        "market_only_test_metrics": _metric_summary(subset_labels, market_probabilities, subset_rows),
        "model_subset_test_metrics": _metric_summary(subset_labels, model_subset, subset_rows),
        "fused_test_metrics": _metric_summary(subset_labels, fused_subset, subset_rows),
    }


def _best_summary(
    candidate: Mapping[str, Any],
    baseline_test: Mapping[str, float],
    selection_basis: str = "validation",
) -> Dict[str, object]:
    test_metrics = dict(candidate["test_metrics"])
    return {
        "name": candidate["name"],
        "preset": candidate.get("preset"),
        "selection_basis": selection_basis,
        "top_k": candidate["top_k"],
        "epochs": candidate["epochs"],
        "weights": candidate["weights"],
        "rating_mode": candidate.get("rating_mode", _DEFAULT_RATING_MODE),
        "inject_elo": candidate.get("inject_elo", True),
        "validation_metrics": candidate["validation_metrics"],
        "test_metrics": test_metrics,
        "accuracy_delta_vs_baseline": test_metrics["accuracy"] - float(baseline_test["accuracy"]),
        "log_loss_delta_vs_baseline": test_metrics["log_loss"] - float(baseline_test["log_loss"]),
        "selected_feature_names": candidate.get("selected_feature_names", []),
        "feature_preparation": candidate.get("feature_preparation", {}),
        "calibration": candidate.get("calibration", {}),
        "probability_selection": candidate.get("probability_selection", {}),
        "market_fusion": candidate.get("market_fusion", {}),
        "rolling_validation": candidate.get("rolling_validation", {}),
    }


def _prediction_rows(rows: Sequence[dict], probabilities: Sequence[float]) -> List[Dict[str, object]]:
    output = []
    for row, probability in zip(rows, probabilities):
        predicted = row.get("team1") if probability >= 0.5 else row.get("team2")
        output.append({
            "date": row.get("date"),
            "event": row.get("event"),
            "team1": row.get("team1"),
            "team2": row.get("team2"),
            "winner": row.get("winner"),
            "probability_team1": probability,
            "predicted_winner": predicted,
            "correct": predicted == row.get("winner"),
        })
    return output


def _labels(rows: Sequence[dict]) -> List[int]:
    return [1 if row.get("winner") == row.get("team1") else 0 for row in rows]


def _picked_odds(row: dict, probability: float) -> float:
    key = "odds_team1" if probability >= 0.5 else "odds_team2"
    try:
        return float(row.get(key, 2.0) or 2.0)
    except (TypeError, ValueError):
        return 2.0


def _clip_probability(value: float) -> float:
    return max(0.000001, min(0.999999, float(value)))
