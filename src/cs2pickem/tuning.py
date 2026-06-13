from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .calibration import ProbabilityCalibrator, make_calibrator
from .cleaning import clean_matches
from .evaluation import accuracy, auc, brier_score, calibration_table, log_loss, profit_loss
from .features import FeatureBuilder
from .imbalance import rebalance_training_data
from .models import default_ensemble
from .odds import DEFAULT_DEVIG_METHOD, market_probability_from_row, normalize_devig_method
from .reliability import PLAYER_STATUS_REQUIRED_FEATURES, UNSTABLE_IDENTITY_FEATURES, prepare_reliability_features
from .selection import FeatureSelector
from .splitting import time_series_folds, time_series_split
from .strategy import (
    DEFAULT_MODEL_WEIGHT,
    adjust_probability_toward_market_probability,
)

# Tuning reports the default (linear-blend) path under the canonical name
# 'legacy'; strategy's own 'legacy_clip' is accepted as an alias by the resolver.
# 'logit_pool' is the opt-in geometric/log-odds pool. The real legacy-vs-logit_pool
# A/B adjudication is deferred to WF-2F; this stage only wires the switch.
_DEFAULT_TUNING_FUSION_METHOD = "legacy"


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
    inject_bt_modes: Sequence[bool] | None = None,
    calibration_methods: Sequence[str] | None = None,
    fusion_method: str = _DEFAULT_TUNING_FUSION_METHOD,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
    include_unverified_features: bool = False,
    devig_method: str = DEFAULT_DEVIG_METHOD,
    calibration_method_modes: Sequence[str] | None = None,
    fusion_method_modes: Sequence[str] | None = None,
    model_weight_values: Sequence[float] | None = None,
    include_unverified_modes: Sequence[bool] | None = None,
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
            or _candidate_grid(
                top_k_values,
                epochs_values,
                candidate_names,
                elo_modes=elo_modes,
                rating_modes=rating_modes,
                inject_bt_modes=inject_bt_modes,
                calibration_method_modes=calibration_method_modes,
                fusion_method_modes=fusion_method_modes,
                model_weight_values=model_weight_values,
                include_unverified_modes=include_unverified_modes,
            )
        )
    ]
    baseline_validation = _constant_metrics(split.validation, 0.5)
    baseline_test = _constant_metrics(split.test, 0.5)
    # Rows depend on which Elo gets injected, which rating engine is selected, AND whether the
    # Bradley-Terry strength prior is injected, so the split / feature-prep caches are keyed on
    # (inject_elo, rating_mode, inject_bt). The default ('elo', no BT) keeps
    # inject_glicko=inject_bt=False -> rows are bit-identical to the historic path. Folding
    # inject_bt into the key is load-bearing: a BT-on candidate must NOT reuse BT-off rows, or
    # bt_strength_diff would be selected on (non-zero) training rows yet scored constant-0.
    #
    # The matrix caches additionally fold in include_unverified: the FeatureBuilder feature set
    # (and therefore the selected matrices) differs when the gated/unverified columns are opted
    # in, so an unverified-on candidate must NOT reuse the unverified-off matrices (same anti-skew
    # reasoning as inject_bt). The default include_unverified=False keeps the matrices byte-identical.
    default_rating_mode = _DEFAULT_RATING_MODE
    split_cache: Dict[tuple[bool, str, bool], object] = {(True, default_rating_mode, False): split}
    feature_preparation_cache: Dict[tuple[bool, str, bool], Dict[str, object]] = {(True, default_rating_mode, False): feature_preparation}
    prepared_cache: Dict[tuple[int, bool, str, bool, bool], Dict[str, object]] = {}
    rolling_cache: Dict[tuple[int, bool, str, bool, bool], List[Dict[str, object]]] = {}
    candidate_results = []
    for index, config in enumerate(configs):
        top_k = int(config.get("top_k", 25))
        inject_elo = _config_inject_elo(config)
        rating_mode = _config_rating_mode(config)
        inject_bt = _config_inject_bt(config)
        include_unverified = _config_include_unverified(config)
        rating_key = (inject_elo, rating_mode, inject_bt)
        if rating_key not in split_cache:
            prepared_cleaned, _, candidate_feature_preparation = prepare_reliability_features(
                raw_cleaned,
                inject_elo=inject_elo,
                inject_glicko=(rating_mode == "glicko"),
                inject_bt=inject_bt,
            )
            split_cache[rating_key] = time_series_split(prepared_cleaned, train_ratio=train_ratio, validation_ratio=validation_ratio)
            feature_preparation_cache[rating_key] = candidate_feature_preparation
        candidate_split = split_cache[rating_key]
        cache_key = (top_k, inject_elo, rating_mode, inject_bt, include_unverified)
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = _prepare_split_matrices(
                candidate_split.train,
                candidate_split.validation,
                candidate_split.test,
                top_k=top_k,
                feature_preparation=feature_preparation_cache[rating_key],
                include_unverified_features=include_unverified,
            )
        if cache_key not in rolling_cache:
            rolling_cache[cache_key] = _prepare_rolling_fold_matrices(
                candidate_split.train + candidate_split.validation + candidate_split.test,
                top_k=top_k,
                rolling_folds=rolling_folds,
                feature_preparation=feature_preparation_cache[rating_key],
                include_unverified_features=include_unverified,
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
                calibration_methods=_config_calibration_methods(config, calibration_methods),
                fusion_method=_resolve_fusion_method(_config_fusion_method(config, fusion_method)),
                model_weight=_config_model_weight(config, model_weight),
                devig_method=_config_devig_method(config, devig_method),
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
        "fusion_method": _resolve_fusion_method(fusion_method),
        "model_weight": float(model_weight),
        "probability_objective": probability_objective,
        "calibration_methods": _normalize_calibration_methods(calibration_methods),
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
    inject_bt_modes: Sequence[bool] | None = None,
    calibration_method_modes: Sequence[str] | None = None,
    fusion_method_modes: Sequence[str] | None = None,
    model_weight_values: Sequence[float] | None = None,
    include_unverified_modes: Sequence[bool] | None = None,
) -> List[Dict[str, object]]:
    top_values = [int(value) for value in (top_k_values or [12, 18, 25])]
    epoch_values = [int(value) for value in (epochs_values or [8])]
    names = [str(name) for name in (candidate_names or ["fast_logistic", "random_forest", "no_nn"])]
    modes = [_normalize_elo_mode(mode) for mode in (elo_modes or ["with"])]
    rating_mode_values = [_normalize_rating_mode(mode) for mode in (rating_modes or [_DEFAULT_RATING_MODE])]
    # BT axis defaults to OFF-only so the default grid is byte-identical (no BT candidates, no
    # name suffix). Pass [False, True] to A/B the Bradley-Terry strength prior. As with the
    # rating-mode suffix, the BT-on variant gets a disambiguating '_bt' suffix whenever it is
    # non-default so its names never collide with the BT-off baseline in reports.
    bt_mode_values = [bool(value) for value in (inject_bt_modes or [False])]
    # Probability-quality axes (WF-2F same-口径 A/B). Each defaults to the single
    # production value so the default grid is byte-identical (no extra candidates, no
    # name suffix). Passing a multi-value list (or a single non-default value) fans out
    # the grid AND attaches a disambiguating suffix so the variant never collides with
    # the byte-identical default name in reports -- mirroring the elo/rating/bt axes.
    #   * calibration_method: fixes the candidate's single calibration method (one of
    #     platt/beta/temperature) so {platt vs beta vs temperature} can be compared
    #     same-口径 within one backtest run. (The default 'platt' keeps the historic
    #     single-method validation-selection path unchanged.)
    #   * fusion_method + model_weight: pick the market-fusion blend (legacy_clip vs
    #     logit_pool) and the logit-pool model-weight prior.
    #   * include_unverified: opts the gated/unverified feature columns back into the
    #     selector's candidate pool (5E event-grade, odds audit, weak-prior magnitudes).
    calibration_mode_values = [_normalize_calibration_method(mode) for mode in (calibration_method_modes or ["platt"])]
    fusion_mode_values = [_resolve_fusion_method(mode) for mode in (fusion_method_modes or [_DEFAULT_TUNING_FUSION_METHOD])]
    model_weight_grid = [float(value) for value in (model_weight_values or [DEFAULT_MODEL_WEIGHT])]
    unverified_mode_values = [bool(value) for value in (include_unverified_modes or [False])]
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
                        for inject_bt in bt_mode_values:
                            bt_suffix = "_bt" if (len(bt_mode_values) > 1 or inject_bt) and inject_bt else ""
                            for calibration_method in calibration_mode_values:
                                cal_suffix = (
                                    f"_{calibration_method}"
                                    if len(calibration_mode_values) > 1 or calibration_method != "platt"
                                    else ""
                                )
                                for fusion_method in fusion_mode_values:
                                    for model_weight in model_weight_grid:
                                        fusion_suffix = _fusion_grid_suffix(
                                            fusion_method,
                                            model_weight,
                                            fusion_mode_values,
                                            model_weight_grid,
                                        )
                                        for include_unverified in unverified_mode_values:
                                            unv_suffix = (
                                                "_unv"
                                                if (len(unverified_mode_values) > 1 or include_unverified)
                                                and include_unverified
                                                else ""
                                            )
                                            configs.append({
                                                "name": (
                                                    f"{name}{mode_suffix}{rating_suffix}{bt_suffix}"
                                                    f"{cal_suffix}{fusion_suffix}{unv_suffix}_k{top_k}_e{epochs}"
                                                ),
                                                "preset": name,
                                                "top_k": top_k,
                                                "epochs": epochs,
                                                "weights": dict(weights),
                                                "inject_elo": mode == "with",
                                                "rating_mode": rating_mode,
                                                "inject_bt": inject_bt,
                                                "calibration_methods": [calibration_method],
                                                "fusion_method": fusion_method,
                                                "model_weight": model_weight,
                                                "include_unverified_features": include_unverified,
                                            })
    return configs


def _fusion_grid_suffix(
    fusion_method: str,
    model_weight: float,
    fusion_mode_values: Sequence[str],
    model_weight_grid: Sequence[float],
) -> str:
    """Disambiguating name suffix for the market-fusion grid axis.

    Stays empty for the byte-identical default (single legacy method, single default
    model_weight) so existing candidate names are unchanged. A non-default fusion method
    or a fanned-out model_weight grid gets a ``_<method>`` and/or ``_mw<weight>`` tag so
    variants never collide in reports.
    """
    parts = ""
    if len(fusion_mode_values) > 1 or fusion_method != _DEFAULT_TUNING_FUSION_METHOD:
        parts += f"_{fusion_method}"
    if len(model_weight_grid) > 1:
        parts += f"_mw{model_weight:g}".replace(".", "")
    return parts


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


# Orthogonal Bradley-Terry strength-prior switch. It is deliberately NOT folded into
# RATING_MODES (which stays {elo, glicko}): BT can ride alongside either rating engine, so a
# separate inject_bt config key keeps the rating-engine axis and the BT axis independent. The
# WF-2C/D backtest never injected BT (inject_bt was hard-defaulted False), leaving
# bt_strength_diff / bt_map_strength_diff as constant-0 dead columns in every candidate. With
# this switch a candidate can opt BT in so the selector actually competes those columns. It
# defaults False so the existing hot path is bit-for-bit unchanged, and -- crucially -- it is
# folded into the rating cache key so a BT-on candidate never reuses BT-off rows (which would
# silently feed it constant-0 BT columns = the very train/serve skew we are guarding against).
def _config_inject_bt(config: Mapping[str, Any]) -> bool:
    return bool(config.get("inject_bt", False))


# Probability-quality / feature axes resolved per-candidate (config overrides the top-level
# default). Keeping the top-level default as the fallback means an explicit candidate_configs
# list (which omits these keys) inherits the run-wide setting exactly as before -- and the grid
# path, which now writes these keys explicitly, drives them per candidate. All default to the
# production-safe value so the default grid stays byte-identical.
def _config_include_unverified(config: Mapping[str, Any]) -> bool:
    return bool(config.get("include_unverified_features", False))


def _config_calibration_methods(
    config: Mapping[str, Any], default_methods: Sequence[str] | None
) -> List[str]:
    """Resolve the calibration methods for one candidate.

    A grid candidate PINS a single method (``calibration_methods=[m]``) so the
    {platt vs beta vs temperature} A/B is same-口径: that one method is used verbatim,
    WITHOUT the run-wide ``_normalize_calibration_methods`` platt-prepend, so a 'beta'
    candidate scores beta-only and never silently re-competes platt. Explicit
    ``candidate_configs`` that omit the key inherit the run-wide default list (which keeps
    the historic platt-first normalization for back-compat). A pinned multi-element list
    still flows through normalization so an opt-in multi-method candidate behaves as before.
    """
    pinned = config.get("calibration_methods")
    if pinned not in (None, ""):
        materialized = [str(method).strip().lower() for method in pinned]
        if len(materialized) == 1:
            return [_normalize_calibration_method(materialized[0])]
        return _normalize_calibration_methods(materialized)
    return _normalize_calibration_methods(default_methods)


def _config_fusion_method(config: Mapping[str, Any], default_method: str) -> str:
    value = config.get("fusion_method")
    return str(value) if value not in (None, "") else default_method


def _config_model_weight(config: Mapping[str, Any], default_weight: float) -> float:
    value = config.get("model_weight")
    if value in (None, ""):
        return float(default_weight)
    return float(value)


def _config_devig_method(config: Mapping[str, Any], default_method: str) -> str:
    """Resolve the de-vig method for one candidate (config overrides run-wide default).

    The de-vig method only affects how raw two-way odds are turned into a fair market
    probability inside the market-fusion report, so it is purely an odds-subset axis. It
    defaults to ``multiplicative`` (the historic behaviour), keeping the no-odds hot path
    and every existing market-fusion number byte-identical until a candidate opts in.
    """
    value = config.get("devig_method")
    return normalize_devig_method(str(value) if value not in (None, "") else default_method)


CALIBRATION_METHODS = ("platt", "beta", "temperature")


def _normalize_calibration_method(method: str | None) -> str:
    """Validate / normalise a single calibration method for the grid axis.

    Unlike ``_normalize_calibration_methods`` (which always keeps platt first as the
    headline block), this returns exactly the one requested method so a grid candidate
    can be pinned to platt, beta, OR temperature for a same-口径 A/B. Defaults to platt.
    """
    normalized = str(method or "platt").strip().lower()
    if normalized not in CALIBRATION_METHODS:
        raise ValueError(f"unknown calibration method: {method!r}")
    return normalized


def _resolve_evaluation_methods(methods: Sequence[str] | None) -> List[str]:
    """Resolve the per-candidate calibration method list at evaluation time.

    A single pinned method is returned verbatim (only validated) so the grid's same-口径
    {platt|beta|temperature} A/B uses exactly that one calibrator. Empty / multi-element
    requests fall back to ``_normalize_calibration_methods`` (platt-first), preserving the
    historic default and the opt-in multi-method validation-selection path.
    """
    materialized = [str(method).strip().lower() for method in (methods or [])]
    if len(materialized) == 1:
        return [_normalize_calibration_method(materialized[0])]
    return _normalize_calibration_methods(materialized)


def _normalize_calibration_methods(methods: Sequence[str] | None) -> List[str]:
    """De-duplicate / validate the requested calibration methods, platt first.

    Platt is always evaluated and kept first so its report stays the headline
    'calibration' block (back-compat). Unknown methods raise; an empty / falsy
    request collapses to the default platt-only behaviour.
    """
    requested = [str(method).strip().lower() for method in (methods or ["platt"])]
    ordered: List[str] = ["platt"]
    for method in requested:
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"unknown calibration method: {method!r}")
        if method not in ordered:
            ordered.append(method)
    return ordered


_METHOD_SELECTION_CV_FOLDS = 3


def _out_of_fold_validation_probabilities(
    method: str,
    validation_probabilities: Sequence[float],
    validation_labels: Sequence[int],
) -> List[float] | None:
    """Expanding-window out-of-fold calibrated probabilities on the validation set.

    Method *selection* must not reward a calibrator for fitting the very rows it
    is then scored on -- an in-sample comparison systematically prefers the
    highest-DOF method (beta) regardless of generalisation (review red-line (b):
    choose by held-out, not in-sample, validation quality). For each expanding
    fold we fit the calibrator ONLY on the rows before the fold and predict the
    fold rows it has never seen; the concatenated out-of-fold predictions give an
    honest validation score. The calibrator never sees test labels here either.

    Returns ``None`` when there are too few rows to carve at least two folds, so
    the caller can fall back to the (full-fit) in-sample metric for that method.
    """
    n = len(validation_labels)
    folds = _METHOD_SELECTION_CV_FOLDS
    if n < folds + 1:
        return None
    fold_size = max(1, n // (folds + 1))
    predictions: List[float] = [float(p) for p in validation_probabilities]
    covered = False
    for fold_index in range(1, folds + 1):
        train_end = fold_size * fold_index
        validation_end = min(n, train_end + fold_size)
        if train_end <= 0 or validation_end <= train_end:
            break
        train_probs = list(validation_probabilities[:train_end])
        train_labels = list(validation_labels[:train_end])
        if not train_probs:
            continue
        calibrator = make_calibrator(method).fit(train_probs, train_labels)
        held = calibrator.transform(list(validation_probabilities[train_end:validation_end]))
        for offset, value in enumerate(held):
            predictions[train_end + offset] = value
        covered = True
    return predictions if covered else None


def _resolve_fusion_method(fusion_method: str | None) -> str:
    """Validate / normalise the market-fusion method (defaults to legacy).

    The real A/B between ``legacy`` (the current convex linear blend, behaviour
    unchanged) and ``logit_pool`` (the geometric/log-odds pool with the frozen
    ``model_weight`` prior) is deferred to WF-2F's multi-season adjudication; this
    stage only wires the switch and reports the resolved method. ``'legacy'`` and
    strategy's ``'legacy_clip'`` are accepted as aliases for the default path,
    which tuning reports under the canonical name ``'legacy'``.
    """
    resolved = str(fusion_method or _DEFAULT_TUNING_FUSION_METHOD).strip().lower()
    if resolved in {"legacy", "legacy_clip"}:
        return _DEFAULT_TUNING_FUSION_METHOD
    if resolved == "logit_pool":
        return "logit_pool"
    raise ValueError(
        f"unknown fusion method: {fusion_method!r}; expected one of "
        f"{('legacy', 'logit_pool')}"
    )


def _evaluate_candidate(
    prepared: Mapping[str, object],
    config: Mapping[str, Any],
    seed: int,
    calibrate: bool,
    rolling_prepared: Sequence[Mapping[str, object]],
    market_weight: float,
    probability_objective: str,
    calibration_methods: Sequence[str] = ("platt",),
    fusion_method: str = _DEFAULT_TUNING_FUSION_METHOD,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
    devig_method: str = DEFAULT_DEVIG_METHOD,
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
    # Resolve the requested calibration methods. A single pinned method (e.g. the grid's
    # {platt|beta|temperature} same-口径 A/B) is honoured VERBATIM -- it is not run through
    # the platt-prepend normalization, so a beta candidate scores beta-only and never
    # silently re-competes platt. Any multi-element request keeps the historic platt-first
    # normalization (headline 'calibration' block stays back-compat for existing readers).
    methods = _resolve_evaluation_methods(calibration_methods)
    method_candidates: list[dict] = []
    multi_method = len(methods) > 1
    if calibrated_available:
        for method in methods:
            # The calibrator is fit ONLY on validation (never on test): no test
            # label ever participates in fitting or selecting the calibrator.
            calibrator = make_calibrator(method).fit(validation_probabilities, validation_labels)
            method_validation_probabilities = calibrator.transform(validation_probabilities)
            method_test_probabilities = calibrator.transform(raw_test_probabilities)
            report = calibrator.report()
            report["basis"] = f"validation_{method}" if method != "platt" else "validation_platt_logistic"
            # Selection metric: when multiple methods compete, score them on
            # EXPANDING-WINDOW OUT-OF-FOLD validation predictions so the choice is
            # not biased toward the highest-DOF method by in-sample over-fit
            # (review red-line (b)). The single (platt) path keeps the historic
            # in-sample validation metric so default behaviour is unchanged.
            selection_probabilities = method_validation_probabilities
            selection_basis = "validation_in_sample"
            if multi_method:
                out_of_fold = _out_of_fold_validation_probabilities(
                    method, validation_probabilities, validation_labels
                )
                if out_of_fold is not None:
                    selection_probabilities = out_of_fold
                    selection_basis = "validation_out_of_fold"
            method_candidates.append(
                {
                    "method": method,
                    "basis": f"calibrated_{method}",
                    "report": report,
                    "validation_probabilities": method_validation_probabilities,
                    "validation_metrics": _metric_summary(validation_labels, selection_probabilities, validation_rows),
                    "selection_metric_basis": selection_basis,
                    "test_probabilities": method_test_probabilities,
                }
            )
        # The headline calibration block / the legacy calibrated-on-test arrays
        # track the first (platt) method so existing reports are unchanged.
        primary = method_candidates[0]
        calibration_report = dict(primary["report"])
        calibrated_test_probabilities = list(primary["test_probabilities"])
        calibrated_validation_probabilities = list(primary["validation_probabilities"])
    # The raw-vs-calibrated decision is made ONLY on the validation split so that
    # no test label ever participates in model/probability selection. The chosen
    # basis is then applied to the test probabilities; test metrics are reported
    # for both bases but never drive the choice. With a single (platt) method the
    # method_candidates path reduces exactly to the historic raw-vs-platt pick.
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
        method_candidates=method_candidates if (calibrated_available and len(methods) > 1) else None,
    )
    test_probabilities = probability_selection["selected_probabilities"]
    market_fusion = _market_fusion_report(
        test_labels,
        test_probabilities,
        test_rows,
        market_weight=market_weight,
        fusion_method=fusion_method,
        model_weight=model_weight,
        devig_method=devig_method,
    )
    return {
        "name": str(config.get("name") or config.get("preset") or "candidate"),
        "preset": config.get("preset"),
        "top_k": top_k,
        "epochs": epochs,
        "weights": weights,
        "inject_elo": _config_inject_elo(config),
        "rating_mode": _config_rating_mode(config),
        "inject_bt": _config_inject_bt(config),
        "include_unverified_features": _config_include_unverified(config),
        "calibration_methods": list(methods),
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
    include_unverified_features: bool = False,
) -> Dict[str, object]:
    feature_preparation = feature_preparation or {
        "elo": {"basis": "not_applied", "rows": len(train_rows), "teams": 0},
        "excluded_feature_names": list(UNSTABLE_IDENTITY_FEATURES),
        "required_feature_names": list(PLAYER_STATUS_REQUIRED_FEATURES),
    }
    # One builder per (split, include_unverified) -> the SAME feature set is fit on train and
    # applied to validation/test, so the candidate never sees a column at fit time that is
    # absent at score time. include_unverified opts the gated columns into the candidate pool.
    builder = FeatureBuilder(include_unverified_features=include_unverified_features)
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
    include_unverified_features: bool = False,
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
                include_unverified_features=include_unverified_features,
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


def _objective_is_better(objective: str, candidate: Mapping[str, float], incumbent: Mapping[str, float]) -> bool:
    """Lower-is-better for log_loss/brier_score/ece, higher-is-better for accuracy."""
    if objective == "accuracy":
        return candidate["accuracy"] > incumbent["accuracy"]
    return candidate[objective] < incumbent[objective]


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
    method_candidates: "Sequence[Mapping[str, object]] | None" = None,
) -> Dict[str, object]:
    objective = objective if objective in {"accuracy", "log_loss", "brier_score", "ece"} else "log_loss"
    # Decide the probability basis purely on the validation split: test labels
    # must never influence which basis is selected. Per the review red-line,
    # tuning chooses among {raw, platt, beta, temperature} on VALIDATION only.
    validation_raw_metrics = _metric_summary(validation_labels, validation_raw_probabilities, validation_rows)
    validation_calibrated_metrics = _metric_summary(validation_labels, validation_calibrated_probabilities, validation_rows)
    raw_test_metrics = _metric_summary(test_labels, test_raw_probabilities, test_rows)
    calibrated_test_metrics = _metric_summary(test_labels, test_calibrated_probabilities, test_rows)

    # Build the candidate pool. The default single-calibrator path keeps the
    # historic 'raw_model' / 'calibrated_model' basis labels byte-identical; a
    # method_candidates list (one entry per fitted calibration method) opts into
    # the multi-method validation-based selection. 'raw_model' is always present.
    pool: list[dict] = [
        {
            "basis": "raw_model",
            "method": "raw",
            "validation_metrics": validation_raw_metrics,
            "test_probabilities": list(test_raw_probabilities),
        }
    ]
    if method_candidates:
        for candidate in method_candidates:
            pool.append(
                {
                    "basis": str(candidate.get("basis", f"calibrated_{candidate.get('method', 'platt')}")),
                    "method": str(candidate.get("method", "platt")),
                    "validation_metrics": dict(candidate["validation_metrics"]),
                    "test_probabilities": list(candidate["test_probabilities"]),
                }
            )
    elif calibrated_available:
        pool.append(
            {
                "basis": "calibrated_model",
                "method": "platt",
                "validation_metrics": validation_calibrated_metrics,
                "test_probabilities": list(test_calibrated_probabilities),
            }
        )

    # Greedy pick over the validation objective; ties keep the earlier (rawer)
    # candidate so an uninformative calibrator never displaces the raw model.
    chosen = pool[0]
    for candidate in pool[1:]:
        if _objective_is_better(objective, candidate["validation_metrics"], chosen["validation_metrics"]):
            chosen = candidate
    selected_basis = chosen["basis"]
    selected_probabilities = list(chosen["test_probabilities"])

    result = {
        "objective": objective,
        "selected_basis": selected_basis,
        "selected_method": chosen["method"],
        "selection_basis": "validation_only",
        "validation_raw_metrics": validation_raw_metrics,
        "validation_calibrated_metrics": validation_calibrated_metrics,
        "raw_test_metrics": raw_test_metrics,
        "calibrated_test_metrics": calibrated_test_metrics,
        "selected_probabilities": selected_probabilities,
    }
    if method_candidates:
        result["method_validation_metrics"] = {
            candidate["method"]: candidate["validation_metrics"] for candidate in pool
        }
    return result


def _market_fusion_report(
    labels: Sequence[int],
    model_probabilities: Sequence[float],
    rows: Sequence[dict],
    market_weight: float,
    fusion_method: str = _DEFAULT_TUNING_FUSION_METHOD,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
    devig_method: str = DEFAULT_DEVIG_METHOD,
) -> Dict[str, object]:
    market_weight = max(0.0, min(1.0, float(market_weight)))
    fusion_method = _resolve_fusion_method(fusion_method)
    devig_method = normalize_devig_method(devig_method)
    signals = [(index, market_probability_from_row(row, method=devig_method)) for index, row in enumerate(rows)]
    market_signals = [(index, signal) for index, signal in signals if signal is not None]
    if not market_signals:
        return {
            "market_weight": market_weight,
            "fusion_method": fusion_method,
            "model_weight": float(model_weight),
            "devig_method": devig_method,
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
        _fuse_probability(
            model_probability,
            market_probability,
            market_weight=market_weight,
            fusion_method=fusion_method,
            model_weight=model_weight,
        )
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
        "fusion_method": fusion_method,
        "model_weight": float(model_weight),
        "devig_method": devig_method,
        "test_rows_with_market": len(market_indexes),
        "proxy_rows": proxy_rows,
        "signal_counts": signal_counts,
        "source_counts": sources,
        "market_only_test_metrics": _metric_summary(subset_labels, market_probabilities, subset_rows),
        "model_subset_test_metrics": _metric_summary(subset_labels, model_subset, subset_rows),
        "fused_test_metrics": _metric_summary(subset_labels, fused_subset, subset_rows),
    }


def _fuse_probability(
    model_probability: float,
    market_probability: float,
    market_weight: float,
    fusion_method: str,
    model_weight: float,
) -> float:
    """Blend one model/market probability per the resolved ``fusion_method``.

    ``legacy`` keeps the historic convex linear blend
    ``(1 - market_weight)*p_model + market_weight*p_market`` byte-identical (so the
    existing market_weight-passthrough tests stay green). ``logit_pool`` delegates
    to strategy's logarithmic opinion pool with the frozen ``model_weight`` prior
    (``market_weight`` is ignored on that path, by design).
    """
    if fusion_method == "logit_pool":
        return adjust_probability_toward_market_probability(
            model_probability,
            market_probability,
            fusion_method="logit_pool",
            model_weight=model_weight,
        )
    return _clip_probability((1.0 - market_weight) * model_probability + market_weight * market_probability)


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
        "inject_bt": candidate.get("inject_bt", False),
        "include_unverified_features": candidate.get("include_unverified_features", False),
        "calibration_methods": candidate.get("calibration_methods", ["platt"]),
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
