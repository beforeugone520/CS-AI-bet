from __future__ import annotations

from datetime import date
from typing import Callable, Dict, List, Sequence, Tuple

from .cleaning import clean_matches
from .data import read_matches_csv, write_json, write_matches_csv
from .enrichment import build_team_profiles, enrich_match_history
from .evaluation import accuracy, auc, log_loss, profit_loss
from .features import FeatureBuilder
from .imbalance import rebalance_training_data
from .models import default_ensemble, model_hyperparameters
from .selection import FeatureSelector
from .splitting import time_series_date_split, time_series_folds, time_series_split
from .strategy import choose_pickems
from .swiss import TeamSeed, simulate_swiss


def run_demo() -> Dict[str, object]:
    matches = _demo_matches()
    cleaned = clean_matches(matches, reference_date="2026-05-31")
    builder = FeatureBuilder()
    dataset = builder.fit_transform(cleaned)
    model = default_ensemble(seed=11, epochs=8).fit(dataset.rows, dataset.labels)
    team_strength = _team_strength_from_matches(cleaned)
    teams = [TeamSeed(name, seed) for seed, name in enumerate(sorted(team_strength, key=team_strength.get, reverse=True), start=1)]

    def predictor(team_a: TeamSeed, team_b: TeamSeed, best_of: int, state) -> float:
        strength_a = team_strength.get(team_a.name, 0.5)
        strength_b = team_strength.get(team_b.name, 0.5)
        base = 0.5 + (strength_a - strength_b) * (0.22 if best_of == 1 else 0.28)
        return min(0.92, max(0.08, base))

    simulation = simulate_swiss(teams, predictor, simulations=500, seed=17)
    rankings = {team.name: team.seed for team in teams}
    pickems = choose_pickems(simulation.team_probabilities, rankings=rankings, slots={"3-0": 1, "advance": 2, "0-3": 1})
    probabilities = model.predict_proba(dataset.rows)
    return {
        "cleaned_matches": len(cleaned),
        "feature_count": len(dataset.feature_names),
        "training_probabilities": probabilities,
        "team_probabilities": simulation.team_probabilities,
        "pickems": pickems,
    }


def train_evaluate(
    rows: List[dict],
    reference_date: str,
    epochs: int = 50,
    top_k: int = 25,
    cv_folds: int = 5,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    max_age_days: int = 90,
    train_end_date: str | None = None,
    validation_end_date: str | None = None,
) -> Dict[str, object]:
    cleaned = sorted(clean_matches(rows, reference_date=reference_date, max_age_days=max_age_days), key=lambda row: row["date"])
    if bool(train_end_date) != bool(validation_end_date):
        raise ValueError("both train_end_date and validation_end_date must be provided for calendar split")

    split_strategy = "date_boundaries" if train_end_date and validation_end_date else "ratio"
    if len(cleaned) < 6 and split_strategy != "date_boundaries":
        return _train_evaluate_in_sample(cleaned, epochs=epochs, top_k=top_k, max_age_days=max_age_days)

    if split_strategy == "date_boundaries":
        _validate_calendar_boundaries(str(train_end_date), str(validation_end_date))
        split = time_series_date_split(cleaned, train_end_date=str(train_end_date), validation_end_date=str(validation_end_date))
        _validate_non_empty_date_split(split)
    else:
        split = time_series_split(cleaned, train_ratio=train_ratio, validation_ratio=validation_ratio)
    prepared = _prepare_time_split(split.train, [split.validation, split.test], top_k=top_k)
    train_x, train_y = prepared["train_rows"], prepared["train_labels"]
    validation_x, validation_y = prepared["eval_rows"][0], _labels(split.validation)
    test_x, test_y = prepared["eval_rows"][1], _labels(split.test)

    ensemble = default_ensemble(seed=19, epochs=epochs).fit(train_x, train_y, sample_weights=prepared["sample_weights"])
    validation_probabilities = ensemble.predict_proba(validation_x) if validation_x else []
    test_probabilities = ensemble.predict_proba(test_x) if test_x else []
    primary_rows = split.test if split.test else split.validation
    primary_labels = test_y if split.test else validation_y
    primary_probabilities = test_probabilities if split.test else validation_probabilities
    ensemble_weights = _default_ensemble_weights()

    return {
        "cleaned_matches": len(cleaned),
        "max_age_days": max_age_days,
        "split_strategy": split_strategy,
        "split_boundaries": {
            "train_end_date": train_end_date,
            "validation_end_date": validation_end_date,
        },
        "split_counts": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "leakage_guard": _leakage_guard(split.train, split.validation, split.test),
        "feature_names": prepared["feature_names"],
        "selected_feature_names": prepared["selected_feature_names"],
        "feature_importance": prepared["feature_importance"],
        "imbalance": prepared["imbalance"],
        "ensemble_weights": ensemble_weights,
        "model_hyperparameters": model_hyperparameters(epochs=epochs),
        "validation_tuned_ensemble_weights": _validation_tuned_weights(
            train_x,
            train_y,
            validation_x,
            validation_y,
            epochs=epochs,
            sample_weights=prepared["sample_weights"],
            fallback_weights=ensemble_weights,
        ),
        "metrics": _metric_summary(primary_labels, primary_probabilities, primary_rows),
        "holdout_metrics": {
            "validation": _metric_summary(validation_y, validation_probabilities, split.validation),
            "test": _metric_summary(test_y, test_probabilities, split.test),
        },
        "segment_metrics": _segment_metrics(primary_labels, primary_probabilities, primary_rows),
        "cv_metrics": _cross_validate(cleaned, cv_folds=cv_folds, top_k=top_k, epochs=epochs),
        "model_comparison": _model_comparison(
            train_x,
            train_y,
            test_x or validation_x,
            test_y or validation_y,
            split.test or split.validation,
            epochs=epochs,
            sample_weights=prepared["sample_weights"],
        ),
        "probabilities": _probability_rows(primary_rows, primary_probabilities),
    }


def _validate_calendar_boundaries(train_end_date: str, validation_end_date: str) -> None:
    train_boundary = date.fromisoformat(train_end_date)
    validation_boundary = date.fromisoformat(validation_end_date)
    if train_boundary > validation_boundary:
        raise ValueError("train_end_date must be on or before validation_end_date")


def _validate_non_empty_date_split(split) -> None:
    empty_buckets = [
        name
        for name, rows in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        )
        if not rows
    ]
    if empty_buckets:
        counts = {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)}
        raise ValueError(
            "date boundary split produced empty bucket(s): "
            + ", ".join(empty_buckets)
            + f"; counts={counts}. Adjust boundaries or omit explicit date boundaries for compact sample data."
        )


def simulate_from_team_rows(team_rows: List[dict], simulations: int = 100000, seed: int = 13) -> Dict[str, object]:
    teams = [TeamSeed(str(row["team"]), int(row["seed"])) for row in team_rows]
    strengths = {str(row["team"]): float(row.get("strength", 0.5)) for row in team_rows}

    def predictor(team_a: TeamSeed, team_b: TeamSeed, best_of: int, state) -> float:
        edge = strengths.get(team_a.name, 0.5) - strengths.get(team_b.name, 0.5)
        multiplier = 0.35 if best_of == 3 else 0.28
        return min(0.94, max(0.06, 0.5 + edge * multiplier))

    simulation = simulate_swiss(teams, predictor, simulations=simulations, seed=seed)
    rankings = {team.name: team.seed for team in teams}
    return {
        "simulations": simulations,
        "team_probabilities": simulation.team_probabilities,
        "pickems": choose_pickems(simulation.team_probabilities, rankings=rankings),
    }


def enrich_matches_file(input_path: str, output_path: str, profiles_path: str | None = None) -> Dict[str, object]:
    raw_rows = read_matches_csv(input_path)
    enriched = enrich_match_history(raw_rows)
    profiles = build_team_profiles(raw_rows)
    write_matches_csv(output_path, enriched)
    if profiles_path:
        write_json(profiles_path, profiles)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "profiles_path": profiles_path,
        "rows": len(enriched),
        "profiles": len(profiles),
    }


def _train_evaluate_in_sample(cleaned: List[dict], epochs: int, top_k: int, max_age_days: int) -> Dict[str, object]:
    builder = FeatureBuilder()
    dataset = builder.fit_transform(cleaned)
    selector = FeatureSelector(top_k=top_k)
    selected = selector.fit_transform(dataset.rows, dataset.labels, dataset.feature_names)
    rebalanced = rebalance_training_data(selected.rows, dataset.labels)
    model = default_ensemble(seed=19, epochs=epochs).fit(rebalanced.rows, rebalanced.labels, sample_weights=rebalanced.sample_weights)
    probabilities = model.predict_proba(selected.rows)
    ensemble_weights = _default_ensemble_weights()
    return {
        "cleaned_matches": len(cleaned),
        "max_age_days": max_age_days,
        "split_strategy": "in_sample",
        "split_boundaries": {"train_end_date": None, "validation_end_date": None},
        "split_counts": {"train": len(cleaned), "validation": 0, "test": 0},
        "leakage_guard": _leakage_guard(cleaned, [], []),
        "feature_names": dataset.feature_names,
        "selected_feature_names": selected.feature_names,
        "feature_importance": selector.importance_scores,
        "imbalance": rebalanced.report,
        "ensemble_weights": ensemble_weights,
        "model_hyperparameters": model_hyperparameters(epochs=epochs),
        "validation_tuned_ensemble_weights": {
            "basis": "no_validation_rows",
            "validation_count": 0,
            "weights": ensemble_weights,
            "model_log_loss": {},
        },
        "metrics": _metric_summary(dataset.labels, probabilities, dataset.raw_rows),
        "holdout_metrics": {"validation": _metric_summary([], [], []), "test": _metric_summary([], [], [])},
        "segment_metrics": _segment_metrics(dataset.labels, probabilities, dataset.raw_rows),
        "cv_metrics": [],
        "model_comparison": _model_comparison(
            rebalanced.rows,
            rebalanced.labels,
            selected.rows,
            dataset.labels,
            dataset.raw_rows,
            epochs=epochs,
            sample_weights=rebalanced.sample_weights,
        ),
        "probabilities": _probability_rows(dataset.raw_rows, probabilities),
    }


def _prepare_time_split(train_rows: List[dict], eval_groups: Sequence[List[dict]], top_k: int) -> Dict[str, object]:
    builder = FeatureBuilder()
    train_dataset = builder.fit_transform(train_rows)
    selector = FeatureSelector(top_k=top_k)
    selected_train = selector.fit_transform(train_dataset.rows, train_dataset.labels, train_dataset.feature_names)
    rebalanced = rebalance_training_data(selected_train.rows, train_dataset.labels)
    selected_eval_rows = []
    for group in eval_groups:
        transformed = builder.transform(group)
        selected_eval_rows.append(selector.transform(transformed).rows)
    return {
        "train_rows": rebalanced.rows,
        "train_labels": rebalanced.labels,
        "sample_weights": rebalanced.sample_weights,
        "imbalance": rebalanced.report,
        "eval_rows": selected_eval_rows,
        "feature_names": train_dataset.feature_names,
        "selected_feature_names": selected_train.feature_names,
        "feature_importance": selector.importance_scores,
    }


def _cross_validate(cleaned: List[dict], cv_folds: int, top_k: int, epochs: int) -> List[Dict[str, object]]:
    metrics = []
    for fold_number, (train_rows, validation_rows) in enumerate(time_series_folds(cleaned, folds=cv_folds), start=1):
        prepared = _prepare_time_split(train_rows, [validation_rows], top_k=top_k)
        model = default_ensemble(seed=23 + fold_number, epochs=epochs).fit(
            prepared["train_rows"],
            prepared["train_labels"],
            sample_weights=prepared["sample_weights"],
        )
        validation_x = prepared["eval_rows"][0]
        probabilities = model.predict_proba(validation_x)
        metrics.append(
            {
                "fold": fold_number,
                "train_count": len(train_rows),
                "validation_count": len(validation_rows),
                "max_train_date": train_rows[-1]["date"],
                "min_validation_date": validation_rows[0]["date"],
                "imbalance": prepared["imbalance"],
                "metrics": _metric_summary(_labels(validation_rows), probabilities, validation_rows),
            }
        )
    return metrics


def _model_comparison(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    eval_x: Sequence[Sequence[float]],
    eval_y: Sequence[int],
    eval_rows: Sequence[dict],
    epochs: int,
    sample_weights: Sequence[float] | None = None,
) -> Dict[str, Dict[str, float]]:
    comparison: Dict[str, Dict[str, float]] = {}
    for name, factory in _model_factories(epochs=epochs).items():
        model = factory()
        model.fit(train_x, train_y, sample_weights=sample_weights)
        probabilities = model.predict_proba(eval_x)
        comparison[name] = _metric_summary(eval_y, probabilities, eval_rows)
    return comparison


def _model_factories(epochs: int) -> Dict[str, Callable[[], object]]:
    return {
        "logistic": lambda: default_ensemble(seed=31, epochs=epochs).models["logistic"],
        "random_forest": lambda: default_ensemble(seed=31, epochs=epochs).models["random_forest"],
        "xgboost": lambda: default_ensemble(seed=31, epochs=epochs).models["xgboost"],
        "neural_network": lambda: default_ensemble(seed=37, epochs=epochs).models["neural_network"],
        "ensemble": lambda: default_ensemble(seed=41, epochs=epochs),
    }


def _base_model_factories(epochs: int) -> Dict[str, Callable[[], object]]:
    return {name: factory for name, factory in _model_factories(epochs).items() if name != "ensemble"}


def _default_ensemble_weights() -> Dict[str, float]:
    return dict(default_ensemble().weights)


def _validation_tuned_weights(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[int],
    validation_x: Sequence[Sequence[float]],
    validation_y: Sequence[int],
    epochs: int,
    sample_weights: Sequence[float] | None,
    fallback_weights: Dict[str, float],
) -> Dict[str, object]:
    if not validation_x or not validation_y:
        return {
            "basis": "no_validation_rows",
            "validation_count": 0,
            "weights": dict(fallback_weights),
            "model_log_loss": {},
        }
    losses: Dict[str, float] = {}
    scores: Dict[str, float] = {}
    for name, factory in _base_model_factories(epochs).items():
        model = factory()
        model.fit(train_x, train_y, sample_weights=sample_weights)
        probabilities = model.predict_proba(validation_x)
        loss = log_loss(validation_y, probabilities)
        losses[name] = loss
        scores[name] = 1.0 / max(loss, 1e-6)
    total = sum(scores.values())
    weights = {name: score / total for name, score in scores.items()} if total > 0 else dict(fallback_weights)
    return {
        "basis": "validation_log_loss",
        "validation_count": len(validation_y),
        "weights": weights,
        "model_log_loss": losses,
    }


def _metric_summary(labels: Sequence[int], probabilities: Sequence[float], rows: Sequence[dict]) -> Dict[str, float]:
    odds = [_picked_odds(row, probability) for row, probability in zip(rows, probabilities)]
    return {
        "accuracy": accuracy(labels, probabilities),
        "auc": auc(labels, probabilities),
        "log_loss": log_loss(labels, probabilities),
        "profit_loss": profit_loss(labels, probabilities, odds),
    }


def _segment_metrics(labels: Sequence[int], probabilities: Sequence[float], rows: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    segments = {"BO1": [], "BO3": []}
    for index, row in enumerate(rows):
        key = "BO3" if int(row.get("best_of", 1)) == 3 else "BO1"
        segments[key].append(index)
    output: Dict[str, Dict[str, float]] = {}
    for key, indexes in segments.items():
        output[key] = _metric_summary(
            [labels[index] for index in indexes],
            [probabilities[index] for index in indexes],
            [rows[index] for index in indexes],
        )
    return output


def _labels(rows: Sequence[dict]) -> List[int]:
    return [1 if row.get("winner") == row.get("team1") else 0 for row in rows]


def _probability_rows(rows: Sequence[dict], probabilities: Sequence[float]) -> List[Dict[str, object]]:
    return [
        {
            "date": row.get("date"),
            "team1": row.get("team1"),
            "team2": row.get("team2"),
            "best_of": row.get("best_of"),
            "winner_probability_team1": probability,
        }
        for row, probability in zip(rows, probabilities)
    ]


def _leakage_guard(train_rows: Sequence[dict], validation_rows: Sequence[dict], test_rows: Sequence[dict]) -> Dict[str, object]:
    return {
        "max_train_date": train_rows[-1]["date"] if train_rows else None,
        "min_validation_date": validation_rows[0]["date"] if validation_rows else None,
        "max_validation_date": validation_rows[-1]["date"] if validation_rows else None,
        "min_test_date": test_rows[0]["date"] if test_rows else None,
    }


def _picked_odds(row: dict, probability: float) -> float:
    key = "odds_team1" if probability >= 0.5 else "odds_team2"
    try:
        return float(row.get(key, 2.0) or 2.0)
    except (TypeError, ValueError):
        return 2.0


def _team_strength_from_matches(matches: List[dict]) -> Dict[str, float]:
    scores: Dict[str, List[float]] = {}
    for row in matches:
        scores.setdefault(row["team1"], []).append(1.0 if row["winner"] == row["team1"] else 0.0)
        scores.setdefault(row["team2"], []).append(1.0 if row["winner"] == row["team2"] else 0.0)
    return {team: sum(values) / len(values) for team, values in scores.items()}


def _demo_matches() -> List[dict]:
    return [
        _match("2026-05-20", "Alpha", "Bravo", "Alpha", 1, 3, 18, 0.8, 0.4, 0.74, 0.44, 1.16, 1.01, 1.45, 2.7),
        _match("2026-05-21", "Charlie", "Delta", "Delta", 3, 25, 8, 0.45, 0.75, 0.47, 0.69, 1.0, 1.17, 2.4, 1.55),
        _match("2026-05-22", "Alpha", "Charlie", "Alpha", 1, 3, 25, 0.82, 0.45, 0.68, 0.46, 1.18, 1.0, 1.35, 3.1),
        _match("2026-05-23", "Bravo", "Delta", "Delta", 3, 18, 8, 0.42, 0.76, 0.51, 0.7, 1.02, 1.19, 2.2, 1.65),
    ]


def _match(date: str, team1: str, team2: str, winner: str, best_of: int, rank1: int, rank2: int, recent1: float, recent2: float, map1: float, map2: float, rating1: float, rating2: float, odds1: float, odds2: float) -> dict:
    return {
        "date": date,
        "event": "IEM Cologne Demo",
        "event_tier": "S",
        "status": "completed",
        "team1": team1,
        "team2": team2,
        "winner": winner,
        "best_of": best_of,
        "map": "mirage",
        "team1_rank": rank1,
        "team2_rank": rank2,
        "team1_rmr_points": max(100, 1000 - rank1 * 20),
        "team2_rmr_points": max(100, 1000 - rank2 * 20),
        "team1_recent_winrate_10": recent1,
        "team2_recent_winrate_10": recent2,
        "team1_bo1_winrate_6m": recent1,
        "team2_bo1_winrate_6m": recent2,
        "team1_bo3_winrate_6m": recent1,
        "team2_bo3_winrate_6m": recent2,
        "team1_map_winrate": map1,
        "team2_map_winrate": map2,
        "team1_rating": rating1,
        "team2_rating": rating2,
        "team1_kd": rating1,
        "team2_kd": rating2,
        "team1_opening_success": 0.5 + (rating1 - 1.0) / 3,
        "team2_opening_success": 0.5 + (rating2 - 1.0) / 3,
        "team1_clutch_winrate": 0.5 + (rating1 - 1.0) / 3,
        "team2_clutch_winrate": 0.5 + (rating2 - 1.0) / 3,
        "team1_star_rating": rating1 + 0.12,
        "team2_star_rating": rating2 + 0.12,
        "h2h_team1_winrate": 0.5,
        "odds_team1": odds1,
        "odds_team2": odds2,
        "swiss_round": 1,
        "team1_wins": 0,
        "team1_losses": 0,
        "team2_wins": 0,
        "team2_losses": 0,
    }
