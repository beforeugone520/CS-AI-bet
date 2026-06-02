from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .bp import merge_bp_into_fixtures
from .data import read_matches_csv
from .odds import market_probability_from_row
from .predictor import MatchPredictor
from .strategy import adjust_probability_toward_market_probability, single_match_pick


def forecast_fixtures(
    history_rows: Iterable[Mapping[str, Any]],
    fixture_rows: Iterable[Mapping[str, Any]],
    reference_date: str,
    profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    top_k: int = 25,
    epochs: int = 50,
    bp_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    max_age_days: int = 90,
    ensemble_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    fixtures = [dict(row) for row in fixture_rows]
    bp_report = None
    if bp_rows is not None:
        fixtures, bp_report = merge_bp_into_fixtures(fixtures, bp_rows)
    predictor = MatchPredictor.train(
        history_rows,
        reference_date=reference_date,
        top_k=top_k,
        epochs=epochs,
        max_age_days=max_age_days,
        ensemble_weights=dict(ensemble_weights) if ensemble_weights else None,
    )

    predictions = []
    for fixture in fixtures:
        raw_probability, map_details = predictor.predict_with_maps(fixture, profiles)
        market_signal = market_probability_from_row(fixture)
        market_adjustment_applied = bool(market_signal and not market_signal.get("proxy"))
        adjusted = raw_probability
        if market_adjustment_applied:
            adjusted = adjust_probability_toward_market_probability(
                raw_probability,
                market_probability=_num(market_signal.get("probability_team1"), 0.5),
            )
        pick = single_match_pick(adjusted, str(fixture.get("team1")), str(fixture.get("team2")))
        confidence_margin = abs(adjusted - 0.5)
        predictions.append(
            {
                "date": fixture.get("date"),
                "event": fixture.get("event"),
                "team1": fixture.get("team1"),
                "team2": fixture.get("team2"),
                "best_of": fixture.get("best_of", 1),
                "map": fixture.get("map", "unknown"),
                "model_probability_team1": raw_probability,
                "adjusted_probability_team1": adjusted,
                "market_adjustment_applied": market_adjustment_applied,
                "market_signal": market_signal or {},
                "pick": pick,
                "confidence_margin": confidence_margin,
                "low_confidence": pick == "avoid",
                "bp_applied": fixture.get("bp_applied", 0),
                "bp_source": fixture.get("bp_source"),
                "bp_confidence": fixture.get("bp_confidence"),
                **map_details,
            }
        )

    return {
        "trained_matches": predictor.trained_matches,
        "fixtures": len(fixtures),
        "selected_feature_names": predictor.selected_feature_names,
        "imbalance": predictor.imbalance_report,
        "ensemble_weights": predictor.ensemble_weights,
        "model_hyperparameters": predictor.model_hyperparameters,
        "probability_calibration": predictor.calibration_report,
        "feature_preparation": predictor.feature_preparation,
        "bp_report": bp_report,
        "predictions": predictions,
        "decision_summary": _decision_summary(predictions),
    }


def forecast_fixtures_file(
    history_path: str,
    fixtures_path: str,
    reference_date: str,
    profiles_path: Optional[str] = None,
    bp_path: Optional[str] = None,
    top_k: int = 25,
    epochs: int = 50,
    max_age_days: int = 90,
    ensemble_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    profiles: Optional[Mapping[str, Mapping[str, Any]]] = None
    if profiles_path:
        import json

        with open(profiles_path, encoding="utf-8") as handle:
            profiles = json.load(handle)
    return forecast_fixtures(
        read_matches_csv(history_path),
        read_matches_csv(fixtures_path),
        reference_date=reference_date,
        profiles=profiles,
        bp_rows=read_matches_csv(bp_path) if bp_path else None,
        top_k=top_k,
        epochs=epochs,
        max_age_days=max_age_days,
        ensemble_weights=ensemble_weights,
    )

def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _decision_summary(predictions: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    materialized = list(predictions)
    low_confidence = sum(1 for row in materialized if row.get("low_confidence"))
    return {
        "fixtures": len(materialized),
        "actionable_picks": len(materialized) - low_confidence,
        "low_confidence_avoids": low_confidence,
    }
