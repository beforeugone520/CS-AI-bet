from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .bp import merge_bp_into_fixtures
from .data import read_matches_csv, write_json
from .odds import market_probability_from_row
from .predictor import MatchPredictor
from .strategy import (
    PRODUCTION_FUSION_METHOD,
    PRODUCTION_MODEL_WEIGHT,
    adjust_probability_toward_market_probability,
    single_match_pick,
)


SWISS_PRESSURE_FIELDS = (
    "swiss_round",
    "team1_wins",
    "team1_losses",
    "team2_wins",
    "team2_losses",
    "team1_record",
    "team2_record",
    "team1_record_status",
    "team2_record_status",
    "standings_source",
    "swiss_match_type",
)


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
    minimum_margin: float = 0.02,
    bo1_minimum_margin: float | None = None,
    avoid_player_form_counter_signal: bool = False,
    player_form_counter_min_confidence: float = 0.4,
    avoid_market_favorite_player_form_counter_signal: bool = False,
    market_favorite_counter_min_probability: float = 0.6,
    avoid_player_status_risk: bool = False,
    player_status_min_confidence: float = 0.4,
    player_status_min_margin: float = 0.06,
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
            # Production fusion (WF-2F): logarithmic opinion pool leaning on the market
            # (model weight ~0.30). The library default stays legacy_clip; this call point
            # opts in explicitly via the centralised production constants.
            adjusted = adjust_probability_toward_market_probability(
                raw_probability,
                market_probability=_num(market_signal.get("probability_team1"), 0.5),
                fusion_method=PRODUCTION_FUSION_METHOD,
                model_weight=PRODUCTION_MODEL_WEIGHT,
            )
        confidence_margin = abs(adjusted - 0.5)
        effective_minimum_margin = _effective_minimum_margin(fixture, minimum_margin, bo1_minimum_margin)
        player_form_summary = _player_form_summary(fixture)
        pick = single_match_pick(
            adjusted,
            str(fixture.get("team1")),
            str(fixture.get("team2")),
            minimum_margin=effective_minimum_margin,
            player_form_score_diff=_num(player_form_summary.get("diff", {}).get("score"), 0.0),
            player_form_sample_confidence=_player_form_sample_confidence(player_form_summary),
            player_form_counter_min_confidence=player_form_counter_min_confidence,
            avoid_player_form_counter_signal=avoid_player_form_counter_signal,
            avoid_player_status_risk=avoid_player_status_risk,
            player_status_min_confidence=player_status_min_confidence,
            player_status_min_margin=player_status_min_margin,
            **_player_status_pick_kwargs(player_form_summary),
        )
        avoid_reason = _avoid_reason(
            pick=pick,
            adjusted_probability_team1=adjusted,
            minimum_margin=effective_minimum_margin,
            player_form_summary=player_form_summary,
            avoid_player_form_counter_signal=avoid_player_form_counter_signal,
            player_form_counter_min_confidence=player_form_counter_min_confidence,
            market_probability_team1=_market_probability_team1_from_signal(market_signal or {}),
            avoid_market_favorite_player_form_counter_signal=avoid_market_favorite_player_form_counter_signal,
            market_favorite_counter_min_probability=market_favorite_counter_min_probability,
            avoid_player_status_risk=avoid_player_status_risk,
            player_status_min_confidence=player_status_min_confidence,
            player_status_min_margin=player_status_min_margin,
        )
        if avoid_reason == "market_favorite_player_form_counter_signal":
            pick = "avoid"
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
                "avoid_reason": avoid_reason,
                "confidence_margin": confidence_margin,
                "effective_minimum_margin": effective_minimum_margin,
                "low_confidence": avoid_reason == "low_confidence",
                "player_form_summary": player_form_summary,
                "bp_applied": fixture.get("bp_applied", 0),
                "bp_source": fixture.get("bp_source"),
                "bp_confidence": fixture.get("bp_confidence"),
                **_swiss_pressure_fields(fixture),
                **map_details,
            }
        )

    return {
        "trained_matches": predictor.trained_matches,
        "fixtures": len(fixtures),
        "selected_feature_names": predictor.selected_feature_names,
        "feature_selection": {
            "required_features": predictor.selector.required_feature_report,
        },
        "imbalance": predictor.imbalance_report,
        "ensemble_weights": predictor.ensemble_weights,
        "model_hyperparameters": predictor.model_hyperparameters,
        "probability_calibration": predictor.calibration_report,
        "feature_preparation": predictor.feature_preparation,
        "decision_policy": {
            "minimum_margin": minimum_margin,
            "bo1_minimum_margin": bo1_minimum_margin,
            "avoid_player_form_counter_signal": avoid_player_form_counter_signal,
            "player_form_counter_min_confidence": player_form_counter_min_confidence,
            "avoid_market_favorite_player_form_counter_signal": avoid_market_favorite_player_form_counter_signal,
            "market_favorite_counter_min_probability": market_favorite_counter_min_probability,
            "avoid_player_status_risk": avoid_player_status_risk,
            "player_status_min_confidence": player_status_min_confidence,
            "player_status_min_margin": player_status_min_margin,
        },
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
    minimum_margin: float = 0.02,
    bo1_minimum_margin: float | None = None,
    avoid_player_form_counter_signal: bool = False,
    player_form_counter_min_confidence: float = 0.4,
    avoid_market_favorite_player_form_counter_signal: bool = False,
    market_favorite_counter_min_probability: float = 0.6,
    avoid_player_status_risk: bool = False,
    player_status_min_confidence: float = 0.4,
    player_status_min_margin: float = 0.06,
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
        minimum_margin=minimum_margin,
        bo1_minimum_margin=bo1_minimum_margin,
        avoid_player_form_counter_signal=avoid_player_form_counter_signal,
        player_form_counter_min_confidence=player_form_counter_min_confidence,
        avoid_market_favorite_player_form_counter_signal=avoid_market_favorite_player_form_counter_signal,
        market_favorite_counter_min_probability=market_favorite_counter_min_probability,
        avoid_player_status_risk=avoid_player_status_risk,
        player_status_min_confidence=player_status_min_confidence,
        player_status_min_margin=player_status_min_margin,
    )


def apply_forecast_policy(
    forecast_report: Mapping[str, Any],
    fixture_rows: Optional[Iterable[Mapping[str, Any]]] = None,
    minimum_margin: float = 0.02,
    bo1_minimum_margin: float | None = None,
    avoid_player_form_counter_signal: bool = False,
    player_form_counter_min_confidence: float = 0.4,
    avoid_market_favorite_player_form_counter_signal: bool = False,
    market_favorite_counter_min_probability: float = 0.6,
    avoid_player_status_risk: bool = False,
    player_status_min_confidence: float = 0.4,
    player_status_min_margin: float = 0.06,
) -> Dict[str, object]:
    raw_predictions = forecast_report.get("predictions", [])
    if not isinstance(raw_predictions, list):
        raise ValueError("forecast report must contain a predictions list")
    fixture_lookup = _fixture_lookup(fixture_rows or [])
    predictions = []
    fixtures_augmented = 0
    for prediction in raw_predictions:
        updated = dict(prediction)
        fixture = _lookup_fixture(updated, fixture_lookup)
        if fixture is not None:
            aligned_fixture = _aligned_fixture_row(updated, fixture)
            player_form_summary = _player_form_summary({**updated, **aligned_fixture})
            updated.update(_swiss_pressure_fields(aligned_fixture))
            fixtures_augmented += 1
        else:
            player_form_summary = updated.get("player_form_summary")
            if not isinstance(player_form_summary, Mapping):
                player_form_summary = _player_form_summary(updated)
        _apply_decision_policy_to_prediction(
            updated,
            player_form_summary=player_form_summary,
            minimum_margin=minimum_margin,
            bo1_minimum_margin=bo1_minimum_margin,
            avoid_player_form_counter_signal=avoid_player_form_counter_signal,
            player_form_counter_min_confidence=player_form_counter_min_confidence,
            avoid_market_favorite_player_form_counter_signal=avoid_market_favorite_player_form_counter_signal,
            market_favorite_counter_min_probability=market_favorite_counter_min_probability,
            avoid_player_status_risk=avoid_player_status_risk,
            player_status_min_confidence=player_status_min_confidence,
            player_status_min_margin=player_status_min_margin,
        )
        predictions.append(updated)
    report = dict(forecast_report)
    report["predictions"] = predictions
    report["decision_policy"] = {
        "minimum_margin": minimum_margin,
        "bo1_minimum_margin": bo1_minimum_margin,
        "avoid_player_form_counter_signal": avoid_player_form_counter_signal,
        "player_form_counter_min_confidence": player_form_counter_min_confidence,
        "avoid_market_favorite_player_form_counter_signal": avoid_market_favorite_player_form_counter_signal,
        "market_favorite_counter_min_probability": market_favorite_counter_min_probability,
        "avoid_player_status_risk": avoid_player_status_risk,
        "player_status_min_confidence": player_status_min_confidence,
        "player_status_min_margin": player_status_min_margin,
    }
    report["decision_summary"] = _decision_summary(predictions)
    report["policy_application"] = {
        "basis": "posthoc_forecast_policy",
        "fixtures_augmented": fixtures_augmented,
    }
    return report


def apply_forecast_policy_file(
    forecast_report_path: str,
    fixtures_path: Optional[str] = None,
    output_path: Optional[str] = None,
    minimum_margin: float = 0.02,
    bo1_minimum_margin: float | None = None,
    avoid_player_form_counter_signal: bool = False,
    player_form_counter_min_confidence: float = 0.4,
    avoid_market_favorite_player_form_counter_signal: bool = False,
    market_favorite_counter_min_probability: float = 0.6,
    avoid_player_status_risk: bool = False,
    player_status_min_confidence: float = 0.4,
    player_status_min_margin: float = 0.06,
) -> Dict[str, object]:
    with open(forecast_report_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("forecast report must be a JSON object")
    report = apply_forecast_policy(
        payload,
        fixture_rows=read_matches_csv(fixtures_path) if fixtures_path else None,
        minimum_margin=minimum_margin,
        bo1_minimum_margin=bo1_minimum_margin,
        avoid_player_form_counter_signal=avoid_player_form_counter_signal,
        player_form_counter_min_confidence=player_form_counter_min_confidence,
        avoid_market_favorite_player_form_counter_signal=avoid_market_favorite_player_form_counter_signal,
        market_favorite_counter_min_probability=market_favorite_counter_min_probability,
        avoid_player_status_risk=avoid_player_status_risk,
        player_status_min_confidence=player_status_min_confidence,
        player_status_min_margin=player_status_min_margin,
    )
    report["forecast_report_path"] = forecast_report_path
    if fixtures_path:
        report["fixtures_path"] = fixtures_path
    if output_path:
        write_json(output_path, report)
    return report


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _effective_minimum_margin(
    row: Mapping[str, Any],
    minimum_margin: float,
    bo1_minimum_margin: float | None,
) -> float:
    if bo1_minimum_margin is not None and _best_of(row) == 1:
        return bo1_minimum_margin
    return minimum_margin


def _best_of(row: Mapping[str, Any]) -> int:
    try:
        return int(float(row.get("best_of", 1)))
    except (TypeError, ValueError):
        return 1


def _swiss_pressure_fields(row: Mapping[str, Any]) -> Dict[str, object]:
    return {
        field: row[field]
        for field in SWISS_PRESSURE_FIELDS
        if field in row and row.get(field) not in (None, "")
    }


def _aligned_fixture_row(
    prediction: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> Dict[str, object]:
    direct = dict(fixture)
    prediction_team1 = _team_key(prediction.get("team1"))
    prediction_team2 = _team_key(prediction.get("team2"))
    fixture_team1 = _team_key(fixture.get("team1"))
    fixture_team2 = _team_key(fixture.get("team2"))
    if prediction_team1 == fixture_team1 and prediction_team2 == fixture_team2:
        return direct
    if prediction_team1 != fixture_team2 or prediction_team2 != fixture_team1:
        return direct
    swapped: Dict[str, object] = {}
    for key, value in direct.items():
        if key == "team1":
            swapped[key] = direct.get("team2", value)
        elif key == "team2":
            swapped[key] = direct.get("team1", value)
        elif key.startswith("team1_"):
            source_key = "team2_" + key[len("team1_"):]
            swapped[key] = direct.get(source_key, value)
        elif key.startswith("team2_"):
            source_key = "team1_" + key[len("team2_"):]
            swapped[key] = direct.get(source_key, value)
        elif key == "odds_team1":
            swapped[key] = direct.get("odds_team2", value)
        elif key == "odds_team2":
            swapped[key] = direct.get("odds_team1", value)
        else:
            swapped[key] = value
    return swapped


def _player_form_summary(row: Mapping[str, Any]) -> Dict[str, object]:
    team1 = _player_form_side(row, "team1")
    team2 = _player_form_side(row, "team2")
    return {
        "team1": team1,
        "team2": team2,
        "diff": {
            "score": float(team1["score"]) - float(team2["score"]),
            "trend": float(team1["trend"]) - float(team2["trend"]),
            "sample_confidence": float(team1["sample_confidence"]) - float(team2["sample_confidence"]),
        },
    }


def _player_form_side(row: Mapping[str, Any], prefix: str) -> Dict[str, object]:
    return {
        "team": row.get(prefix),
        "score": _num(row.get(f"{prefix}_player_form_score"), 0.0),
        "trend": _num(row.get(f"{prefix}_player_form_trend"), 0.0),
        "sample_confidence": _num(row.get(f"{prefix}_player_sample_confidence"), 0.0),
        "player_sample": int(_num(row.get(f"{prefix}_player_sample"), 0.0)),
        "substitute_flag": int(_num(row.get(f"{prefix}_substitute_flag"), 0.0)),
    }


def _avoid_reason(
    pick: str,
    adjusted_probability_team1: float,
    minimum_margin: float,
    player_form_summary: Mapping[str, Any],
    avoid_player_form_counter_signal: bool,
    player_form_counter_min_confidence: float,
    market_probability_team1: float | None = None,
    avoid_market_favorite_player_form_counter_signal: bool = False,
    market_favorite_counter_min_probability: float = 0.6,
    avoid_player_status_risk: bool = False,
    player_status_min_confidence: float = 0.4,
    player_status_min_margin: float = 0.06,
) -> str | None:
    diff = player_form_summary.get("diff", {})
    form_score_diff = _num(diff.get("score") if isinstance(diff, Mapping) else None, 0.0)
    directional_form_score = form_score_diff if adjusted_probability_team1 >= 0.5 else -form_score_diff
    if pick == "avoid":
        if abs(adjusted_probability_team1 - 0.5) <= minimum_margin:
            return "low_confidence"
        if (
            avoid_player_form_counter_signal
            and _player_form_sample_confidence(player_form_summary) >= player_form_counter_min_confidence
            and directional_form_score < 0
        ):
            return "player_form_counter_signal"
        if (
            avoid_player_status_risk
            and abs(adjusted_probability_team1 - 0.5) <= player_status_min_margin
            and _player_status_risk(
                adjusted_probability_team1,
                player_form_summary,
                player_status_min_confidence,
            )
        ):
            return "player_status_risk"
        return "avoid"
    if (
        avoid_market_favorite_player_form_counter_signal
        and _market_favorite_player_form_counter_signal(
            adjusted_probability_team1=adjusted_probability_team1,
            market_probability_team1=market_probability_team1,
            directional_form_score=directional_form_score,
            min_probability=market_favorite_counter_min_probability,
        )
    ):
        return "market_favorite_player_form_counter_signal"
    return None


def _market_favorite_player_form_counter_signal(
    adjusted_probability_team1: float,
    market_probability_team1: float | None,
    directional_form_score: float,
    min_probability: float,
) -> bool:
    if market_probability_team1 is None:
        return False
    adjusted_picks_team1 = adjusted_probability_team1 >= 0.5
    market_favors_team1 = market_probability_team1 >= 0.5
    market_favorite_probability = market_probability_team1 if market_favors_team1 else 1.0 - market_probability_team1
    return (
        adjusted_picks_team1 == market_favors_team1
        and market_favorite_probability >= min_probability
        and directional_form_score < 0
    )


def _apply_decision_policy_to_prediction(
    prediction: Dict[str, Any],
    player_form_summary: Mapping[str, Any],
    minimum_margin: float,
    bo1_minimum_margin: float | None,
    avoid_player_form_counter_signal: bool,
    player_form_counter_min_confidence: float,
    avoid_market_favorite_player_form_counter_signal: bool,
    market_favorite_counter_min_probability: float,
    avoid_player_status_risk: bool,
    player_status_min_confidence: float,
    player_status_min_margin: float,
) -> None:
    adjusted = _num(
        prediction.get("adjusted_probability_team1"),
        _num(prediction.get("model_probability_team1"), 0.5),
    )
    effective_minimum_margin = _effective_minimum_margin(
        prediction,
        minimum_margin,
        bo1_minimum_margin,
    )
    prediction["previous_pick"] = prediction.get("pick")
    prediction["confidence_margin"] = abs(adjusted - 0.5)
    prediction["effective_minimum_margin"] = effective_minimum_margin
    prediction["player_form_summary"] = player_form_summary
    prediction["pick"] = single_match_pick(
        adjusted,
        str(prediction.get("team1")),
        str(prediction.get("team2")),
        minimum_margin=effective_minimum_margin,
        player_form_score_diff=_num(player_form_summary.get("diff", {}).get("score"), 0.0),
        player_form_sample_confidence=_player_form_sample_confidence(player_form_summary),
        player_form_counter_min_confidence=player_form_counter_min_confidence,
        avoid_player_form_counter_signal=avoid_player_form_counter_signal,
        avoid_player_status_risk=avoid_player_status_risk,
        player_status_min_confidence=player_status_min_confidence,
        player_status_min_margin=player_status_min_margin,
        **_player_status_pick_kwargs(player_form_summary),
    )
    prediction["avoid_reason"] = _avoid_reason(
        pick=str(prediction["pick"]),
        adjusted_probability_team1=adjusted,
        minimum_margin=effective_minimum_margin,
        player_form_summary=player_form_summary,
        avoid_player_form_counter_signal=avoid_player_form_counter_signal,
        player_form_counter_min_confidence=player_form_counter_min_confidence,
        market_probability_team1=_market_probability_team1_from_prediction(prediction),
        avoid_market_favorite_player_form_counter_signal=avoid_market_favorite_player_form_counter_signal,
        market_favorite_counter_min_probability=market_favorite_counter_min_probability,
        avoid_player_status_risk=avoid_player_status_risk,
        player_status_min_confidence=player_status_min_confidence,
        player_status_min_margin=player_status_min_margin,
    )
    if prediction["avoid_reason"] == "market_favorite_player_form_counter_signal":
        prediction["pick"] = "avoid"
    prediction["low_confidence"] = prediction["avoid_reason"] == "low_confidence"


def _player_form_sample_confidence(player_form_summary: Mapping[str, Any]) -> float:
    team1 = player_form_summary.get("team1", {})
    team2 = player_form_summary.get("team2", {})
    if not isinstance(team1, Mapping) or not isinstance(team2, Mapping):
        return 0.0
    return min(_num(team1.get("sample_confidence"), 0.0), _num(team2.get("sample_confidence"), 0.0))


def _player_status_pick_kwargs(player_form_summary: Mapping[str, Any]) -> Dict[str, object]:
    team1 = player_form_summary.get("team1", {})
    team2 = player_form_summary.get("team2", {})
    if not isinstance(team1, Mapping):
        team1 = {}
    if not isinstance(team2, Mapping):
        team2 = {}
    return {
        "team1_player_sample_confidence": _num(team1.get("sample_confidence"), 0.0),
        "team2_player_sample_confidence": _num(team2.get("sample_confidence"), 0.0),
        "team1_substitute_flag": _num(team1.get("substitute_flag"), 0.0),
        "team2_substitute_flag": _num(team2.get("substitute_flag"), 0.0),
    }


def _player_status_risk(
    adjusted_probability_team1: float,
    player_form_summary: Mapping[str, Any],
    player_status_min_confidence: float,
) -> bool:
    side_key = "team1" if adjusted_probability_team1 >= 0.5 else "team2"
    side = player_form_summary.get(side_key, {})
    if not isinstance(side, Mapping):
        return False
    sample_confidence = _num(side.get("sample_confidence"), 1.0)
    substitute_flag = _num(side.get("substitute_flag"), 0.0)
    return sample_confidence < player_status_min_confidence or substitute_flag >= 1.0


def _market_probability_team1_from_signal(market_signal: Mapping[str, Any]) -> float | None:
    if market_signal.get("proxy"):
        return None
    return _optional_probability(market_signal.get("probability_team1"))


def _market_probability_team1_from_prediction(prediction: Mapping[str, Any]) -> float | None:
    market_signal = prediction.get("market_signal")
    if isinstance(market_signal, Mapping):
        probability = _market_probability_team1_from_signal(market_signal)
        if probability is not None:
            return probability
    return _optional_probability(prediction.get("market_probability_team1"))


def _optional_probability(value: Any) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= probability <= 1.0:
        return probability
    return None


def _fixture_lookup(fixture_rows: Iterable[Mapping[str, Any]]) -> Dict[tuple[str, str, str], Mapping[str, Any]]:
    lookup: Dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for fixture in fixture_rows:
        lookup[_match_key(fixture)] = fixture
    return lookup


def _lookup_fixture(
    prediction: Mapping[str, Any],
    fixture_lookup: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    return fixture_lookup.get(_match_key(prediction))


def _match_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    teams = sorted((_team_key(row.get("team1")), _team_key(row.get("team2"))))
    return (str(row.get("date") or "")[:10], teams[0], teams[1])


def _team_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "")


def _decision_summary(predictions: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    materialized = list(predictions)
    low_confidence = sum(1 for row in materialized if row.get("low_confidence"))
    avoid_picks = sum(1 for row in materialized if row.get("pick") == "avoid")
    player_form_counter_signal_avoids = sum(1 for row in materialized if row.get("avoid_reason") == "player_form_counter_signal")
    market_form_counter_avoids = sum(
        1
        for row in materialized
        if row.get("avoid_reason") == "market_favorite_player_form_counter_signal"
    )
    player_status_risk_avoids = sum(1 for row in materialized if row.get("avoid_reason") == "player_status_risk")
    return {
        "fixtures": len(materialized),
        "actionable_picks": len(materialized) - avoid_picks,
        "avoid_picks": avoid_picks,
        "low_confidence_avoids": low_confidence,
        "player_form_counter_signal_avoids": player_form_counter_signal_avoids,
        "market_favorite_player_form_counter_signal_avoids": market_form_counter_avoids,
        "player_status_risk_avoids": player_status_risk_avoids,
    }
