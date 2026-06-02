from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional

from .data import read_matches_csv, read_teams_csv
from .predictor import MatchPredictor
from .strategy import adjust_probability_with_market, choose_pickems, describe_pickem_risk, describe_pickems
from .swiss import TeamSeed, simulate_swiss


def model_driven_pickems(
    history_rows: Iterable[Mapping[str, Any]],
    team_rows: Iterable[Mapping[str, Any]],
    reference_date: str,
    profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    simulations: int = 100000,
    seed: int = 13,
    top_k: int = 25,
    epochs: int = 50,
    slots: Optional[Mapping[str, int]] = None,
    stage: str = "default",
    max_age_days: int = 90,
    ensemble_weights: Optional[Mapping[str, float]] = None,
    fixture_rows: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, object]:
    teams_data = {str(row["team"]): dict(row) for row in team_rows}
    teams = [TeamSeed(name, int(row.get("seed", index + 1))) for index, (name, row) in enumerate(teams_data.items())]
    teams.sort(key=lambda team: team.seed)
    fixture_odds = _fixture_odds_lookup(fixture_rows or [])
    predictor = MatchPredictor.train(
        history_rows,
        reference_date=reference_date,
        top_k=top_k,
        epochs=epochs,
        seed=67,
        max_age_days=max_age_days,
        ensemble_weights=dict(ensemble_weights) if ensemble_weights else None,
    )
    rankings = {team.name: int(teams_data[team.name].get("rank", team.seed)) for team in teams}
    probability_cache: Dict[str, float] = {}
    detail_cache: Dict[str, Dict[str, object]] = {}

    def swiss_predictor(team_a: TeamSeed, team_b: TeamSeed, best_of: int, state) -> float:
        key = f"{team_a.name}__{team_b.name}__bo{best_of}"
        if key not in probability_cache:
            fixture = _fixture_from_team_rows(teams_data[team_a.name], teams_data[team_b.name], best_of=best_of)
            _apply_fixture_odds(fixture, fixture_odds)
            model_probability, details = predictor.predict_with_maps(fixture, profiles)
            market_adjustment_applied = bool(fixture.get("market_odds_available"))
            adjusted_probability = model_probability
            if market_adjustment_applied:
                adjusted_probability = adjust_probability_with_market(
                    model_probability,
                    odds_team1=_num(fixture.get("odds_team1"), 2.0),
                    odds_team2=_num(fixture.get("odds_team2"), 2.0),
                )
            probability_cache[key] = adjusted_probability
            detail_cache[key] = {
                **details,
                "model_probability_team1": model_probability,
                "adjusted_probability_team1": adjusted_probability,
                "market_adjustment_applied": market_adjustment_applied,
            }
        return probability_cache[key]

    simulation = simulate_swiss(teams, swiss_predictor, simulations=simulations, seed=seed)
    sample_match_probabilities = _sample_probabilities(probability_cache, teams, swiss_predictor)
    pickems = choose_pickems(simulation.team_probabilities, rankings=rankings, slots=slots, stage=stage, team_features=teams_data)
    pickem_risk_details = describe_pickem_risk(
        simulation.team_probabilities,
        rankings=rankings,
        stage=stage,
        team_features=teams_data,
    )
    return {
        "trained_matches": predictor.trained_matches,
        "teams": len(teams),
        "simulations": simulations,
        "selected_feature_names": predictor.selected_feature_names,
        "imbalance": predictor.imbalance_report,
        "ensemble_weights": predictor.ensemble_weights,
        "model_hyperparameters": predictor.model_hyperparameters,
        "stage_strategy": _stage_strategy(stage),
        "team_probabilities": simulation.team_probabilities,
        "pickems": pickems,
        "pickem_details": describe_pickems(simulation.team_probabilities, pickems, rankings=rankings, risk_details=pickem_risk_details),
        "pickem_risk_details": pickem_risk_details,
        "sample_match_probabilities": sample_match_probabilities,
        "sample_match_details": _sample_details(detail_cache, sample_match_probabilities),
        "market_adjustment_summary": _market_adjustment_summary(detail_cache),
    }


def model_driven_pickems_file(
    history_path: str,
    teams_path: str,
    reference_date: str,
    profiles_path: Optional[str] = None,
    simulations: int = 100000,
    seed: int = 13,
    top_k: int = 25,
    epochs: int = 50,
    stage: str = "default",
    max_age_days: int = 90,
    ensemble_weights: Optional[Mapping[str, float]] = None,
    fixtures_path: Optional[str] = None,
) -> Dict[str, object]:
    profiles: Optional[Mapping[str, Mapping[str, Any]]] = None
    if profiles_path:
        with open(profiles_path, encoding="utf-8") as handle:
            profiles = json.load(handle)
    return model_driven_pickems(
        read_matches_csv(history_path),
        read_teams_csv(teams_path),
        reference_date=reference_date,
        profiles=profiles,
        simulations=simulations,
        seed=seed,
        top_k=top_k,
        epochs=epochs,
        stage=stage,
        max_age_days=max_age_days,
        ensemble_weights=ensemble_weights,
        fixture_rows=read_matches_csv(fixtures_path) if fixtures_path else None,
    )


def _fixture_from_team_rows(team1: Mapping[str, Any], team2: Mapping[str, Any], best_of: int) -> Dict[str, Any]:
    market_odds_available = team1.get("odds") not in (None, "") and team2.get("odds") not in (None, "")
    return {
        "date": "prediction",
        "event": "IEM Cologne Major",
        "event_tier": "S",
        "status": "scheduled",
        "team1": team1["team"],
        "team2": team2["team"],
        "best_of": best_of,
        "map": "unknown",
        "team1_rank": team1.get("rank", team1.get("seed", 80)),
        "team2_rank": team2.get("rank", team2.get("seed", 80)),
        "team1_rmr_points": team1.get("rmr_points", 0),
        "team2_rmr_points": team2.get("rmr_points", 0),
        "team1_major_best_placement": team1.get("major_best_placement", 32),
        "team2_major_best_placement": team2.get("major_best_placement", 32),
        "team1_recent_winrate_10": team1.get("recent_winrate_10", 0.5),
        "team2_recent_winrate_10": team2.get("recent_winrate_10", 0.5),
        "team1_bo1_winrate_6m": team1.get("bo1_winrate_6m", 0.5),
        "team2_bo1_winrate_6m": team2.get("bo1_winrate_6m", 0.5),
        "team1_bo3_winrate_6m": team1.get("bo3_winrate_6m", 0.5),
        "team2_bo3_winrate_6m": team2.get("bo3_winrate_6m", 0.5),
        "team1_rating": team1.get("rating", 1.0),
        "team2_rating": team2.get("rating", 1.0),
        "team1_kd": team1.get("kd", 1.0),
        "team2_kd": team2.get("kd", 1.0),
        "team1_opening_success": team1.get("opening_success", 0.5),
        "team2_opening_success": team2.get("opening_success", 0.5),
        "team1_clutch_winrate": team1.get("clutch_winrate", 0.5),
        "team2_clutch_winrate": team2.get("clutch_winrate", 0.5),
        "team1_star_rating": team1.get("star_rating", team1.get("rating", 1.0)),
        "team2_star_rating": team2.get("star_rating", team2.get("rating", 1.0)),
        "h2h_team1_winrate": 0.5,
        "odds_team1": team1.get("odds", 2.0),
        "odds_team2": team2.get("odds", 2.0),
        "market_odds_available": int(market_odds_available),
    }


def _sample_probabilities(cache: Dict[str, float], teams: list[TeamSeed], predictor) -> Dict[str, float]:
    if len(teams) >= 2:
        key = f"{teams[0].name}__{teams[1].name}__bo1"
        if key not in cache:
            predictor(teams[0], teams[1], 1, {})
    return dict(sorted(cache.items())[:12])


def _sample_details(cache: Dict[str, Dict[str, object]], sample_probabilities: Mapping[str, float]) -> Dict[str, Dict[str, object]]:
    return {key: cache[key] for key in sample_probabilities if key in cache}


def _market_adjustment_summary(cache: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    adjusted = sorted(key for key, details in cache.items() if details.get("market_adjustment_applied"))
    return {
        "cached_matchups": len(cache),
        "adjusted_matchups": len(adjusted),
        "unadjusted_matchups": len(cache) - len(adjusted),
        "adjusted_matchup_keys": adjusted,
    }


def _fixture_odds_lookup(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    odds_by_pair: Dict[str, list[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row.get("odds_team1") in (None, "") or row.get("odds_team2") in (None, ""):
            continue
        team1 = str(row.get("team1", ""))
        team2 = str(row.get("team2", ""))
        if not team1 or not team2:
            continue
        canonical_team1, _ = _canonical_pair(team1, team2)
        odds_left = _num(row.get("odds_team1"), 2.0)
        odds_right = _num(row.get("odds_team2"), 2.0)
        if _team_key(team1) == _team_key(canonical_team1):
            odds_team1, odds_team2 = odds_left, odds_right
        else:
            odds_team1, odds_team2 = odds_right, odds_left
        odds_by_pair[_pair_key(team1, team2)].append({"odds_team1": odds_team1, "odds_team2": odds_team2})
    return {
        key: {
            "odds_team1": sum(row["odds_team1"] for row in values) / len(values),
            "odds_team2": sum(row["odds_team2"] for row in values) / len(values),
        }
        for key, values in odds_by_pair.items()
        if values
    }


def _apply_fixture_odds(fixture: Dict[str, Any], fixture_odds: Mapping[str, Mapping[str, float]]) -> None:
    pair_key = _pair_key(fixture.get("team1", ""), fixture.get("team2", ""))
    odds = fixture_odds.get(pair_key)
    if not odds:
        return
    canonical_team1, _ = _canonical_pair(fixture["team1"], fixture["team2"])
    if _team_key(fixture["team1"]) == _team_key(canonical_team1):
        fixture["odds_team1"] = odds["odds_team1"]
        fixture["odds_team2"] = odds["odds_team2"]
    else:
        fixture["odds_team1"] = odds["odds_team2"]
        fixture["odds_team2"] = odds["odds_team1"]
    fixture["market_odds_available"] = 1


def _pair_key(team1: Any, team2: Any) -> str:
    left, right = _canonical_pair(team1, team2)
    return f"{_team_key(left)}__{_team_key(right)}"


def _canonical_pair(team1: Any, team2: Any) -> tuple[str, str]:
    left = str(team1)
    right = str(team2)
    return tuple(sorted([left, right], key=_team_key))  # type: ignore[return-value]


def _team_key(value: Any) -> str:
    return str(value).strip().lower()


def _stage_strategy(stage: str) -> Dict[str, str]:
    normalized = stage.strip().lower()
    if normalized in {"challengers", "challenger", "opening"}:
        return {
            "stage": normalized,
            "focus": "BO1 map depth, recent volatility, and low-confidence avoidance",
        }
    if normalized in {"legends", "legend", "elimination"}:
        return {
            "stage": normalized,
            "focus": "elite rank, rating stability, and deep-round survival strength",
        }
    return {
        "stage": normalized or "default",
        "focus": "model probability, rankings, odds-adjusted risk, and upset limits",
    }


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
