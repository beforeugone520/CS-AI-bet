from __future__ import annotations

from typing import Dict, List, Mapping, Optional


DEFAULT_SLOTS = {"3-0": 2, "advance": 6, "0-3": 2}


def adjust_probability_with_market(
    model_probability: float,
    odds_team1: float,
    odds_team2: float,
    max_adjustment: float = 0.03,
) -> float:
    market_probability = _market_probability(odds_team1, odds_team2)
    delta = market_probability - model_probability
    capped_delta = max(-max_adjustment, min(max_adjustment, delta))
    return _clip(model_probability + capped_delta)


def single_match_pick(probability_team1: float, team1: str, team2: str, threshold: float = 0.52) -> str:
    if max(probability_team1, 1.0 - probability_team1) <= threshold:
        return "avoid"
    return team1 if probability_team1 >= 0.5 else team2


def choose_pickems(
    team_probabilities: Mapping[str, Mapping[str, float]],
    rankings: Optional[Mapping[str, int]] = None,
    slots: Optional[Mapping[str, int]] = None,
    upset_rank_limit: int = 15,
    stage: str = "default",
    team_features: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, List[str]]:
    slots = dict(slots or DEFAULT_SLOTS)
    rankings = dict(rankings or {})
    team_features = dict(team_features or {})
    picked: set[str] = set()

    three_zero = _top_teams(team_probabilities, "3-0", slots.get("3-0", 0), rankings, upset_rank_limit, picked, stage=stage, team_features=team_features)
    picked.update(three_zero)

    zero_three = _top_teams(team_probabilities, "0-3", slots.get("0-3", 0), rankings, upset_rank_limit, picked, prefer_weak=True, stage=stage, team_features=team_features)
    picked.update(zero_three)

    advance = _top_teams(team_probabilities, "advance", slots.get("advance", 0), rankings, upset_rank_limit, picked, stage=stage, team_features=team_features)

    return {"3-0": three_zero, "advance": advance, "0-3": zero_three}


def describe_pickems(
    team_probabilities: Mapping[str, Mapping[str, float]],
    pickems: Mapping[str, List[str]],
    rankings: Optional[Mapping[str, int]] = None,
    risk_details: Optional[Mapping[str, List[Mapping[str, object]]]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    rankings = dict(rankings or {})
    details: Dict[str, List[Dict[str, object]]] = {}
    selected_anywhere = {team for teams in pickems.values() for team in teams}
    for category, teams in pickems.items():
        selected = list(teams)
        score_lookup = _risk_score_lookup(risk_details, category)
        if score_lookup:
            unselected_scores = [
                score
                for team, score in score_lookup.items()
                if team not in selected_anywhere
            ]
            next_best = max(unselected_scores) if unselected_scores else None
        else:
            unselected_probabilities = [
                float(values.get(category, 0.0))
                for team, values in team_probabilities.items()
                if team not in selected
            ]
            next_best = max(unselected_probabilities) if unselected_probabilities else None
        details[category] = []
        for team in selected:
            probability = float(team_probabilities.get(team, {}).get(category, 0.0))
            selection_score = score_lookup.get(team) if score_lookup else probability
            details[category].append(
                {
                    "team": team,
                    "category": category,
                    "probability": probability,
                    "rank": rankings.get(team),
                    "next_best_probability": next_best if not score_lookup else None,
                    "selection_score": selection_score,
                    "next_best_score": next_best if score_lookup else None,
                    "selection_margin": selection_score - next_best if next_best is not None and selection_score is not None else None,
                }
            )
    return details


def _risk_score_lookup(risk_details: Optional[Mapping[str, List[Mapping[str, object]]]], category: str) -> Dict[str, float]:
    if not risk_details:
        return {}
    entries = risk_details.get(category, [])
    lookup: Dict[str, float] = {}
    for entry in entries:
        team = entry.get("team")
        if team in (None, ""):
            continue
        try:
            lookup[str(team)] = float(entry.get("final_score"))
        except (TypeError, ValueError):
            continue
    return lookup


def describe_pickem_risk(
    team_probabilities: Mapping[str, Mapping[str, float]],
    rankings: Optional[Mapping[str, int]] = None,
    upset_rank_limit: int = 15,
    stage: str = "default",
    team_features: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    rankings = dict(rankings or {})
    team_features = dict(team_features or {})
    best_rank = _best_rank(team_probabilities, rankings)
    details: Dict[str, List[Dict[str, object]]] = {}
    for key in ("3-0", "advance", "0-3"):
        entries = []
        for team, values in team_probabilities.items():
            rank = rankings.get(team, 80)
            base_probability = float(values.get(key, 0.0))
            stage_adjustment = _stage_adjustment(stage, key, team_features.get(team, {}), rank)
            upset_rank_gap = max(0, rank - best_rank)
            upset_penalty_multiplier = _upset_penalty_multiplier(key, upset_rank_gap, upset_rank_limit)
            prefer_weak_multiplier = _prefer_weak_multiplier(rank) if key == "0-3" else 1.0
            final_score = (base_probability + stage_adjustment) * upset_penalty_multiplier * prefer_weak_multiplier
            entries.append(
                {
                    "team": team,
                    "category": key,
                    "rank": rank,
                    "base_probability": base_probability,
                    "stage_adjustment": stage_adjustment,
                    "upset_rank_gap": upset_rank_gap,
                    "upset_penalty_multiplier": upset_penalty_multiplier,
                    "prefer_weak_multiplier": prefer_weak_multiplier,
                    "final_score": final_score,
                }
            )
        details[key] = sorted(entries, key=lambda entry: (-float(entry["final_score"]), int(entry["rank"]), str(entry["team"])))
    return details


def _top_teams(
    probabilities: Mapping[str, Mapping[str, float]],
    key: str,
    count: int,
    rankings: Mapping[str, int],
    upset_rank_limit: int,
    excluded: set[str],
    prefer_weak: bool = False,
    stage: str = "default",
    team_features: Mapping[str, Mapping[str, float]] | None = None,
) -> List[str]:
    team_features = team_features or {}
    scored = []
    best_rank = _best_rank(probabilities, rankings)
    for team, values in probabilities.items():
        if team in excluded:
            continue
        rank = rankings.get(team, 80)
        score = _candidate_score(
            float(values.get(key, 0.0)),
            key,
            rank,
            best_rank,
            upset_rank_limit,
            stage,
            team_features.get(team, {}),
            prefer_weak=prefer_weak,
        )
        scored.append((score, -rank if prefer_weak else rank, team))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [team for _, _, team in scored[:count]]


def _candidate_score(
    base_probability: float,
    key: str,
    rank: int,
    best_rank: int,
    upset_rank_limit: int,
    stage: str,
    features: Mapping[str, float],
    prefer_weak: bool = False,
) -> float:
    score = base_probability + _stage_adjustment(stage, key, features, rank)
    score *= _upset_penalty_multiplier(key, max(0, rank - best_rank), upset_rank_limit)
    if prefer_weak:
        score *= _prefer_weak_multiplier(rank)
    return score


def _best_rank(probabilities: Mapping[str, Mapping[str, float]], rankings: Mapping[str, int]) -> int:
    return min((rankings.get(team, 80) for team in probabilities), default=1)


def _upset_penalty_multiplier(key: str, rank_gap: int, upset_rank_limit: int) -> float:
    if key in {"3-0", "advance"} and rank_gap > upset_rank_limit:
        return 0.75
    return 1.0


def _prefer_weak_multiplier(rank: int) -> float:
    return 1.0 + min(rank, 80) / 200.0


def _market_probability(odds_team1: float, odds_team2: float) -> float:
    inv1 = 1.0 / odds_team1 if odds_team1 > 0 else 0.5
    inv2 = 1.0 / odds_team2 if odds_team2 > 0 else 0.5
    total = inv1 + inv2
    return inv1 / total if total else 0.5


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _stage_adjustment(stage: str, key: str, features: Mapping[str, float], rank: int) -> float:
    if key == "0-3":
        return 0.0
    normalized_stage = stage.strip().lower()
    if normalized_stage in {"challengers", "challenger", "opening"}:
        bo1 = _num(features.get("bo1_winrate_6m", features.get("bo1_winrate", 0.5)), 0.5)
        map_depth = _num(features.get("map_depth", features.get("map_pool_score", 0.5)), 0.5)
        return max(-0.04, min(0.06, (bo1 - 0.5) * 0.10 + (map_depth - 0.5) * 0.08))
    if normalized_stage in {"legends", "legend", "elimination"}:
        rating = _num(features.get("rating", features.get("team_rating", 1.0)), 1.0)
        rank_bonus = (80.0 - min(max(rank, 1), 80)) / 80.0
        return max(-0.03, min(0.06, (rating - 1.0) * 0.10 + rank_bonus * 0.04))
    return 0.0


def _num(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
