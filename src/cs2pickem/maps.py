from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping


DEFAULT_MAP_POOL = ["mirage", "inferno", "ancient", "anubis", "nuke", "overpass", "train"]


def candidate_maps_from_bp(
    team1_profile: Mapping[str, object],
    team2_profile: Mapping[str, object],
    map_pool: Iterable[str] | None = None,
    top_n: int = 3,
) -> List[str]:
    pool = [_normalize_map(name) for name in (map_pool or DEFAULT_MAP_POOL)]
    banned = {_normalize_map(name) for name in _list(team1_profile.get("ban_top3")) + _list(team2_profile.get("ban_top3"))}
    scored = []
    for map_name in pool:
        if map_name in banned:
            continue
        score = _preference_score(map_name, team1_profile) + _preference_score(map_name, team2_profile)
        score += _balance_score(map_name, team1_profile, team2_profile)
        score += _combined_strength(map_name, team1_profile, team2_profile)
        scored.append((score, map_name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [map_name for _, map_name in scored[:top_n]]


def average_unknown_map_prediction(
    base_row: Mapping[str, object],
    team1_profile: Mapping[str, object],
    team2_profile: Mapping[str, object],
    predictor: Callable[[Dict[str, object]], float],
    map_pool: Iterable[str] | None = None,
    top_n: int = 3,
) -> Dict[str, object]:
    candidates = candidate_maps_from_bp(team1_profile, team2_profile, map_pool=map_pool, top_n=top_n)
    per_map = {}
    for map_name in candidates:
        row = dict(base_row)
        row["map"] = map_name
        row["team1_map_winrate"] = _map_winrate(team1_profile, map_name)
        row["team2_map_winrate"] = _map_winrate(team2_profile, map_name)
        per_map[map_name] = min(1.0, max(0.0, float(predictor(row))))
    average = sum(per_map.values()) / len(per_map) if per_map else 0.5
    return {
        "candidate_maps": candidates,
        "per_map_probability_team1": per_map,
        "average_probability_team1": average,
    }


def _preference_score(map_name: str, profile: Mapping[str, object]) -> float:
    preferences = [_normalize_map(name) for name in _list(profile.get("prefer_top3"))]
    if map_name not in preferences:
        return 0.0
    return 3.0 - preferences.index(map_name)


def _balance_score(map_name: str, team1_profile: Mapping[str, object], team2_profile: Mapping[str, object]) -> float:
    gap = abs(_map_winrate(team1_profile, map_name) - _map_winrate(team2_profile, map_name))
    return 1.0 - min(1.0, gap)


def _combined_strength(map_name: str, team1_profile: Mapping[str, object], team2_profile: Mapping[str, object]) -> float:
    return (_map_winrate(team1_profile, map_name) + _map_winrate(team2_profile, map_name)) / 2.0


def _map_winrate(profile: Mapping[str, object], map_name: str) -> float:
    winrates = profile.get("map_winrates") or {}
    if not isinstance(winrates, Mapping):
        return 0.5
    value = winrates.get(map_name, winrates.get(map_name.title(), 0.5))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.5


def _list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    return [str(item) for item in value]


def _normalize_map(value: object) -> str:
    return str(value).strip().lower().replace("de_", "")
