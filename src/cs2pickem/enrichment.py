from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

from .cleaning import parse_date
from .ratings import compute_elo_ratings

ELO_TIER_K = {"S": 32.0, "A": 20.0, "B": 14.0, "C": 10.0}


PLAYER_FIELDS = (
    "rating",
    "kd",
    "opening_success",
    "clutch_winrate",
    "star_rating",
)


@dataclass
class TeamHistory:
    results: Deque[tuple[str, bool, int, str]] = field(default_factory=deque)
    player_values: Dict[str, Deque[float]] = field(default_factory=lambda: defaultdict(deque))
    current_streak: int = 0


def enrich_match_history(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    histories: Dict[str, TeamHistory] = defaultdict(TeamHistory)
    h2h: Dict[frozenset[str], List[str]] = defaultdict(list)
    enriched_rows: List[Dict[str, Any]] = []

    ordered = sorted((dict(row) for row in rows), key=lambda item: item["date"])
    elo_per_match, _ = compute_elo_ratings(ordered, base=1500.0, k=24.0, tier_k=ELO_TIER_K)
    for row, elo in zip(ordered, elo_per_match):
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        map_name = _map_name(row.get("map"))
        played_at = parse_date(row["date"])
        enriched = dict(row)
        enriched["team1_elo"] = elo["team1_elo_pre"]
        enriched["team2_elo"] = elo["team2_elo_pre"]

        _apply_team_features(enriched, "team1", team1, team2, map_name, played_at, histories, h2h)
        _apply_team_features(enriched, "team2", team2, team1, map_name, played_at, histories, h2h)
        enriched["h2h_team1_winrate"] = _h2h_winrate(h2h[frozenset({team1, team2})], team1)
        enriched_rows.append(enriched)

        winner = str(row.get("winner", ""))
        _record_result(histories[team1], row, team1, winner == team1)
        _record_result(histories[team2], row, team2, winner == team2)
        h2h[frozenset({team1, team2})].append(winner)

    return enriched_rows


def build_team_profiles(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    map_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    map_wins: Dict[str, Counter[str]] = defaultdict(Counter)
    map_seen: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        team1 = str(row["team1"])
        team2 = str(row["team2"])
        winner = str(row.get("winner", ""))
        map_name = _map_name(row.get("map"))
        for team in (team1, team2):
            map_counts[team][map_name] += 1
            map_seen[team][map_name] += 1
        if winner in (team1, team2):
            map_wins[winner][map_name] += 1

    profiles: Dict[str, Dict[str, Any]] = {}
    all_maps = sorted({map_name for counter in map_counts.values() for map_name in counter})
    for team, counts in map_counts.items():
        prefer_top3 = [name for name, _ in counts.most_common(3)]
        low_play_maps = sorted(all_maps, key=lambda name: (counts.get(name, 0), name))[:3]
        profiles[team] = {
            "prefer_top3": prefer_top3,
            "ban_top3": low_play_maps,
            "map_winrates": {
                name: map_wins[team].get(name, 0) / map_seen[team].get(name, 1)
                for name in all_maps
                if map_seen[team].get(name, 0) > 0
            },
        }
    return profiles


def _apply_team_features(
    row: Dict[str, Any],
    prefix: str,
    team: str,
    opponent: str,
    map_name: str,
    played_at,
    histories: Dict[str, TeamHistory],
    h2h: Dict[frozenset[str], List[str]],
) -> None:
    history = histories[team]
    row[f"{prefix}_matches_30d"] = _matches_since(history.results, played_at, days=30)
    row[f"{prefix}_recent_winrate_5"] = _recent_winrate(history.results, limit=5)
    row[f"{prefix}_recent_winrate_10"] = _recent_winrate(history.results, limit=10)
    row[f"{prefix}_bo1_winrate_6m"] = _mode_winrate(history.results, best_of=1)
    row[f"{prefix}_bo3_winrate_6m"] = _mode_winrate(history.results, best_of=3)
    row[f"{prefix}_map_winrate"] = _map_winrate(history.results, map_name)
    row[f"{prefix}_current_streak"] = history.current_streak
    row[f"{prefix}_h2h_winrate_vs_opponent"] = _h2h_winrate(h2h[frozenset({team, opponent})], team)

    for field_name in PLAYER_FIELDS:
        key = f"{prefix}_{field_name}"
        if row.get(key) in (None, ""):
            row[key] = _recent_mean(history.player_values[field_name], default=_default_player_value(field_name))


def _record_result(history: TeamHistory, row: Mapping[str, Any], team: str, won: bool) -> None:
    best_of = int(_num(row.get("best_of"), 1))
    map_name = _map_name(row.get("map"))
    played_at = str(row["date"])
    history.results.append((played_at, won, best_of, map_name))
    history.current_streak = history.current_streak + 1 if won and history.current_streak >= 0 else 1 if won else history.current_streak - 1 if history.current_streak <= 0 else -1

    prefix = "team1" if row.get("team1") == team else "team2"
    for field_name in PLAYER_FIELDS:
        value = _optional_float(row.get(f"{prefix}_{field_name}"))
        if value is not None:
            history.player_values[field_name].append(value)


def _matches_since(results: Iterable[tuple[str, bool, int, str]], played_at, days: int) -> int:
    cutoff = played_at - timedelta(days=days)
    return sum(1 for raw_date, _, _, _ in results if parse_date(raw_date) >= cutoff)


def _recent_winrate(results: Deque[tuple[str, bool, int, str]], limit: int) -> float:
    selected = list(results)[-limit:]
    if not selected:
        return 0.5
    return sum(1 for _, won, _, _ in selected if won) / len(selected)


def _mode_winrate(results: Deque[tuple[str, bool, int, str]], best_of: int) -> float:
    selected = [won for _, won, mode, _ in results if mode == best_of]
    if not selected:
        return 0.5
    return sum(1 for won in selected if won) / len(selected)


def _map_winrate(results: Deque[tuple[str, bool, int, str]], map_name: str) -> float:
    selected = [won for _, won, _, played_map in results if played_map == map_name]
    if not selected:
        return 0.5
    return sum(1 for won in selected if won) / len(selected)


def _h2h_winrate(winners: List[str], team: str, limit: int = 3) -> float:
    if not winners:
        return 0.5
    selected = winners[-limit:]
    return sum(1 for winner in selected if winner == team) / len(selected)


def _recent_mean(values: Deque[float], default: float) -> float:
    if not values:
        return default
    return sum(values) / len(values)


def _default_player_value(field_name: str) -> float:
    if field_name in {"rating", "kd", "star_rating"}:
        return 1.0
    return 0.5


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_name(value: Any) -> str:
    return str(value or "unknown").strip().lower().replace("de_", "")
