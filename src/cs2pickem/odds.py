from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .data import read_matches_csv, write_matches_csv


def normalize_odds_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        date = str(row.get("date", ""))[:10]
        left = str(row.get("team1", ""))
        right = str(row.get("team2", ""))
        canonical_team1, canonical_team2 = _canonical_pair(left, right)
        odds_left = _num(row.get("odds_team1"), 2.0)
        odds_right = _num(row.get("odds_team2"), 2.0)
        if _team_key(left) == _team_key(canonical_team1):
            odds_team1, odds_team2 = odds_left, odds_right
        else:
            odds_team1, odds_team2 = odds_right, odds_left
        market_probability = _market_probability(odds_team1, odds_team2)
        normalized.append(
            {
                "date": date,
                "provider": row.get("provider", "unknown"),
                "team1": canonical_team1,
                "team2": canonical_team2,
                "odds_team1": odds_team1,
                "odds_team2": odds_team2,
                "market_probability_team1": market_probability,
                "canonical_key": _canonical_key(date, canonical_team1, canonical_team2),
            }
        )
    return normalized


def merge_odds_into_matches(matches: Iterable[Dict[str, Any]], odds_rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    odds_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in normalize_odds_rows(odds_rows):
        odds_by_key[row["canonical_key"]].append(row)

    output = []
    matched = 0
    for match in matches:
        copied = dict(match)
        key = _canonical_key(str(copied.get("date", ""))[:10], copied.get("team1", ""), copied.get("team2", ""))
        candidates = odds_by_key.get(key, [])
        if candidates:
            matched += 1
            canonical_team1, _ = _canonical_pair(copied.get("team1", ""), copied.get("team2", ""))
            canonical_probability = _average([row["market_probability_team1"] for row in candidates])
            canonical_odds_team1 = _average([row["odds_team1"] for row in candidates])
            canonical_odds_team2 = _average([row["odds_team2"] for row in candidates])
            if _team_key(copied.get("team1", "")) == _team_key(canonical_team1):
                copied["odds_team1"] = canonical_odds_team1
                copied["odds_team2"] = canonical_odds_team2
                copied["market_probability_team1"] = canonical_probability
            else:
                copied["odds_team1"] = canonical_odds_team2
                copied["odds_team2"] = canonical_odds_team1
                copied["market_probability_team1"] = 1.0 - canonical_probability
            copied["odds_provider_count"] = len(candidates)
            copied["odds_providers"] = sorted({str(row["provider"]) for row in candidates})
        output.append(copied)

    return output, {"matches": len(output), "matched": matched, "unmatched": len(output) - matched}


def merge_odds_file(matches_path: str, odds_path: str, output_path: str) -> Dict[str, int]:
    merged, report = merge_odds_into_matches(read_matches_csv(matches_path), read_matches_csv(odds_path))
    write_matches_csv(output_path, merged)
    return report


def _canonical_pair(team1: Any, team2: Any) -> Tuple[str, str]:
    left = str(team1)
    right = str(team2)
    return tuple(sorted([left, right], key=lambda value: _team_key(value)))  # type: ignore[return-value]


def _canonical_key(date: str, team1: Any, team2: Any) -> str:
    left, right = _canonical_pair(team1, team2)
    return f"{date}__{_team_key(left)}__{_team_key(right)}"


def _team_key(value: Any) -> str:
    return str(value).strip().lower()


def _market_probability(odds_team1: float, odds_team2: float) -> float:
    inv1 = 1.0 / odds_team1 if odds_team1 > 0 else 0.5
    inv2 = 1.0 / odds_team2 if odds_team2 > 0 else 0.5
    total = inv1 + inv2
    return inv1 / total if total else 0.5


def _average(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
