from __future__ import annotations

import csv
import json
from typing import Any, Dict, Iterable, List


NUMERIC_FIELDS = {
    "best_of": int,
    "seed": int,
    "strength": float,
    "rmr_points": float,
    "team1_rank": int,
    "team2_rank": int,
    "team1_rmr_points": float,
    "team2_rmr_points": float,
    "team1_major_best_placement": int,
    "team2_major_best_placement": int,
    "team1_recent_winrate_10": float,
    "team2_recent_winrate_10": float,
    "team1_bo1_winrate_6m": float,
    "team2_bo1_winrate_6m": float,
    "team1_bo3_winrate_6m": float,
    "team2_bo3_winrate_6m": float,
    "team1_map_winrate": float,
    "team2_map_winrate": float,
    "team1_rating": float,
    "team2_rating": float,
    "team1_kd": float,
    "team2_kd": float,
    "team1_opening_success": float,
    "team2_opening_success": float,
    "team1_clutch_winrate": float,
    "team2_clutch_winrate": float,
    "team1_star_rating": float,
    "team2_star_rating": float,
    "h2h_team1_winrate": float,
    "odds_team1": float,
    "odds_team2": float,
    "team1_odds": float,
    "team2_odds": float,
    "market_probability_team1": float,
    "market_signal_proxy": int,
    "hltv_poll_team1": float,
    "hltv_poll_team2": float,
    "swiss_round": int,
    "team1_wins": int,
    "team1_losses": int,
    "team2_wins": int,
    "team2_losses": int,
    "team1_matches_30d": int,
    "team2_matches_30d": int,
    "team1_recent_winrate_5": float,
    "team2_recent_winrate_5": float,
    "team1_current_streak": int,
    "team2_current_streak": int,
    "team1_substitute_flag": int,
    "team2_substitute_flag": int,
    "team1_player_sample": int,
    "team2_player_sample": int,
    "rating": float,
    "kd": float,
    "opening_success": float,
    "clutch_winrate": float,
    "is_substitute": int,
    "confidence": float,
    "bp_confidence": float,
    "bp_applied": int,
    "world_rank": int,
    "rank": int,
    "points": int,
}


def read_matches_csv(path: str) -> List[Dict[str, Any]]:
    return _read_csv(path)


def read_teams_csv(path: str) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    for index, row in enumerate(rows, start=1):
        row.setdefault("seed", index)
        row.setdefault("strength", max(0.05, 1.0 - (float(row["seed"]) - 1.0) / max(1.0, len(rows))))
    return rows


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def write_matches_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames = _fieldnames(materialized)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def _read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: _convert(key, value) for key, value in row.items()} for row in reader]


def _fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    ordered = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def _convert(key: str, value: str) -> Any:
    if value == "":
        return None
    converter = NUMERIC_FIELDS.get(key)
    if not converter:
        return value
    try:
        return converter(float(value)) if converter is int else converter(value)
    except (TypeError, ValueError):
        return value
