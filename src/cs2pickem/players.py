from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Iterable, List

from .cleaning import parse_date
from .data import read_matches_csv, write_matches_csv


STAT_FIELDS = ("rating", "kd", "opening_success", "clutch_winrate")


def merge_player_stats_into_matches(
    matches: Iterable[Dict[str, Any]],
    player_rows: Iterable[Dict[str, Any]],
    window_days: int = 15,
) -> List[Dict[str, Any]]:
    players = [dict(row) for row in player_rows]
    output = []
    for match in matches:
        copied = dict(match)
        match_date = parse_date(copied["date"])
        for prefix in ("team1", "team2"):
            team = str(copied.get(prefix, ""))
            summary = _team_summary(team, prefix, copied, match_date, players, window_days)
            copied[f"{prefix}_rating"] = summary["rating"]
            copied[f"{prefix}_kd"] = summary["kd"]
            copied[f"{prefix}_opening_success"] = summary["opening_success"]
            copied[f"{prefix}_clutch_winrate"] = summary["clutch_winrate"]
            copied[f"{prefix}_star_rating"] = summary["star_rating"]
            copied[f"{prefix}_substitute_flag"] = summary["substitute_flag"]
            copied[f"{prefix}_player_sample"] = summary["player_sample"]
            copied[f"{prefix}_player_form_score"] = summary["player_form_score"]
            copied[f"{prefix}_player_form_trend"] = summary["player_form_trend"]
            copied[f"{prefix}_player_sample_confidence"] = summary["player_sample_confidence"]
        output.append(copied)
    return output


def merge_player_stats_file(matches_path: str, players_path: str, output_path: str, window_days: int = 15) -> Dict[str, int]:
    merged = merge_player_stats_into_matches(read_matches_csv(matches_path), read_matches_csv(players_path), window_days=window_days)
    write_matches_csv(output_path, merged)
    return {
        "matches": len(merged),
        "teams_augmented": len(merged) * 2,
        "window_days": window_days,
    }


def _team_summary(team: str, prefix: str, match: Dict[str, Any], match_date, players: List[Dict[str, Any]], window_days: int) -> Dict[str, Any]:
    cutoff = match_date - timedelta(days=window_days)
    selected = [
        row
        for row in players
        if str(row.get("team")) == team and cutoff <= parse_date(row.get("date")) < match_date
    ]
    defaults = {
        "rating": _num(match.get(f"{prefix}_rating"), 1.0),
        "kd": _num(match.get(f"{prefix}_kd"), 1.0),
        "opening_success": _num(match.get(f"{prefix}_opening_success"), 0.5),
        "clutch_winrate": _num(match.get(f"{prefix}_clutch_winrate"), 0.5),
        "star_rating": _num(match.get(f"{prefix}_star_rating"), _num(match.get(f"{prefix}_rating"), 1.0)),
        "substitute_flag": int(_num(match.get(f"{prefix}_substitute_flag"), 0.0)),
        "player_form_score": _num(match.get(f"{prefix}_player_form_score"), 0.0),
        "player_form_trend": _num(match.get(f"{prefix}_player_form_trend"), 0.0),
        "player_sample_confidence": _num(match.get(f"{prefix}_player_sample_confidence"), 0.0),
        "has_player_form_score": match.get(f"{prefix}_player_form_score") not in (None, ""),
    }
    selected_scores = _player_form_scores(selected)
    return {
        "rating": _average(selected, "rating", defaults["rating"]),
        "kd": _average(selected, "kd", defaults["kd"]),
        "opening_success": _average(selected, "opening_success", defaults["opening_success"]),
        "clutch_winrate": _average(selected, "clutch_winrate", defaults["clutch_winrate"]),
        "star_rating": max([_num(row.get("rating"), defaults["star_rating"]) for row in selected], default=defaults["star_rating"]),
        "substitute_flag": 1 if any(_truthy(row.get("is_substitute")) for row in selected) else defaults["substitute_flag"],
        "player_sample": len(selected),
        "player_form_score": sum(selected_scores) / len(selected_scores) if selected_scores else _default_player_form_score(defaults),
        "player_form_trend": _player_form_trend(selected) if selected_scores else defaults["player_form_trend"],
        "player_sample_confidence": min(1.0, len(selected) / 5.0) if selected else defaults["player_sample_confidence"],
    }


def _average(rows: List[Dict[str, Any]], key: str, default: float) -> float:
    values = [_num(row.get(key), default) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else default


def _player_form_scores(rows: List[Dict[str, Any]]) -> List[float]:
    return [_player_form_score(row) for row in rows]


def _player_form_trend(rows: List[Dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    ordered = sorted(rows, key=lambda row: parse_date(row.get("date")))
    return _player_form_score(ordered[-1]) - _player_form_score(ordered[0])


def _player_form_score(row: Dict[str, Any]) -> float:
    score = (
        (_num(row.get("rating"), 1.0) - 1.0) * 0.45
        + (_num(row.get("kd"), 1.0) - 1.0) * 0.30
        + (_num(row.get("opening_success"), 0.5) - 0.5) * 0.15
        + (_num(row.get("clutch_winrate"), 0.5) - 0.5) * 0.10
    )
    return max(-0.25, min(0.25, score))


def _default_player_form_score(defaults: Dict[str, Any]) -> float:
    if defaults.get("has_player_form_score"):
        return _num(defaults.get("player_form_score"), 0.0)
    return _player_form_score(defaults)


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "sub", "substitute"}
