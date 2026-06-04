from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from .maps import DEFAULT_MAP_POOL


MAP_FEATURE_NAMES = [f"map_{name}" for name in DEFAULT_MAP_POOL]


@dataclass
class Dataset:
    rows: List[List[float]]
    labels: List[int]
    feature_names: List[str]
    raw_rows: List[Dict[str, Any]]


class FeatureBuilder:
    """Build normalized static, dynamic, cross, and Swiss-state features."""

    feature_names = [
        "rank_diff",
        "elo_diff",
        "rmr_points_diff",
        "major_best_placement_diff",
        "matches_30d_diff",
        "recent_winrate_5_diff",
        "recent_winrate_10_diff",
        "bo1_winrate_diff",
        "bo3_winrate_diff",
        "map_winrate_diff",
        *MAP_FEATURE_NAMES,
        "rating_diff",
        "kd_diff",
        "opening_success_diff",
        "clutch_winrate_diff",
        "star_rating_diff",
        "substitute_flag_diff",
        "player_sample_diff",
        "player_form_score_diff",
        "player_form_trend_diff",
        "player_sample_confidence_diff",
        "h2h_team1_winrate",
        "odds_implied_diff",
        "is_bo1",
        "is_bo3",
        "swiss_round",
        "team1_wins",
        "team1_losses",
        "team2_wins",
        "team2_losses",
        "swiss_score_diff",
        "wins_needed_to_advance_diff",
        "losses_until_elimination_diff",
        "current_streak_diff",
        "team1_code",
        "team2_code",
        "event_code",
        "event_tier_code",
        "version_tag_code",
    ]

    def __init__(self) -> None:
        self._minimums: Dict[str, float] = {}
        self._maximums: Dict[str, float] = {}
        self._version_codes: Dict[str, int] = {}
        self._category_codes: Dict[str, Dict[str, int]] = {}

    def fit_transform(self, rows: Iterable[Dict[str, Any]]) -> Dataset:
        raw_rows = list(rows)
        raw_matrix = [self._raw_features(row) for row in raw_rows]
        self._fit_scaler(raw_matrix)
        return Dataset(
            rows=[self._normalize(feature_row) for feature_row in raw_matrix],
            labels=[1 if row.get("winner") == row.get("team1") else 0 for row in raw_rows],
            feature_names=list(self.feature_names),
            raw_rows=raw_rows,
        )

    def transform(self, rows: Iterable[Dict[str, Any]]) -> List[List[float]]:
        return [self._normalize(self._raw_features(row)) for row in rows]

    def _fit_scaler(self, rows: Sequence[Dict[str, float]]) -> None:
        for name in self.feature_names:
            values = [row[name] for row in rows] or [0.0]
            self._minimums[name] = min(values)
            self._maximums[name] = max(values)

    def _normalize(self, row: Dict[str, float]) -> List[float]:
        normalized = []
        for name in self.feature_names:
            low = self._minimums.get(name, 0.0)
            high = self._maximums.get(name, 1.0)
            if high == low:
                normalized.append(row[name] if _passthrough_binary(name) else 0.5)
            else:
                normalized.append((row[name] - low) / (high - low))
        return [min(1.0, max(0.0, value)) for value in normalized]

    def _raw_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        odds_team1 = _num(row, "odds_team1", 2.0)
        odds_team2 = _num(row, "odds_team2", 2.0)
        implied_1, implied_2 = _implied_market_pair(odds_team1, odds_team2)
        team1_score = _num(row, "team1_wins") - _num(row, "team1_losses")
        team2_score = _num(row, "team2_wins") - _num(row, "team2_losses")
        best_of = int(_num(row, "best_of", 1))
        map_name = _normalize_map_name(row.get("map", "unknown"))
        team1_wins = _num(row, "team1_wins")
        team2_wins = _num(row, "team2_wins")
        team1_losses = _num(row, "team1_losses")
        team2_losses = _num(row, "team2_losses")
        team1_wins_needed = max(0.0, 3.0 - team1_wins)
        team2_wins_needed = max(0.0, 3.0 - team2_wins)
        team1_losses_until_elimination = max(0.0, 3.0 - team1_losses)
        team2_losses_until_elimination = max(0.0, 3.0 - team2_losses)

        features = {
            "rank_diff": _num(row, "team2_rank", 80) - _num(row, "team1_rank", 80),
            "elo_diff": _num(row, "team1_elo", 1500.0) - _num(row, "team2_elo", 1500.0),
            "rmr_points_diff": _num(row, "team1_rmr_points") - _num(row, "team2_rmr_points"),
            "major_best_placement_diff": _num(row, "team2_major_best_placement", 32) - _num(row, "team1_major_best_placement", 32),
            "matches_30d_diff": _num(row, "team1_matches_30d") - _num(row, "team2_matches_30d"),
            "recent_winrate_5_diff": _num(row, "team1_recent_winrate_5", 0.5) - _num(row, "team2_recent_winrate_5", 0.5),
            "recent_winrate_10_diff": _num(row, "team1_recent_winrate_10", 0.5) - _num(row, "team2_recent_winrate_10", 0.5),
            "bo1_winrate_diff": _num(row, "team1_bo1_winrate_6m", 0.5) - _num(row, "team2_bo1_winrate_6m", 0.5),
            "bo3_winrate_diff": _num(row, "team1_bo3_winrate_6m", 0.5) - _num(row, "team2_bo3_winrate_6m", 0.5),
            "map_winrate_diff": _num(row, "team1_map_winrate", 0.5) - _num(row, "team2_map_winrate", 0.5),
            "rating_diff": _num(row, "team1_rating", 1.0) - _num(row, "team2_rating", 1.0),
            "kd_diff": _num(row, "team1_kd", 1.0) - _num(row, "team2_kd", 1.0),
            "opening_success_diff": _num(row, "team1_opening_success", 0.5) - _num(row, "team2_opening_success", 0.5),
            "clutch_winrate_diff": _num(row, "team1_clutch_winrate", 0.5) - _num(row, "team2_clutch_winrate", 0.5),
            "star_rating_diff": _num(row, "team1_star_rating", 1.0) - _num(row, "team2_star_rating", 1.0),
            "substitute_flag_diff": _num(row, "team1_substitute_flag") - _num(row, "team2_substitute_flag"),
            "player_sample_diff": _num(row, "team1_player_sample") - _num(row, "team2_player_sample"),
            "player_form_score_diff": _num(row, "team1_player_form_score") - _num(row, "team2_player_form_score"),
            "player_form_trend_diff": _num(row, "team1_player_form_trend") - _num(row, "team2_player_form_trend"),
            "player_sample_confidence_diff": _num(row, "team1_player_sample_confidence") - _num(row, "team2_player_sample_confidence"),
            "h2h_team1_winrate": _num(row, "h2h_team1_winrate", 0.5),
            "odds_implied_diff": implied_1 - implied_2,
            "is_bo1": 1.0 if best_of == 1 else 0.0,
            "is_bo3": 1.0 if best_of == 3 else 0.0,
            "swiss_round": _num(row, "swiss_round", 1),
            "team1_wins": team1_wins,
            "team1_losses": team1_losses,
            "team2_wins": team2_wins,
            "team2_losses": team2_losses,
            "swiss_score_diff": team1_score - team2_score,
            "wins_needed_to_advance_diff": team2_wins_needed - team1_wins_needed,
            "losses_until_elimination_diff": team1_losses_until_elimination - team2_losses_until_elimination,
            "current_streak_diff": _num(row, "team1_current_streak") - _num(row, "team2_current_streak"),
            "team1_code": float(self._category_code("team", str(row.get("team1", "unknown")))),
            "team2_code": float(self._category_code("team", str(row.get("team2", "unknown")))),
            "event_code": float(self._category_code("event", str(row.get("event", "unknown")))),
            "event_tier_code": float(self._category_code("event_tier", str(row.get("event_tier", "unknown")))),
            "version_tag_code": float(self._version_code(str(row.get("version_tag", "unknown")))),
        }
        for name in DEFAULT_MAP_POOL:
            features[f"map_{name}"] = 1.0 if map_name == name else 0.0
        return features

    def _version_code(self, tag: str) -> int:
        if tag not in self._version_codes:
            self._version_codes[tag] = len(self._version_codes)
        return self._version_codes[tag]

    def _category_code(self, namespace: str, value: str) -> int:
        codes = self._category_codes.setdefault(namespace, {})
        if value not in codes:
            codes[value] = len(codes)
        return codes[value]


def _num(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _implied_market_pair(odds_team1: float, odds_team2: float) -> tuple[float, float]:
    inv1 = 1.0 / odds_team1 if odds_team1 > 0 else 0.5
    inv2 = 1.0 / odds_team2 if odds_team2 > 0 else 0.5
    total = inv1 + inv2
    if total == 0:
        return 0.5, 0.5
    return inv1 / total, inv2 / total


def _normalize_map_name(value: Any) -> str:
    return str(value).strip().lower().replace("de_", "")


def _passthrough_binary(name: str) -> bool:
    return name in {"is_bo1", "is_bo3"} or name.startswith("map_")
