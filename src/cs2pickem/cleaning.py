from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, Iterable, List, Optional


VALID_TIERS = {"S", "A"}
INVALID_STATUSES = {
    "aborted",
    "cancelled",
    "canceled",
    "default",
    "forfeit",
    "remake",
    "restart",
    "restarted",
    "retired",
    "walkover",
}
SECONDARY_MARKERS = ("academy", "jr", "junior", "youth", "u21", "u18", "talent")
TEMPORARY_MARKERS = ("mix", "stack", "stand-in", "stand in", "temporary", "pickup", "pick-up", "pug")
OUTLIER_FIELDS = ("team1_rating", "team2_rating", "team1_kd", "team2_kd")


def parse_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not value:
        raise ValueError("date is required")
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def clean_matches(
    rows: Iterable[Dict[str, Any]],
    reference_date: Optional[Any] = None,
    max_age_days: int = 90,
    valid_tiers: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Return high-quality CS2 match rows suitable for model training."""
    ref = parse_date(reference_date) if reference_date else date.today()
    cutoff = ref - timedelta(days=max_age_days)
    tiers = {normalize_tier(tier) for tier in (valid_tiers or VALID_TIERS)}

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        try:
            played_at = parse_date(row.get("date"))
        except ValueError:
            continue
        if played_at < cutoff:
            continue
        if normalize_tier(row.get("event_tier", "")) not in tiers:
            continue
        if str(row.get("status", "")).strip().lower() in INVALID_STATUSES:
            continue
        if _is_secondary_team(row, "team1") or _is_secondary_team(row, "team2"):
            continue
        if _is_temporary_team(row, "team1") or _is_temporary_team(row, "team2"):
            continue

        copied = dict(row)
        copied["date"] = played_at.isoformat()
        if copied.get("h2h_team1_winrate") in (None, ""):
            copied["h2h_team1_winrate"] = 0.5
        candidates.append(copied)

    return _filter_statistical_outliers(candidates)


def _is_secondary_team(row: Dict[str, Any], prefix: str) -> bool:
    if _truthy(row.get(f"{prefix}_is_secondary")):
        return True
    team_name = str(row.get(prefix, "")).lower()
    return any(marker in team_name for marker in SECONDARY_MARKERS)


def _is_temporary_team(row: Dict[str, Any], prefix: str) -> bool:
    if _truthy(row.get(f"{prefix}_is_temporary")):
        return True
    team_name = str(row.get(prefix, "")).lower()
    return any(marker in team_name for marker in TEMPORARY_MARKERS)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_tier(value: Any) -> str:
    return str(value).strip().upper()


def _filter_statistical_outliers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats = _population_stats(rows)
    return [row for row in rows if not _is_statistical_outlier(row, stats)]


def _population_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for field in OUTLIER_FIELDS:
        values = [_to_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        stats[field] = {
            "count": float(len(values)),
            "sum": sum(values),
            "sum_sq": sum(value * value for value in values),
        }
    return stats


def _is_statistical_outlier(row: Dict[str, Any], stats: Dict[str, Dict[str, float]]) -> bool:
    for field in OUTLIER_FIELDS:
        value = _to_float(row.get(field))
        if value is None:
            continue
        field_stats = stats.get(field, {})
        count = int(field_stats.get("count", 0.0))
        count_without_row = count - 1
        if count_without_row < 3:
            continue
        total = field_stats.get("sum", 0.0) - value
        total_sq = field_stats.get("sum_sq", 0.0) - value * value
        center = total / count_without_row
        variance = max(0.0, total_sq / count_without_row - center * center)
        sigma = math.sqrt(variance)
        if sigma == 0:
            continue
        if abs(value - center) > 3 * sigma:
            return True
    return False


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
