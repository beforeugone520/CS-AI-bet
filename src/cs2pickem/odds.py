from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .data import read_matches_csv, write_matches_csv


def market_probability_from_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extract an auditable team1 market signal from odds, explicit probability, or poll proxy."""

    odds_team1 = _optional_num(row.get("odds_team1"))
    odds_team2 = _optional_num(row.get("odds_team2"))
    if odds_team1 is not None and odds_team2 is not None:
        return {
            "probability_team1": _market_probability(odds_team1, odds_team2),
            "basis": "real_odds",
            "source": row.get("market_signal_source") or row.get("provider") or row.get("odds_providers") or "odds",
            "proxy": False,
        }

    explicit_probability = _optional_num(row.get("market_probability_team1"))
    if explicit_probability is not None:
        basis = str(row.get("market_signal_basis") or "explicit_market_probability")
        proxy_value = row.get("market_signal_proxy")
        proxy = _truthy(proxy_value) if proxy_value not in (None, "") else bool(row.get("market_proxy_source")) or basis == "poll_proxy"
        return {
            "probability_team1": _clip_probability(explicit_probability),
            "basis": basis,
            "source": row.get("market_signal_source") or row.get("market_proxy_source") or "market_probability_team1",
            "proxy": proxy,
        }

    poll_team1 = _optional_num(row.get("hltv_poll_team1"))
    poll_team2 = _optional_num(row.get("hltv_poll_team2"))
    if poll_team1 is not None and poll_team2 is not None and poll_team1 + poll_team2 > 0:
        probability = poll_team1 / (poll_team1 + poll_team2)
        return {
            "probability_team1": _clip_probability(probability),
            "basis": "poll_proxy",
            "source": row.get("market_proxy_source") or "hltv_fan_poll",
            "proxy": True,
        }

    return None


def normalize_odds_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        date = str(row.get("date", ""))[:10]
        left = str(row.get("team1", ""))
        right = str(row.get("team2", ""))
        if not left or not right:
            continue
        canonical_team1, canonical_team2 = _canonical_pair(left, right)
        odds_left = _decimal_odds_from_row(row, ("odds_team1", "team1_odds", "decimal_odds_team1", "team1_decimal_odds"), ("odds_team1_american", "team1_american_odds", "american_odds_team1"))
        odds_right = _decimal_odds_from_row(row, ("odds_team2", "team2_odds", "decimal_odds_team2", "team2_decimal_odds"), ("odds_team2_american", "team2_american_odds", "american_odds_team2"))
        if odds_left is None or odds_right is None:
            continue
        if _team_key(left) == _team_key(canonical_team1):
            odds_team1, odds_team2 = odds_left, odds_right
        else:
            odds_team1, odds_team2 = odds_right, odds_left
        market_probability = _market_probability(odds_team1, odds_team2)
        provider = row.get("provider") or row.get("source") or "unknown"
        normalized.append(
            {
                "date": date,
                "provider": provider,
                "team1": canonical_team1,
                "team2": canonical_team2,
                "odds_team1": odds_team1,
                "odds_team2": odds_team2,
                "market_probability_team1": market_probability,
                "market_signal_source": provider,
                "market_signal_basis": "real_odds",
                "market_signal_proxy": False,
                "source_match_url": row.get("source_match_url"),
                "canonical_key": _canonical_key(date, canonical_team1, canonical_team2),
            }
        )
    return normalized


def merge_odds_into_matches(matches: Iterable[Dict[str, Any]], odds_rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    odds_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    odds_by_url: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    normalized_odds = normalize_odds_rows(odds_rows)
    for row in normalized_odds:
        odds_by_key[row["canonical_key"]].append(row)
        url = str(row.get("source_match_url") or "")
        if url:
            odds_by_url[url].append(row)

    output = []
    matched = 0
    matched_by_source_url = 0
    matched_by_canonical = 0
    for match in matches:
        copied = dict(match)
        key = _canonical_key(str(copied.get("date", ""))[:10], copied.get("team1", ""), copied.get("team2", ""))
        source_url = str(copied.get("source_match_url") or "")
        if source_url and source_url in odds_by_url:
            candidates = odds_by_url[source_url]
            matched_by_source_url += 1
        else:
            candidates = odds_by_key.get(key, [])
            if candidates:
                matched_by_canonical += 1
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
            copied["market_signal_source"] = "odds_provider_average"
            copied["market_signal_basis"] = "real_odds"
            copied["market_signal_proxy"] = False
        output.append(copied)

    return output, {
        "matches": len(output),
        "matched": matched,
        "unmatched": len(output) - matched,
        "odds_rows": len(normalized_odds),
        "matched_by_source_url": matched_by_source_url,
        "matched_by_canonical": matched_by_canonical,
    }


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


def _optional_num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decimal_odds_from_row(row: Dict[str, Any], decimal_keys: Tuple[str, ...], american_keys: Tuple[str, ...]) -> float | None:
    for key in decimal_keys:
        value = _optional_num(row.get(key))
        if value is not None:
            return value
    for key in american_keys:
        value = _optional_num(row.get(key))
        if value is not None:
            return _american_to_decimal(value)
    return None


def _american_to_decimal(value: float) -> float | None:
    if value < 0:
        return 1.0 + 100.0 / abs(value)
    if value > 0:
        return 1.0 + value / 100.0
    return None


def _clip_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y"}
