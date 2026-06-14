from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .data import read_matches_csv, write_matches_csv


DEVIG_METHODS = ("multiplicative", "power", "shin")
DEFAULT_DEVIG_METHOD = "multiplicative"


def market_probability_from_row(
    row: Dict[str, Any],
    method: str = DEFAULT_DEVIG_METHOD,
) -> Dict[str, Any] | None:
    """Extract an auditable team1 market signal from odds, explicit probability, or poll proxy.

    When raw two-way odds are present the team1 probability is the de-vigged
    fair probability. ``method`` selects the de-vig technique
    (``multiplicative`` by default for backward compatibility, or ``power`` /
    ``shin``); the returned signal also carries the audit fields
    ``overround``, ``devig_z``, ``devig_power_k`` and ``devig_method``.
    """

    odds_team1 = _optional_num(row.get("odds_team1"))
    odds_team2 = _optional_num(row.get("odds_team2"))
    if odds_team1 is not None and odds_team2 is not None:
        audit = devig_market(odds_team1, odds_team2, method=method)
        return {
            "probability_team1": audit["fair_prob_team1"],
            "basis": "real_odds",
            "source": row.get("market_signal_source") or row.get("provider") or row.get("odds_providers") or "odds",
            "proxy": False,
            "overround": audit["overround"],
            "devig_z": audit["devig_z"],
            "devig_power_k": audit["devig_power_k"],
            "devig_method": audit["devig_method"],
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


def normalize_odds_rows(
    rows: Iterable[Dict[str, Any]],
    method: str = DEFAULT_DEVIG_METHOD,
) -> List[Dict[str, Any]]:
    resolved_method = _resolve_method(method)
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
        audit = devig_market(odds_team1, odds_team2, method=resolved_method)
        market_probability = audit["fair_prob_team1"]
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
                "fair_prob_team1": audit["fair_prob_team1"],
                "fair_prob_team2": audit["fair_prob_team2"],
                "overround": audit["overround"],
                "devig_z": audit["devig_z"],
                "devig_power_k": audit["devig_power_k"],
                "devig_method": audit["devig_method"],
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
            # Merge the de-vig audit magnitudes back onto the match row so the (previously
            # constant-0) odds_overround / odds_devig_z feature columns carry real signal once
            # the join lands. overround is always present; devig_z is None for multiplicative /
            # power de-vig, so average only the providers that report it (and omit the key when
            # none do, leaving the feature's neutral 0 default untouched -- no silent skew).
            overrounds = [row["overround"] for row in candidates if row.get("overround") is not None]
            if overrounds:
                copied["overround"] = _average(overrounds)
            devig_zs = [row["devig_z"] for row in candidates if row.get("devig_z") is not None]
            if devig_zs:
                copied["devig_z"] = _average(devig_zs)
            copied["devig_method"] = candidates[0].get("devig_method")
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


def devig_market(
    odds_team1: float,
    odds_team2: float,
    method: str = DEFAULT_DEVIG_METHOD,
) -> Dict[str, Any]:
    """De-vig a two-outcome market into auditable fair probabilities.

    Supports three pluggable methods (``multiplicative``, ``power``, ``shin``).
    Returns a dict with ``fair_prob_team1``/``fair_prob_team2`` plus the audit
    fields ``overround``, ``devig_z`` (Shin insider proportion, ``None`` outside
    Shin), ``devig_power_k`` (solved exponent, ``None`` outside power) and
    ``devig_method``. Degenerate odds (non-positive) fall back to an even split
    rather than raising.
    """

    resolved = _resolve_method(method)
    inv1 = 1.0 / odds_team1 if odds_team1 and odds_team1 > 0 else None
    inv2 = 1.0 / odds_team2 if odds_team2 and odds_team2 > 0 else None
    if inv1 is None or inv2 is None or (inv1 + inv2) <= 0.0:
        return {
            "fair_prob_team1": 0.5,
            "fair_prob_team2": 0.5,
            "overround": 0.0,
            "devig_z": None,
            "devig_power_k": None,
            "devig_method": resolved,
        }

    overround = inv1 + inv2 - 1.0
    fair, z, power_k = _devig((inv1, inv2), resolved)
    return {
        "fair_prob_team1": fair[0],
        "fair_prob_team2": fair[1],
        "overround": overround,
        "devig_z": z,
        "devig_power_k": power_k,
        "devig_method": resolved,
    }


def _resolve_method(method: Optional[str]) -> str:
    resolved = str(method or DEFAULT_DEVIG_METHOD).strip().lower()
    if resolved not in DEVIG_METHODS:
        raise ValueError(f"unknown devig method: {method!r}; expected one of {DEVIG_METHODS}")
    return resolved


def normalize_devig_method(method: Optional[str]) -> str:
    """Public validator/normaliser for the de-vig method axis.

    Thin wrapper over the internal resolver so callers (e.g. the WF-2F backtest's
    devig A/B) can validate/normalise a requested method without reaching into a
    private name. Defaults to ``multiplicative`` for backward compatibility.
    """
    return _resolve_method(method)


def _devig(
    implied: Sequence[float],
    method: str,
) -> Tuple[List[float], Optional[float], Optional[float]]:
    """Normalize raw implied probabilities into fair probabilities.

    Returns ``(fair_probs, z, power_k)`` where ``z`` is the Shin insider
    proportion (``None`` unless Shin) and ``power_k`` is the solved power
    exponent (``None`` unless power).
    """

    raw = [float(value) for value in implied]
    total = math.fsum(raw)
    if total <= 0.0:
        even = [1.0 / len(raw)] * len(raw) if raw else []
        return even, None, None
    if method == "multiplicative":
        return _devig_multiplicative(raw, total)
    if method == "power":
        return _devig_power(raw)
    if method == "shin":
        return _devig_shin(raw, total)
    raise ValueError(f"unknown devig method: {method!r}")


def _devig_multiplicative(raw: Sequence[float], total: float) -> Tuple[List[float], None, None]:
    fair = [value / total for value in raw]
    return fair, None, None


def _devig_power(raw: Sequence[float]) -> Tuple[List[float], None, float]:
    total = math.fsum(raw)
    if total == 1.0 or all(value <= 0.0 for value in raw):
        # No overround (or degenerate) -> identity exponent.
        return list(raw), None, 1.0

    def g(exponent: float) -> float:
        return math.fsum(_safe_pow(value, exponent) for value in raw) - 1.0

    # g(k) = sum(raw_i ** k) - 1 is strictly decreasing in k when sum(raw) > 1
    # (every raw_i < 1 in that regime), and increasing-from-below when sum < 1.
    # Bracket the root with a monotone bisection that adapts to either side.
    lo, hi = 0.0, 1.0
    if total > 1.0:
        # Root k > 1: push hi out until g(hi) <= 0.
        hi = 1.0
        while g(hi) > 0.0 and hi < 1e6:
            hi *= 2.0
    else:
        # Overround < 0 (rare): root k < 1; g(lo)=len-1>0, g(1)=total-1<0.
        hi = 1.0
        if g(lo) <= 0.0:  # g(0) == len(raw) - 1; only <=0 for a degenerate 1-way market
            return list(raw), None, 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = g(mid)
        if abs(value) < 1e-12 or (hi - lo) < 1e-10:
            break
        if value > 0.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    powered = [_safe_pow(value, k) for value in raw]
    denom = math.fsum(powered)
    fair = [value / denom for value in powered] if denom > 0.0 else _devig_multiplicative(raw, math.fsum(raw))[0]
    return fair, None, k


def _devig_shin(raw: Sequence[float], total: float) -> Tuple[List[float], float, None]:
    """Shin (1992/1993) two-outcome closed form.

    The Shin model treats observed (over-round) booking probabilities as a
    mixture of an insider fraction ``z`` of informed money and ``1 - z`` of
    uninformed money. For ``n = 2`` outcomes the fair probabilities admit an
    exact analytic inverse:

        p_i = ( sqrt(z**2 + 4*(1 - z)*raw_i**2 / o) - z ) / (2*(1 - z))

    where ``raw_i = 1/odds_i`` and ``o = sum_i raw_i = overround + 1``. The
    insider proportion ``z`` is the unique root in ``[0, 1)`` of
    ``sum_i p_i = 1``. Imposing that constraint and solving the resulting
    quadratic exactly (no numerical search, no post-hoc renormalisation) gives

        z = 1 - 2*(1 - a - b) / (1 - (b - a)**2),   a = raw_1**2/o, b = raw_2**2/o.

    The returned probabilities sum to 1 by construction; ``z`` is exposed as a
    disagreement diagnostic (larger ``z`` => more implied insider/skew money).
    """

    if len(raw) != 2:
        # Outside the 2-way case there is no scalar closed form here.
        return _devig_multiplicative(raw, total)[0], 0.0, None

    o = total  # overround + 1
    if o <= 1.0:
        # No vig (or negative margin): no insider money to back out.
        return _devig_multiplicative(raw, total)[0], 0.0, None

    raw1, raw2 = float(raw[0]), float(raw[1])
    a = raw1 * raw1 / o
    b = raw2 * raw2 / o
    denom = 1.0 - (b - a) * (b - a)
    if denom <= 0.0:
        return _devig_multiplicative(raw, total)[0], 0.0, None
    z = 1.0 - 2.0 * (1.0 - a - b) / denom
    z = max(0.0, min(1.0, z))
    if z <= 0.0:
        return _devig_multiplicative(raw, total)[0], 0.0, None
    if z >= 1.0:
        return _devig_multiplicative(raw, total)[0], z, None

    fair = []
    for value in (raw1, raw2):
        inner = z * z + 4.0 * (1.0 - z) * value * value / o
        inner = max(0.0, inner)
        fair.append((math.sqrt(inner) - z) / (2.0 * (1.0 - z)))
    return fair, z, None


def _safe_pow(base: float, exponent: float) -> float:
    if base <= 0.0:
        return 0.0
    return math.pow(base, exponent)


def _market_probability(
    odds_team1: float,
    odds_team2: float,
    method: str = DEFAULT_DEVIG_METHOD,
) -> float:
    """Fair team1 probability after removing the bookmaker margin.

    Defaults to multiplicative de-vig for backward compatibility; pass
    ``method`` to switch to ``power`` or ``shin``.
    """

    return devig_market(odds_team1, odds_team2, method=method)["fair_prob_team1"]


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
