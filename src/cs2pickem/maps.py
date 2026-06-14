from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional

from .series import score_distribution, series_win_prob


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
    veto_weighted: bool = False,
    best_of: Optional[int] = None,
) -> Dict[str, object]:
    """Predict team1's win probability on an unknown (not-yet-vetoed) map.

    Default behavior (``veto_weighted=False``, ``best_of=None``) is preserved
    exactly for backward compatibility: candidate maps are scored from the ban/
    pick profiles and the per-map probabilities are averaged with **equal**
    weight (BO1 single-map view).

    Optional upgrades (all additive, default-off):

    * ``veto_weighted=True`` — weight each candidate map by a veto-likelihood
      score (how likely it is to actually be played given both teams' pick/ban
      tendencies) instead of a flat mean. The weights are the normalized
      candidate scores from :func:`candidate_maps_from_bp`'s scoring, so a map
      both teams prefer (and neither bans) carries more weight. This is a
      heuristic stand-in for the true sequential pick/ban game tree, not a
      step-by-step veto solver.
    * ``best_of`` — when given (3 or 5, ...), also compose the per-map
      probabilities into a **series** win probability and score-line
      distribution via :mod:`cs2pickem.series`. The ordered candidate
      probabilities are fed to the heterogeneous path enumeration. NOTE: this
      treats each map as independent (a known first-order approximation; see
      ``series`` module docstring).

      Two further caveats on the series outputs:
      - ``series_win_probability_team1`` is order-invariant (the enumeration sums
        over all winning paths), so it is robust to map ordering.
      - ``series_score_distribution`` is NOT order-invariant: the 2-0 vs 2-1 (or
        3-0/3-1/3-2) split depends on WHICH map sits in WHICH slot. The order
        used here is the candidate PREFERENCE-RANK order (``candidates``), which
        is a proxy and NOT the true veto/play order of the actual series. Treat
        the score-line split as approximate when maps are heterogeneous.
      - ``veto_weighted`` does NOT propagate into the series composition. It
        reshapes only the scalar BO1 ``average`` (and, via that average, the
        padded tail slots when fewer candidate maps than ``best_of`` exist); the
        per-map probabilities fed to the series enumeration remain unweighted, so
        the veto-likelihood signal does not currently inform the series path
        probabilities.
    """
    candidates = candidate_maps_from_bp(team1_profile, team2_profile, map_pool=map_pool, top_n=top_n)
    per_map = {}
    for map_name in candidates:
        row = dict(base_row)
        row["map"] = map_name
        row["team1_map_winrate"] = _map_winrate(team1_profile, map_name)
        row["team2_map_winrate"] = _map_winrate(team2_profile, map_name)
        per_map[map_name] = min(1.0, max(0.0, float(predictor(row))))

    weights = _veto_weights(candidates, team1_profile, team2_profile) if veto_weighted else None
    if weights and per_map:
        average = sum(per_map[name] * weights[name] for name in candidates)
    else:
        average = sum(per_map.values()) / len(per_map) if per_map else 0.5

    result: Dict[str, object] = {
        "candidate_maps": candidates,
        "per_map_probability_team1": per_map,
        "average_probability_team1": average,
    }
    if weights:
        result["map_weights"] = weights

    if best_of is not None:
        ordered = [per_map[name] for name in candidates]
        # Pad the ordered map list so a full best-of-N series can be enumerated
        # even when fewer candidate maps than `best_of` were surfaced; the
        # representative `average` win prob fills the remaining slots.
        if ordered:
            while len(ordered) < best_of:
                ordered.append(average)
            result["series_win_probability_team1"] = series_win_prob(ordered, best_of)
            result["series_score_distribution"] = score_distribution(ordered, best_of)
        else:
            result["series_win_probability_team1"] = series_win_prob(0.5, best_of)
            result["series_score_distribution"] = score_distribution(0.5, best_of)

    return result


def _veto_weights(
    candidates: List[str],
    team1_profile: Mapping[str, object],
    team2_profile: Mapping[str, object],
) -> Dict[str, float]:
    """Normalized "likely to be played" weights over candidate maps.

    Heuristic: a map's veto weight grows with how strongly both teams prefer it
    (and shrinks when it sits low in the pool). We reuse the same preference +
    balance components used to *select* candidate maps, then normalize to a
    probability simplex. This is an approximation of the real sequential pick/
    ban process, not a full veto game tree.
    """
    raw = {
        name: max(
            0.0,
            _preference_score(name, team1_profile)
            + _preference_score(name, team2_profile)
            + _balance_score(name, team1_profile, team2_profile),
        )
        for name in candidates
    }
    total = sum(raw.values())
    if total <= 0.0:
        # Degenerate (no preference signal at all): fall back to uniform.
        uniform = 1.0 / len(candidates) if candidates else 0.0
        return {name: uniform for name in candidates}
    return {name: value / total for name, value in raw.items()}


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
