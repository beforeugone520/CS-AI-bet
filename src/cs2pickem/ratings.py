"""Opponent-strength (Elo) ratings derived from match results.

Elo naturally de-inflates strength-of-schedule: beating a weak team gains little,
beating a strong team gains a lot. Unlike the static world-rank column (which is a
constant placeholder in the 5E corpus), Elo is populated for every team, varies, and
is computable for upcoming-fixture teams from their own match history.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .maps import DEFAULT_MAP_POOL


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    matches: Sequence[Mapping[str, Any]],
    base: float = 1500.0,
    k: float = 24.0,
    initial_ratings: Optional[Mapping[str, float]] = None,
    tier_k: Optional[Mapping[str, float]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Process matches in chronological order and return pre-match Elo per match plus
    final ratings.

    Each returned per-match dict carries ``team1_elo_pre`` / ``team2_elo_pre`` — the
    ratings BEFORE that match is applied, so the feature is leakage-free. ``tier_k``
    optionally scales K by ``event_tier`` (e.g. Major wins move ratings more).
    """
    ratings: Dict[str, float] = dict(initial_ratings or {})
    ordered = sorted(range(len(matches)), key=lambda i: str(matches[i].get("date") or ""))
    per_match_by_index: Dict[int, Dict[str, float]] = {}

    for i in ordered:
        row = matches[i]
        team1 = str(row.get("team1") or "")
        team2 = str(row.get("team2") or "")
        winner = str(row.get("winner") or "")
        r1 = ratings.get(team1, base)
        r2 = ratings.get(team2, base)
        per_match_by_index[i] = {"team1_elo_pre": r1, "team2_elo_pre": r2}
        if not team1 or not team2 or winner not in (team1, team2):
            continue  # skip unscored/invalid rows but still expose pre-match ratings
        match_k = k
        if tier_k:
            match_k = tier_k.get(str(row.get("event_tier") or ""), k)
        s1 = 1.0 if winner == team1 else 0.0
        e1 = _expected_score(r1, r2)
        delta = match_k * (s1 - e1)
        ratings[team1] = r1 + delta
        ratings[team2] = r2 - delta

    per_match = [per_match_by_index[i] for i in range(len(matches))]
    return per_match, ratings


# ---------------------------------------------------------------------------
# Bradley-Terry strength ratings (batch MLE)
# ---------------------------------------------------------------------------
#
# Bradley-Terry is the no-intercept logistic model P(i beats j) = pi_i/(pi_i+pi_j),
# equivalently P = sigmoid(theta_i - theta_j) with theta_i = log(pi_i). The batch MLE
# is solved by the standard MM (minorization-maximization) fixed-point sweep:
#
#     pi_i <- (W_i + ridge*prior_pi_i) / ( sum_{j!=i} N_ij/(pi_i+pi_j) + ridge )
#
# where W_i is i's total wins and N_ij the games played between i and j. The ridge term
# is a Gaussian-style shrinkage pseudo-count toward a prior strength (uniform for the
# overall fit, the global BT strength for sparse per-map fits) so undefeated/winless
# teams and data-poor maps stay finite and identifiable. ``ridge``, ``max_iter`` and
# ``tol`` are defaulted, FeatureSelector-visible hyperparameters, not baked-in magic.

_BT_DEFAULT_RIDGE = 1.0
_BT_MAP_DEFAULT_RIDGE = 4.0
_BT_MAX_ITER = 200
_BT_TOL = 1e-6


def _collect_pairwise(
    matches: Sequence[Mapping[str, Any]],
    map_filter: Optional[str] = None,
) -> Tuple[List[str], Dict[str, float], Dict[Tuple[int, int], float]]:
    """Aggregate wins and pairwise game counts from scored result rows.

    Returns ``(teams, wins_by_index, pair_games)`` where ``wins_by_index`` maps a team's
    positional index to its total wins and ``pair_games`` maps an ``(i, j)`` index pair
    (i < j) to the number of games played between them. ``map_filter`` restricts to a
    single normalized map when provided. Uses only the supplied rows (callers pass
    pre-match history slices for leakage-free fits).
    """
    index_of: Dict[str, int] = {}
    teams: List[str] = []
    wins: Dict[int, float] = {}
    pair_games: Dict[Tuple[int, int], float] = {}

    def _idx(name: str) -> int:
        if name not in index_of:
            index_of[name] = len(teams)
            teams.append(name)
            wins[index_of[name]] = 0.0
        return index_of[name]

    for row in matches:
        team1 = str(row.get("team1") or "")
        team2 = str(row.get("team2") or "")
        winner = str(row.get("winner") or "")
        if not team1 or not team2 or team1 == team2:
            continue
        if winner not in (team1, team2):
            continue
        if map_filter is not None and _normalize_map(row.get("map")) != map_filter:
            continue
        i = _idx(team1)
        j = _idx(team2)
        key = (i, j) if i < j else (j, i)
        pair_games[key] = pair_games.get(key, 0.0) + 1.0
        wins[_idx(winner)] = wins.get(_idx(winner), 0.0) + 1.0

    return teams, wins, pair_games


def _bt_iterate(
    teams: Sequence[str],
    wins: Mapping[int, float],
    pair_games: Mapping[Tuple[int, int], float],
    prior_pi: Sequence[float],
    ridge: float,
    max_iter: int,
    tol: float,
) -> List[float]:
    """Run the MM fixed-point sweeps and return per-team strengths ``pi`` (positive)."""
    n = len(teams)
    if n == 0:
        return []
    pi = [1.0] * n
    win_arr = [float(wins.get(idx, 0.0)) for idx in range(n)]
    pairs = list(pair_games.items())
    ridge = max(0.0, float(ridge))

    for _ in range(max(1, max_iter)):
        denom = [ridge] * n  # ridge pseudo-count anchors the denominator
        for (i, j), games in pairs:
            inv = games / (pi[i] + pi[j])
            denom[i] += inv
            denom[j] += inv
        max_delta = 0.0
        new_pi = [0.0] * n
        for idx in range(n):
            numer = win_arr[idx] + ridge * prior_pi[idx]
            value = numer / denom[idx] if denom[idx] > 0 else prior_pi[idx]
            new_pi[idx] = value
        # Geometric-mean normalisation for identifiability (mean log-strength = 0).
        log_sum = sum(math.log(max(v, 1e-300)) for v in new_pi)
        scale = math.exp(-log_sum / n)
        for idx in range(n):
            normalised = new_pi[idx] * scale
            delta = abs(math.log(max(normalised, 1e-300)) - math.log(max(pi[idx], 1e-300)))
            if delta > max_delta:
                max_delta = delta
            new_pi[idx] = normalised
        pi = new_pi
        if max_delta < tol:
            break
    return pi


def compute_bradley_terry(
    matches: Sequence[Mapping[str, Any]],
    *,
    ridge: float = _BT_DEFAULT_RIDGE,
    max_iter: int = _BT_MAX_ITER,
    tol: float = _BT_TOL,
    prior: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Batch Bradley-Terry MLE over all results; returns mean-centered log-strength.

    ``theta_i = log(pi_i)`` per team, centered so the mean is 0 for identifiability.
    ``ridge`` shrinks toward ``prior`` (a {team: theta} map; uniform/neutral when None),
    keeping undefeated/winless teams finite. Iterates the MM fixed-point to ``tol`` on the
    mean absolute change in log-strength or ``max_iter`` sweeps, whichever comes first.
    Uses only the rows passed in (callers slice pre-match history for leakage-free fits).
    """
    teams, wins, pair_games = _collect_pairwise(matches)
    if not teams:
        return {}
    prior_pi = _prior_pi(teams, prior)
    pi = _bt_iterate(teams, wins, pair_games, prior_pi, ridge, max_iter, tol)
    theta = {team: math.log(max(pi[idx], 1e-300)) for idx, team in enumerate(teams)}
    return _mean_center(theta)


def compute_map_bradley_terry(
    matches: Sequence[Mapping[str, Any]],
    *,
    ridge: float = _BT_MAP_DEFAULT_RIDGE,
    global_theta: Optional[Mapping[str, float]] = None,
    map_pool: Optional[Sequence[str]] = None,
    max_iter: int = _BT_MAX_ITER,
    tol: float = _BT_TOL,
) -> Dict[str, Dict[str, float]]:
    """Per-map Bradley-Terry, one independent fit per map, shrunk toward the global BT.

    Sparse per-map samples are ridge-shrunk toward ``global_theta`` (the overall BT
    strength, computed from ``matches`` when not supplied). A map with no games collapses
    entirely to the (centered) global prior. ``ridge`` defaults conservative-high so
    data-poor maps lean on the global signal. Returns ``{map: {team: theta}}`` with each
    map's strengths mean-centered.
    """
    if global_theta is None:
        global_theta = compute_bradley_terry(matches, max_iter=max_iter, tol=tol)
    pool = [_normalize_map(name) for name in (map_pool if map_pool is not None else DEFAULT_MAP_POOL)]
    # Also fit any map seen in the data that is outside the pool, so nothing is dropped.
    seen = {_normalize_map(row.get("map")) for row in matches if row.get("map") not in (None, "")}
    ordered_maps = list(dict.fromkeys(pool + sorted(seen - set(pool))))

    per_map: Dict[str, Dict[str, float]] = {}
    for map_name in ordered_maps:
        teams, wins, pair_games = _collect_pairwise(matches, map_filter=map_name)
        if not teams:
            # No games on this map: fall back to the centered global prior over all
            # globally-known teams so downstream lookups still resolve.
            per_map[map_name] = _mean_center(dict(global_theta)) if global_theta else {}
            continue
        prior_pi = _prior_pi(teams, global_theta)
        pi = _bt_iterate(teams, wins, pair_games, prior_pi, ridge, max_iter, tol)
        theta = {team: math.log(max(pi[idx], 1e-300)) for idx, team in enumerate(teams)}
        per_map[map_name] = _mean_center(theta)
    return per_map


def _prior_pi(teams: Sequence[str], prior: Optional[Mapping[str, float]]) -> List[float]:
    """Convert a {team: theta} prior into per-team strengths ``pi`` (neutral=1.0)."""
    if not prior:
        return [1.0] * len(teams)
    return [math.exp(float(prior.get(team, 0.0))) for team in teams]


def _mean_center(theta: Dict[str, float]) -> Dict[str, float]:
    if not theta:
        return {}
    mean = sum(theta.values()) / len(theta)
    return {team: value - mean for team, value in theta.items()}


def _normalize_map(value: object) -> str:
    return str(value or "").strip().lower().replace("de_", "")
