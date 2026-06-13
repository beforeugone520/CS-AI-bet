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


# ---------------------------------------------------------------------------
# Glicko-2 ratings (period-batched, with inactivity RD inflation + MOV damping)
# ---------------------------------------------------------------------------
#
# Glicko-2 (Glickman, 2013, "Example of the Glicko-2 system") tracks three state
# variables per team: mu (rating, original 1500-scale), phi (rating deviation / RD)
# and sigma (volatility). The standard update batches all of a team's games inside a
# *rating period* (here a calendar day) and runs five steps in the internal Glicko-2
# scale (divide rating offsets by SCALE=173.7178, RD by SCALE). The volatility step
# solves f(x)=0 with the Illinois (regula-falsi) variant, exactly as in the paper.
#
# Anti-leakage: we order rows by date, group by day, snapshot every row in a day with
# the state from BEFORE that day, and only then batch-update with the day's results.
# A day's games never feed the snapshot that scores them, and the rolling state never
# reads future days. This mirrors compute_elo_ratings' "use then update" but at period
# (not per-game) granularity: same-day games do not influence one another.
#
# All system constants below are FROZEN literature values, NOT tuned on the corpus
# (tuning a rating constant on the full corpus would leak future information). They are
# exposed only as default kwargs so a later held-out stage (WF-2F) can probe them.

GLICKO_SCALE = 173.7178  # Glicko-2 scale factor (Glickman 2013)
GLICKO_MU0 = 1500.0      # default initial rating
GLICKO_PHI0 = 350.0      # default initial RD (also the inactivity cap)
GLICKO_SIGMA0 = 0.06     # default initial volatility (Glickman 2013)
GLICKO_TAU = 0.5         # volatility constraint; Glickman recommends 0.3-1.2, 0.5 typical
GLICKO_EPS = 1e-6        # Illinois convergence threshold |B - A|
GLICKO_MAX_RD = 350.0    # RD ceiling so a long inactivity gap cannot diverge

# Margin-of-victory damping. The MULTIPLIER EXPRESSION is taken verbatim from the
# FiveThirtyEight NFL Elo margin-of-victory multiplier (Silver / Boice, FiveThirtyEight,
# "How Our NFL Predictions Work"):
#     mult = ln(round_diff + 1) * (beta / (beta + gamma * |elo_diff|)),
# with the published constants beta=2.2, gamma=0.001. The ln(.) term gives diminishing
# returns to blowouts; the second factor is the autocorrelation correction that damps a
# strong favourite running up the score. alpha is an overall scale (1.0 = neutral).
#
# DEVIATION FROM THE SOURCE (made explicit per WF-2C review): FiveThirtyEight applies this
# multiplier to the Elo K-FACTOR only, i.e. it scales the rating DISPLACEMENT. Here the
# weight is fed into BOTH the Glicko-2 information sum v (line ~530) AND the displacement
# sum Delta (line ~531), i.e. a blowout is treated as a higher-information observation, so
# it both moves the rating more AND shrinks RD more (the model gets more certain). This is
# a deliberate modelling CHOICE, not the 538 semantics; it is defensible (a decisive map is
# more informative) but reviewers correctly note it diverges from "MOV only amplifies the
# rating move". WF-2F should A/B this against the displacement-only variant (weight on Delta
# only, v left as the standard per-map information) before relying on MOV-weighted RD.
#
# The multiplier is applied ONLY inside the Glicko update; it never enters the pre-match
# snapshot, so it cannot create leakage. With no round scores the multiplier is exactly 1.0
# (standard Glicko-2). These are frozen literature values exposed only as default kwargs
# (do NOT tune on the corpus -- that would leak future information).
MOV_ALPHA = 1.0
MOV_BETA = 2.2
MOV_GAMMA = 0.001


def glicko_g(rd: float) -> float:
    """g(phi): attenuation factor for an opponent of rating deviation ``rd`` (orig scale).

    Decreasing in ``rd`` -> a more uncertain opponent flattens the win expectation toward
    0.5. Computed in the internal Glicko-2 scale (phi = rd / SCALE).
    """
    phi = float(rd) / GLICKO_SCALE
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def glicko_expected_score(mu: float, opp_mu: float, opp_rd: float = GLICKO_PHI0) -> float:
    """E: expected score (win prob) of a team at rating ``mu`` vs an opponent at ``opp_mu``
    with deviation ``opp_rd``. Inputs are original 1500-scale; the high-RD opponent pulls
    the expectation toward 0.5 (cold-start shrinkage)."""
    g = glicko_g(opp_rd)
    # Glicko-2 scale difference of the ratings.
    diff = (float(mu) - float(opp_mu)) / GLICKO_SCALE
    return 1.0 / (1.0 + math.exp(-g * diff))


def _mov_multiplier(
    round_diff: float,
    rating_gap: float,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """FiveThirtyEight-style margin-of-victory observation weight (>= 0).

    ``round_diff`` is the absolute map round difference; ``rating_gap`` the absolute
    pre-match rating gap |mu_winner - mu_loser| (original scale). Returns 1.0 when
    ``round_diff <= 0`` (the win/loss fallback path)."""
    if round_diff <= 0:
        return 1.0
    autocorr = beta / (beta + gamma * abs(rating_gap))
    return alpha * math.log(1.0 + round_diff) * autocorr


def _round_diff(row: Mapping[str, Any]) -> Optional[float]:
    """Absolute round-score difference, or None when scores are missing / sum to zero."""
    s1 = row.get("team1_score")
    s2 = row.get("team2_score")
    if s1 in (None, "") or s2 in (None, ""):
        return None
    try:
        v1 = float(s1)
        v2 = float(s2)
    except (TypeError, ValueError):
        return None
    if v1 + v2 == 0:
        return None
    return abs(v1 - v2)


def _solve_volatility(delta_sq: float, phi: float, v: float, sigma: float, tau: float) -> float:
    """Illinois (regula-falsi) solution of the Glicko-2 volatility equation.

    Solves f(x) = 0 for x = ln(sigma'^2) where
        f(x) = e^x (Delta^2 - phi^2 - v - e^x) / (2 (phi^2 + v + e^x)^2) - (x - a) / tau^2,
    a = ln(sigma^2). Returns sigma' = exp(A/2). Pure standard Glicko-2; no MOV here.
    """
    a = math.log(sigma * sigma)

    def f(x: float) -> float:
        ex = math.exp(x)
        denom = phi * phi + v + ex
        return (ex * (delta_sq - phi * phi - v - ex)) / (2.0 * denom * denom) - (x - a) / (tau * tau)

    big_A = a
    if delta_sq > phi * phi + v:
        big_B = math.log(delta_sq - phi * phi - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        big_B = a - k * tau

    f_A = f(big_A)
    f_B = f(big_B)
    while abs(big_B - big_A) > GLICKO_EPS:
        big_C = big_A + (big_A - big_B) * f_A / (f_B - f_A)
        f_C = f(big_C)
        if f_C * f_B <= 0:
            big_A = big_B
            f_A = f_B
        else:
            f_A = f_A / 2.0
        big_B = big_C
        f_B = f_C
    return math.exp(big_A / 2.0)


def _default_glicko_state() -> Dict[str, Dict[str, float]]:
    return {"ratings": {}, "rds": {}, "sigmas": {}}


def compute_glicko_ratings(
    matches: Sequence[Mapping[str, Any]],
    *,
    mu0: float = GLICKO_MU0,
    phi0: float = GLICKO_PHI0,
    sigma0: float = GLICKO_SIGMA0,
    tau: float = GLICKO_TAU,
    max_rd: float = GLICKO_MAX_RD,
    use_mov: bool = True,
    mov_alpha: float = MOV_ALPHA,
    mov_beta: float = MOV_BETA,
    mov_gamma: float = MOV_GAMMA,
    initial_state: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Standard Glicko-2 over period-batched (per calendar day) results.

    Returns ``(per_match_pre_snapshots, final_state)`` mirroring ``compute_elo_ratings``'s
    ``(per_match, ratings)`` two-tuple. Each per-match dict carries the leakage-free
    pre-period snapshot ``team{1,2}_glicko_pre`` (mu) and ``team{1,2}_rd_pre`` (phi). The
    final state is ``{"ratings": {team: mu}, "rds": {team: phi}, "sigmas": {team: sigma}}``.

    Semantics (locked by tests):
    - Rating period = calendar day. Rows are date-sorted then grouped; every row in a day
      is snapshotted with the state from BEFORE that day, then the day is batch-updated.
      Same-day games never influence one another (period batching, unlike per-game Elo).
    - Inactivity: a team that does NOT play in a period inflates only its RD via the
      internal-scale step phi*_int = sqrt(phi_int^2 + sigma^2 * t) (then back to original
      scale), capped at ``max_rd``; mu and sigma are unchanged. ``t`` counts skipped
      periods relative to the team's last *active* period. CAVEAT: a "period" here is a
      calendar day on which SOME match occurred (periods are sparse, not equidistant
      wall-clock weeks), so ``t`` is the number of intervening match-days, NOT the real
      elapsed idle time -- a team idle for months across which few match-days fell inflates
      less than the Glicko-2 fixed-duration-period intent. WF-2F may revisit the period
      definition (e.g. weekly buckets) on held-out data.
    - MOV (``use_mov=True``): each map's contribution to the Glicko-2 v / Delta sums is
      weighted by the frozen FiveThirtyEight margin-of-victory multiplier. With no round
      scores the weight is 1.0, so behaviour collapses to standard Glicko-2 (the dominant
      path on the corpus, where ~90% of map rows lack round scores).

    All system / MOV constants are frozen literature values exposed only as defaults; they
    must not be tuned on the full corpus (that would leak future information).
    """
    ratings: Dict[str, float] = dict((initial_state or {}).get("ratings", {}))
    rds: Dict[str, float] = dict((initial_state or {}).get("rds", {}))
    sigmas: Dict[str, float] = dict((initial_state or {}).get("sigmas", {}))

    def _mu(team: str) -> float:
        return ratings.get(team, mu0)

    def _phi(team: str) -> float:
        return rds.get(team, phi0)

    def _sigma(team: str) -> float:
        return sigmas.get(team, sigma0)

    ordered = sorted(range(len(matches)), key=lambda i: str(matches[i].get("date") or ""))
    # Group ordered indices by calendar day (rating period).
    period_groups: List[Tuple[str, List[int]]] = []
    for i in ordered:
        date = str(matches[i].get("date") or "")
        if period_groups and period_groups[-1][0] == date:
            period_groups[-1][1].append(i)
        else:
            period_groups.append((date, [i]))

    per_match_by_index: Dict[int, Dict[str, float]] = {}
    # last_active_period[team] = index (into period_groups) of the team's last active period.
    last_active_period: Dict[str, int] = {}

    for period_idx, (_date, indices) in enumerate(period_groups):
        # (1) Snapshot every row in this period with the pre-period state (leakage-free).
        for i in indices:
            row = matches[i]
            t1 = str(row.get("team1") or "")
            t2 = str(row.get("team2") or "")
            per_match_by_index[i] = {
                "team1_glicko_pre": _mu(t1),
                "team2_glicko_pre": _mu(t2),
                "team1_rd_pre": _phi(t1),
                "team2_rd_pre": _phi(t2),
            }

        # (2) Accumulate this period's game observations per team (batched update).
        #   obs[team] = list of (opp_mu, opp_phi, s, weight)
        obs: Dict[str, List[Tuple[float, float, float, float]]] = {}
        active: List[str] = []

        def _ensure(team: str) -> None:
            if team not in obs:
                obs[team] = []
                active.append(team)

        for i in indices:
            row = matches[i]
            t1 = str(row.get("team1") or "")
            t2 = str(row.get("team2") or "")
            winner = str(row.get("winner") or "")
            if not t1 or not t2 or t1 == t2 or winner not in (t1, t2):
                continue
            mu1, mu2 = _mu(t1), _mu(t2)
            phi1, phi2 = _phi(t1), _phi(t2)
            weight = 1.0
            if use_mov:
                rd = _round_diff(row)
                if rd is not None:
                    gap = abs(mu1 - mu2)
                    weight = _mov_multiplier(
                        rd, gap, alpha=mov_alpha, beta=mov_beta, gamma=mov_gamma
                    )
            s1 = 1.0 if winner == t1 else 0.0
            _ensure(t1)
            _ensure(t2)
            obs[t1].append((mu2, phi2, s1, weight))
            obs[t2].append((mu1, phi1, 1.0 - s1, weight))

        # (3) Inactivity RD inflation for teams that have played before but not this period.
        for team in list(last_active_period.keys()):
            if team in obs:
                continue
            t = period_idx - last_active_period[team]
            if t <= 0:
                continue
            phi = _phi(team)
            sig = _sigma(team)
            # Glicko-2's pre-rating-period inflation runs ENTIRELY in the internal
            # scale: phi*_int = sqrt(phi_int^2 + sigma^2 * t), then back to original
            # scale. phi is original-scale (~290), sigma is internal-scale (~0.06);
            # converting phi to internal scale before adding sigma^2*t is required, else
            # sigma^2*t (added to phi^2 ~84100) is ~3e4x too small and inflation is a
            # no-op. Mirrors the correct phi_g usage in steps (4) below.
            phi_g = phi / GLICKO_SCALE
            inflated = math.sqrt(phi_g * phi_g + sig * sig * float(t)) * GLICKO_SCALE
            rds[team] = min(inflated, max_rd)
            # Mark current as the new "last seen for inflation" baseline so a team idle
            # across many periods inflates relative to its last active period only once
            # per gap (re-anchored here to avoid double counting on the next idle period).
            last_active_period[team] = period_idx

        # (4) Batch Glicko-2 update for every team active this period.
        new_state: Dict[str, Tuple[float, float, float]] = {}
        for team in active:
            mu = _mu(team)
            phi = _phi(team)
            sigma = _sigma(team)
            phi_g = phi / GLICKO_SCALE
            mu_g = (mu - GLICKO_MU0) / GLICKO_SCALE

            v_inv = 0.0
            delta_sum = 0.0
            for opp_mu, opp_phi, s, weight in obs[team]:
                g = glicko_g(opp_phi)
                opp_mu_g = (opp_mu - GLICKO_MU0) / GLICKO_SCALE
                e = 1.0 / (1.0 + math.exp(-g * (mu_g - opp_mu_g)))
                v_inv += weight * g * g * e * (1.0 - e)
                delta_sum += weight * g * (s - e)
            if v_inv <= 0:
                # No effective observations (e.g. all-zero weights): only RD inflation.
                phi_star = math.sqrt(phi_g * phi_g + sigma * sigma)
                new_phi = min(phi_star * GLICKO_SCALE, max_rd)
                new_state[team] = (mu, new_phi, sigma)
                continue
            v = 1.0 / v_inv
            delta = v * delta_sum

            sigma_prime = _solve_volatility(delta * delta, phi_g, v, sigma, tau)
            phi_star = math.sqrt(phi_g * phi_g + sigma_prime * sigma_prime)
            new_phi_g = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
            new_mu_g = mu_g + new_phi_g * new_phi_g * delta_sum

            new_mu = new_mu_g * GLICKO_SCALE + GLICKO_MU0
            new_phi = min(new_phi_g * GLICKO_SCALE, max_rd)
            new_state[team] = (new_mu, new_phi, sigma_prime)

        # (5) Commit the batched updates and mark these teams active this period.
        for team, (m, p, sg) in new_state.items():
            ratings[team] = m
            rds[team] = p
            sigmas[team] = sg
            last_active_period[team] = period_idx

    per_match = [per_match_by_index[i] for i in range(len(matches))]
    final_state = {"ratings": ratings, "rds": rds, "sigmas": sigmas}
    return per_match, final_state


def _normalize_map(value: object) -> str:
    return str(value or "").strip().lower().replace("de_", "")
