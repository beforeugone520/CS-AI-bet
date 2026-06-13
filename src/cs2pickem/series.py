"""Map → series probability composition (pure functions, no IO).

This module turns *per-map* win probabilities into *series* win probabilities
and score-line distributions for best-of-N CS2 matches (BO1/BO3/BO5/...).

Two regimes are supported by the same public helpers:

* **Homogeneous** — a single per-map probability ``p`` is assumed identical for
  every map. The series win probability then has a closed form (negative
  binomial / "first-to-k" sum), e.g. BO3 ``= p^2 (3 - 2p)`` and
  BO5 ``= p^3 (10 - 15p + 6p^2)``.
* **Heterogeneous** — an *ordered* list of per-map probabilities (one entry per
  map in veto order). We enumerate every ordered "winning path" to a clinch and
  sum the product of the relevant per-map outcomes. The number of paths is a
  small constant (BO3 has 3 winning prefixes, BO5 has 10), so this is O(1)
  enumeration, never a brute-force search.

IMPORTANT MODELLING ASSUMPTION — MAP INDEPENDENCE
-------------------------------------------------
Both regimes assume each map's outcome is **statistically independent** of the
others (no momentum / tilt / fatigue carryover). This is a deliberate
first-order approximation: real series exhibit some autocorrelation, but
modelling it would require joint per-(map, score-state) probabilities we do not
estimate. Callers relying on score-line EV should treat the tails (e.g. the
2-1 vs 2-0 split) as approximate. See the project review notes (red line d).
"""

from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence, Union


MapProb = Union[float, Sequence[float]]


def _maps_to_win(best_of: int) -> int:
    """Number of map wins required to clinch a best-of-N series."""
    if best_of < 1 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd integer (1, 3, 5, 7, ...)")
    return best_of // 2 + 1


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def series_win_prob(p_maps: MapProb, best_of: int) -> float:
    """Probability that team1 wins a best-of-``best_of`` series.

    ``p_maps`` may be either:

    * a scalar ``p`` — homogeneous closed form (every map has win prob ``p``); or
    * an ordered sequence of per-map win probabilities ``[p_1, p_2, ...]`` —
      heterogeneous path enumeration (entry ``i`` is the win prob of the i-th
      map actually played, in veto order). At least ``best_of`` entries are
      required so every possible series length can be evaluated.

    Each map outcome is assumed independent (see module docstring).
    """
    needed = _maps_to_win(best_of)

    if _is_scalar(p_maps):
        p = _clamp(p_maps)  # type: ignore[arg-type]
        # First-to-`needed`: sum over k losses-before-clinch of the negative
        # binomial term C(needed-1+k, k) p^needed (1-p)^k, k = 0..needed-1.
        return sum(
            comb(needed - 1 + k, k) * (p ** needed) * ((1.0 - p) ** k)
            for k in range(needed)
        )

    probs = [_clamp(value) for value in p_maps]
    if len(probs) < best_of:
        raise ValueError(
            f"heterogeneous series needs at least {best_of} ordered map probabilities, got {len(probs)}"
        )
    return _heterogeneous_win_prob(probs, needed)


def score_distribution(p_maps: MapProb, best_of: int) -> Dict[str, float]:
    """Distribution over final series score-lines for a best-of-``best_of`` match.

    Returns a mapping like ``{"2-0": .., "2-1": .., "1-2": .., "0-2": ..}`` whose
    values sum to 1. Keys are ``"<team1_wins>-<team2_wins>"``. The winning
    score-lines (team1 reaching ``needed`` wins) sum to :func:`series_win_prob`.

    Each map outcome is assumed independent (see module docstring).
    """
    needed = _maps_to_win(best_of)

    if _is_scalar(p_maps):
        p = _clamp(p_maps)  # type: ignore[arg-type]
        return _homogeneous_score_distribution(p, needed)

    probs = [_clamp(value) for value in p_maps]
    if len(probs) < best_of:
        raise ValueError(
            f"heterogeneous series needs at least {best_of} ordered map probabilities, got {len(probs)}"
        )
    return _heterogeneous_score_distribution(probs, needed)


def map_prob_from_series(series_prob: float, best_of: int, tolerance: float = 1e-9) -> float:
    """Diagnostic inverse of the homogeneous :func:`series_win_prob`.

    Given a series win probability, recover the implied constant per-map win
    probability via monotone bisection. Intended for sanity-checking / display
    only; not used in the prediction path.
    """
    target = _clamp(series_prob)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        value = series_win_prob(mid, best_of)
        if abs(value - target) <= tolerance:
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _is_scalar(value: object) -> bool:
    return isinstance(value, (int, float))


def _homogeneous_score_distribution(p: float, needed: int) -> Dict[str, float]:
    q = 1.0 - p
    distribution: Dict[str, float] = {}
    # Team1 wins with `needed` map wins and k opponent wins (k = 0..needed-1):
    # the last map is a team1 win, the first (needed-1+k) maps contain k losses.
    for k in range(needed):
        weight = comb(needed - 1 + k, k) * (p ** needed) * (q ** k)
        distribution[f"{needed}-{k}"] = weight
    for k in range(needed):
        weight = comb(needed - 1 + k, k) * (q ** needed) * (p ** k)
        distribution[f"{k}-{needed}"] = weight
    return distribution


def _heterogeneous_win_prob(probs: List[float], needed: int) -> float:
    distribution = _heterogeneous_score_distribution(probs, needed)
    return sum(value for key, value in distribution.items() if int(key.split("-")[0]) == needed)


def _heterogeneous_score_distribution(probs: List[float], needed: int) -> Dict[str, float]:
    """Enumerate ordered map outcomes until one side clinches `needed` wins.

    DFS over the ordered map list; each node multiplies in the i-th map's win
    (probs[i]) or loss (1 - probs[i]) probability. We stop a branch the instant
    either side reaches `needed` wins and bucket the leaf by its final score.
    Branch count is bounded by the number of winning paths (small constant), so
    this is O(1) enumeration rather than a search over outcomes.
    """
    distribution: Dict[str, float] = {}

    def walk(index: int, team1_wins: int, team2_wins: int, prob: float) -> None:
        if team1_wins == needed:
            key = f"{team1_wins}-{team2_wins}"
            distribution[key] = distribution.get(key, 0.0) + prob
            return
        if team2_wins == needed:
            key = f"{team1_wins}-{team2_wins}"
            distribution[key] = distribution.get(key, 0.0) + prob
            return
        p = probs[index]
        walk(index + 1, team1_wins + 1, team2_wins, prob * p)
        walk(index + 1, team1_wins, team2_wins + 1, prob * (1.0 - p))

    walk(0, 0, 0, 1.0)
    return distribution
