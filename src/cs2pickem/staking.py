"""Bankroll staking helpers: fractional Kelly sizing + a portfolio exposure cap.

The Kelly criterion maximises the long-run expected log-growth of a bankroll, but
it does so under the *assumption that the win probability ``p`` is known exactly*.
Here ``p`` is a model estimate carrying real error, and full (1x) Kelly under an
over-estimated ``p`` produces ruinous variance and drawdowns -- a single bad run
can wipe most of the bankroll. We therefore ship ONLY fractional Kelly with a
conservative default coefficient of ``0.5`` (half-Kelly) and the API discourages
coefficients above ``0.5``. On top of per-bet fractional Kelly,
:func:`portfolio_exposure_cap` enforces a hard aggregate ceiling across the
simultaneous Pick'em / match bets so correlated downside stays bounded.

Pure standard-library, no I/O, no third-party dependencies.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


# Conventionally we never stake more than half-Kelly per bet (see module docstring
# on the ruin risk of full Kelly under an over-estimated ``p``). This is BOTH the
# default per-bet coefficient and a hard ceiling: coefficients above it are clamped
# down rather than honoured. (Half-Kelly, not quarter-Kelly, is the documented
# default; callers wanting an even more conservative stake pass ``fraction=0.25``.)
MAX_KELLY_COEFFICIENT = 0.5
# Default aggregate exposure ceiling, as a fraction of bankroll, across all
# simultaneous bets. A documented, conservative default (<= half the bankroll).
DEFAULT_MAX_TOTAL_EXPOSURE = 0.25


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Full-Kelly stake fraction for a single binary bet.

    With net decimal payout ``b = decimal_odds - 1`` and loss probability
    ``q = 1 - p``, the growth-optimal fraction is ``f* = (b*p - q) / b``. A
    non-positive edge yields ``f* <= 0``; we clamp such bets to ``0.0`` (no bet).

    WARNING: this is FULL Kelly and is provided as a building block only. Do not
    stake it directly -- ``p`` is an estimate, and full Kelly under an
    over-estimated ``p`` is catastrophic. Use :func:`fractional_kelly`.
    """
    probability = _validate_probability(p)
    odds = _validate_decimal_odds(decimal_odds)
    b = odds - 1.0
    q = 1.0 - probability
    f_star = (b * probability - q) / b
    return f_star if f_star > 0.0 else 0.0


def fractional_kelly(
    p: float,
    decimal_odds: float,
    fraction: float = MAX_KELLY_COEFFICIENT,
) -> float:
    """Fractional-Kelly stake fraction: ``fraction * kelly_fraction(p, odds)``.

    ``fraction`` defaults to ``0.5`` (half-Kelly) and is, by convention, capped at
    :data:`MAX_KELLY_COEFFICIENT` (``0.5``): values above it are clamped DOWN to
    ``0.5`` and negative values are clamped to ``0.0``. Half-Kelly trades a small
    amount of theoretical log-growth for a large reduction in variance and
    drawdown, which is the right trade when ``p`` is an estimate rather than a
    known truth.

    WARNING: full Kelly (``fraction = 1``) maximises log-growth only when ``p`` is
    exactly correct; under realistic estimation error it has ruinous variance and
    drawdown risk. That is why the coefficient is clamped to ``<= 0.5`` here.
    """
    coefficient = max(0.0, min(MAX_KELLY_COEFFICIENT, float(fraction)))
    return coefficient * kelly_fraction(p, decimal_odds)


def portfolio_exposure_cap(
    stakes: Sequence[float] | Mapping[object, float],
    max_total_exposure: float = DEFAULT_MAX_TOTAL_EXPOSURE,
) -> List[float] | Dict[object, float]:
    """Scale proposed bankroll-fraction stakes down to a total-exposure ceiling.

    ``stakes`` are per-bet stakes expressed as fractions of the bankroll (e.g. the
    output of :func:`fractional_kelly`). If their (non-negative) sum exceeds
    ``max_total_exposure`` every stake is scaled by the same factor so the total
    equals the cap; otherwise the stakes are returned unchanged. Negative entries
    are floored to ``0.0``. A list input returns a list, a mapping returns a dict
    keyed identically -- preserving the caller's bet identifiers.

    This is a HARD aggregate-risk ceiling layered on top of each bet's per-bet
    fractional Kelly, bounding correlated downside across simultaneous bets.
    """
    cap = max(0.0, float(max_total_exposure))
    if isinstance(stakes, Mapping):
        cleaned = {key: max(0.0, float(value)) for key, value in stakes.items()}
        total = sum(cleaned.values())
        if total <= cap or total <= 0.0:
            return cleaned
        scale = cap / total
        return {key: value * scale for key, value in cleaned.items()}
    cleaned_list = [max(0.0, float(value)) for value in stakes]
    total = sum(cleaned_list)
    if total <= cap or total <= 0.0:
        return cleaned_list
    scale = cap / total
    return [value * scale for value in cleaned_list]


def kelly_report(
    p: float,
    decimal_odds: float,
    fraction: float = MAX_KELLY_COEFFICIENT,
) -> Dict[str, float]:
    """Per-bet audit dict: edge, raw (full) Kelly, fractional Kelly.

    Mirrors the ``report()`` style used elsewhere so a caller can log/inspect why
    a stake came out the size it did. ``capped_stake`` is left to
    :func:`portfolio_exposure_cap`, which needs the whole portfolio at once.
    """
    probability = _validate_probability(p)
    odds = _validate_decimal_odds(decimal_odds)
    b = odds - 1.0
    raw = kelly_fraction(probability, odds)
    coefficient = max(0.0, min(MAX_KELLY_COEFFICIENT, float(fraction)))
    return {
        "p": probability,
        "decimal_odds": odds,
        # Expected net return per unit staked = b*p - q (the Kelly numerator).
        "edge": b * probability - (1.0 - probability),
        "raw_kelly": raw,
        "fraction": coefficient,
        "fractional_kelly": coefficient * raw,
    }


def _validate_probability(p: float) -> float:
    probability = float(p)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p!r}")
    return probability


def _validate_decimal_odds(decimal_odds: float) -> float:
    odds = float(decimal_odds)
    if odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0 (got {decimal_odds!r})")
    return odds
