"""Opponent-strength (Elo) ratings derived from match results.

Elo naturally de-inflates strength-of-schedule: beating a weak team gains little,
beating a strong team gains a lot. Unlike the static world-rank column (which is a
constant placeholder in the 5E corpus), Elo is populated for every team, varies, and
is computable for upcoming-fixture teams from their own match history.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


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
