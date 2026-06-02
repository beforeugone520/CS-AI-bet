from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from .ratings import compute_elo_ratings


ELO_BASE = 1500.0
ELO_K = 24.0
ELO_TIER_K = {"S": 32.0, "A": 20.0, "B": 14.0, "C": 10.0}
UNSTABLE_IDENTITY_FEATURES = ("team1_code", "team2_code", "event_code", "version_tag_code")


def prepare_reliability_features(
    rows: Iterable[Mapping[str, Any]],
    inject_elo: bool = True,
    excluded_feature_names: Sequence[str] | None = None,
) -> tuple[list[dict], Dict[str, float], Dict[str, object]]:
    """Return chronologically ordered rows with leakage-free pre-match Elo features."""

    materialized = sorted((dict(row) for row in rows), key=lambda row: str(row.get("date") or ""))
    excluded = tuple(excluded_feature_names if excluded_feature_names is not None else UNSTABLE_IDENTITY_FEATURES)
    final_elo: Dict[str, float] = {}
    elo_report: Dict[str, object] = {
        "basis": "not_applied",
        "rows": len(materialized),
        "teams": 0,
        "base": ELO_BASE,
        "k": ELO_K,
        "tier_k": dict(ELO_TIER_K),
    }
    if inject_elo and materialized:
        per_match, final_elo = compute_elo_ratings(materialized, base=ELO_BASE, k=ELO_K, tier_k=ELO_TIER_K)
        for row, elo in zip(materialized, per_match):
            row["team1_elo"] = elo["team1_elo_pre"]
            row["team2_elo"] = elo["team2_elo_pre"]
        elo_report.update({"basis": "chronological_pre_match_online", "teams": len(final_elo)})
    return materialized, final_elo, {
        "elo": elo_report,
        "excluded_feature_names": list(excluded),
    }


def apply_final_elo_to_match(row: Mapping[str, Any], ratings: Mapping[str, float]) -> dict:
    output = dict(row)
    if output.get("team1_elo") in (None, ""):
        output["team1_elo"] = float(ratings.get(str(output.get("team1") or ""), ELO_BASE))
    if output.get("team2_elo") in (None, ""):
        output["team2_elo"] = float(ratings.get(str(output.get("team2") or ""), ELO_BASE))
    return output
