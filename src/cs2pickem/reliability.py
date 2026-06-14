from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .ratings import (
    GLICKO_MU0,
    GLICKO_PHI0,
    GLICKO_TAU,
    MOV_ALPHA,
    MOV_BETA,
    MOV_GAMMA,
    compute_bradley_terry,
    compute_elo_ratings,
    compute_glicko_ratings,
    compute_map_bradley_terry,
)


ELO_BASE = 1500.0
ELO_K = 24.0
ELO_TIER_K = {"S": 32.0, "A": 20.0, "B": 14.0, "C": 10.0}

# Bradley-Terry strength prior. Neutral strength is 0 (mean-centered log-strength),
# mirroring Elo's base=1500 fallback for teams with no prior history.
BT_NEUTRAL = 0.0
BT_RIDGE = 1.0
BT_MAP_RIDGE = 4.0
BT_INJECTED_COLUMNS = (
    "bt_team1_strength",
    "bt_team2_strength",
    "bt_strength_diff",
    "bt_team1_map_strength",
    "bt_team2_map_strength",
    "bt_map_strength_diff",
)
# The leakage-free *directional diffs* are the headline Bradley-Terry signal. Per the
# WF-2 review red-line they are CANDIDATE features (FeatureSelector competes them on
# importance), not forced into the required set. Listed here only for documentation /
# downstream visibility; they are intentionally NOT added to required_feature_names.
BT_CANDIDATE_FEATURES = (
    "bt_strength_diff",
    "bt_map_strength_diff",
)

# Glicko-2 pre-match injection. Neutral rating is GLICKO_MU0=1500 (cold start), neutral
# RD is GLICKO_PHI0=350, mirroring the Elo base=1500 fallback.
GLICKO_NEUTRAL = GLICKO_MU0
GLICKO_RD_NEUTRAL = GLICKO_PHI0
GLICKO_INJECTED_COLUMNS = (
    "team1_glicko_pre",
    "team2_glicko_pre",
    "team1_rd_pre",
    "team2_rd_pre",
    "glicko_diff",
    "glicko_rd_sum",
)
# The leakage-free directional rating diff (glicko_diff, antisymmetric, neutral=0) and the
# symmetric uncertainty tape (glicko_rd_sum, swap-invariant) are the headline Glicko-2
# signals for FeatureBuilder. Per the WF-2 review red-line (c) they are CANDIDATE features
# (the selector competes them on importance), NOT forced into the required set. Listed here
# only for downstream visibility.
#
# Note the two columns enter FeatureBuilder's two-tier candidate pool at DIFFERENT tiers
# (see features.py): glicko_diff is a strong directional signal wired end-to-end, so it sits
# in the default-active pool; glicko_rd_sum is a weak-prior magnitude (a large RD-sum mostly
# flags cold-start/inactive teams, a confounder), so it sits in UNVERIFIED_FEATURE_NAMES and
# is opted in only for the WF-2F A/B. Neither is in the required/excluded lists below -> no
# name drift with selection.py.
GLICKO_CANDIDATE_FEATURES = (
    "glicko_diff",
    "glicko_rd_sum",
)
UNSTABLE_IDENTITY_FEATURES = ("team1_code", "team2_code", "event_code", "version_tag_code")
PLAYER_STATUS_REQUIRED_FEATURES = (
    "rating_diff",
    "kd_diff",
    "opening_success_diff",
    "clutch_winrate_diff",
    "star_rating_diff",
    "substitute_flag_diff",
    "player_sample_diff",
    "player_form_score_diff",
    "player_form_trend_diff",
    "player_sample_confidence_diff",
)


def prepare_reliability_features(
    rows: Iterable[Mapping[str, Any]],
    inject_elo: bool = True,
    excluded_feature_names: Sequence[str] | None = None,
    *,
    inject_bt: bool = False,
    bt_ridge: float = BT_RIDGE,
    bt_map_ridge: float = BT_MAP_RIDGE,
    bt_half_life_days: Optional[float] = None,
    inject_glicko: bool = False,
    glicko_tau: float = GLICKO_TAU,
    glicko_use_mov: bool = True,
    glicko_mov_alpha: float = MOV_ALPHA,
    glicko_mov_beta: float = MOV_BETA,
    glicko_mov_gamma: float = MOV_GAMMA,
) -> tuple[list[dict], Dict[str, float], Dict[str, object]]:
    """Return chronologically ordered rows with leakage-free pre-match Elo + BT features.

    Bradley-Terry strength is injected with the same pre-match discipline as Elo: for each
    distinct match date we refit BT on the strictly-earlier history and snapshot every row
    on that date with those pre-date strengths (``bt_team1_strength`` / ``bt_team2_strength``
    and their per-map counterparts). The current result is never in the fit that scores it,
    and the rolling refit never reads future rows. ``bt_ridge`` / ``bt_map_ridge`` are the
    tunable shrinkage hyperparameters (FeatureSelector-visible via the diff columns).
    ``bt_half_life_days`` is reserved for time-decayed refits and defaults to ``None`` (no
    decay) so behaviour is unchanged until a decay weighting is wired in a later stage.

    ``inject_bt`` defaults to ``False`` so the existing training / prediction / tuning hot
    paths keep their (cheap) Elo-only behaviour: the rolling per-date BT refit is O(dates x
    maps x iterations) and costs ~30s on the full ~13.6k-row history, which is not worth
    paying for columns that are still only *candidates*. It also avoids a train/serve skew --
    ``predict_probability_details`` does not yet call ``apply_final_bt_to_match``, so injecting
    BT into training rows but not serving rows would feed the model a column that is non-zero
    at fit time and constant-0 at score time. WF-2F turns ``inject_bt=True`` on deliberately
    (and must wire the matching serve-side injection) when it A/B-adjudicates the BT signal.

    ``inject_glicko`` (Glicko-2 pre-match, also defaulting to ``False`` for the same two
    reasons) snapshots ``team{1,2}_glicko_pre`` / ``team{1,2}_rd_pre`` with strict period
    (calendar-day) batching -- the day's results never feed the snapshot that scores them --
    and derives the candidate signals ``glicko_diff`` (antisymmetric rating gap) and
    ``glicko_rd_sum`` (symmetric uncertainty tape). The rolling refit is the O(periods x teams
    x volatility-iterations) Glicko-2 hot path, and ``predict_probability_details`` does not
    yet call ``apply_final_glicko_to_match``; WF-2F must turn ``inject_glicko=True`` on AND
    wire the serve-side injection together so there is no train/serve skew. Margin-of-victory
    damping (``glicko_use_mov``) is applied only inside the rating update, never to the
    leakage-free snapshot, and falls back to win/loss when round scores are absent.
    """

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

    bt_report: Dict[str, object] = {
        "basis": "not_applied",
        "rows": len(materialized),
        "teams": 0,
        "ridge": bt_ridge,
        "map_ridge": bt_map_ridge,
        "half_life_days": bt_half_life_days,
        "neutral": BT_NEUTRAL,
        "columns": list(BT_INJECTED_COLUMNS),
        "final": {},
        "final_map": {},
    }
    if inject_bt and materialized:
        final_bt, final_bt_map = _inject_bt_features(
            materialized, ridge=bt_ridge, map_ridge=bt_map_ridge
        )
        bt_report.update(
            {
                "basis": "chronological_pre_match_rolling",
                "teams": len(final_bt),
                "final": final_bt,
                "final_map": final_bt_map,
            }
        )

    glicko_report: Dict[str, object] = {
        "basis": "not_applied",
        "rows": len(materialized),
        "teams": 0,
        "tau": glicko_tau,
        "neutral": GLICKO_NEUTRAL,
        "rd_neutral": GLICKO_RD_NEUTRAL,
        "columns": list(GLICKO_INJECTED_COLUMNS),
        "final": {"ratings": {}, "rds": {}, "sigmas": {}},
        "mov": {
            "enabled": glicko_use_mov,
            "alpha": glicko_mov_alpha,
            "beta": glicko_mov_beta,
            "gamma": glicko_mov_gamma,
            "rows_total": len(materialized),
            "rows_with_round_diff": 0,
            "coverage": 0.0,
        },
    }
    if inject_glicko and materialized:
        final_glicko, mov_covered = _inject_glicko_features(
            materialized,
            tau=glicko_tau,
            use_mov=glicko_use_mov,
            mov_alpha=glicko_mov_alpha,
            mov_beta=glicko_mov_beta,
            mov_gamma=glicko_mov_gamma,
        )
        rows_total = len(materialized)
        glicko_report.update(
            {
                "basis": "chronological_pre_match_rolling",
                "teams": len(final_glicko.get("ratings", {})),
                "final": final_glicko,
            }
        )
        glicko_report["mov"].update(
            {
                "rows_with_round_diff": mov_covered,
                "coverage": (mov_covered / rows_total) if rows_total else 0.0,
            }
        )

    # Per the WF-2 review red-line (c), newly-injected BT / Glicko diff columns are NOT
    # forced into the required set: they enter the candidate pool and FeatureSelector
    # competes them on real importance (A/B significance is adjudicated in WF-2F). Only the
    # player-status diffs remain protected here, exactly as before -> no name drift.
    return materialized, final_elo, {
        "elo": elo_report,
        "bt": bt_report,
        "glicko": glicko_report,
        "excluded_feature_names": list(excluded),
        "required_feature_names": list(PLAYER_STATUS_REQUIRED_FEATURES),
    }


def _inject_bt_features(
    materialized: List[dict],
    *,
    ridge: float,
    map_ridge: float,
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Rolling, pre-match BT injection (mutates rows in place).

    Refits at each distinct date boundary on strictly-earlier history, so all rows sharing
    a date receive the same strictly pre-date snapshot. Returns the final full-history
    overall + per-map fits for scoring upcoming fixtures.
    """
    # Group row indices by date, preserving chronological order of the (already sorted) list.
    date_groups: List[tuple[str, List[int]]] = []
    for idx, row in enumerate(materialized):
        date = str(row.get("date") or "")
        if date_groups and date_groups[-1][0] == date:
            date_groups[-1][1].append(idx)
        else:
            date_groups.append((date, [idx]))

    history: List[dict] = []
    for _date, indices in date_groups:
        # Fit on everything strictly before this date (history excludes the current group).
        if history:
            global_theta = compute_bradley_terry(history, ridge=ridge)
            map_theta = compute_map_bradley_terry(history, ridge=map_ridge, global_theta=global_theta)
        else:
            global_theta = {}
            map_theta = {}
        for idx in indices:
            _write_bt_snapshot(materialized[idx], global_theta, map_theta)
        # Only after snapshotting do these rows join the history for later dates.
        for idx in indices:
            history.append(materialized[idx])

    # Final full-history fit for upcoming fixtures (analogous to apply_final_elo_to_match).
    if history:
        final_bt = compute_bradley_terry(history, ridge=ridge)
        final_bt_map = compute_map_bradley_terry(history, ridge=map_ridge, global_theta=final_bt)
    else:
        final_bt = {}
        final_bt_map = {}
    return final_bt, final_bt_map


def _write_bt_snapshot(
    row: dict,
    global_theta: Mapping[str, float],
    map_theta: Mapping[str, Mapping[str, float]],
) -> None:
    team1 = str(row.get("team1") or "")
    team2 = str(row.get("team2") or "")
    s1 = float(global_theta.get(team1, BT_NEUTRAL))
    s2 = float(global_theta.get(team2, BT_NEUTRAL))
    row["bt_team1_strength"] = s1
    row["bt_team2_strength"] = s2
    row["bt_strength_diff"] = s1 - s2

    map_name = _normalize_map(row.get("map"))
    map_strengths = map_theta.get(map_name, {}) if map_name else {}
    m1 = float(map_strengths.get(team1, global_theta.get(team1, BT_NEUTRAL)))
    m2 = float(map_strengths.get(team2, global_theta.get(team2, BT_NEUTRAL)))
    row["bt_team1_map_strength"] = m1
    row["bt_team2_map_strength"] = m2
    row["bt_map_strength_diff"] = m1 - m2


def apply_final_elo_to_match(row: Mapping[str, Any], ratings: Mapping[str, float]) -> dict:
    output = dict(row)
    if output.get("team1_elo") in (None, ""):
        output["team1_elo"] = float(ratings.get(str(output.get("team1") or ""), ELO_BASE))
    if output.get("team2_elo") in (None, ""):
        output["team2_elo"] = float(ratings.get(str(output.get("team2") or ""), ELO_BASE))
    return output


def apply_final_bt_to_match(
    row: Mapping[str, Any],
    strengths: Mapping[str, float],
    map_strengths: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> dict:
    """Score an upcoming fixture with the final full-history BT fit (leakage-free: the
    fixture has no result yet). Fills only missing columns, mirroring the Elo helper."""
    output = dict(row)
    team1 = str(output.get("team1") or "")
    team2 = str(output.get("team2") or "")
    if output.get("bt_team1_strength") in (None, ""):
        output["bt_team1_strength"] = float(strengths.get(team1, BT_NEUTRAL))
    if output.get("bt_team2_strength") in (None, ""):
        output["bt_team2_strength"] = float(strengths.get(team2, BT_NEUTRAL))
    output["bt_strength_diff"] = output["bt_team1_strength"] - output["bt_team2_strength"]

    map_name = _normalize_map(output.get("map"))
    per_map = (map_strengths or {}).get(map_name, {}) if map_name else {}
    if output.get("bt_team1_map_strength") in (None, ""):
        output["bt_team1_map_strength"] = float(per_map.get(team1, output["bt_team1_strength"]))
    if output.get("bt_team2_map_strength") in (None, ""):
        output["bt_team2_map_strength"] = float(per_map.get(team2, output["bt_team2_strength"]))
    output["bt_map_strength_diff"] = output["bt_team1_map_strength"] - output["bt_team2_map_strength"]
    return output


def _inject_glicko_features(
    materialized: List[dict],
    *,
    tau: float,
    use_mov: bool,
    mov_alpha: float,
    mov_beta: float,
    mov_gamma: float,
) -> tuple[Dict[str, Dict[str, float]], int]:
    """Rolling, period-batched Glicko-2 injection (mutates rows in place).

    ``compute_glicko_ratings`` already snapshots each row with its strictly-pre-period
    state (calendar-day rating periods), so a single full-history pass is leakage-free:
    same-day games never feed the snapshot scoring them, and the rolling state never reads
    future days. We copy those pre snapshots onto the rows and derive ``glicko_diff`` /
    ``glicko_rd_sum``. Returns ``(final_state, rows_with_round_diff)`` -- the final state is
    used to score upcoming fixtures (apply_final_glicko_to_match), the count feeds the MOV
    coverage report.
    """
    per_match, final_state = compute_glicko_ratings(
        materialized,
        tau=tau,
        use_mov=use_mov,
        mov_alpha=mov_alpha,
        mov_beta=mov_beta,
        mov_gamma=mov_gamma,
    )
    covered = 0
    for row, snap in zip(materialized, per_match):
        row["team1_glicko_pre"] = snap["team1_glicko_pre"]
        row["team2_glicko_pre"] = snap["team2_glicko_pre"]
        row["team1_rd_pre"] = snap["team1_rd_pre"]
        row["team2_rd_pre"] = snap["team2_rd_pre"]
        row["glicko_diff"] = snap["team1_glicko_pre"] - snap["team2_glicko_pre"]
        row["glicko_rd_sum"] = snap["team1_rd_pre"] + snap["team2_rd_pre"]
        if _row_has_round_diff(row):
            covered += 1
    return final_state, covered


def _row_has_round_diff(row: Mapping[str, Any]) -> bool:
    s1 = row.get("team1_score")
    s2 = row.get("team2_score")
    if s1 in (None, "") or s2 in (None, ""):
        return False
    try:
        v1 = float(s1)
        v2 = float(s2)
    except (TypeError, ValueError):
        return False
    return (v1 + v2) != 0


def apply_final_glicko_to_match(
    row: Mapping[str, Any],
    glicko_state: Mapping[str, Mapping[str, float]],
) -> dict:
    """Score an upcoming fixture with the final full-history Glicko-2 state (leakage-free:
    the fixture has no result yet). Fills only missing snapshot columns, mirroring the Elo /
    BT helpers, then (re)derives ``glicko_diff`` / ``glicko_rd_sum``."""
    output = dict(row)
    team1 = str(output.get("team1") or "")
    team2 = str(output.get("team2") or "")
    ratings = glicko_state.get("ratings", {})
    rds = glicko_state.get("rds", {})

    if output.get("team1_glicko_pre") in (None, ""):
        output["team1_glicko_pre"] = float(ratings.get(team1, GLICKO_NEUTRAL))
    if output.get("team2_glicko_pre") in (None, ""):
        output["team2_glicko_pre"] = float(ratings.get(team2, GLICKO_NEUTRAL))
    if output.get("team1_rd_pre") in (None, ""):
        output["team1_rd_pre"] = float(rds.get(team1, GLICKO_RD_NEUTRAL))
    if output.get("team2_rd_pre") in (None, ""):
        output["team2_rd_pre"] = float(rds.get(team2, GLICKO_RD_NEUTRAL))

    output["glicko_diff"] = output["team1_glicko_pre"] - output["team2_glicko_pre"]
    output["glicko_rd_sum"] = output["team1_rd_pre"] + output["team2_rd_pre"]
    return output


def _normalize_map(value: object) -> str:
    return str(value or "").strip().lower().replace("de_", "")
