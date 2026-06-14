from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from .bp import bp_structured_features
from .maps import DEFAULT_MAP_POOL


MAP_FEATURE_NAMES = [f"map_{name}" for name in DEFAULT_MAP_POOL]

# Glicko-2 pre-match neutral fallbacks (cold start), mirroring the reliability.py
# injection defaults (mu0=1500, phi0=350). When the Glicko columns are not injected
# the diff is a neutral 0 and the uncertainty tape is two cold-start RDs (350+350=700),
# so an un-injected row never silently looks like a confident or low-uncertainty match.
GLICKO_NEUTRAL_DIFF = 0.0
GLICKO_NEUTRAL_RD = 350.0
GLICKO_NEUTRAL_RD_SUM = GLICKO_NEUTRAL_RD * 2.0


@dataclass
class Dataset:
    rows: List[List[float]]
    labels: List[int]
    feature_names: List[str]
    raw_rows: List[Dict[str, Any]]


class FeatureBuilder:
    """Build normalized static, dynamic, cross, and Swiss-state features.

    New WF-2B signal columns are split into two tiers so in-sample feature
    selection is not diluted by dead or weak-prior columns (review red-line c):

    - *Default-active* (in ``feature_names``): only columns that are genuinely
      wired end-to-end AND carry a strong prior -> ``bt_strength_diff`` /
      ``bt_map_strength_diff`` (real pre-match BT signal), ``odds_provider_count``
      / ``odds_is_proxy`` (written by the odds merge), and ``bp_applied`` (a
      zero-cost has-intel gate).
    - *Gated / unverified* (``UNVERIFIED_FEATURE_NAMES``, OFF by default): columns
      that are currently dead in the real pipeline (no join wires them onto match
      rows -> constant 0) or whose predictive prior is unproven / sparse. These are
      still *computed* by ``_raw_features`` (so nothing downstream breaks and they
      can be inspected), but they are excluded from the selector's candidate pool
      until WF-2F adjudicates them with a held-out significance gate. Pass
      ``include_unverified_features=True`` to opt them back in for that A/B.
    """

    # Strong-signal, end-to-end-wired columns: these compete in the selector.
    _BASE_FEATURE_NAMES = [
        "rank_diff",
        "elo_diff",
        "rmr_points_diff",
        "major_best_placement_diff",
        "matches_30d_diff",
        "recent_winrate_5_diff",
        "recent_winrate_10_diff",
        "bo1_winrate_diff",
        "bo3_winrate_diff",
        "map_winrate_diff",
        *MAP_FEATURE_NAMES,
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
        "h2h_team1_winrate",
        "odds_implied_diff",
        # Bradley-Terry pre-match strength (injected by reliability.py). Pure
        # team1-minus-team2 diffs -> antisymmetric, neutral (0) default when absent.
        # Genuinely wired (reliability rolling refit) -> stays in the default pool.
        "bt_strength_diff",
        "bt_map_strength_diff",
        # Glicko-2 pre-match rating gap (injected by reliability.py, leakage-free
        # period-batched snapshot). Pure team1-minus-team2 diff -> antisymmetric,
        # neutral (0) default when absent. It is a strong directional rating signal
        # wired end-to-end the same way as the BT diffs, so it competes in the
        # selector by default. The symmetric uncertainty tape (glicko_rd_sum) is a
        # weak-prior magnitude and lives in UNVERIFIED_FEATURE_NAMES instead.
        "glicko_diff",
        # Ban/Pick gate: bp_applied is a zero-cost has-intel indicator that is
        # written by merge_bp_into_fixtures, so it carries real (if sparse) signal.
        "bp_applied",
        # Odds market metadata that the merge actually writes onto the match row.
        "odds_provider_count",
        "odds_is_proxy",
        "is_bo1",
        "is_bo3",
        "swiss_round",
        "team1_wins",
        "team1_losses",
        "team2_wins",
        "team2_losses",
        "swiss_score_diff",
        "wins_needed_to_advance_diff",
        "losses_until_elimination_diff",
        "current_streak_diff",
        "team1_code",
        "team2_code",
        "event_code",
        "event_tier_code",
        "version_tag_code",
    ]

    # Unverified / currently-dead / weak-prior columns. OFF by default so they do
    # not dilute in-sample selection; opt in with include_unverified_features=True
    # (WF-2F A/B). They are still computed in _raw_features for inspection.
    #   - event_grade_*: no 5E profile->match join exists yet -> constant 0 in the
    #     real pipeline (dead until the join is wired in a later stage).
    #   - odds_overround / odds_devig_z: dropped by merge_odds_into_matches and
    #     (devig_z) None under the default multiplicative de-vig -> constant 0.
    #   - bp_confidence / bp_total_* / bp_ban_overlap: sparse magnitude columns with
    #     an unproven win/loss prior; high false-correlation risk on small samples.
    #   - glicko_rd_sum: the symmetric Glicko-2 uncertainty tape (team1_rd_pre +
    #     team2_rd_pre). It IS wired end-to-end alongside glicko_diff, but as a
    #     standalone win/loss feature its prior is unproven: a large RD-sum mostly
    #     flags "both teams are cold-start / inactive" (a confounder, not a clean
    #     directional win signal), so forcing it into the default pool would dilute
    #     in-sample selection. It is gated OFF until WF-2F A/B-adjudicates it,
    #     mirroring how the strong gate (glicko_diff / bp_applied) competes by default
    #     while the weak-prior magnitude lives here.
    UNVERIFIED_FEATURE_NAMES = (
        "event_grade_sum",
        "team_event_grade_sum",
        "team_event_grade_diff",
        "odds_overround",
        "odds_devig_z",
        "bp_confidence",
        "bp_total_bans",
        "bp_ban_overlap",
        "bp_total_picks",
        "glicko_rd_sum",
    )

    # Backward-compatible full ordering (kept stable for any caller that reads the
    # superset). The default instance exposes only _BASE_FEATURE_NAMES.
    feature_names = [
        *_BASE_FEATURE_NAMES,
        *UNVERIFIED_FEATURE_NAMES,
    ]

    def __init__(self, include_unverified_features: bool = False) -> None:
        self._minimums: Dict[str, float] = {}
        self._maximums: Dict[str, float] = {}
        self._version_codes: Dict[str, int] = {}
        self._category_codes: Dict[str, Dict[str, int]] = {}
        self.include_unverified_features = include_unverified_features
        # Per-instance active column list: strong-signal pool by default, plus the
        # gated columns only when explicitly opted in for WF-2F adjudication.
        self.feature_names = list(self._BASE_FEATURE_NAMES)
        if include_unverified_features:
            self.feature_names = self.feature_names + list(self.UNVERIFIED_FEATURE_NAMES)

    def fit_transform(self, rows: Iterable[Dict[str, Any]]) -> Dataset:
        raw_rows = list(rows)
        raw_matrix = [self._raw_features(row) for row in raw_rows]
        self._fit_scaler(raw_matrix)
        return Dataset(
            rows=[self._normalize(feature_row) for feature_row in raw_matrix],
            labels=[1 if row.get("winner") == row.get("team1") else 0 for row in raw_rows],
            feature_names=list(self.feature_names),
            raw_rows=raw_rows,
        )

    def transform(self, rows: Iterable[Dict[str, Any]]) -> List[List[float]]:
        return [self._normalize(self._raw_features(row)) for row in rows]

    def _fit_scaler(self, rows: Sequence[Dict[str, float]]) -> None:
        for name in self.feature_names:
            values = [row[name] for row in rows] or [0.0]
            self._minimums[name] = min(values)
            self._maximums[name] = max(values)

    def _normalize(self, row: Dict[str, float]) -> List[float]:
        normalized = []
        for name in self.feature_names:
            low = self._minimums.get(name, 0.0)
            high = self._maximums.get(name, 1.0)
            if high == low:
                normalized.append(row[name] if _passthrough_binary(name) else 0.5)
            else:
                normalized.append((row[name] - low) / (high - low))
        return [min(1.0, max(0.0, value)) for value in normalized]

    def _raw_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        odds_team1 = _num(row, "odds_team1", 2.0)
        odds_team2 = _num(row, "odds_team2", 2.0)
        implied_1, implied_2 = _implied_market_pair(odds_team1, odds_team2)
        team1_score = _num(row, "team1_wins") - _num(row, "team1_losses")
        team2_score = _num(row, "team2_wins") - _num(row, "team2_losses")
        best_of = int(_num(row, "best_of", 1))
        map_name = _normalize_map_name(row.get("map", "unknown"))
        team1_wins = _num(row, "team1_wins")
        team2_wins = _num(row, "team2_wins")
        team1_losses = _num(row, "team1_losses")
        team2_losses = _num(row, "team2_losses")
        team1_wins_needed = max(0.0, 3.0 - team1_wins)
        team2_wins_needed = max(0.0, 3.0 - team2_wins)
        team1_losses_until_elimination = max(0.0, 3.0 - team1_losses)
        team2_losses_until_elimination = max(0.0, 3.0 - team2_losses)

        # --- 5E event-grade magnitude (symmetric sum) + schedule-quality diff ---
        match_grade = _num(row, "event_grade", 0.0)
        team1_sched_grade = _num(row, "team1_fivee_6m_avg_event_grade", 0.0)
        team2_sched_grade = _num(row, "team2_fivee_6m_avg_event_grade", 0.0)

        # --- Ban/Pick structured features (gated by bp_applied; single source in bp.py) ---
        bp = bp_structured_features(row)

        features = {
            "rank_diff": _num(row, "team2_rank", 80) - _num(row, "team1_rank", 80),
            "elo_diff": _num(row, "team1_elo", 1500.0) - _num(row, "team2_elo", 1500.0),
            "rmr_points_diff": _num(row, "team1_rmr_points") - _num(row, "team2_rmr_points"),
            "major_best_placement_diff": _num(row, "team2_major_best_placement", 32) - _num(row, "team1_major_best_placement", 32),
            "matches_30d_diff": _num(row, "team1_matches_30d") - _num(row, "team2_matches_30d"),
            "recent_winrate_5_diff": _num(row, "team1_recent_winrate_5", 0.5) - _num(row, "team2_recent_winrate_5", 0.5),
            "recent_winrate_10_diff": _num(row, "team1_recent_winrate_10", 0.5) - _num(row, "team2_recent_winrate_10", 0.5),
            "bo1_winrate_diff": _num(row, "team1_bo1_winrate_6m", 0.5) - _num(row, "team2_bo1_winrate_6m", 0.5),
            "bo3_winrate_diff": _num(row, "team1_bo3_winrate_6m", 0.5) - _num(row, "team2_bo3_winrate_6m", 0.5),
            "map_winrate_diff": _num(row, "team1_map_winrate", 0.5) - _num(row, "team2_map_winrate", 0.5),
            "rating_diff": _num(row, "team1_rating", 1.0) - _num(row, "team2_rating", 1.0),
            "kd_diff": _num(row, "team1_kd", 1.0) - _num(row, "team2_kd", 1.0),
            "opening_success_diff": _num(row, "team1_opening_success", 0.5) - _num(row, "team2_opening_success", 0.5),
            "clutch_winrate_diff": _num(row, "team1_clutch_winrate", 0.5) - _num(row, "team2_clutch_winrate", 0.5),
            "star_rating_diff": _num(row, "team1_star_rating", 1.0) - _num(row, "team2_star_rating", 1.0),
            "substitute_flag_diff": _num(row, "team1_substitute_flag") - _num(row, "team2_substitute_flag"),
            "player_sample_diff": _num(row, "team1_player_sample") - _num(row, "team2_player_sample"),
            "player_form_score_diff": _num(row, "team1_player_form_score") - _num(row, "team2_player_form_score"),
            "player_form_trend_diff": _num(row, "team1_player_form_trend") - _num(row, "team2_player_form_trend"),
            "player_sample_confidence_diff": _num(row, "team1_player_sample_confidence") - _num(row, "team2_player_sample_confidence"),
            "h2h_team1_winrate": _num(row, "h2h_team1_winrate", 0.5),
            "odds_implied_diff": implied_1 - implied_2,
            # Bradley-Terry diffs: pure antisymmetric, neutral default when not injected.
            "bt_strength_diff": _num(row, "bt_strength_diff", 0.0),
            "bt_map_strength_diff": _num(row, "bt_map_strength_diff", 0.0),
            # Glicko-2 pre-match signals. glicko_diff is antisymmetric (neutral 0 when
            # not injected). glicko_rd_sum is the symmetric uncertainty tape; its neutral
            # default is two cold-start RDs (350+350=700) so an un-injected row reads as
            # maximally uncertain rather than spuriously confident. Prefer the injected
            # column, else reconstruct from the pre snapshots, else fall back to neutral.
            "glicko_diff": _glicko_diff(row),
            "glicko_rd_sum": _glicko_rd_sum(row),
            # 5E magnitude (symmetric sum = swap-invariant) + directional schedule diff.
            # event_grade is a single match-level scalar (no team orientation), so it is
            # intrinsically swap-invariant; exposed under a "_sum" name for consistency.
            "event_grade_sum": match_grade,
            "team_event_grade_sum": team1_sched_grade + team2_sched_grade,
            "team_event_grade_diff": team1_sched_grade - team2_sched_grade,
            # BP structured (already swap-invariant or gated to 0 when no intel).
            "bp_applied": bp["bp_applied"],
            "bp_confidence": bp["bp_confidence"],
            "bp_total_bans": bp["bp_total_bans"],
            "bp_ban_overlap": bp["bp_ban_overlap"],
            "bp_total_picks": bp["bp_total_picks"],
            # Odds metadata: match-level / swap-invariant magnitudes.
            "odds_provider_count": _num(row, "odds_provider_count", 0.0),
            "odds_overround": _num(row, "overround", 0.0),
            "odds_devig_z": _num(row, "devig_z", 0.0),
            "odds_is_proxy": 1.0 if _truthy(row.get("market_signal_proxy")) else 0.0,
            "is_bo1": 1.0 if best_of == 1 else 0.0,
            "is_bo3": 1.0 if best_of == 3 else 0.0,
            "swiss_round": _num(row, "swiss_round", 1),
            "team1_wins": team1_wins,
            "team1_losses": team1_losses,
            "team2_wins": team2_wins,
            "team2_losses": team2_losses,
            "swiss_score_diff": team1_score - team2_score,
            "wins_needed_to_advance_diff": team2_wins_needed - team1_wins_needed,
            "losses_until_elimination_diff": team1_losses_until_elimination - team2_losses_until_elimination,
            "current_streak_diff": _num(row, "team1_current_streak") - _num(row, "team2_current_streak"),
            "team1_code": float(self._category_code("team", str(row.get("team1", "unknown")))),
            "team2_code": float(self._category_code("team", str(row.get("team2", "unknown")))),
            "event_code": float(self._category_code("event", str(row.get("event", "unknown")))),
            "event_tier_code": float(self._category_code("event_tier", str(row.get("event_tier", "unknown")))),
            "version_tag_code": float(self._version_code(str(row.get("version_tag", "unknown")))),
        }
        for name in DEFAULT_MAP_POOL:
            features[f"map_{name}"] = 1.0 if map_name == name else 0.0
        return features

    def _version_code(self, tag: str) -> int:
        if tag not in self._version_codes:
            self._version_codes[tag] = len(self._version_codes)
        return self._version_codes[tag]

    def _category_code(self, namespace: str, value: str) -> int:
        codes = self._category_codes.setdefault(namespace, {})
        if value not in codes:
            codes[value] = len(codes)
        return codes[value]


def _num(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _glicko_diff(row: Dict[str, Any]) -> float:
    """Antisymmetric pre-match Glicko-2 rating gap (team1 - team2).

    Prefers the injected ``glicko_diff`` column; if only the pre snapshots are present
    it reconstructs the diff from them; otherwise falls back to a neutral 0. Keeping
    reconstruction symmetric with the snapshots guarantees the team-swap antisymmetry
    contract holds whichever upstream wiring produced the row.
    """
    if row.get("glicko_diff") not in (None, ""):
        return _num(row, "glicko_diff", GLICKO_NEUTRAL_DIFF)
    pre1 = row.get("team1_glicko_pre")
    pre2 = row.get("team2_glicko_pre")
    if pre1 not in (None, "") and pre2 not in (None, ""):
        return _num(row, "team1_glicko_pre", 0.0) - _num(row, "team2_glicko_pre", 0.0)
    return GLICKO_NEUTRAL_DIFF


def _glicko_rd_sum(row: Dict[str, Any]) -> float:
    """Symmetric (swap-invariant) Glicko-2 uncertainty tape (rd1 + rd2).

    Prefers the injected ``glicko_rd_sum`` column; reconstructs from the pre RD
    snapshots when only those are present; otherwise falls back to two cold-start RDs
    (700) so an un-injected row reads as maximally uncertain instead of confident.
    """
    if row.get("glicko_rd_sum") not in (None, ""):
        return _num(row, "glicko_rd_sum", GLICKO_NEUTRAL_RD_SUM)
    rd1 = row.get("team1_rd_pre")
    rd2 = row.get("team2_rd_pre")
    if rd1 not in (None, "") and rd2 not in (None, ""):
        return _num(row, "team1_rd_pre", GLICKO_NEUTRAL_RD) + _num(row, "team2_rd_pre", GLICKO_NEUTRAL_RD)
    return GLICKO_NEUTRAL_RD_SUM


def _implied_market_pair(odds_team1: float, odds_team2: float) -> tuple[float, float]:
    inv1 = 1.0 / odds_team1 if odds_team1 > 0 else 0.5
    inv2 = 1.0 / odds_team2 if odds_team2 > 0 else 0.5
    total = inv1 + inv2
    if total == 0:
        return 0.5, 0.5
    return inv1 / total, inv2 / total


def _normalize_map_name(value: Any) -> str:
    return str(value).strip().lower().replace("de_", "")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    text = str(value).strip().lower()
    if text in {"0", "false", "no", "n", "none"}:
        return False
    if text in {"1", "true", "yes", "y"}:
        return True
    try:
        return float(text) != 0.0
    except (TypeError, ValueError):
        return bool(text)


def _passthrough_binary(name: str) -> bool:
    return name in {"is_bo1", "is_bo3", "bp_applied", "odds_is_proxy"} or name.startswith("map_")
