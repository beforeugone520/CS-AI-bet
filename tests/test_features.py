import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


# Field-pair aliases that carry a per-team orientation without a literal
# ``team1_``/``team2_`` prefix. Swapping teams must also swap these.
_PAIR_ALIASES = {
    "odds_team1": "odds_team2",
    "odds_team2": "odds_team1",
    "team1_bans": "team2_bans",
    "team2_bans": "team1_bans",
    "team1_picks": "team2_picks",
    "team2_picks": "team1_picks",
    "team1_pick": "team2_pick",
    "team2_pick": "team1_pick",
    "bt_team1_strength": "bt_team2_strength",
    "bt_team2_strength": "bt_team1_strength",
    "bt_team1_map_strength": "bt_team2_map_strength",
    "bt_team2_map_strength": "bt_team1_map_strength",
}

# Pre-injected diff columns: upstream computes them as ``team1 - team2`` from the
# team identity, so a genuine team swap negates them. The test models that by
# negating these carried-over values. ``glicko_rd_sum`` is intentionally absent --
# it is swap-invariant, so a genuine swap carries it over unchanged.
_NEGATED_ON_SWAP = ("bt_strength_diff", "bt_map_strength_diff", "glicko_diff")


def _swap_teams(row):
    """Return a team-swapped copy of a feature row (team1 <-> team2), label inverted."""
    swapped = {}
    for key, value in row.items():
        if key in _PAIR_ALIASES:
            swapped[_PAIR_ALIASES[key]] = value
        elif key in _NEGATED_ON_SWAP:
            swapped[key] = -value
        elif key.startswith("team1_"):
            swapped["team2_" + key[len("team1_"):]] = value
        elif key.startswith("team2_"):
            swapped["team1_" + key[len("team2_"):]] = value
        elif key == "team1":
            swapped["team2"] = value
        elif key == "team2":
            swapped["team1"] = value
        else:
            swapped[key] = value
    # Winner identity is preserved (it names a team), label flips because team1/team2 swapped.
    return swapped


class FeatureBuilderNewColumnTests(unittest.TestCase):
    def test_consumes_bradley_terry_diff_columns(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {
                "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage",
                "bt_team1_strength": 0.8, "bt_team2_strength": 0.2, "bt_strength_diff": 0.6,
                "bt_team1_map_strength": 0.9, "bt_team2_map_strength": 0.1, "bt_map_strength_diff": 0.8,
            },
            {
                "team1": "Charlie", "team2": "Delta", "winner": "Delta", "best_of": 3, "map": "inferno",
                "bt_team1_strength": -0.5, "bt_team2_strength": 0.5, "bt_strength_diff": -1.0,
                "bt_team1_map_strength": -0.6, "bt_team2_map_strength": 0.4, "bt_map_strength_diff": -1.0,
            },
        ]
        dataset = FeatureBuilder().fit_transform(rows)

        for name in ("bt_strength_diff", "bt_map_strength_diff"):
            self.assertIn(name, dataset.feature_names)
            index = dataset.feature_names.index(name)
            # The team1-favoured row should score higher than the team2-favoured one.
            self.assertGreater(dataset.rows[0][index], dataset.rows[1][index])

    def test_bt_strength_diff_defaults_to_neutral_when_absent(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {"team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1},
            {"team1": "Charlie", "team2": "Delta", "winner": "Delta", "best_of": 3},
        ]
        builder = FeatureBuilder()
        builder.fit_transform(rows)
        raw = builder._raw_features({"team1": "Alpha", "team2": "Bravo"})
        self.assertEqual(raw["bt_strength_diff"], 0.0)
        self.assertEqual(raw["bt_map_strength_diff"], 0.0)

    def test_consumes_5e_enrichment_diff_and_symmetric_sum(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {
                "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1,
                "event_grade": 9, "team1_fivee_6m_avg_event_grade": 8.0,
                "team2_fivee_6m_avg_event_grade": 5.0,
            },
            {
                "team1": "Charlie", "team2": "Delta", "winner": "Delta", "best_of": 3,
                "event_grade": 3, "team1_fivee_6m_avg_event_grade": 4.0,
                "team2_fivee_6m_avg_event_grade": 4.0,
            },
        ]
        # The 5E columns are gated OFF by default (dead in the real pipeline until a
        # profile->match join is wired), so they are computed but not exposed to the
        # selector. Opting in (WF-2F) puts them back into the candidate pool.
        builder = FeatureBuilder()
        raw = builder._raw_features(rows[0])
        self.assertEqual(raw["event_grade_sum"], 9.0)
        self.assertEqual(raw["team_event_grade_sum"], 13.0)
        self.assertEqual(raw["team_event_grade_diff"], 3.0)

        opted_in = FeatureBuilder(include_unverified_features=True).fit_transform(rows)
        self.assertIn("event_grade_sum", opted_in.feature_names)
        self.assertIn("team_event_grade_sum", opted_in.feature_names)
        self.assertIn("team_event_grade_diff", opted_in.feature_names)

    def test_event_grade_sum_is_symmetric_under_team_swap(self):
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        row = {
            "team1": "Alpha", "team2": "Bravo", "best_of": 1, "event_grade": 7,
            "team1_fivee_6m_avg_event_grade": 8.0, "team2_fivee_6m_avg_event_grade": 5.0,
        }
        raw = builder._raw_features(row)
        swapped = builder._raw_features(_swap_teams(row))
        # Symmetric-sum magnitude features are invariant under team swap.
        self.assertAlmostEqual(raw["event_grade_sum"], swapped["event_grade_sum"])
        self.assertAlmostEqual(raw["team_event_grade_sum"], swapped["team_event_grade_sum"])

    def test_consumes_bp_structured_features(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {
                "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 3,
                "bp_applied": 1, "bp_confidence": 0.82,
                "team1_bans": "nuke|ancient", "team2_bans": "anubis|nuke",
            },
            {
                "team1": "Charlie", "team2": "Delta", "winner": "Delta", "best_of": 3,
                "bp_applied": 0,
            },
        ]
        # bp_applied (the has-intel gate) competes by default; the sparse magnitude
        # columns are gated OFF (weak prior) until WF-2F opts them in.
        dataset = FeatureBuilder().fit_transform(rows)
        self.assertIn("bp_applied", dataset.feature_names)
        for name in ("bp_confidence", "bp_total_bans", "bp_ban_overlap", "bp_total_picks"):
            self.assertNotIn(name, dataset.feature_names)
        opted_in = FeatureBuilder(include_unverified_features=True).fit_transform(rows)
        for name in ("bp_applied", "bp_confidence", "bp_total_bans", "bp_ban_overlap", "bp_total_picks"):
            self.assertIn(name, opted_in.feature_names)

    def test_bp_ban_overlap_counts_shared_bans(self):
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        raw = builder._raw_features(
            {
                "team1": "Alpha", "team2": "Bravo", "best_of": 3,
                "bp_applied": 1, "team1_bans": "nuke|ancient", "team2_bans": "anubis|nuke",
            }
        )
        # "nuke" is banned by both teams.
        self.assertEqual(raw["bp_ban_overlap"], 1.0)
        self.assertEqual(raw["bp_total_bans"], 4.0)

    def test_bp_features_neutral_when_not_applied(self):
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        raw = builder._raw_features({"team1": "Alpha", "team2": "Bravo", "best_of": 3})
        self.assertEqual(raw["bp_applied"], 0.0)
        self.assertEqual(raw["bp_confidence"], 0.0)
        self.assertEqual(raw["bp_total_bans"], 0.0)
        self.assertEqual(raw["bp_ban_overlap"], 0.0)

    def test_bp_features_symmetric_under_team_swap(self):
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        row = {
            "team1": "Alpha", "team2": "Bravo", "best_of": 3,
            "bp_applied": 1, "bp_confidence": 0.7,
            "team1_bans": "nuke|ancient", "team2_bans": "anubis|nuke",
        }
        raw = builder._raw_features(row)
        swapped = builder._raw_features(_swap_teams(row))
        for name in ("bp_applied", "bp_confidence", "bp_total_bans", "bp_ban_overlap", "bp_total_picks"):
            self.assertAlmostEqual(raw[name], swapped[name])

    def test_consumes_odds_meta_features(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {
                "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1,
                "odds_team1": 1.5, "odds_team2": 2.6, "odds_provider_count": 3,
                "overround": 0.05, "devig_z": 0.02, "market_signal_proxy": False,
            },
            {
                "team1": "Charlie", "team2": "Delta", "winner": "Delta", "best_of": 3,
                "odds_team1": 2.0, "odds_team2": 1.8, "odds_provider_count": 1,
                "overround": 0.08, "devig_z": 0.0, "market_signal_proxy": True,
            },
        ]
        # The merge-written columns (odds_provider_count / odds_is_proxy) compete by
        # default; overround / devig_z are gated OFF (dropped by the odds merge ->
        # dead in the real pipeline) until WF-2F wires them in and opts them in.
        dataset = FeatureBuilder().fit_transform(rows)
        for name in ("odds_provider_count", "odds_is_proxy"):
            self.assertIn(name, dataset.feature_names)
        for name in ("odds_overround", "odds_devig_z"):
            self.assertNotIn(name, dataset.feature_names)
        opted_in = FeatureBuilder(include_unverified_features=True).fit_transform(rows)
        for name in ("odds_provider_count", "odds_overround", "odds_devig_z", "odds_is_proxy"):
            self.assertIn(name, opted_in.feature_names)

    def test_odds_is_proxy_reads_flag(self):
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        proxy = builder._raw_features(
            {"team1": "A", "team2": "B", "market_signal_proxy": True}
        )
        real = builder._raw_features(
            {"team1": "A", "team2": "B", "market_signal_proxy": False}
        )
        self.assertEqual(proxy["odds_is_proxy"], 1.0)
        self.assertEqual(real["odds_is_proxy"], 0.0)

    def test_full_antisymmetry_preserved_for_all_diff_features(self):
        """Swapping teams must negate every *_diff column and leave swap-invariant
        magnitude/sum columns unchanged. This is the correctness contract."""
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        row = {
            "team1": "Alpha", "team2": "Bravo", "best_of": 3, "map": "mirage",
            "team1_elo": 1600.0, "team2_elo": 1450.0,
            "team1_rank": 4, "team2_rank": 12,
            "bt_strength_diff": 0.6, "bt_team1_strength": 0.8, "bt_team2_strength": 0.2,
            "bt_map_strength_diff": 0.5, "bt_team1_map_strength": 0.7, "bt_team2_map_strength": 0.2,
            "event_grade": 8, "team1_fivee_6m_avg_event_grade": 7.0, "team2_fivee_6m_avg_event_grade": 5.0,
            "bp_applied": 1, "bp_confidence": 0.7, "team1_bans": "nuke", "team2_bans": "nuke|anubis",
            "odds_team1": 1.5, "odds_team2": 2.6, "odds_provider_count": 2,
            "overround": 0.05, "devig_z": 0.01, "market_signal_proxy": False,
        }
        raw = builder._raw_features(row)
        swapped = builder._raw_features(_swap_teams(row))
        for name, value in raw.items():
            if name.endswith("_diff"):
                self.assertAlmostEqual(
                    swapped[name], -value, places=9, msg=f"{name} must negate on team swap"
                )

    def test_strong_signal_new_columns_compete_by_default(self):
        """Only genuinely-wired, strong-prior new columns are in the default
        candidate pool so they compete in the selector without dilution."""
        from cs2pickem.features import FeatureBuilder

        active = set(FeatureBuilder().feature_names)
        for name in (
            "bt_strength_diff",
            "bt_map_strength_diff",
            "bp_applied",
            "odds_provider_count",
            "odds_is_proxy",
        ):
            self.assertIn(name, active)

    def test_unverified_columns_are_off_by_default(self):
        """Dead / weak-prior columns must NOT be in the default candidate pool so
        in-sample selection is not diluted (review red-line c). They are still
        computed by _raw_features for inspection."""
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        active = set(builder.feature_names)
        for name in (
            "event_grade_sum",
            "team_event_grade_sum",
            "team_event_grade_diff",
            "odds_overround",
            "odds_devig_z",
            "bp_confidence",
            "bp_total_bans",
            "bp_ban_overlap",
            "bp_total_picks",
        ):
            self.assertNotIn(name, active)
            # Still computed (just not exposed to the selector) -> back-compat.
            self.assertIn(name, builder._raw_features({"team1": "A", "team2": "B"}))

    def test_unverified_columns_opt_in_for_wf2f_ab(self):
        """WF-2F can opt the gated columns back into the candidate pool for an
        A/B significance adjudication."""
        from cs2pickem.features import FeatureBuilder

        active = set(FeatureBuilder(include_unverified_features=True).feature_names)
        for name in (
            "event_grade_sum",
            "team_event_grade_sum",
            "team_event_grade_diff",
            "odds_overround",
            "odds_devig_z",
            "bp_confidence",
            "bp_total_bans",
            "bp_ban_overlap",
            "bp_total_picks",
        ):
            self.assertIn(name, active)

    def test_full_feature_name_superset_is_stable(self):
        """The class-level feature_names superset still lists every new column so
        callers reading the full ordering keep working."""
        from cs2pickem.features import FeatureBuilder

        names = set(FeatureBuilder.feature_names)
        for name in (
            "bt_strength_diff",
            "bt_map_strength_diff",
            "glicko_diff",
            "glicko_rd_sum",
            "event_grade_sum",
            "team_event_grade_sum",
            "team_event_grade_diff",
            "bp_applied",
            "bp_confidence",
            "bp_total_bans",
            "bp_ban_overlap",
            "bp_total_picks",
            "odds_provider_count",
            "odds_overround",
            "odds_devig_z",
            "odds_is_proxy",
        ):
            self.assertIn(name, names)


class FeatureBuilderGlickoTests(unittest.TestCase):
    """WF-2C: FeatureBuilder consumes the leakage-free pre-match Glicko-2 signals.

    glicko_diff (strong antisymmetric rating gap) competes in the default candidate
    pool; glicko_rd_sum (weak-prior symmetric uncertainty tape) is gated OFF until the
    WF-2F A/B opts it in. Neither must break the antisymmetry / back-compat contracts.
    """

    def test_consumes_glicko_diff_column(self):
        from cs2pickem.features import FeatureBuilder

        rows = [
            {
                "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage",
                "team1_glicko_pre": 1700.0, "team2_glicko_pre": 1400.0,
                "team1_rd_pre": 80.0, "team2_rd_pre": 120.0,
                "glicko_diff": 300.0, "glicko_rd_sum": 200.0,
            },
            {
                "team1": "Charlie", "team2": "Delta", "winner": "Delta", "best_of": 3, "map": "inferno",
                "team1_glicko_pre": 1450.0, "team2_glicko_pre": 1650.0,
                "team1_rd_pre": 110.0, "team2_rd_pre": 90.0,
                "glicko_diff": -200.0, "glicko_rd_sum": 200.0,
            },
        ]
        dataset = FeatureBuilder().fit_transform(rows)
        self.assertIn("glicko_diff", dataset.feature_names)
        index = dataset.feature_names.index("glicko_diff")
        # The team1-favoured row scores higher on glicko_diff than the team2-favoured one.
        self.assertGreater(dataset.rows[0][index], dataset.rows[1][index])

    def test_glicko_diff_is_default_active_candidate(self):
        """glicko_diff is a strong directional signal wired end-to-end -> it competes in
        the default candidate pool (mirrors the BT diffs), not gated to UNVERIFIED."""
        from cs2pickem.features import FeatureBuilder

        active = set(FeatureBuilder().feature_names)
        self.assertIn("glicko_diff", active)

    def test_glicko_rd_sum_is_unverified_off_by_default(self):
        """glicko_rd_sum is a weak-prior magnitude (large RD-sum mostly flags
        cold-start/inactive teams) -> gated OFF by default, still computed for
        inspection, opted back in for the WF-2F A/B."""
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        active = set(builder.feature_names)
        self.assertNotIn("glicko_rd_sum", active)
        # Still computed (just not exposed to the selector) -> back-compat.
        self.assertIn("glicko_rd_sum", builder._raw_features({"team1": "A", "team2": "B"}))
        self.assertIn("glicko_rd_sum", FeatureBuilder.UNVERIFIED_FEATURE_NAMES)
        # Opt-in (WF-2F) puts it back into the candidate pool.
        opted_in = set(FeatureBuilder(include_unverified_features=True).feature_names)
        self.assertIn("glicko_rd_sum", opted_in)
        self.assertIn("glicko_diff", opted_in)

    def test_glicko_defaults_to_neutral_when_absent(self):
        """An un-injected row reads as neutral rating gap (0) and maximally uncertain
        (two cold-start RDs = 700), never spuriously confident."""
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        builder.fit_transform([
            {"team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1},
        ])
        raw = builder._raw_features({"team1": "Alpha", "team2": "Bravo"})
        self.assertEqual(raw["glicko_diff"], 0.0)
        self.assertEqual(raw["glicko_rd_sum"], 700.0)

    def test_glicko_reconstructs_from_pre_snapshots(self):
        """When only the pre snapshots are present, glicko_diff / glicko_rd_sum are
        reconstructed from them (team1 - team2 / team1 + team2)."""
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        raw = builder._raw_features({
            "team1": "A", "team2": "B",
            "team1_glicko_pre": 1620.0, "team2_glicko_pre": 1500.0,
            "team1_rd_pre": 100.0, "team2_rd_pre": 220.0,
        })
        self.assertAlmostEqual(raw["glicko_diff"], 120.0, places=9)
        self.assertAlmostEqual(raw["glicko_rd_sum"], 320.0, places=9)

    def test_glicko_diff_antisymmetric_rd_sum_symmetric_under_swap(self):
        """glicko_diff negates under team swap; glicko_rd_sum is swap-invariant."""
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        row = {
            "team1": "Alpha", "team2": "Bravo", "best_of": 3, "map": "mirage",
            "team1_glicko_pre": 1640.0, "team2_glicko_pre": 1490.0,
            "team1_rd_pre": 90.0, "team2_rd_pre": 160.0,
            "glicko_diff": 150.0, "glicko_rd_sum": 250.0,
        }
        raw = builder._raw_features(row)
        swapped_row = _swap_teams(row)
        # _swap_teams negates carried-over *_diff columns and swaps team1_/team2_ prefixes,
        # so this exercises both the injected column and the snapshot reconstruction paths.
        swapped = builder._raw_features(swapped_row)
        self.assertAlmostEqual(swapped["glicko_diff"], -raw["glicko_diff"], places=9)
        self.assertAlmostEqual(swapped["glicko_rd_sum"], raw["glicko_rd_sum"], places=9)

    def test_glicko_diff_in_full_antisymmetry_contract(self):
        """glicko_diff participates in the all-*_diff-columns antisymmetry contract."""
        from cs2pickem.features import FeatureBuilder

        builder = FeatureBuilder()
        row = {
            "team1": "Alpha", "team2": "Bravo", "best_of": 3, "map": "mirage",
            "team1_glicko_pre": 1700.0, "team2_glicko_pre": 1400.0,
            "team1_rd_pre": 70.0, "team2_rd_pre": 130.0,
            "glicko_diff": 300.0, "glicko_rd_sum": 200.0,
        }
        raw = builder._raw_features(row)
        swapped = builder._raw_features(_swap_teams(row))
        self.assertIn("glicko_diff", raw)
        for name, value in raw.items():
            if name.endswith("_diff"):
                self.assertAlmostEqual(
                    swapped[name], -value, places=9, msg=f"{name} must negate on team swap"
                )

    def test_glicko_columns_excluded_from_unstable_identity_and_required(self):
        """No name drift: glicko candidate columns are neither in the excluded
        (unstable-identity) set nor force-added to the player-status required set."""
        from cs2pickem.reliability import (
            PLAYER_STATUS_REQUIRED_FEATURES,
            UNSTABLE_IDENTITY_FEATURES,
            GLICKO_CANDIDATE_FEATURES,
        )

        for name in ("glicko_diff", "glicko_rd_sum"):
            self.assertIn(name, GLICKO_CANDIDATE_FEATURES)
            self.assertNotIn(name, UNSTABLE_IDENTITY_FEATURES)
            self.assertNotIn(name, PLAYER_STATUS_REQUIRED_FEATURES)


if __name__ == "__main__":
    unittest.main()
