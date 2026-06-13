import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _history():
    """A's strength accrues over time; the final match should see A favoured."""
    rows = []
    day = 1
    for _ in range(6):
        rows.append({"date": f"2026-03-{day:02d}", "team1": "A", "team2": "B", "winner": "A", "map": "mirage"})
        day += 1
    for _ in range(4):
        rows.append({"date": f"2026-03-{day:02d}", "team1": "B", "team2": "A", "winner": "B", "map": "inferno"})
        day += 1
    # Final pre-match fixture (still has a winner, used only to score, never to fit itself).
    rows.append({"date": f"2026-03-{day:02d}", "team1": "A", "team2": "B", "winner": "A", "map": "mirage"})
    return rows


class ReliabilityBtInjectionTests(unittest.TestCase):
    def test_default_still_injects_elo_only_when_bt_disabled(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        materialized, _, report = prepare_reliability_features(rows, inject_bt=False)
        self.assertIn("team1_elo", materialized[0])
        self.assertNotIn("bt_team1_strength", materialized[0])
        self.assertEqual(report["bt"]["basis"], "not_applied")

    def test_bt_columns_present_when_enabled(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        materialized, _final_elo, report = prepare_reliability_features(rows, inject_bt=True)
        first = materialized[0]
        for col in (
            "bt_team1_strength",
            "bt_team2_strength",
            "bt_strength_diff",
            "bt_team1_map_strength",
            "bt_team2_map_strength",
            "bt_map_strength_diff",
        ):
            self.assertIn(col, first)
        # diff columns are the literal difference of the two snapshots.
        last = materialized[-1]
        self.assertAlmostEqual(
            last["bt_strength_diff"], last["bt_team1_strength"] - last["bt_team2_strength"], places=9
        )
        self.assertAlmostEqual(
            last["bt_map_strength_diff"],
            last["bt_team1_map_strength"] - last["bt_team2_map_strength"],
            places=9,
        )
        self.assertEqual(report["bt"]["basis"], "chronological_pre_match_rolling")
        self.assertGreaterEqual(report["bt"]["teams"], 2)
        # final full-history fit is exposed for upcoming fixtures.
        self.assertIn("A", report["bt"]["final"])

    def test_first_row_has_no_pre_history_so_strengths_are_neutral(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        materialized, _, _ = prepare_reliability_features(rows, inject_bt=True)
        # No prior history before the first chronological match -> neutral (0) snapshots.
        self.assertAlmostEqual(materialized[0]["bt_team1_strength"], 0.0, places=9)
        self.assertAlmostEqual(materialized[0]["bt_team2_strength"], 0.0, places=9)

    def test_pre_match_snapshot_excludes_the_match_being_scored(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        materialized, _, _ = prepare_reliability_features(rows, inject_bt=True)
        # The final match: A has dominated history -> A's pre-match strength > B's,
        # and that snapshot must NOT depend on the final match itself.
        last = materialized[-1]
        self.assertGreater(last["bt_team1_strength"], last["bt_team2_strength"])

    def test_rolling_refit_does_not_read_future(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        # The strength A has at the final match must be identical whether or not
        # we append a future result that would (if leaked) change the fit.
        baseline, _, _ = prepare_reliability_features(rows, inject_bt=True)
        baseline_last = baseline[-1]["bt_team1_strength"]

        with_future = list(rows)
        with_future.append(
            {"date": "2026-04-30", "team1": "B", "team2": "A", "winner": "B", "map": "mirage"}
        )
        future, _, _ = prepare_reliability_features(with_future, inject_bt=True)
        # Find the same final-fixture row inside the extended run (same date).
        target_date = rows[-1]["date"]
        same = next(r for r in future if r["date"] == target_date and r["team1"] == "A")
        self.assertAlmostEqual(same["bt_team1_strength"], baseline_last, places=9)

    def test_apply_final_bt_to_upcoming_fixture(self):
        from cs2pickem.reliability import prepare_reliability_features, apply_final_bt_to_match

        rows = _history()
        _, _, report = prepare_reliability_features(rows, inject_bt=True)
        final_bt = report["bt"]["final"]
        fixture = {"team1": "A", "team2": "B", "map": "mirage"}
        out = apply_final_bt_to_match(fixture, final_bt)
        self.assertIn("bt_team1_strength", out)
        self.assertIn("bt_team2_strength", out)
        self.assertAlmostEqual(
            out["bt_strength_diff"], out["bt_team1_strength"] - out["bt_team2_strength"], places=9
        )

    def test_backward_compatible_default_call(self):
        from cs2pickem.reliability import prepare_reliability_features

        # Old call sites pass only rows / inject_elo and must keep working.
        rows = _history()
        materialized, final_elo, report = prepare_reliability_features(rows)
        self.assertIn("elo", report)
        self.assertIn("team1_elo", materialized[0])

    def test_bt_injection_is_off_by_default(self):
        from cs2pickem.reliability import prepare_reliability_features

        # Default (no inject_bt kwarg) must NOT pay the ~30s rolling-BT refit nor
        # inject BT columns: the hot training/prediction/tuning paths stay Elo-only,
        # and there is no train/serve skew because serving also has no BT. WF-2F
        # opts in explicitly.
        rows = _history()
        materialized, _, report = prepare_reliability_features(rows)
        self.assertNotIn("bt_team1_strength", materialized[0])
        self.assertNotIn("bt_strength_diff", materialized[0])
        self.assertEqual(report["bt"]["basis"], "not_applied")

    def test_bt_diffs_are_candidate_features_not_forced_required(self):
        from cs2pickem.reliability import (
            BT_CANDIDATE_FEATURES,
            PLAYER_STATUS_REQUIRED_FEATURES,
            prepare_reliability_features,
        )

        # Per the WF-2 red-line, BT diffs compete in FeatureSelector; they must NOT be
        # force-added to the required (protected) list. Player-status diffs stay required.
        self.assertIn("bt_strength_diff", BT_CANDIDATE_FEATURES)
        self.assertIn("bt_map_strength_diff", BT_CANDIDATE_FEATURES)

        rows = _history()
        _, _, report = prepare_reliability_features(rows, inject_bt=True)
        required = report["required_feature_names"]
        self.assertNotIn("bt_strength_diff", required)
        self.assertNotIn("bt_map_strength_diff", required)
        # The unchanged player-status protection still holds (no name drift).
        self.assertEqual(required, list(PLAYER_STATUS_REQUIRED_FEATURES))

    def test_required_and_excluded_lists_do_not_overlap(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        _, _, report = prepare_reliability_features(rows, inject_bt=True)
        excluded = set(report["excluded_feature_names"])
        required = set(report["required_feature_names"])
        # No name drift: a feature cannot be both excluded and required.
        self.assertEqual(excluded & required, set())


if __name__ == "__main__":
    unittest.main()
