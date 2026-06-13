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


def _scored_history():
    """Same shape as _history but with round scores so MOV is exercised."""
    rows = []
    day = 1
    for _ in range(6):
        rows.append({
            "date": f"2026-03-{day:02d}", "team1": "A", "team2": "B", "winner": "A",
            "map": "mirage", "team1_score": 16, "team2_score": 9,
        })
        day += 1
    for _ in range(4):
        rows.append({
            "date": f"2026-03-{day:02d}", "team1": "B", "team2": "A", "winner": "B",
            "map": "inferno", "team1_score": 16, "team2_score": 11,
        })
        day += 1
    rows.append({
        "date": f"2026-03-{day:02d}", "team1": "A", "team2": "B", "winner": "A",
        "map": "mirage", "team1_score": 16, "team2_score": 4,
    })
    return rows


class ReliabilityGlickoInjectionTests(unittest.TestCase):
    def test_glicko_off_by_default_does_not_change_elo_or_bt_defaults(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        materialized, _final_elo, report = prepare_reliability_features(rows)
        # Default call must remain Elo-only: no Glicko columns, basis not_applied.
        self.assertNotIn("glicko_diff", materialized[0])
        self.assertNotIn("team1_glicko_pre", materialized[0])
        self.assertNotIn("glicko_rd_sum", materialized[0])
        self.assertIn("glicko", report)
        self.assertEqual(report["glicko"]["basis"], "not_applied")
        # Elo still injected exactly as before.
        self.assertIn("team1_elo", materialized[0])

    def test_glicko_columns_present_when_enabled(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        materialized, _final_elo, report = prepare_reliability_features(rows, inject_glicko=True)
        first = materialized[0]
        for col in (
            "team1_glicko_pre",
            "team2_glicko_pre",
            "team1_rd_pre",
            "team2_rd_pre",
            "glicko_diff",
            "glicko_rd_sum",
        ):
            self.assertIn(col, first)
        last = materialized[-1]
        # glicko_diff is the antisymmetric pre-match rating difference.
        self.assertAlmostEqual(
            last["glicko_diff"], last["team1_glicko_pre"] - last["team2_glicko_pre"], places=9
        )
        # glicko_rd_sum is the symmetric (swap-invariant) uncertainty tape.
        self.assertAlmostEqual(
            last["glicko_rd_sum"], last["team1_rd_pre"] + last["team2_rd_pre"], places=9
        )
        self.assertEqual(report["glicko"]["basis"], "chronological_pre_match_rolling")
        self.assertGreaterEqual(report["glicko"]["teams"], 2)
        self.assertIn("A", report["glicko"]["final"]["ratings"])

    def test_first_period_rows_get_cold_start_snapshot(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        materialized, _, _ = prepare_reliability_features(rows, inject_glicko=True)
        # No prior history -> cold start mu0=1500, phi0=350 -> diff 0, rd_sum 700.
        self.assertAlmostEqual(materialized[0]["team1_glicko_pre"], 1500.0, places=6)
        self.assertAlmostEqual(materialized[0]["team2_glicko_pre"], 1500.0, places=6)
        self.assertAlmostEqual(materialized[0]["glicko_diff"], 0.0, places=6)
        self.assertAlmostEqual(materialized[0]["glicko_rd_sum"], 700.0, places=6)

    def test_pre_match_snapshot_reflects_history_not_self(self):
        from cs2pickem.reliability import prepare_reliability_features

        # A dominant-throughout history: A beats B on every prior date, then a final
        # fixture. Glicko-2 is recency-sensitive, so we keep A winning right up to the
        # final match -> A's pre-match Glicko must lead B's, and it must NOT depend on the
        # final match itself (leakage-free snapshot).
        rows = []
        for day in range(1, 11):
            rows.append({
                "date": f"2026-05-{day:02d}", "team1": "A", "team2": "B", "winner": "A",
                "map": "mirage",
            })
        rows.append({
            "date": "2026-05-20", "team1": "A", "team2": "B", "winner": "A", "map": "mirage",
        })
        materialized, _, _ = prepare_reliability_features(rows, inject_glicko=True)
        last = materialized[-1]
        self.assertGreater(last["team1_glicko_pre"], last["team2_glicko_pre"])
        self.assertGreater(last["glicko_diff"], 0.0)

    def test_recent_results_dominate_glicko_snapshot(self):
        from cs2pickem.reliability import prepare_reliability_features

        # Unlike the order-invariant BT MLE, Glicko-2 is recency-weighted: if B wins the
        # most-recent games, B leads at the next fixture even with an even/older record.
        # This locks the documented semantic difference so WF-2F does not misattribute it.
        rows = _history()  # A wins days 1-6, then B wins days 7-10, then a final A-vs-B row
        materialized, _, _ = prepare_reliability_features(rows, inject_glicko=True)
        last = materialized[-1]
        self.assertGreater(last["team2_glicko_pre"], last["team1_glicko_pre"])

    def test_rolling_refit_does_not_read_future(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()
        baseline, _, _ = prepare_reliability_features(rows, inject_glicko=True)
        baseline_last = baseline[-1]["team1_glicko_pre"]

        with_future = list(rows)
        with_future.append(
            {"date": "2026-04-30", "team1": "B", "team2": "A", "winner": "B", "map": "mirage"}
        )
        future, _, _ = prepare_reliability_features(with_future, inject_glicko=True)
        target_date = rows[-1]["date"]
        same = next(r for r in future if r["date"] == target_date and r["team1"] == "A")
        self.assertAlmostEqual(same["team1_glicko_pre"], baseline_last, places=9)

    def test_same_period_rows_share_pre_snapshot(self):
        from cs2pickem.reliability import prepare_reliability_features

        # Two matches A faces on the SAME date must see the identical pre-period
        # snapshot for A (period batching: same-day games do not influence each other).
        rows = [
            {"date": "2026-05-01", "team1": "A", "team2": "B", "winner": "A", "map": "mirage"},
            {"date": "2026-05-01", "team1": "A", "team2": "C", "winner": "A", "map": "inferno"},
            {"date": "2026-05-02", "team1": "A", "team2": "B", "winner": "A", "map": "mirage"},
        ]
        materialized, _, _ = prepare_reliability_features(rows, inject_glicko=True)
        a_rows_day1 = [r for r in materialized if r["date"] == "2026-05-01"]
        self.assertEqual(len(a_rows_day1), 2)
        self.assertAlmostEqual(
            a_rows_day1[0]["team1_glicko_pre"], a_rows_day1[1]["team1_glicko_pre"], places=12
        )
        self.assertAlmostEqual(
            a_rows_day1[0]["team1_rd_pre"], a_rows_day1[1]["team1_rd_pre"], places=12
        )

    def test_mov_coverage_reported(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _scored_history()
        _, _, report = prepare_reliability_features(rows, inject_glicko=True)
        mov = report["glicko"]["mov"]
        # All scored rows have round diffs -> full MOV coverage; constants reported.
        self.assertEqual(mov["rows_with_round_diff"], len(rows))
        self.assertEqual(mov["rows_total"], len(rows))
        self.assertGreater(mov["coverage"], 0.99)
        self.assertIn("alpha", mov)
        self.assertIn("beta", mov)
        self.assertIn("gamma", mov)

    def test_mov_coverage_zero_when_no_scores(self):
        from cs2pickem.reliability import prepare_reliability_features

        rows = _history()  # no team*_score columns
        _, _, report = prepare_reliability_features(rows, inject_glicko=True)
        mov = report["glicko"]["mov"]
        self.assertEqual(mov["rows_with_round_diff"], 0)
        self.assertAlmostEqual(mov["coverage"], 0.0, places=9)

    def test_apply_final_glicko_to_upcoming_fixture(self):
        from cs2pickem.reliability import prepare_reliability_features, apply_final_glicko_to_match

        # A wins every prior date -> A's final-state rating leads, so the upcoming fixture
        # is scored in A's favour.
        rows = [
            {"date": f"2026-05-{day:02d}", "team1": "A", "team2": "B", "winner": "A", "map": "mirage"}
            for day in range(1, 11)
        ]
        _, _, report = prepare_reliability_features(rows, inject_glicko=True)
        final_glicko = report["glicko"]["final"]
        fixture = {"team1": "A", "team2": "B", "map": "mirage"}
        out = apply_final_glicko_to_match(fixture, final_glicko)
        for col in ("team1_glicko_pre", "team2_glicko_pre", "team1_rd_pre", "team2_rd_pre",
                    "glicko_diff", "glicko_rd_sum"):
            self.assertIn(col, out)
        self.assertAlmostEqual(
            out["glicko_diff"], out["team1_glicko_pre"] - out["team2_glicko_pre"], places=9
        )
        self.assertAlmostEqual(
            out["glicko_rd_sum"], out["team1_rd_pre"] + out["team2_rd_pre"], places=9
        )
        # A dominated history -> still favoured on the fixture.
        self.assertGreater(out["glicko_diff"], 0.0)

    def test_apply_final_glicko_only_fills_missing(self):
        from cs2pickem.reliability import apply_final_glicko_to_match

        final_glicko = {
            "ratings": {"A": 1700.0, "B": 1400.0},
            "rds": {"A": 60.0, "B": 80.0},
            "sigmas": {"A": 0.06, "B": 0.06},
        }
        # Pre-filled snapshot must be preserved (only missing columns filled).
        fixture = {
            "team1": "A", "team2": "B",
            "team1_glicko_pre": 1234.0, "team2_glicko_pre": 1111.0,
            "team1_rd_pre": 10.0, "team2_rd_pre": 20.0,
        }
        out = apply_final_glicko_to_match(fixture, final_glicko)
        self.assertEqual(out["team1_glicko_pre"], 1234.0)
        self.assertEqual(out["team2_rd_pre"], 20.0)
        self.assertAlmostEqual(out["glicko_diff"], 1234.0 - 1111.0, places=9)
        self.assertAlmostEqual(out["glicko_rd_sum"], 10.0 + 20.0, places=9)

    def test_apply_final_glicko_unknown_team_uses_cold_start(self):
        from cs2pickem.reliability import apply_final_glicko_to_match

        final_glicko = {"ratings": {"A": 1700.0}, "rds": {"A": 60.0}, "sigmas": {"A": 0.06}}
        fixture = {"team1": "A", "team2": "ZZZ", "map": "mirage"}
        out = apply_final_glicko_to_match(fixture, final_glicko)
        # Unknown team -> mu0=1500, phi0=350 cold start.
        self.assertAlmostEqual(out["team2_glicko_pre"], 1500.0, places=6)
        self.assertAlmostEqual(out["team2_rd_pre"], 350.0, places=6)
        self.assertAlmostEqual(out["team1_glicko_pre"], 1700.0, places=6)

    def test_elo_default_behavior_unchanged_with_glicko_flag(self):
        from cs2pickem.reliability import prepare_reliability_features

        # Turning Glicko on must not perturb the Elo columns at all.
        rows = _history()
        elo_only, _, _ = prepare_reliability_features(rows)
        with_glicko, _, _ = prepare_reliability_features(rows, inject_glicko=True)
        for a, b in zip(elo_only, with_glicko):
            self.assertEqual(a["team1_elo"], b["team1_elo"])
            self.assertEqual(a["team2_elo"], b["team2_elo"])


if __name__ == "__main__":
    unittest.main()
