import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class ExampleDataTests(unittest.TestCase):
    def test_cologne_2026_participants_cover_all_major_start_stages(self):
        from cs2pickem.data import read_matches_csv

        rows = read_matches_csv(os.path.join(ROOT, "examples", "cologne2026_participants.csv"))
        teams = {row["team"] for row in rows}
        stages = {}
        for row in rows:
            stages.setdefault(row["start_stage"], 0)
            stages[row["start_stage"]] += 1

        self.assertEqual(len(rows), 32)
        self.assertEqual(len(teams), 32)
        self.assertEqual(stages, {"stage1": 16, "stage2": 8, "stage3": 8})
        self.assertTrue(all(row.get("rmr_points") for row in rows))
        self.assertTrue(all(row.get("source_url") for row in rows))

    def test_cologne_2026_stage1_team_points_are_numeric(self):
        from cs2pickem.data import read_teams_csv

        rows = read_teams_csv(os.path.join(ROOT, "examples", "cologne2026_stage1_teams.csv"))

        self.assertIsInstance(rows[0]["rmr_points"], float)
        self.assertGreater(rows[0]["rmr_points"], rows[-1]["rmr_points"])

    def test_pickem_backtest_suite_sample_meets_documented_pass_rate_gate(self):
        from cs2pickem.backtest import backtest_pickem_suite_file

        report = backtest_pickem_suite_file(
            os.path.join(ROOT, "examples", "pickem_backtest_suite_sample.json"),
            pass_rate_target=0.38,
        )

        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["passed_cases"], 1)
        self.assertGreaterEqual(report["pass_rate"], 0.38)
        self.assertTrue(report["meets_pass_rate_target"])

    def test_cologne_final_fused_checkpoint_keeps_player_status_fields(self):
        from cs2pickem.backtest import checkpoint_pickem_file

        report = checkpoint_pickem_file(
            os.path.join(
                ROOT,
                "data",
                "cologne2026",
                "predictions",
                "fivee_6m_stage1_2026-06-01",
                "final_fused_pickem_2026-06-01.json",
            ),
            os.path.join(
                ROOT,
                "data",
                "cologne2026",
                "source_inputs",
                "stage1_round3_standings_2026-06-04.csv",
            ),
        )

        picks = {(row["category"], row["team"]): row for row in report["picks"]}
        self.assertIn("player_sample_confidence", picks[("3-0", "MIBR")])
        self.assertIn("substitute_flag", picks[("3-0", "MIBR")])
        self.assertIn("player_status_risk", picks[("3-0", "MIBR")])
        self.assertIn("status_adjusted_score", picks[("3-0", "MIBR")])
        self.assertEqual(picks[("3-0", "MIBR")]["player_sample_confidence"], 0.0)
        self.assertTrue(picks[("3-0", "MIBR")]["player_status_risk"])
        self.assertLess(
            picks[("3-0", "MIBR")]["status_adjusted_score"],
            picks[("3-0", "MIBR")]["raw_fused_score"],
        )
        self.assertGreater(report["category_diagnostics"]["3-0"]["player_status_risk_picks"], 0)
        self.assertEqual(
            report["category_diagnostics"]["advance"]["alive_status_risk_pressure_picks"],
            4,
        )

    def test_cologne_final_fused_pickems_apply_player_status_to_scores(self):
        import json

        with open(
            os.path.join(
                ROOT,
                "data",
                "cologne2026",
                "predictions",
                "fivee_6m_stage1_2026-06-01",
                "final_fused_pickem_2026-06-01.json",
            ),
            encoding="utf-8",
        ) as handle:
            report = json.load(handle)

        mibr = report["picks"]["3-0"][0]
        self.assertIn("raw_fused_score", mibr)
        self.assertIn("player_availability_multiplier", mibr)
        self.assertIn("status_adjusted_score", mibr)
        self.assertLess(mibr["status_adjusted_score"], mibr["raw_fused_score"])
        self.assertAlmostEqual(mibr["player_availability_multiplier"], 0.88)
        self.assertAlmostEqual(
            mibr["status_adjusted_score"],
            mibr["raw_fused_score"] * mibr["player_availability_multiplier"],
        )

        risk_detail = report["pickem_risk_details"]["3-0"][0]
        self.assertEqual(risk_detail["team"], mibr["team"])
        self.assertEqual(risk_detail["status_adjusted_score"], mibr["status_adjusted_score"])


if __name__ == "__main__":
    unittest.main()
