import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def pickem_report():
    return {
        "simulations": 100000,
        "stage_strategy": {"stage": "challengers", "focus": "BO1/map pool volatility"},
        "pickems": {
            "3-0": ["Alpha"],
            "advance": ["Bravo"],
            "0-3": ["Delta"],
        },
        "team_probabilities": {
            "Alpha": {"3-0": 0.31, "3-1": 0.34, "3-2": 0.22, "advance": 0.87, "0-3": 0.01, "1-3": 0.04, "2-3": 0.08, "eliminate": 0.13},
            "Bravo": {"3-0": 0.12, "3-1": 0.28, "3-2": 0.34, "advance": 0.74, "0-3": 0.03, "1-3": 0.08, "2-3": 0.15, "eliminate": 0.26},
            "Delta": {"3-0": 0.02, "3-1": 0.08, "3-2": 0.18, "advance": 0.28, "0-3": 0.42, "1-3": 0.20, "2-3": 0.10, "eliminate": 0.72},
        },
        "pickem_details": {
            "3-0": [
                {"team": "Alpha", "category": "3-0", "probability": 0.31, "rank": 4, "selection_score": 0.33, "next_best_score": 0.305, "selection_margin": 0.025},
            ],
            "advance": [
                {"team": "Bravo", "category": "advance", "probability": 0.74, "rank": 20, "selection_score": 0.74, "next_best_score": 0.68, "selection_margin": 0.06},
            ],
            "0-3": [
                {"team": "Delta", "category": "0-3", "probability": 0.42, "rank": 24, "selection_score": 0.47, "next_best_score": 0.40, "selection_margin": 0.07},
            ],
        },
        "market_adjustment_summary": {"cached_matchups": 2, "adjusted_matchups": 0, "adjusted_matchup_keys": []},
    }


class ExportTests(unittest.TestCase):
    def test_build_pickem_answer_sheet_flattens_picks_and_surfaces_warnings(self):
        from cs2pickem.export import build_pickem_answer_sheet

        sheet = build_pickem_answer_sheet(
            pickem_report(),
            readiness_report={"ready": False, "failed_checks": ["pickem_selection_margin"]},
        )

        self.assertEqual(sheet["pickems"]["3-0"], ["Alpha"])
        self.assertFalse(sheet["ready"])
        self.assertEqual(sheet["failed_checks"], ["pickem_selection_margin"])
        self.assertEqual(sheet["stage"], "challengers")
        self.assertEqual(sheet["simulations"], 100000)
        self.assertEqual([pick["team"] for pick in sheet["picks"]], ["Alpha", "Bravo", "Delta"])
        self.assertEqual(sheet["picks"][0]["category"], "3-0")
        self.assertAlmostEqual(sheet["picks"][0]["selection_margin"], 0.025)
        self.assertAlmostEqual(sheet["confidence"]["minimum_selection_margin"], 0.025)
        self.assertEqual(sheet["confidence"]["market_adjusted_matchups"], 0)
        self.assertEqual([row["team"] for row in sheet["team_outcomes"]], ["Alpha", "Bravo", "Delta"])
        self.assertAlmostEqual(sheet["team_outcomes"][0]["advance"], 0.87)
        self.assertAlmostEqual(sheet["team_outcomes"][0]["eliminate"], 0.13)
        self.assertEqual(sheet["team_outcomes"][0]["most_likely_record"], "3-1")
        self.assertEqual(sheet["team_outcomes"][2]["most_likely_record"], "0-3")
        self.assertIn("readiness_not_ready", [warning["code"] for warning in sheet["warnings"]])
        self.assertIn("low_selection_margin", [warning["code"] for warning in sheet["warnings"]])
        self.assertIn("no_market_adjusted_matchups", [warning["code"] for warning in sheet["warnings"]])

    def test_pickem_answer_sheet_file_workflow_writes_json(self):
        from cs2pickem.export import build_pickem_answer_sheet_file

        with tempfile.TemporaryDirectory() as tmpdir:
            pickem_path = os.path.join(tmpdir, "pickem.json")
            readiness_path = os.path.join(tmpdir, "readiness.json")
            output_path = os.path.join(tmpdir, "answer_sheet.json")
            with open(pickem_path, "w", encoding="utf-8") as handle:
                json.dump(pickem_report(), handle)
            with open(readiness_path, "w", encoding="utf-8") as handle:
                json.dump({"ready": True, "failed_checks": []}, handle)

            sheet = build_pickem_answer_sheet_file(pickem_path, readiness_path, output_path)

            self.assertTrue(os.path.exists(output_path))
            with open(output_path, encoding="utf-8") as handle:
                saved = json.load(handle)

        self.assertTrue(sheet["ready"])
        self.assertEqual(saved["picks"][1]["team"], "Bravo")

    def test_cli_exports_pickem_answer_sheet(self):
        from cs2pickem.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            pickem_path = os.path.join(tmpdir, "pickem.json")
            readiness_path = os.path.join(tmpdir, "readiness.json")
            output_path = os.path.join(tmpdir, "answer_sheet.json")
            with open(pickem_path, "w", encoding="utf-8") as handle:
                json.dump(pickem_report(), handle)
            with open(readiness_path, "w", encoding="utf-8") as handle:
                json.dump({"ready": False, "failed_checks": ["data_volume"]}, handle)

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "answer-sheet",
                "--pickem-report",
                pickem_path,
                "--readiness-report",
                readiness_path,
                "--output",
                output_path,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            with open(output_path, encoding="utf-8") as handle:
                sheet = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual(sheet["failed_checks"], ["data_volume"])
        self.assertEqual(sheet["picks"][2]["team"], "Delta")


if __name__ == "__main__":
    unittest.main()
