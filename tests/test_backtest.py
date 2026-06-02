import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class PickemBacktestTests(unittest.TestCase):
    def test_evaluate_pickem_result_scores_categories_and_pass_threshold(self):
        from cs2pickem.backtest import evaluate_pickem_result

        report = evaluate_pickem_result(
            {
                "3-0": ["Alpha", "Bravo"],
                "advance": ["Charlie", "Delta", "Echo"],
                "0-3": ["Foxtrot", "Golf"],
            },
            [
                {"team": "Alpha", "wins": 3, "losses": 0},
                {"team": "Bravo", "wins": 3, "losses": 1},
                {"team": "Charlie", "wins": 3, "losses": 2},
                {"team": "Delta", "wins": 2, "losses": 3},
                {"team": "Echo", "wins": 3, "losses": 1},
                {"team": "Foxtrot", "wins": 0, "losses": 3},
                {"team": "Golf", "wins": 1, "losses": 3},
            ],
            pass_threshold=4,
        )

        self.assertEqual(report["correct"], 4)
        self.assertEqual(report["total_picks"], 7)
        self.assertTrue(report["passed"])
        self.assertEqual(report["category_scores"]["3-0"]["correct"], 1)
        self.assertEqual(report["category_scores"]["advance"]["correct"], 2)
        self.assertEqual(report["category_scores"]["0-3"]["correct"], 1)
        self.assertIn({"category": "3-0", "team": "Bravo"}, report["missed_picks"])

    def test_backtest_cli_reads_pickem_json_and_standings_csv(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            pickems_path = os.path.join(tmpdir, "pickems.json")
            results_path = os.path.join(tmpdir, "standings.csv")
            output_path = os.path.join(tmpdir, "backtest.json")
            with open(pickems_path, "w", encoding="utf-8") as handle:
                json.dump({"pickems": {"3-0": ["Alpha"], "advance": ["Bravo"], "0-3": ["Charlie"]}}, handle)
            write_matches_csv(
                results_path,
                [
                    {"team": "Alpha", "wins": 3, "losses": 0},
                    {"team": "Bravo", "wins": 3, "losses": 2},
                    {"team": "Charlie", "wins": 0, "losses": 3},
                ],
            )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "backtest-pickem",
                "--pickems",
                pickems_path,
                "--results",
                results_path,
                "--pass-threshold",
                "3",
                "--output",
                output_path,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv
            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["correct"], 3)
        self.assertTrue(report["passed"])

    def test_evaluate_pickem_backtest_suite_reports_historical_pass_rate(self):
        from cs2pickem.backtest import evaluate_pickem_backtest_suite

        cases = [
            {
                "name": "major-a",
                "pickems": {"3-0": ["Alpha"], "advance": ["Bravo"], "0-3": ["Charlie"]},
                "results": [
                    {"team": "Alpha", "wins": 3, "losses": 0},
                    {"team": "Bravo", "wins": 3, "losses": 1},
                    {"team": "Charlie", "wins": 0, "losses": 3},
                ],
            },
            {
                "name": "major-b",
                "pickems": {"3-0": ["Delta"], "advance": ["Echo"], "0-3": ["Foxtrot"]},
                "results": [
                    {"team": "Delta", "wins": 3, "losses": 1},
                    {"team": "Echo", "wins": 2, "losses": 3},
                    {"team": "Foxtrot", "wins": 1, "losses": 3},
                ],
            },
        ]

        report = evaluate_pickem_backtest_suite(cases, pass_threshold=2, pass_rate_target=0.38)

        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["passed_cases"], 1)
        self.assertEqual(report["pass_rate"], 0.5)
        self.assertTrue(report["meets_pass_rate_target"])
        self.assertEqual(report["case_reports"][0]["name"], "major-a")

    def test_backtest_suite_cli_reads_case_paths_and_writes_pass_rate_report(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = os.path.join(tmpdir, "suite.json")
            output_path = os.path.join(tmpdir, "suite-report.json")
            case_a_pickems = os.path.join(tmpdir, "case-a-pickems.json")
            case_a_results = os.path.join(tmpdir, "case-a-results.csv")
            case_b_pickems = os.path.join(tmpdir, "case-b-pickems.json")
            case_b_results = os.path.join(tmpdir, "case-b-results.csv")
            with open(case_a_pickems, "w", encoding="utf-8") as handle:
                json.dump({"pickems": {"3-0": ["Alpha"], "advance": ["Bravo"], "0-3": ["Charlie"]}}, handle)
            write_matches_csv(
                case_a_results,
                [
                    {"team": "Alpha", "wins": 3, "losses": 0},
                    {"team": "Bravo", "wins": 3, "losses": 1},
                    {"team": "Charlie", "wins": 0, "losses": 3},
                ],
            )
            with open(case_b_pickems, "w", encoding="utf-8") as handle:
                json.dump({"pickems": {"3-0": ["Delta"], "advance": ["Echo"], "0-3": ["Foxtrot"]}}, handle)
            write_matches_csv(
                case_b_results,
                [
                    {"team": "Delta", "wins": 3, "losses": 1},
                    {"team": "Echo", "wins": 2, "losses": 3},
                    {"team": "Foxtrot", "wins": 1, "losses": 3},
                ],
            )
            with open(suite_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cases": [
                            {"name": "major-a", "pickems_path": case_a_pickems, "results_path": case_a_results},
                            {"name": "major-b", "pickems_path": case_b_pickems, "results_path": case_b_results},
                        ]
                    },
                    handle,
                )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "backtest-pickem-suite",
                "--suite",
                suite_path,
                "--pass-threshold",
                "2",
                "--pass-rate-target",
                "0.38",
                "--output",
                output_path,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv
            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["passed_cases"], 1)
        self.assertEqual(report["pass_rate"], 0.5)
        self.assertTrue(report["meets_pass_rate_target"])
        self.assertEqual(report["case_reports"][0]["pickems_path"], case_a_pickems)


if __name__ == "__main__":
    unittest.main()
