import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tests.test_forecast import history_rows
from tests.test_pickem import profiles, team_rows


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

    def test_evaluate_forecast_result_reports_accuracy_and_player_form_diagnostics(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "Alpha",
                    "adjusted_probability_team1": 0.61,
                    "confidence_margin": 0.11,
                    "low_confidence": False,
                    "player_form_summary": {"diff": {"score": 0.08, "trend": 0.03, "sample_confidence": 0.6}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "Charlie",
                    "adjusted_probability_team1": 0.57,
                    "confidence_margin": 0.07,
                    "low_confidence": False,
                    "player_form_summary": {"diff": {"score": 0.12, "trend": -0.05, "sample_confidence": -0.4}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Echo",
                    "team2": "Foxtrot",
                    "pick": "avoid",
                    "adjusted_probability_team1": 0.49,
                    "confidence_margin": 0.01,
                    "low_confidence": True,
                    "player_form_summary": {"diff": {"score": -0.02, "trend": 0.0, "sample_confidence": 0.0}},
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "score": "13-8", "map": "Mirage"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Delta", "score": "16-14", "map": "Ancient", "note": "OT"},
                {"date": "2026-06-02", "team1": "Echo", "team2": "Foxtrot", "winner": "Foxtrot"},
            ],
        )

        self.assertEqual(report["matched"], 3)
        self.assertEqual(report["actionable_picks"], 2)
        self.assertEqual(report["correct_actionable"], 1)
        self.assertEqual(report["avoid_picks"], 1)
        self.assertEqual(report["avoid_directional_correct"], 1)
        self.assertEqual(report["model_upsets"], 1)
        self.assertAlmostEqual(report["actionable_accuracy"], 0.5)
        self.assertAlmostEqual(report["directional_accuracy"], 2 / 3)
        self.assertAlmostEqual(report["player_form_diagnostics"]["correct_actionable_avg_score_diff"], 0.08)
        self.assertAlmostEqual(report["player_form_diagnostics"]["missed_actionable_avg_score_diff"], 0.12)
        self.assertEqual(report["matches"][0]["score"], "13-8")
        self.assertEqual(report["matches"][0]["map"], "Mirage")
        self.assertEqual(report["matches"][1]["result_note"], "OT")
        self.assertEqual(report["matches"][1]["correct"], False)
        self.assertEqual(report["matches"][1]["directional_pick"], "Charlie")

    def test_backtest_forecast_cli_reads_report_json_and_results_csv(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            forecast_path = os.path.join(tmpdir, "forecast.json")
            results_path = os.path.join(tmpdir, "results.csv")
            output_path = os.path.join(tmpdir, "forecast-backtest.json")
            with open(forecast_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "predictions": [
                            {
                                "date": "2026-06-02",
                                "team1": "Alpha",
                                "team2": "Bravo",
                                "pick": "Alpha",
                                "adjusted_probability_team1": 0.61,
                                "player_form_summary": {"diff": {"score": 0.08}},
                            }
                        ]
                    },
                    handle,
                )
            write_matches_csv(
                results_path,
                [{"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha"}],
            )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "backtest-forecast",
                "--forecast-report",
                forecast_path,
                "--results",
                results_path,
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
        self.assertEqual(report["matched"], 1)
        self.assertEqual(report["correct_actionable"], 1)
        self.assertEqual(report["forecast_report_path"], forecast_path)
        self.assertEqual(report["results_path"], results_path)

    def test_backtest_cli_reads_final_fused_pickem_json(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            pickems_path = os.path.join(tmpdir, "final-fused-pickems.json")
            results_path = os.path.join(tmpdir, "standings.csv")
            output_path = os.path.join(tmpdir, "backtest.json")
            with open(pickems_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "weights": {"experts": 0.3, "market": 0.2, "model": 0.5},
                        "picks": {
                            "3-0": [{"team": "Alpha", "confidence": 0.81}],
                            "advance": [{"team": "Bravo", "confidence": 0.74}],
                            "0-3": [{"team": "Charlie", "confidence": 0.69}],
                        },
                    },
                    handle,
                )
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
        self.assertEqual(report["total_picks"], 3)
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

    def test_replay_pickem_backtest_suite_trains_generates_and_scores_pickems(self):
        from cs2pickem.backtest import replay_pickem_backtest_suite

        cases = [
            {
                "name": "replay-major-a",
                "history": history_rows(),
                "teams": team_rows(),
                "profiles": profiles(),
                "results": [
                    {"team": "Alpha", "wins": 3, "losses": 0},
                    {"team": "Bravo", "wins": 0, "losses": 3},
                    {"team": "Charlie", "wins": 1, "losses": 3},
                    {"team": "Delta", "wins": 3, "losses": 1},
                ],
                "reference_date": "2026-05-31",
                "simulations": 20,
                "seed": 5,
                "top_k": 6,
                "epochs": 3,
                "slots": {"3-0": 1, "advance": 2, "0-3": 1},
            }
        ]

        report = replay_pickem_backtest_suite(cases, pass_threshold=0, pass_rate_target=1.0)

        self.assertEqual(report["cases"], 1)
        self.assertEqual(report["passed_cases"], 1)
        self.assertTrue(report["meets_pass_rate_target"])
        case_report = report["case_reports"][0]
        self.assertEqual(case_report["name"], "replay-major-a")
        self.assertIn("generated_pickems", case_report)
        self.assertEqual(case_report["generated_summary"]["trained_matches"], 8)
        self.assertEqual(case_report["generated_summary"]["simulations"], 20)
        self.assertIn("score_report", case_report)

    def test_replay_pickem_suite_cli_reads_paths_and_writes_report(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = os.path.join(tmpdir, "replay-suite.json")
            history_path = os.path.join(tmpdir, "history.csv")
            teams_path = os.path.join(tmpdir, "teams.csv")
            results_path = os.path.join(tmpdir, "results.csv")
            profiles_path = os.path.join(tmpdir, "profiles.json")
            output_path = os.path.join(tmpdir, "replay-report.json")
            write_matches_csv(history_path, history_rows())
            write_matches_csv(teams_path, team_rows())
            write_matches_csv(
                results_path,
                [
                    {"team": "Alpha", "wins": 3, "losses": 0},
                    {"team": "Bravo", "wins": 0, "losses": 3},
                    {"team": "Charlie", "wins": 1, "losses": 3},
                    {"team": "Delta", "wins": 3, "losses": 1},
                ],
            )
            with open(profiles_path, "w", encoding="utf-8") as handle:
                json.dump(profiles(), handle)
            with open(suite_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cases": [
                            {
                                "name": "replay-major-a",
                                "history_path": history_path,
                                "teams_path": teams_path,
                                "profiles_path": profiles_path,
                                "results_path": results_path,
                                "reference_date": "2026-05-31",
                                "slots": {"3-0": 1, "advance": 2, "0-3": 1},
                            }
                        ]
                    },
                    handle,
                )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "replay-pickem-suite",
                "--suite",
                suite_path,
                "--pass-threshold",
                "0",
                "--pass-rate-target",
                "1.0",
                "--simulations",
                "20",
                "--top-k",
                "6",
                "--epochs",
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
        self.assertEqual(report["cases"], 1)
        self.assertEqual(report["case_reports"][0]["history_path"], history_path)


if __name__ == "__main__":
    unittest.main()
