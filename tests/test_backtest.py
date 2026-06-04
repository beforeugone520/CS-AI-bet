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

    def test_checkpoint_pickem_cli_reports_locked_alive_and_broken_slots(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            pickems_path = os.path.join(tmpdir, "pickems.json")
            standings_path = os.path.join(tmpdir, "standings.csv")
            output_path = os.path.join(tmpdir, "checkpoint.json")
            with open(pickems_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pickems": {
                            "3-0": ["Alpha", "Bravo"],
                            "advance": ["Charlie", "Delta", "Echo"],
                            "0-3": ["Foxtrot", "Golf"],
                        }
                    },
                    handle,
                )
            write_matches_csv(
                standings_path,
                [
                    {"team": "Alpha", "wins": 3, "losses": 0},
                    {"team": "Bravo", "wins": 2, "losses": 1},
                    {"team": "Charlie", "wins": 3, "losses": 1},
                    {"team": "Delta", "wins": 2, "losses": 1},
                    {"team": "Echo", "wins": 0, "losses": 3},
                    {"team": "Foxtrot", "wins": 0, "losses": 3},
                    {"team": "Golf", "wins": 0, "losses": 2},
                ],
            )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "checkpoint-pickem",
                "--pickems",
                pickems_path,
                "--standings",
                standings_path,
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
        self.assertEqual(report["summary"], {"locked": 3, "alive": 2, "broken": 2, "missing": 0})
        statuses = {(row["category"], row["team"]): row["status"] for row in report["picks"]}
        self.assertEqual(statuses[("3-0", "Alpha")], "locked")
        self.assertEqual(statuses[("3-0", "Bravo")], "broken")
        self.assertEqual(statuses[("advance", "Delta")], "alive")
        self.assertEqual(statuses[("0-3", "Foxtrot")], "locked")
        self.assertEqual(statuses[("0-3", "Golf")], "alive")

    def test_checkpoint_pickem_cli_keeps_pick_confidence_diagnostics(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            pickems_path = os.path.join(tmpdir, "pickems.json")
            standings_path = os.path.join(tmpdir, "standings.csv")
            output_path = os.path.join(tmpdir, "checkpoint.json")
            with open(pickems_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "picks": {
                            "3-0": [
                                {
                                    "team": "Alpha",
                                    "confidence": 0.8,
                                    "tier": "High",
                                    "market_win_prob_r1": 0.66,
                                    "model": {"3-0": 0.25, "advance": 0.7, "0-3": 0.03},
                                }
                            ],
                            "advance": [
                                {
                                    "team": "Bravo",
                                    "confidence": 0.4,
                                    "tier": "Low",
                                    "market_win_prob_r1": 0.51,
                                    "model": {"3-0": 0.1, "advance": 0.49, "0-3": 0.2},
                                }
                            ],
                            "0-3": [],
                        }
                    },
                    handle,
                )
            write_matches_csv(
                standings_path,
                [
                    {"team": "Alpha", "wins": 2, "losses": 1},
                    {"team": "Bravo", "wins": 3, "losses": 1},
                ],
            )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "checkpoint-pickem",
                "--pickems",
                pickems_path,
                "--standings",
                standings_path,
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
        picks = {(row["category"], row["team"]): row for row in report["picks"]}
        self.assertEqual(picks[("3-0", "Alpha")]["status"], "broken")
        self.assertEqual(picks[("3-0", "Alpha")]["confidence"], 0.8)
        self.assertEqual(picks[("3-0", "Alpha")]["tier"], "High")
        self.assertEqual(picks[("3-0", "Alpha")]["market_win_prob_r1"], 0.66)
        self.assertEqual(picks[("3-0", "Alpha")]["model"]["advance"], 0.7)
        self.assertEqual(report["status_diagnostics"]["broken"]["picks"], 1)
        self.assertEqual(report["status_diagnostics"]["broken"]["avg_confidence"], 0.8)
        self.assertEqual(report["status_diagnostics"]["locked"]["avg_confidence"], 0.4)

    def test_evaluate_pickem_checkpoint_reports_category_diagnostics(self):
        from cs2pickem.backtest import evaluate_pickem_checkpoint

        report = evaluate_pickem_checkpoint(
            {
                "3-0": ["Alpha", "Bravo"],
                "advance": ["Charlie"],
                "0-3": ["Delta"],
            },
            [
                {"team": "Alpha", "wins": 2, "losses": 1},
                {"team": "Bravo", "wins": 3, "losses": 0},
                {"team": "Charlie", "wins": 2, "losses": 1},
                {"team": "Delta", "wins": 0, "losses": 3},
            ],
            pick_details={
                ("3-0", "alpha"): {"confidence": 0.8, "tier": "High"},
                ("3-0", "bravo"): {"confidence": 0.4, "tier": "Low"},
                ("advance", "charlie"): {"confidence": 0.6, "tier": "Medium"},
                ("0-3", "delta"): {"confidence": 0.7, "tier": "High"},
            },
        )

        self.assertIn("category_diagnostics", report)
        categories = report["category_diagnostics"]
        self.assertEqual(categories["3-0"]["picks"], 2)
        self.assertEqual(categories["3-0"]["locked"], 1)
        self.assertEqual(categories["3-0"]["broken"], 1)
        self.assertAlmostEqual(categories["3-0"]["avg_confidence"], 0.6)
        self.assertAlmostEqual(categories["3-0"]["broken_avg_confidence"], 0.8)
        self.assertEqual(categories["3-0"]["high_tier_broken"], 1)
        self.assertEqual(categories["advance"]["alive"], 1)
        self.assertEqual(categories["0-3"]["locked"], 1)

    def test_standings_from_results_cli_derives_current_swiss_records(self):
        from cs2pickem.cli import main
        from cs2pickem.data import read_matches_csv, write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.csv")
            output_path = os.path.join(tmpdir, "standings.csv")
            write_matches_csv(
                results_path,
                [
                    {"team1": "Alpha", "team2": "Bravo", "winner": "Alpha"},
                    {"team1": "Alpha", "team2": "Charlie", "winner": "Alpha"},
                    {"team1": "Alpha", "team2": "Delta", "winner": "Alpha"},
                    {"team1": "Bravo", "team2": "Charlie", "winner": "Charlie"},
                    {"team1": "Bravo", "team2": "Echo", "winner": "Echo"},
                ],
            )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "standings-from-results",
                "--results",
                results_path,
                "--source",
                "unit-test",
                "--output",
                output_path,
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv
            with open(output_path, "rb") as handle:
                raw_output = handle.read()
            standings = read_matches_csv(output_path)

        self.assertEqual(exit_code, 0)
        self.assertNotIn(b"\r\n", raw_output)
        by_team = {row["team"]: row for row in standings}
        self.assertEqual(by_team["Alpha"]["wins"], "3")
        self.assertEqual(by_team["Alpha"]["losses"], "0")
        self.assertEqual(by_team["Alpha"]["status"], "advanced")
        self.assertEqual(by_team["Bravo"]["wins"], "0")
        self.assertEqual(by_team["Bravo"]["losses"], "3")
        self.assertEqual(by_team["Bravo"]["status"], "eliminated")
        self.assertEqual(by_team["Bravo"]["source"], "unit-test")
        self.assertEqual(by_team["Charlie"]["status"], "alive")

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

    def test_evaluate_forecast_result_reports_policy_threshold_candidates(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {"date": "2026-06-02", "team1": "A", "team2": "B", "pick": "A", "adjusted_probability_team1": 0.54},
                {"date": "2026-06-02", "team1": "C", "team2": "D", "pick": "C", "adjusted_probability_team1": 0.56},
                {"date": "2026-06-02", "team1": "E", "team2": "F", "pick": "E", "adjusted_probability_team1": 0.61},
                {"date": "2026-06-02", "team1": "G", "team2": "H", "pick": "G", "adjusted_probability_team1": 0.67},
            ],
            [
                {"date": "2026-06-02", "team1": "A", "team2": "B", "winner": "B"},
                {"date": "2026-06-02", "team1": "C", "team2": "D", "winner": "C"},
                {"date": "2026-06-02", "team1": "E", "team2": "F", "winner": "E"},
                {"date": "2026-06-02", "team1": "G", "team2": "H", "winner": "H"},
            ],
        )

        diagnostics = report["policy_diagnostics"]
        self.assertEqual(diagnostics["current_policy"]["actionable_picks"], 4)
        self.assertEqual(diagnostics["current_policy"]["correct_actionable"], 2)
        threshold_rows = {row["minimum_margin"]: row for row in diagnostics["threshold_candidates"]}
        self.assertEqual(threshold_rows[0.05]["actionable_picks"], 3)
        self.assertEqual(threshold_rows[0.05]["correct_actionable"], 2)
        self.assertAlmostEqual(threshold_rows[0.05]["actionable_accuracy"], 2 / 3)
        self.assertEqual(threshold_rows[0.05]["avoided_losses"], 1)
        self.assertEqual(diagnostics["recommended_minimum_margin"], 0.05)
        self.assertEqual(diagnostics["recommendation_basis"], "highest_accuracy_with_minimum_two_picks")

    def test_evaluate_forecast_result_reports_player_form_counter_signal_risk(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "Alpha",
                    "adjusted_probability_team1": 0.58,
                    "player_form_summary": {"diff": {"score": -0.06, "trend": -0.02, "sample_confidence": -0.2}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "Charlie",
                    "adjusted_probability_team1": 0.62,
                    "player_form_summary": {"diff": {"score": 0.04, "trend": 0.01, "sample_confidence": 0.2}},
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Charlie"},
            ],
        )

        risk = report["policy_diagnostics"]["player_form_counter_signal"]
        self.assertEqual(risk["available_matches"], 2)
        self.assertEqual(risk["counter_signal_matches"], 1)
        self.assertEqual(risk["counter_signal_losses"], 1)
        self.assertAlmostEqual(risk["counter_signal_loss_rate"], 1.0)
        self.assertEqual(risk["aligned_matches"], 1)
        self.assertEqual(risk["aligned_losses"], 0)

    def test_evaluate_forecast_result_reports_player_form_policy_candidates(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "Alpha",
                    "adjusted_probability_team1": 0.62,
                    "player_form_summary": {
                        "team1": {"sample_confidence": 0.7},
                        "team2": {"sample_confidence": 0.6},
                        "diff": {"score": -0.05},
                    },
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "Charlie",
                    "adjusted_probability_team1": 0.61,
                    "player_form_summary": {
                        "team1": {"sample_confidence": 0.2},
                        "team2": {"sample_confidence": 0.2},
                        "diff": {"score": -0.04},
                    },
                },
                {
                    "date": "2026-06-02",
                    "team1": "Echo",
                    "team2": "Foxtrot",
                    "pick": "Echo",
                    "adjusted_probability_team1": 0.59,
                    "player_form_summary": {
                        "team1": {"sample_confidence": 0.8},
                        "team2": {"sample_confidence": 0.7},
                        "diff": {"score": 0.06},
                    },
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Charlie"},
                {"date": "2026-06-02", "team1": "Echo", "team2": "Foxtrot", "winner": "Echo"},
            ],
        )

        candidates = {
            row["player_form_counter_min_confidence"]: row
            for row in report["policy_diagnostics"]["player_form_policy_candidates"]
        }
        self.assertEqual(candidates[0.4]["avoided_losses"], 1)
        self.assertEqual(candidates[0.4]["avoided_wins"], 0)
        self.assertEqual(candidates[0.4]["actionable_picks"], 2)
        self.assertEqual(candidates[0.4]["correct_actionable"], 2)
        self.assertAlmostEqual(candidates[0.4]["actionable_accuracy"], 1.0)
        self.assertEqual(candidates[0.8]["counter_signal_matches"], 0)
        self.assertEqual(candidates[0.8]["actionable_picks"], 3)

    def test_evaluate_forecast_result_reports_favorite_upset_diagnostics(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "Alpha",
                    "adjusted_probability_team1": 0.66,
                    "model_probability_team1": 0.62,
                    "market_signal": {"probability_team1": 0.64, "basis": "real_odds"},
                    "player_form_summary": {
                        "team1": {"sample_confidence": 0.7},
                        "team2": {"sample_confidence": 0.6},
                        "diff": {"score": -0.08},
                    },
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "Charlie",
                    "adjusted_probability_team1": 0.60,
                    "model_probability_team1": 0.48,
                    "market_signal": {"probability_team1": 0.65, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": 0.04}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Echo",
                    "team2": "Foxtrot",
                    "pick": "Echo",
                    "adjusted_probability_team1": 0.54,
                    "model_probability_team1": 0.54,
                    "market_signal": {"probability_team1": 0.52, "basis": "real_odds"},
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Charlie"},
                {"date": "2026-06-02", "team1": "Echo", "team2": "Foxtrot", "winner": "Foxtrot"},
            ],
        )

        self.assertIn("favorite_upset_diagnostics", report)
        diagnostics = report["favorite_upset_diagnostics"]
        self.assertEqual(diagnostics["minimum_favorite_probability"], 0.55)
        self.assertEqual(diagnostics["adjusted_favorites"], 2)
        self.assertEqual(diagnostics["adjusted_favorite_losses"], 1)
        self.assertAlmostEqual(diagnostics["adjusted_favorite_loss_rate"], 0.5)
        self.assertEqual(diagnostics["model_favorites"], 1)
        self.assertEqual(diagnostics["model_favorite_losses"], 1)
        self.assertEqual(diagnostics["market_favorites"], 2)
        self.assertEqual(diagnostics["market_favorite_losses"], 1)
        self.assertEqual(diagnostics["model_market_agree_favorites"], 1)
        self.assertEqual(diagnostics["model_market_agree_favorite_losses"], 1)
        self.assertEqual(diagnostics["favorite_losses_with_player_form_counter_signal"], 1)
        self.assertEqual(diagnostics["favorite_loss_examples"][0]["favorite"], "Alpha")
        self.assertEqual(diagnostics["favorite_loss_examples"][0]["winner"], "Bravo")
        self.assertAlmostEqual(diagnostics["favorite_loss_examples"][0]["player_form_directional_score"], -0.08)
        self.assertIn("market_favorite_loss_examples", diagnostics)
        self.assertEqual(diagnostics["market_favorite_loss_examples"][0]["favorite"], "Alpha")
        self.assertEqual(diagnostics["market_favorite_loss_examples"][0]["winner"], "Bravo")

    def test_evaluate_forecast_result_reports_market_favorite_player_form_policy_candidates(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "Alpha",
                    "adjusted_probability_team1": 0.66,
                    "market_signal": {"probability_team1": 0.66, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": -0.08}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "Charlie",
                    "adjusted_probability_team1": 0.62,
                    "market_signal": {"probability_team1": 0.62, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": -0.04}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Echo",
                    "team2": "Foxtrot",
                    "pick": "Echo",
                    "adjusted_probability_team1": 0.64,
                    "market_signal": {"probability_team1": 0.64, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": 0.05}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Golf",
                    "team2": "Hotel",
                    "pick": "Golf",
                    "adjusted_probability_team1": 0.52,
                    "market_signal": {"probability_team1": 0.52, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": -0.03}},
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Charlie"},
                {"date": "2026-06-02", "team1": "Echo", "team2": "Foxtrot", "winner": "Echo"},
                {"date": "2026-06-02", "team1": "Golf", "team2": "Hotel", "winner": "Hotel"},
            ],
        )

        self.assertIn(
            "market_favorite_player_form_policy_candidates",
            report["policy_diagnostics"],
        )
        candidates = {
            row["market_favorite_min_probability"]: row
            for row in report["policy_diagnostics"]["market_favorite_player_form_policy_candidates"]
        }
        self.assertEqual(candidates[0.55]["counter_signal_matches"], 2)
        self.assertEqual(candidates[0.55]["avoided_losses"], 1)
        self.assertEqual(candidates[0.55]["avoided_wins"], 1)
        self.assertEqual(candidates[0.55]["actionable_picks"], 2)
        self.assertEqual(candidates[0.55]["correct_actionable"], 1)
        self.assertAlmostEqual(candidates[0.55]["actionable_accuracy"], 0.5)
        self.assertEqual(candidates[0.65]["counter_signal_matches"], 1)
        self.assertEqual(candidates[0.65]["avoided_losses"], 1)
        self.assertEqual(candidates[0.65]["avoided_wins"], 0)
        self.assertAlmostEqual(candidates[0.65]["actionable_accuracy"], 2 / 3)

    def test_evaluate_forecast_result_summarizes_policy_tradeoffs(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "Alpha",
                    "adjusted_probability_team1": 0.66,
                    "market_signal": {"probability_team1": 0.60, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": -0.08}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "Charlie",
                    "adjusted_probability_team1": 0.62,
                    "market_signal": {"probability_team1": 0.60, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": -0.04}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Echo",
                    "team2": "Foxtrot",
                    "pick": "Echo",
                    "adjusted_probability_team1": 0.64,
                    "market_signal": {"probability_team1": 0.64, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": 0.05}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Golf",
                    "team2": "Hotel",
                    "pick": "Golf",
                    "adjusted_probability_team1": 0.63,
                    "market_signal": {"probability_team1": 0.63, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": 0.06}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "India",
                    "team2": "Juliet",
                    "pick": "India",
                    "adjusted_probability_team1": 0.63,
                    "market_signal": {"probability_team1": 0.52, "basis": "real_odds"},
                    "player_form_summary": {"diff": {"score": -0.02}},
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Charlie"},
                {"date": "2026-06-02", "team1": "Echo", "team2": "Foxtrot", "winner": "Echo"},
                {"date": "2026-06-02", "team1": "Golf", "team2": "Hotel", "winner": "Golf"},
                {"date": "2026-06-02", "team1": "India", "team2": "Juliet", "winner": "India"},
            ],
        )

        summary = report["policy_diagnostics"]["policy_tradeoff_summary"]
        self.assertEqual(summary["current_policy"]["correct_actionable"], 4)
        self.assertAlmostEqual(summary["current_policy"]["coverage"], 1.0)
        self.assertEqual(
            summary["highest_accuracy_candidate"]["source"],
            "market_favorite_player_form_policy_candidates",
        )
        self.assertEqual(summary["highest_accuracy_candidate"]["parameter"], {"market_favorite_min_probability": 0.55})
        self.assertEqual(summary["highest_accuracy_candidate"]["correct_actionable"], 3)
        self.assertAlmostEqual(summary["highest_accuracy_candidate"]["actionable_accuracy"], 1.0)
        self.assertEqual(summary["highest_correct_candidate"]["correct_actionable"], 4)
        self.assertEqual(summary["correct_pick_delta_vs_current"], -1)
        self.assertAlmostEqual(summary["accuracy_gain_over_current"], 0.2)
        self.assertAlmostEqual(summary["coverage_delta_vs_current"], -0.4)
        self.assertEqual(summary["recommendation"], "keep_current_policy")
        self.assertEqual(summary["recommendation_basis"], "accuracy_gain_reduces_total_correct")

    def test_evaluate_forecast_result_reports_player_status_policy_candidates(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "Alpha",
                    "adjusted_probability_team1": 0.57,
                    "player_form_summary": {
                        "team1": {"sample_confidence": 0.2, "substitute_flag": 0},
                        "team2": {"sample_confidence": 0.9, "substitute_flag": 0},
                    },
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "Charlie",
                    "adjusted_probability_team1": 0.61,
                    "player_form_summary": {
                        "team1": {"sample_confidence": 0.2, "substitute_flag": 0},
                        "team2": {"sample_confidence": 0.9, "substitute_flag": 0},
                    },
                },
                {
                    "date": "2026-06-02",
                    "team1": "Echo",
                    "team2": "Foxtrot",
                    "pick": "Foxtrot",
                    "adjusted_probability_team1": 0.43,
                    "player_form_summary": {
                        "team1": {"sample_confidence": 0.9, "substitute_flag": 0},
                        "team2": {"sample_confidence": 0.9, "substitute_flag": 1},
                    },
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Charlie"},
                {"date": "2026-06-02", "team1": "Echo", "team2": "Foxtrot", "winner": "Foxtrot"},
            ],
        )

        self.assertEqual(report["matches"][0]["picked_player_sample_confidence"], 0.2)
        self.assertEqual(report["matches"][2]["picked_substitute_flag"], 1)
        candidates = {
            (row["player_status_min_confidence"], row["player_status_min_margin"]): row
            for row in report["policy_diagnostics"]["player_status_policy_candidates"]
        }
        candidate = candidates[(0.4, 0.08)]
        self.assertEqual(candidate["status_risk_matches"], 2)
        self.assertEqual(candidate["actionable_picks"], 1)
        self.assertEqual(candidate["correct_actionable"], 1)
        self.assertEqual(candidate["avoided_losses"], 1)
        self.assertEqual(candidate["avoided_wins"], 1)
        self.assertEqual(candidate["substitute_risk_matches"], 1)
        self.assertEqual(candidate["low_sample_risk_matches"], 1)

    def test_evaluate_forecast_result_reports_avoid_reason_diagnostics(self):
        from cs2pickem.backtest import evaluate_forecast_result

        report = evaluate_forecast_result(
            [
                {
                    "date": "2026-06-02",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "pick": "avoid",
                    "avoid_reason": "low_confidence",
                    "adjusted_probability_team1": 0.51,
                },
                {
                    "date": "2026-06-02",
                    "team1": "Charlie",
                    "team2": "Delta",
                    "pick": "avoid",
                    "avoid_reason": "market_favorite_player_form_counter_signal",
                    "adjusted_probability_team1": 0.62,
                    "player_form_summary": {"diff": {"score": -0.04}},
                },
                {
                    "date": "2026-06-02",
                    "team1": "Echo",
                    "team2": "Foxtrot",
                    "pick": "Echo",
                    "adjusted_probability_team1": 0.61,
                },
            ],
            [
                {"date": "2026-06-02", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha"},
                {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "winner": "Delta"},
                {"date": "2026-06-02", "team1": "Echo", "team2": "Foxtrot", "winner": "Echo"},
            ],
        )

        self.assertIn("avoid_reason_diagnostics", report)
        self.assertEqual(report["matches"][0]["avoid_reason"], "low_confidence")
        diagnostics = report["avoid_reason_diagnostics"]
        self.assertEqual(diagnostics["low_confidence"]["avoid_picks"], 1)
        self.assertEqual(diagnostics["low_confidence"]["avoided_wins"], 1)
        self.assertEqual(diagnostics["low_confidence"]["avoided_losses"], 0)
        self.assertEqual(diagnostics["market_favorite_player_form_counter_signal"]["avoid_picks"], 1)
        self.assertEqual(diagnostics["market_favorite_player_form_counter_signal"]["avoided_wins"], 0)
        self.assertEqual(diagnostics["market_favorite_player_form_counter_signal"]["avoided_losses"], 1)

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
