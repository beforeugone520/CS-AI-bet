import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class WorkflowTests(unittest.TestCase):
    def test_run_end_to_end_pipeline_writes_core_artifacts(self):
        from cs2pickem.workflow import run_end_to_end_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=os.path.join(ROOT, "examples", "upcoming_fixtures.csv"),
                teams_path=os.path.join(ROOT, "examples", "sample_teams.csv"),
                reference_date="2026-05-31",
                output_dir=tmpdir,
                odds_path=os.path.join(ROOT, "examples", "odds_feed.csv"),
                players_path=os.path.join(ROOT, "examples", "player_stats.csv"),
                bp_path=os.path.join(ROOT, "examples", "bp_intel.csv"),
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                simulations=25,
                top_k=10,
                epochs=4,
            )

            self.assertTrue(os.path.exists(report["artifacts"]["enriched_matches"]))
            self.assertTrue(os.path.exists(report["artifacts"]["train_report"]))
            self.assertTrue(os.path.exists(report["artifacts"]["forecast_report"]))
            self.assertTrue(os.path.exists(report["artifacts"]["pickem_report"]))
            self.assertTrue(os.path.exists(report["artifacts"]["pickem_answer_sheet"]))
            self.assertTrue(os.path.exists(report["artifacts"]["fixtures_with_bp"]))
            self.assertEqual(report["bp"]["map_overrides"], 1)
            self.assertEqual(report["forecast"]["predictions"][0]["map"], "inferno")
            self.assertIn("ready", report["readiness"])
            self.assertEqual(report["forecast"]["fixtures"], 1)
            self.assertIn("pickems", report["pickem"])
            self.assertIn("picks", report["answer_sheet"])
            self.assertEqual(report["answer_sheet"]["pickems"], report["pickem"]["pickems"])
            self.assertIn("participant_coverage", report["readiness"]["checks"])
            self.assertIn("forecast_low_confidence_avoidance", report["readiness"]["checks"])
            self.assertTrue(report["readiness"]["checks"]["forecast_low_confidence_avoidance"]["passed"])
            self.assertIn("validation_tuned_weights", report["readiness"]["checks"])
            self.assertTrue(report["readiness"]["checks"]["validation_tuned_weights"]["passed"])
            tuned_weights = report["train"]["validation_tuned_ensemble_weights"]["weights"]
            for name, weight in tuned_weights.items():
                self.assertAlmostEqual(report["forecast"]["ensemble_weights"][name], weight)
                self.assertAlmostEqual(report["pickem"]["ensemble_weights"][name], weight)

    def test_pipeline_augments_training_history_before_readiness(self):
        from cs2pickem.data import read_matches_csv, write_matches_csv
        from cs2pickem.workflow import run_end_to_end_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            players_path = os.path.join(tmpdir, "history_players.csv")
            write_matches_csv(
                players_path,
                [
                    {"date": "2026-04-20", "team": "Alpha", "player": "alpha_awp", "rating": 1.32, "kd": 1.25, "opening_success": 0.61, "clutch_winrate": 0.64, "is_substitute": 0},
                    {"date": "2026-04-21", "team": "Alpha", "player": "alpha_rifle", "rating": 1.08, "kd": 1.04, "opening_success": 0.52, "clutch_winrate": 0.57, "is_substitute": 0},
                    {"date": "2026-04-20", "team": "Bravo", "player": "bravo_sub", "rating": 0.9, "kd": 0.88, "opening_success": 0.44, "clutch_winrate": 0.46, "is_substitute": 1},
                    {"date": "2026-04-22", "team": "Charlie", "player": "charlie_star", "rating": 1.12, "kd": 1.08, "opening_success": 0.55, "clutch_winrate": 0.58, "is_substitute": 0},
                ],
            )

            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=os.path.join(ROOT, "examples", "upcoming_fixtures.csv"),
                teams_path=os.path.join(ROOT, "examples", "sample_teams.csv"),
                reference_date="2026-05-31",
                output_dir=tmpdir,
                odds_path=os.path.join(ROOT, "examples", "odds_feed.csv"),
                players_path=players_path,
                bp_path=os.path.join(ROOT, "examples", "bp_intel.csv"),
                version_log_path=os.path.join(ROOT, "examples", "version_log.csv"),
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                minimum_rows=1,
                required_teams=3,
                simulations=25,
                top_k=10,
                epochs=4,
            )

            enriched = read_matches_csv(report["artifacts"]["enriched_matches"])

        self.assertEqual(enriched[0]["version_tag"], "cologne-practice-patch")
        self.assertEqual(enriched[0]["team1_major_best_placement"], 1)
        self.assertEqual(enriched[0]["team2_major_best_placement"], 12)
        self.assertEqual(enriched[0]["team1_player_sample"], 2)
        self.assertEqual(enriched[0]["team2_substitute_flag"], 1)
        self.assertEqual(enriched[0]["swiss_round"], 1)
        self.assertEqual(enriched[0]["team1_wins"], 0)
        self.assertTrue(report["readiness"]["checks"]["required_fields"]["passed"])

    def test_pipeline_augments_sparse_fixtures_with_team_metadata(self):
        from cs2pickem.data import read_matches_csv, write_matches_csv
        from cs2pickem.workflow import run_end_to_end_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_path = os.path.join(tmpdir, "sparse_fixtures.csv")
            write_matches_csv(
                fixtures_path,
                [
                    {
                        "date": "2026-06-02",
                        "event": "IEM Cologne Major",
                        "event_tier": "S",
                        "status": "scheduled",
                        "team1": "Alpha",
                        "team2": "Bravo",
                        "best_of": 1,
                        "map": "unknown",
                    }
                ],
            )

            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=fixtures_path,
                teams_path=os.path.join(ROOT, "examples", "sample_teams.csv"),
                reference_date="2026-05-31",
                output_dir=tmpdir,
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                simulations=25,
                top_k=10,
                epochs=4,
            )

            fixtures_ready = read_matches_csv(report["artifacts"]["fixtures_ready"])

        self.assertEqual(fixtures_ready[0]["team1_rank"], 4)
        self.assertEqual(fixtures_ready[0]["team2_rank"], 20)
        self.assertEqual(fixtures_ready[0]["team1_major_best_placement"], 1)
        self.assertEqual(fixtures_ready[0]["team2_major_best_placement"], 12)
        self.assertEqual(fixtures_ready[0]["swiss_round"], 1)
        self.assertEqual(fixtures_ready[0]["team1_wins"], 0)
        self.assertEqual(report["fixtures_augmentation"]["team_metadata"]["teams"], 4)

    def test_pipeline_passes_fixture_level_odds_into_pickem_simulation(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.workflow import run_end_to_end_pipeline
        from tests.test_pickem import team_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            teams_path = os.path.join(tmpdir, "teams.csv")
            write_matches_csv(teams_path, team_rows())
            write_matches_csv(
                fixtures_path,
                [
                    {
                        "date": "2026-06-01",
                        "event": "IEM Cologne Major",
                        "event_tier": "S",
                        "status": "scheduled",
                        "team1": "Alpha",
                        "team2": "Bravo",
                        "best_of": 1,
                        "map": "unknown",
                        "odds_team1": 10.0,
                        "odds_team2": 1.1,
                    }
                ],
            )

            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=fixtures_path,
                teams_path=teams_path,
                reference_date="2026-05-31",
                output_dir=tmpdir,
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                minimum_rows=1,
                required_teams=3,
                simulations=25,
                top_k=10,
                epochs=4,
            )

        details = report["pickem"]["sample_match_details"]["Alpha__Bravo__bo1"]
        self.assertTrue(details["market_adjustment_applied"])

    def test_pipeline_gates_pickem_market_adjustment_when_odds_are_supplied(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.workflow import run_end_to_end_pipeline
        from tests.test_pickem import team_rows

        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            teams_path = os.path.join(tmpdir, "teams.csv")
            odds_path = os.path.join(tmpdir, "unmatched_odds.csv")
            write_matches_csv(teams_path, team_rows())
            write_matches_csv(
                fixtures_path,
                [{"date": "2026-06-01", "event": "IEM Cologne Major", "event_tier": "S", "status": "scheduled", "team1": "Alpha", "team2": "Bravo", "best_of": 1, "map": "unknown"}],
            )
            write_matches_csv(
                odds_path,
                [{"date": "2026-06-01", "provider": "BookA", "team1": "Ghost", "team2": "Phantom", "odds_team1": 1.5, "odds_team2": 2.5}],
            )

            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=fixtures_path,
                teams_path=teams_path,
                reference_date="2026-05-31",
                output_dir=tmpdir,
                odds_path=odds_path,
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                minimum_rows=1,
                required_teams=3,
                simulations=25,
                top_k=10,
                epochs=4,
            )

        self.assertFalse(report["readiness"]["checks"]["pickem_market_adjustment"]["passed"])
        self.assertEqual(report["readiness"]["checks"]["pickem_market_adjustment"]["target"], ">= 1")
        self.assertIn("pickem_market_adjustment", report["readiness"]["failed_checks"])

    def test_pipeline_can_gate_readiness_on_historical_pickem_backtest(self):
        from cs2pickem.workflow import run_end_to_end_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            backtest_path = os.path.join(tmpdir, "pickem_backtest_suite.json")
            with open(backtest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cases": 3,
                        "passed_cases": 1,
                        "pass_rate": 1 / 3,
                        "pass_rate_target": 0.38,
                        "meets_pass_rate_target": False,
                    },
                    handle,
                )

            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=os.path.join(ROOT, "examples", "upcoming_fixtures.csv"),
                teams_path=os.path.join(ROOT, "examples", "sample_teams.csv"),
                reference_date="2026-05-31",
                output_dir=tmpdir,
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                pickem_backtest_report_path=backtest_path,
                pickem_pass_rate_target=0.38,
                minimum_rows=1,
                required_teams=3,
                simulations=25,
                top_k=10,
                epochs=4,
            )

        self.assertFalse(report["readiness"]["ready"])
        self.assertFalse(report["readiness"]["checks"]["pickem_backtest_pass_rate"]["passed"])
        self.assertIn("pickem_backtest_pass_rate", report["readiness"]["failed_checks"])

    def test_pipeline_gates_readiness_on_required_pickem_simulation_count(self):
        from cs2pickem.workflow import run_end_to_end_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=os.path.join(ROOT, "examples", "upcoming_fixtures.csv"),
                teams_path=os.path.join(ROOT, "examples", "sample_teams.csv"),
                reference_date="2026-05-31",
                output_dir=tmpdir,
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                minimum_rows=1,
                required_teams=3,
                simulations=25,
                minimum_pickem_simulations=100000,
                top_k=10,
                epochs=4,
            )

        self.assertEqual(report["pickem"]["simulations"], 25)
        self.assertFalse(report["readiness"]["checks"]["pickem_simulations"]["passed"])
        self.assertEqual(report["readiness"]["checks"]["pickem_simulations"]["target"], ">= 100000")
        self.assertIn("pickem_simulations", report["readiness"]["failed_checks"])

    def test_pipeline_can_gate_readiness_on_pickem_selection_margin(self):
        from cs2pickem.workflow import run_end_to_end_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_end_to_end_pipeline(
                history_path=os.path.join(ROOT, "examples", "raw_match_history.csv"),
                fixtures_path=os.path.join(ROOT, "examples", "upcoming_fixtures.csv"),
                teams_path=os.path.join(ROOT, "examples", "sample_teams.csv"),
                reference_date="2026-05-31",
                output_dir=tmpdir,
                participants_path=os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                top_teams_path=os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                minimum_rows=1,
                required_teams=3,
                simulations=25,
                minimum_pickem_selection_margin=0.99,
                top_k=10,
                epochs=4,
            )

        self.assertFalse(report["readiness"]["checks"]["pickem_selection_margin"]["passed"])
        self.assertEqual(report["readiness"]["checks"]["pickem_selection_margin"]["target"], ">= 0.99")
        self.assertIn("pickem_selection_margin", report["readiness"]["failed_checks"])

    def test_cli_pipeline_runs_the_offline_workflow(self):
        from cs2pickem.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            source_manifest_path = os.path.join(tmpdir, "source_manifest.json")
            with open(source_manifest_path, "w", encoding="utf-8") as handle:
                json.dump({"source": "hltv-results", "updated_at": "2026-05-31T22:00:00+00:00"}, handle)
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "pipeline",
                "--history",
                os.path.join(ROOT, "examples", "raw_match_history.csv"),
                "--fixtures",
                os.path.join(ROOT, "examples", "upcoming_fixtures.csv"),
                "--teams",
                os.path.join(ROOT, "examples", "sample_teams.csv"),
                "--odds",
                os.path.join(ROOT, "examples", "odds_feed.csv"),
                "--players",
                os.path.join(ROOT, "examples", "player_stats.csv"),
                "--bp",
                os.path.join(ROOT, "examples", "bp_intel.csv"),
                "--participants",
                os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                "--top-teams",
                os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                "--reference-date",
                "2026-05-31",
                "--output-dir",
                tmpdir,
                "--simulations",
                "25",
                "--source-manifest",
                source_manifest_path,
                "--required-source",
                "hltv-results",
                "--source-reference-time",
                "2026-06-01T00:00:00+00:00",
                "--maximum-source-age-hours",
                "24",
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            manifest_path = os.path.join(tmpdir, "pipeline_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["bp"]["map_overrides"], 1)
            self.assertEqual(manifest["readiness"]["checks"]["pickem_simulations"]["actual"], 25)
            self.assertEqual(manifest["readiness"]["checks"]["pickem_simulations"]["target"], ">= 100000")
            self.assertTrue(manifest["readiness"]["checks"]["source_freshness"]["passed"])

        self.assertEqual(exit_code, 0)

    def test_cli_pipeline_accepts_pickem_backtest_readiness_gate(self):
        from cs2pickem.cli import main

        with tempfile.TemporaryDirectory() as tmpdir:
            backtest_path = os.path.join(tmpdir, "pickem_backtest_suite.json")
            with open(backtest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cases": 3,
                        "passed_cases": 1,
                        "pass_rate": 1 / 3,
                        "pass_rate_target": 0.38,
                        "meets_pass_rate_target": False,
                    },
                    handle,
                )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "pipeline",
                "--history",
                os.path.join(ROOT, "examples", "raw_match_history.csv"),
                "--fixtures",
                os.path.join(ROOT, "examples", "upcoming_fixtures.csv"),
                "--teams",
                os.path.join(ROOT, "examples", "sample_teams.csv"),
                "--participants",
                os.path.join(ROOT, "examples", "major_participants_sample.csv"),
                "--top-teams",
                os.path.join(ROOT, "examples", "top80_teams_sample.csv"),
                "--reference-date",
                "2026-05-31",
                "--output-dir",
                tmpdir,
                "--simulations",
                "25",
                "--minimum-rows",
                "1",
                "--required-teams",
                "3",
                "--pickem-backtest-report",
                backtest_path,
                "--pickem-pass-rate-target",
                "0.38",
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            finally:
                sys.argv = old_argv

            manifest_path = os.path.join(tmpdir, "pipeline_manifest.json")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(exit_code, 0)
        self.assertIn("pickem_backtest_pass_rate", manifest["readiness"]["failed_checks"])


if __name__ == "__main__":
    unittest.main()
