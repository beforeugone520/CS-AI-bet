import contextlib
import io
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def match_rows():
    return [
        {"date": "2026-05-20", "team1": "Alpha", "team2": "Bravo", "best_of": 1, "map": "mirage"},
        {"date": "2026-06-10", "team1": "Alpha", "team2": "Bravo", "best_of": 3, "map": "inferno"},
    ]


def player_rows():
    return [
        {"date": "2026-05-10", "team": "Alpha", "player": "a1", "rating": 1.10, "kd": 1.05, "opening_success": 0.52, "clutch_winrate": 0.58, "is_substitute": 0},
        {"date": "2026-05-12", "team": "Alpha", "player": "a2", "rating": 1.30, "kd": 1.22, "opening_success": 0.57, "clutch_winrate": 0.62, "is_substitute": 1},
        {"date": "2026-05-19", "team": "Alpha", "player": "a3", "rating": 0.90, "kd": 0.95, "opening_success": 0.48, "clutch_winrate": 0.50, "is_substitute": 0},
        {"date": "2026-05-21", "team": "Alpha", "player": "future", "rating": 2.00, "kd": 2.00, "opening_success": 0.99, "clutch_winrate": 0.99, "is_substitute": 0},
        {"date": "2026-05-11", "team": "Bravo", "player": "b1", "rating": 1.00, "kd": 1.00, "opening_success": 0.50, "clutch_winrate": 0.51, "is_substitute": 0},
        {"date": "2026-05-18", "team": "Bravo", "player": "b2", "rating": 1.08, "kd": 1.03, "opening_success": 0.54, "clutch_winrate": 0.55, "is_substitute": 0},
        {"date": "2026-06-01", "team": "Alpha", "player": "a1", "rating": 1.40, "kd": 1.35, "opening_success": 0.6, "clutch_winrate": 0.65, "is_substitute": 0},
        {"date": "2026-06-03", "team": "Bravo", "player": "b1", "rating": 0.85, "kd": 0.9, "opening_success": 0.45, "clutch_winrate": 0.47, "is_substitute": 1},
    ]


class PlayerStatsTests(unittest.TestCase):
    def test_merge_player_stats_uses_recent_prior_rows_and_flags_substitutes(self):
        from cs2pickem.players import merge_player_stats_into_matches

        merged = merge_player_stats_into_matches(match_rows(), player_rows(), window_days=15)
        first = merged[0]

        self.assertAlmostEqual(first["team1_rating"], (1.10 + 1.30 + 0.90) / 3)
        self.assertAlmostEqual(first["team1_star_rating"], 1.30)
        self.assertEqual(first["team1_substitute_flag"], 1)
        self.assertEqual(first["team1_player_sample"], 3)
        self.assertAlmostEqual(first["team1_player_form_score"], 0.07716666666666668)
        self.assertAlmostEqual(first["team1_player_form_trend"], -0.134)
        self.assertAlmostEqual(first["team1_player_sample_confidence"], 0.6)
        self.assertGreater(first["team1_player_form_score"], first["team2_player_form_score"])
        self.assertLess(first["team1_rating"], 2.0)
        self.assertAlmostEqual(first["team2_rating"], (1.00 + 1.08) / 2)

    def test_merge_player_stats_defaults_when_recent_window_is_empty(self):
        from cs2pickem.players import merge_player_stats_into_matches

        merged = merge_player_stats_into_matches([{"date": "2026-04-01", "team1": "Alpha", "team2": "Bravo"}], player_rows(), window_days=15)

        self.assertEqual(merged[0]["team1_rating"], 1.0)
        self.assertEqual(merged[0]["team2_opening_success"], 0.5)
        self.assertEqual(merged[0]["team1_player_sample"], 0)
        self.assertEqual(merged[0]["team1_player_form_score"], 0.0)
        self.assertEqual(merged[0]["team1_player_form_trend"], 0.0)
        self.assertEqual(merged[0]["team1_player_sample_confidence"], 0.0)

    def test_merge_player_stats_preserves_existing_values_when_recent_window_is_empty(self):
        from cs2pickem.players import merge_player_stats_into_matches

        merged = merge_player_stats_into_matches(
            [
                {
                    "date": "2026-04-01",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "team1_rating": 1.22,
                    "team1_kd": 1.18,
                    "team1_opening_success": 0.57,
                    "team1_clutch_winrate": 0.61,
                    "team1_star_rating": 1.35,
                }
            ],
            player_rows(),
            window_days=15,
        )

        self.assertEqual(merged[0]["team1_player_sample"], 0)
        self.assertAlmostEqual(merged[0]["team1_rating"], 1.22)
        self.assertAlmostEqual(merged[0]["team1_kd"], 1.18)
        self.assertAlmostEqual(merged[0]["team1_opening_success"], 0.57)
        self.assertAlmostEqual(merged[0]["team1_clutch_winrate"], 0.61)
        self.assertAlmostEqual(merged[0]["team1_star_rating"], 1.35)
        self.assertAlmostEqual(merged[0]["team1_player_form_score"], 0.1745)
        self.assertEqual(merged[0]["team1_player_form_trend"], 0.0)
        self.assertEqual(merged[0]["team1_player_sample_confidence"], 0.0)

    def test_merge_player_stats_preserves_existing_player_form_when_recent_window_is_empty(self):
        from cs2pickem.players import merge_player_stats_into_matches

        merged = merge_player_stats_into_matches(
            [
                {
                    "date": "2026-04-01",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "team1_player_form_score": 0.07,
                    "team1_player_form_trend": -0.02,
                    "team1_player_sample_confidence": 0.4,
                }
            ],
            player_rows(),
            window_days=15,
        )

        self.assertEqual(merged[0]["team1_player_form_score"], 0.07)
        self.assertEqual(merged[0]["team1_player_form_trend"], -0.02)
        self.assertEqual(merged[0]["team1_player_sample_confidence"], 0.4)

    def test_player_file_workflow_writes_augmented_matches(self):
        from cs2pickem.data import read_matches_csv, write_matches_csv
        from cs2pickem.players import merge_player_stats_file

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            players_path = os.path.join(tmpdir, "players.csv")
            output_path = os.path.join(tmpdir, "matches_with_players.csv")
            write_matches_csv(matches_path, match_rows())
            write_matches_csv(players_path, player_rows())

            report = merge_player_stats_file(matches_path, players_path, output_path, window_days=15)
            merged = read_matches_csv(output_path)

        self.assertEqual(report["matches"], 2)
        self.assertEqual(report["teams_augmented"], 4)
        self.assertIn("team1_star_rating", merged[0])
        self.assertEqual(merged[0]["team1_substitute_flag"], 1)
        self.assertIn("team1_player_form_score", merged[0])
        self.assertIn("team1_player_form_trend", merged[0])
        self.assertIn("team1_player_sample_confidence", merged[0])
        self.assertIsInstance(merged[0]["team1_player_form_score"], float)

    def test_player_fields_are_consumed_by_feature_builder(self):
        from cs2pickem.features import FeatureBuilder
        from cs2pickem.players import merge_player_stats_into_matches

        matches = [
            {**match_rows()[0], "winner": "Alpha", "event_tier": "S", "status": "completed"},
            {**match_rows()[1], "winner": "Bravo", "event_tier": "S", "status": "completed"},
        ]
        merged = merge_player_stats_into_matches(matches, player_rows(), window_days=15)
        dataset = FeatureBuilder().fit_transform(merged)

        self.assertIn("substitute_flag_diff", dataset.feature_names)
        self.assertIn("player_sample_diff", dataset.feature_names)
        self.assertIn("player_form_score_diff", dataset.feature_names)
        self.assertIn("player_form_trend_diff", dataset.feature_names)
        self.assertIn("player_sample_confidence_diff", dataset.feature_names)

    def test_cli_merge_players_writes_augmented_csv(self):
        from cs2pickem.cli import main
        from cs2pickem.data import read_matches_csv, write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            players_path = os.path.join(tmpdir, "players.csv")
            output_path = os.path.join(tmpdir, "matches_with_players.csv")
            write_matches_csv(matches_path, match_rows())
            write_matches_csv(players_path, player_rows())

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "merge-players",
                "--matches",
                matches_path,
                "--players",
                players_path,
                "--output",
                output_path,
                "--window-days",
                "15",
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main()
            except SystemExit as exc:
                self.fail(f"merge-players CLI should be registered: {exc}")
            finally:
                sys.argv = old_argv

            merged = read_matches_csv(output_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(merged[0]["team1_player_sample"], 3)
        self.assertIn("team2_substitute_flag", merged[0])


if __name__ == "__main__":
    unittest.main()
