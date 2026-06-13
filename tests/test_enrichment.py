import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def raw_history():
    rows = []
    for index, date in enumerate(["2026-05-01", "2026-05-03", "2026-05-05", "2026-05-07", "2026-05-09"]):
        rows.append(
            {
                "date": date,
                "event": "IEM Cologne Qualifier",
                "event_tier": "S",
                "status": "completed",
                "team1": "Alpha",
                "team2": "Bravo" if index % 2 == 0 else "Charlie",
                "winner": "Alpha" if index != 3 else "Charlie",
                "best_of": 1 if index % 2 == 0 else 3,
                "map": "mirage" if index < 4 else "inferno",
                "team1_rank": 4,
                "team2_rank": 18 + index,
                "team1_rmr_points": 900,
                "team2_rmr_points": 520,
                "team1_rating": 1.18,
                "team2_rating": 1.02,
                "team1_kd": 1.15,
                "team2_kd": 0.99,
                "team1_opening_success": 0.56,
                "team2_opening_success": 0.48,
                "team1_clutch_winrate": 0.61,
                "team2_clutch_winrate": 0.44,
                "team1_star_rating": 1.31,
                "team2_star_rating": 1.08,
                "odds_team1": 1.55,
                "odds_team2": 2.35,
            }
        )
    rows.append(
        {
            "date": "2026-05-12",
            "event": "IEM Cologne Qualifier",
            "event_tier": "S",
            "status": "completed",
            "team1": "Bravo",
            "team2": "Alpha",
            "winner": "Alpha",
            "best_of": 1,
            "map": "mirage",
            "team1_rank": 22,
            "team2_rank": 4,
            "team1_rmr_points": 500,
            "team2_rmr_points": 900,
            "team1_rating": 1.01,
            "team2_rating": 1.2,
            "team1_kd": 0.98,
            "team2_kd": 1.17,
            "team1_opening_success": 0.47,
            "team2_opening_success": 0.57,
            "team1_clutch_winrate": 0.43,
            "team2_clutch_winrate": 0.62,
            "team1_star_rating": 1.04,
            "team2_star_rating": 1.34,
            "odds_team1": 2.8,
            "odds_team2": 1.42,
        }
    )
    return rows


class EnrichmentTests(unittest.TestCase):
    def test_enrich_matches_uses_only_prior_rows_for_rolling_features(self):
        from cs2pickem.enrichment import enrich_match_history

        enriched = enrich_match_history(raw_history())
        last = enriched[-1]

        self.assertEqual(last["team1"], "Bravo")
        self.assertEqual(last["team2"], "Alpha")
        self.assertAlmostEqual(last["team2_recent_winrate_5"], 0.8)
        self.assertAlmostEqual(last["team2_recent_winrate_10"], 0.8)
        self.assertAlmostEqual(last["team2_bo1_winrate_6m"], 1.0)
        self.assertAlmostEqual(last["team2_bo3_winrate_6m"], 1 / 2)
        self.assertAlmostEqual(last["team2_map_winrate"], 3 / 4)
        self.assertAlmostEqual(last["h2h_team1_winrate"], 0.0)
        self.assertEqual(last["team2_current_streak"], 1)
        self.assertEqual(last["team2_matches_30d"], 5)

    def test_bo_winrate_6m_excludes_matches_older_than_window(self):
        from cs2pickem.enrichment import enrich_match_history

        # Alpha plays five bo1 matches: four very old wins (well outside the 6-month
        # window) and one recent loss inside the window. The "_6m" winrate must only
        # count the recent match, so it should be 0.0, not 4/5 (all-time).
        rows = [
            {"date": "2024-01-01", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2024-01-08", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2024-01-15", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2024-01-22", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2025-06-01", "team1": "Alpha", "team2": "Charlie", "winner": "Charlie", "best_of": 1, "map": "mirage"},
            {"date": "2025-06-10", "team1": "Alpha", "team2": "Delta", "winner": "Delta", "best_of": 1, "map": "mirage"},
        ]

        enriched = enrich_match_history(rows)
        last = enriched[-1]
        self.assertEqual(last["team1"], "Alpha")
        # Prior to the last match Alpha has 4 old wins (outside window) + 1 recent loss
        # inside the window. The 6-month bo1 winrate must reflect only the recent loss.
        self.assertAlmostEqual(last["team1_bo1_winrate_6m"], 0.0)

    def test_bo_winrate_6m_counts_all_in_window_matches(self):
        from cs2pickem.enrichment import enrich_match_history

        # Two recent bo1 matches within the window (one win, one loss) -> 0.5,
        # plus an in-window bo3 win -> 1.0.
        rows = [
            {"date": "2026-01-05", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-02-05", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo", "best_of": 1, "map": "mirage"},
            {"date": "2026-03-05", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 3, "map": "inferno"},
            {"date": "2026-04-05", "team1": "Alpha", "team2": "Charlie", "winner": "Charlie", "best_of": 1, "map": "nuke"},
        ]

        enriched = enrich_match_history(rows)
        last = enriched[-1]
        self.assertEqual(last["team1"], "Alpha")
        # bo1 prior to last match: win (1/5) + loss (2/5) -> 1/2
        self.assertAlmostEqual(last["team1_bo1_winrate_6m"], 1 / 2)
        # bo3 prior to last match: single win -> 1.0
        self.assertAlmostEqual(last["team1_bo3_winrate_6m"], 1.0)

    def test_enrich_adds_leakage_free_pre_match_elo(self):
        from cs2pickem.enrichment import enrich_match_history

        enriched = enrich_match_history(raw_history())

        # First match: both teams enter at the Elo base (no prior games, no leakage).
        self.assertEqual(enriched[0]["team1_elo"], 1500.0)
        self.assertEqual(enriched[0]["team2_elo"], 1500.0)
        # By the last match Alpha (the dominant winner) has a higher rating than its opponent.
        last = enriched[-1]
        self.assertEqual(last["team2"], "Alpha")
        self.assertGreater(last["team2_elo"], 1500.0)
        self.assertGreater(last["team2_elo"], last["team1_elo"])

    def test_h2h_uses_only_last_three_prior_meetings(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = [
            {"date": "2026-05-01", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-05-02", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "mirage"},
            {"date": "2026-05-03", "team1": "Alpha", "team2": "Bravo", "winner": "Bravo", "best_of": 1, "map": "inferno"},
            {"date": "2026-05-04", "team1": "Bravo", "team2": "Alpha", "winner": "Bravo", "best_of": 3, "map": "inferno"},
            {"date": "2026-05-05", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "nuke"},
            {"date": "2026-05-06", "team1": "Alpha", "team2": "Bravo", "winner": "Alpha", "best_of": 1, "map": "nuke"},
        ]

        enriched = enrich_match_history(rows)

        self.assertAlmostEqual(enriched[-1]["h2h_team1_winrate"], 1 / 3)
        self.assertAlmostEqual(enriched[-1]["team1_h2h_winrate_vs_opponent"], 1 / 3)

    def test_enrich_matches_fills_missing_player_stats_from_prior_team_average(self):
        from cs2pickem.enrichment import enrich_match_history

        rows = raw_history()
        rows[-1]["team2_rating"] = None
        rows[-1]["team2_kd"] = None
        enriched = enrich_match_history(rows)
        last = enriched[-1]

        self.assertAlmostEqual(last["team2_rating"], 1.18)
        self.assertAlmostEqual(last["team2_kd"], 1.15)

    def test_team_profiles_capture_map_preferences_and_bans(self):
        from cs2pickem.enrichment import build_team_profiles

        profiles = build_team_profiles(raw_history())

        self.assertIn("Alpha", profiles)
        self.assertEqual(profiles["Alpha"]["prefer_top3"][0], "mirage")
        self.assertIn("map_winrates", profiles["Alpha"])
        self.assertLessEqual(len(profiles["Alpha"]["ban_top3"]), 3)

    def test_enriched_rolling_fields_are_consumed_by_feature_builder(self):
        from cs2pickem.enrichment import enrich_match_history
        from cs2pickem.features import FeatureBuilder

        enriched = enrich_match_history(raw_history())
        dataset = FeatureBuilder().fit_transform(enriched)

        self.assertIn("recent_winrate_5_diff", dataset.feature_names)
        self.assertIn("matches_30d_diff", dataset.feature_names)
        self.assertIn("current_streak_diff", dataset.feature_names)

    def test_enrich_csv_workflow_writes_training_rows_and_profiles(self):
        from cs2pickem.data import read_matches_csv
        from cs2pickem.pipeline import enrich_matches_file

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "raw.csv")
            output_path = os.path.join(tmpdir, "enriched.csv")
            profiles_path = os.path.join(tmpdir, "profiles.json")
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "date,event,event_tier,status,team1,team2,winner,best_of,map,team1_rank,team2_rank,"
                    "team1_rmr_points,team2_rmr_points,team1_rating,team2_rating,team1_kd,team2_kd\n"
                )
                for row in raw_history():
                    handle.write(
                        f"{row['date']},{row['event']},{row['event_tier']},{row['status']},{row['team1']},{row['team2']},"
                        f"{row['winner']},{row['best_of']},{row['map']},{row['team1_rank']},{row['team2_rank']},"
                        f"{row['team1_rmr_points']},{row['team2_rmr_points']},{row['team1_rating']},{row['team2_rating']},"
                        f"{row['team1_kd']},{row['team2_kd']}\n"
                    )

            manifest = enrich_matches_file(input_path, output_path, profiles_path)
            enriched = read_matches_csv(output_path)

        self.assertEqual(manifest["rows"], 6)
        self.assertEqual(manifest["profiles"], 3)
        self.assertIn("team1_recent_winrate_5", enriched[0])
        self.assertIn("team2_current_streak", enriched[-1])


if __name__ == "__main__":
    unittest.main()
