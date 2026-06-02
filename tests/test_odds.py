import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def fixture_rows():
    return [
        {"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo", "best_of": 1, "map": "unknown"},
        {"date": "2026-06-02", "team1": "Charlie", "team2": "Delta", "best_of": 3, "map": "inferno"},
    ]


def odds_rows():
    return [
        {"date": "2026-06-01", "provider": "BookA", "team1": "Alpha", "team2": "Bravo", "odds_team1": 1.8, "odds_team2": 2.0},
        {"date": "2026-06-01", "provider": "BookB", "team1": "Bravo", "team2": "Alpha", "odds_team1": 2.1, "odds_team2": 1.75},
        {"date": "2026-06-02", "provider": "BookA", "team1": "Charlie", "team2": "Delta", "odds_team1": 2.5, "odds_team2": 1.55},
    ]


class OddsTests(unittest.TestCase):
    def test_normalize_odds_rows_handles_reversed_team_order_and_market_probability(self):
        from cs2pickem.odds import normalize_odds_rows

        normalized = normalize_odds_rows(odds_rows())
        alpha_market = [row for row in normalized if row["canonical_key"] == "2026-06-01__alpha__bravo"]

        self.assertEqual(len(alpha_market), 2)
        self.assertEqual(alpha_market[1]["team1"], "Alpha")
        self.assertEqual(alpha_market[1]["team2"], "Bravo")
        self.assertAlmostEqual(alpha_market[1]["odds_team1"], 1.75)
        self.assertAlmostEqual(alpha_market[1]["odds_team2"], 2.1)
        self.assertTrue(0.0 < alpha_market[0]["market_probability_team1"] < 1.0)

    def test_merge_odds_into_matches_averages_providers_and_tracks_coverage(self):
        from cs2pickem.odds import merge_odds_into_matches

        merged, report = merge_odds_into_matches(fixture_rows(), odds_rows())

        self.assertEqual(report["matches"], 2)
        self.assertEqual(report["matched"], 2)
        self.assertEqual(report["unmatched"], 0)
        self.assertAlmostEqual(merged[0]["odds_team1"], (1.8 + 1.75) / 2)
        self.assertAlmostEqual(merged[0]["odds_team2"], (2.0 + 2.1) / 2)
        self.assertIn("market_probability_team1", merged[0])
        self.assertEqual(sorted(merged[0]["odds_providers"]), ["BookA", "BookB"])

    def test_merge_odds_file_workflow_writes_augmented_csv(self):
        from cs2pickem.data import read_matches_csv, write_matches_csv
        from cs2pickem.odds import merge_odds_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            odds_path = os.path.join(tmpdir, "odds.csv")
            output_path = os.path.join(tmpdir, "fixtures_with_odds.csv")
            write_matches_csv(fixtures_path, fixture_rows())
            write_matches_csv(odds_path, odds_rows())

            report = merge_odds_file(fixtures_path, odds_path, output_path)
            merged = read_matches_csv(output_path)

        self.assertEqual(report["matched"], 2)
        self.assertAlmostEqual(merged[0]["odds_team1"], (1.8 + 1.75) / 2)
        self.assertIn("odds_provider_count", merged[0])


if __name__ == "__main__":
    unittest.main()
