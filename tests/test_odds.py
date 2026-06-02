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
        {"date": "2026-06-03", "source": "5e", "team1": "Echo", "team2": "Foxtrot", "team1_odds": 1.6, "team2_odds": 2.3, "source_match_url": "https://example.test/match/1"},
        {"date": "2026-06-03", "provider": "BookC", "team1": "Golf", "team2": "Hotel", "odds_team1_american": -150, "odds_team2_american": 120},
        {"date": "2026-06-04", "source": "5e", "team1": "Ghost", "team2": "Hotel", "team1_odds": "", "team2_odds": ""},
    ]


class OddsTests(unittest.TestCase):
    def test_market_probability_from_row_prefers_real_odds_then_explicit_probability_then_poll_proxy(self):
        from cs2pickem.odds import market_probability_from_row

        odds_signal = market_probability_from_row({"odds_team1": 1.8, "odds_team2": 2.2, "market_probability_team1": 0.2, "market_signal_source": "book_average", "odds_providers": ["BookA"]})
        explicit_signal = market_probability_from_row({"market_probability_team1": 0.61, "market_signal_source": "consensus_close"})
        averaged_odds_signal = market_probability_from_row(
            {
                "market_probability_team1": 0.58,
                "market_signal_basis": "real_odds",
                "market_signal_proxy": "False",
                "market_signal_source": "odds_provider_average",
            }
        )
        poll_signal = market_probability_from_row({"hltv_poll_team1": 774, "hltv_poll_team2": 226, "market_proxy_source": "hltv_fan_poll_not_odds"})
        missing_signal = market_probability_from_row({"team1": "Alpha", "team2": "Bravo"})

        self.assertAlmostEqual(odds_signal["probability_team1"], (1 / 1.8) / ((1 / 1.8) + (1 / 2.2)))
        self.assertEqual(odds_signal["basis"], "real_odds")
        self.assertEqual(odds_signal["source"], "book_average")
        self.assertFalse(odds_signal["proxy"])
        self.assertEqual(explicit_signal["probability_team1"], 0.61)
        self.assertEqual(explicit_signal["basis"], "explicit_market_probability")
        self.assertEqual(explicit_signal["source"], "consensus_close")
        self.assertEqual(averaged_odds_signal["probability_team1"], 0.58)
        self.assertEqual(averaged_odds_signal["basis"], "real_odds")
        self.assertEqual(averaged_odds_signal["source"], "odds_provider_average")
        self.assertFalse(averaged_odds_signal["proxy"])
        self.assertAlmostEqual(poll_signal["probability_team1"], 0.774)
        self.assertEqual(poll_signal["basis"], "poll_proxy")
        self.assertTrue(poll_signal["proxy"])
        self.assertIsNone(missing_signal)

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
        echo_market = [row for row in normalized if row["canonical_key"] == "2026-06-03__echo__foxtrot"]
        self.assertEqual(len(echo_market), 1)
        self.assertEqual(echo_market[0]["provider"], "5e")
        self.assertEqual(echo_market[0]["source_match_url"], "https://example.test/match/1")
        self.assertAlmostEqual(echo_market[0]["odds_team1"], 1.6)
        self.assertFalse(any(row["canonical_key"] == "2026-06-04__ghost__hotel" for row in normalized))
        golf_market = [row for row in normalized if row["canonical_key"] == "2026-06-03__golf__hotel"]
        self.assertEqual(len(golf_market), 1)
        self.assertAlmostEqual(golf_market[0]["odds_team1"], 1 + 100 / 150)
        self.assertAlmostEqual(golf_market[0]["odds_team2"], 1 + 120 / 100)

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
        self.assertEqual(report["matched_by_canonical"], 2)

    def test_merge_odds_prefers_source_match_url_before_date_team_fallback(self):
        from cs2pickem.odds import merge_odds_into_matches

        matches = [
            {"date": "2026-06-05", "team1": "Alpha", "team2": "Bravo", "source_match_url": "https://example.test/match/exact"},
            {"date": "2026-06-05", "team1": "Alpha", "team2": "Bravo"},
        ]
        odds = [
            {"date": "2026-06-05", "provider": "BookA", "team1": "Alpha", "team2": "Bravo", "odds_team1": 1.4, "odds_team2": 2.9, "source_match_url": "https://example.test/match/exact"},
            {"date": "2026-06-05", "provider": "BookA", "team1": "Alpha", "team2": "Bravo", "odds_team1": 2.4, "odds_team2": 1.6, "source_match_url": "https://example.test/match/other"},
        ]

        merged, report = merge_odds_into_matches(matches, odds)

        self.assertEqual(report["matched"], 2)
        self.assertEqual(report["matched_by_source_url"], 1)
        self.assertEqual(report["matched_by_canonical"], 1)
        self.assertAlmostEqual(merged[0]["odds_team1"], 1.4)
        self.assertAlmostEqual(merged[0]["odds_team2"], 2.9)
        self.assertAlmostEqual(merged[1]["odds_team1"], (1.4 + 2.4) / 2)

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
