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

    def test_merge_odds_into_matches_merges_devig_audit_fields_back(self):
        """WF-2F: the de-vig audit magnitudes (overround / devig_z) now ride back onto the
        match row so the previously constant-0 odds_overround / odds_devig_z feature columns
        carry real signal. overround is averaged across providers; devig_z stays absent under
        the default multiplicative de-vig (None) so the feature keeps its neutral 0 default."""
        from cs2pickem.odds import merge_odds_into_matches

        merged, _ = merge_odds_into_matches(fixture_rows(), odds_rows())
        alpha = merged[0]
        # overround is present and non-zero (real two-way book vig), and matches the
        # provider-average of each candidate's de-vigged overround.
        self.assertIn("overround", alpha)
        self.assertGreater(alpha["overround"], 0.0)
        # Default multiplicative de-vig reports no insider z, so the key is intentionally
        # NOT written -> the feature builder falls back to its neutral 0 (no silent skew).
        self.assertNotIn("devig_z", alpha)
        self.assertEqual(alpha["devig_method"], "multiplicative")

    def test_merge_odds_devig_audit_makes_feature_column_nonzero(self):
        """The merged overround flows through FeatureBuilder into a non-zero odds_overround
        feature -- proving the column is no longer dead in the real (merged) pipeline."""
        from cs2pickem.features import FeatureBuilder
        from cs2pickem.odds import merge_odds_into_matches

        merged, _ = merge_odds_into_matches(fixture_rows(), odds_rows())
        builder = FeatureBuilder(include_unverified_features=True)
        raw = builder._raw_features(merged[0])
        self.assertGreater(raw["odds_overround"], 0.0)

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


class DevigTests(unittest.TestCase):
    def test_three_methods_agree_within_one_percent_on_balanced_market(self):
        from cs2pickem.odds import devig_market

        # Near-symmetric two-way market: with little skew the three de-vig
        # techniques should land within 1 absolute percentage point of each other.
        for odds in [(1.91, 1.91), (1.95, 1.95), (1.85, 1.95), (1.90, 1.92)]:
            multiplicative = devig_market(odds[0], odds[1], "multiplicative")
            power = devig_market(odds[0], odds[1], "power")
            shin = devig_market(odds[0], odds[1], "shin")
            probs = [
                multiplicative["fair_prob_team1"],
                power["fair_prob_team1"],
                shin["fair_prob_team1"],
            ]
            spread = max(probs) - min(probs)
            self.assertLess(spread, 0.01, msg=f"methods diverged >1% for odds {odds}: {probs}")
            for audit in (multiplicative, power, shin):
                self.assertAlmostEqual(audit["fair_prob_team1"] + audit["fair_prob_team2"], 1.0, places=9)

    def test_methods_diverge_on_lopsided_market(self):
        from cs2pickem.odds import devig_market

        # The whole point of pluggable de-vig: on a skewed favourite the methods
        # must NOT collapse to the same answer (otherwise the upgrade is inert).
        multiplicative = devig_market(1.5, 2.6, "multiplicative")["fair_prob_team1"]
        power = devig_market(1.5, 2.6, "power")["fair_prob_team1"]
        shin = devig_market(1.5, 2.6, "shin")["fair_prob_team1"]
        self.assertGreater(power, multiplicative)
        self.assertGreater(shin, multiplicative)

    def test_power_solver_converges_so_powered_probs_sum_to_one(self):
        from cs2pickem.odds import devig_market

        for odds in [(1.6, 2.4), (1.05, 11.0), (1.8, 2.0), (1.5, 2.6)]:
            audit = devig_market(odds[0], odds[1], "power")
            k = audit["devig_power_k"]
            self.assertIsNotNone(k)
            self.assertGreater(k, 0.0)
            raw_sum = (1.0 / odds[0]) ** k + (1.0 / odds[1]) ** k
            self.assertAlmostEqual(raw_sum, 1.0, places=8)
            self.assertAlmostEqual(audit["fair_prob_team1"] + audit["fair_prob_team2"], 1.0, places=9)
            # Vig present -> k must inflate beyond 1 to deflate the implied probs.
            self.assertGreater(k, 1.0)

    def test_shin_z_within_unit_interval_and_self_normalizes(self):
        from cs2pickem.odds import devig_market

        for odds in [(1.6, 2.4), (1.05, 11.0), (1.5, 2.6), (1.91, 1.91), (1.01, 50.0)]:
            audit = devig_market(odds[0], odds[1], "shin")
            z = audit["devig_z"]
            self.assertIsNotNone(z)
            self.assertGreaterEqual(z, 0.0)
            self.assertLessEqual(z, 1.0)
            # Shin closed form must self-normalize without any post-hoc fix-up.
            self.assertAlmostEqual(audit["fair_prob_team1"] + audit["fair_prob_team2"], 1.0, places=9)
            # Favourite stays the favourite after de-vig.
            if odds[0] < odds[1]:
                self.assertGreater(audit["fair_prob_team1"], audit["fair_prob_team2"])

    def test_shin_z_scales_with_overround(self):
        from cs2pickem.odds import devig_market

        low_vig = devig_market(1.98, 1.98, "shin")
        high_vig = devig_market(1.80, 1.80, "shin")
        self.assertLess(low_vig["overround"], high_vig["overround"])
        self.assertLess(low_vig["devig_z"], high_vig["devig_z"])

    def test_overround_matches_sum_of_inverse_odds_minus_one(self):
        from cs2pickem.odds import devig_market

        for odds in [(1.8, 2.0), (1.5, 2.6), (2.5, 1.55)]:
            expected = (1.0 / odds[0]) + (1.0 / odds[1]) - 1.0
            for method in ("multiplicative", "power", "shin"):
                audit = devig_market(odds[0], odds[1], method)
                self.assertAlmostEqual(audit["overround"], expected, places=12)

    def test_multiplicative_default_preserves_legacy_market_probability(self):
        from cs2pickem.odds import _market_probability, devig_market

        for odds in [(1.8, 2.0), (2.5, 1.55), (1.75, 2.1)]:
            inv1 = 1.0 / odds[0]
            inv2 = 1.0 / odds[1]
            legacy = inv1 / (inv1 + inv2)
            self.assertAlmostEqual(_market_probability(odds[0], odds[1]), legacy, places=12)
            self.assertAlmostEqual(devig_market(odds[0], odds[1])["fair_prob_team1"], legacy, places=12)
            self.assertEqual(devig_market(odds[0], odds[1])["devig_method"], "multiplicative")

    def test_audit_fields_present_and_method_specific(self):
        from cs2pickem.odds import devig_market

        multiplicative = devig_market(1.6, 2.4, "multiplicative")
        power = devig_market(1.6, 2.4, "power")
        shin = devig_market(1.6, 2.4, "shin")
        for audit in (multiplicative, power, shin):
            for field in ("fair_prob_team1", "fair_prob_team2", "overround", "devig_z", "devig_power_k", "devig_method"):
                self.assertIn(field, audit)
        # Only the owning method populates its diagnostic; others stay None.
        self.assertIsNone(multiplicative["devig_z"])
        self.assertIsNone(multiplicative["devig_power_k"])
        self.assertIsNone(power["devig_z"])
        self.assertIsNotNone(power["devig_power_k"])
        self.assertIsNotNone(shin["devig_z"])
        self.assertIsNone(shin["devig_power_k"])

    def test_invalid_method_raises(self):
        from cs2pickem.odds import devig_market

        with self.assertRaises(ValueError):
            devig_market(1.8, 2.0, "bananas")

    def test_method_name_is_case_insensitive(self):
        from cs2pickem.odds import devig_market

        upper = devig_market(1.6, 2.4, "SHIN")
        lower = devig_market(1.6, 2.4, "shin")
        self.assertEqual(upper["devig_method"], "shin")
        self.assertAlmostEqual(upper["fair_prob_team1"], lower["fair_prob_team1"], places=12)

    def test_degenerate_inputs_are_robust(self):
        from cs2pickem.odds import devig_market

        for method in ("multiplicative", "power", "shin"):
            for odds in [(0.0, 2.0), (-1.0, 2.0), (0.0, 0.0), (1.8, None)]:
                audit = devig_market(odds[0], odds[1], method)  # type: ignore[arg-type]
                self.assertAlmostEqual(audit["fair_prob_team1"], 0.5, places=12)
                self.assertAlmostEqual(audit["fair_prob_team2"], 0.5, places=12)
                self.assertEqual(audit["overround"], 0.0)
                self.assertEqual(audit["devig_method"], method)

    def test_no_overround_market_returns_inputs_with_unit_exponent_and_zero_insider(self):
        from cs2pickem.odds import devig_market

        # A perfectly fair book (sum of inverse odds == 1) has no margin: power
        # exponent collapses to 1 and Shin insider proportion to 0.
        fair = devig_market(2.0, 2.0, "power")
        self.assertAlmostEqual(fair["overround"], 0.0, places=12)
        self.assertAlmostEqual(fair["devig_power_k"], 1.0, places=9)
        shin = devig_market(2.0, 2.0, "shin")
        self.assertEqual(shin["devig_z"], 0.0)
        self.assertAlmostEqual(shin["fair_prob_team1"], 0.5, places=12)

    def test_market_probability_from_row_attaches_devig_audit(self):
        from cs2pickem.odds import market_probability_from_row

        default_signal = market_probability_from_row({"odds_team1": 1.6, "odds_team2": 2.4})
        self.assertEqual(default_signal["devig_method"], "multiplicative")
        self.assertAlmostEqual(default_signal["overround"], (1 / 1.6) + (1 / 2.4) - 1.0, places=12)
        self.assertIsNone(default_signal["devig_z"])

        shin_signal = market_probability_from_row({"odds_team1": 1.6, "odds_team2": 2.4}, method="shin")
        self.assertEqual(shin_signal["devig_method"], "shin")
        self.assertIsNotNone(shin_signal["devig_z"])
        self.assertGreaterEqual(shin_signal["devig_z"], 0.0)
        self.assertLessEqual(shin_signal["devig_z"], 1.0)
        # Default method keeps legacy multiplicative probability exactly.
        legacy = (1 / 1.6) / ((1 / 1.6) + (1 / 2.4))
        self.assertAlmostEqual(default_signal["probability_team1"], legacy, places=12)

    def test_normalize_odds_rows_persists_devig_audit_fields(self):
        from cs2pickem.odds import normalize_odds_rows

        normalized = normalize_odds_rows(odds_rows())
        sample = normalized[0]
        for field in ("fair_prob_team1", "fair_prob_team2", "overround", "devig_z", "devig_power_k", "devig_method"):
            self.assertIn(field, sample)
        self.assertEqual(sample["devig_method"], "multiplicative")
        # Default normalization must keep the legacy market probability identical.
        self.assertAlmostEqual(sample["market_probability_team1"], sample["fair_prob_team1"], places=12)

        shin_rows = normalize_odds_rows(odds_rows(), method="shin")
        self.assertEqual(shin_rows[0]["devig_method"], "shin")
        self.assertIsNotNone(shin_rows[0]["devig_z"])


if __name__ == "__main__":
    unittest.main()
