import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tests.test_forecast import history_rows


def team_rows():
    return [
        {"team": "Alpha", "seed": 1, "rank": 4, "rmr_points": 900, "rating": 1.18, "kd": 1.16, "recent_winrate_10": 0.8, "bo1_winrate_6m": 0.78, "bo3_winrate_6m": 0.75},
        {"team": "Bravo", "seed": 2, "rank": 20, "rmr_points": 520, "rating": 1.01, "kd": 0.99, "recent_winrate_10": 0.35, "bo1_winrate_6m": 0.42, "bo3_winrate_6m": 0.44},
        {"team": "Charlie", "seed": 3, "rank": 18, "rmr_points": 560, "rating": 1.04, "kd": 1.0, "recent_winrate_10": 0.48, "bo1_winrate_6m": 0.5, "bo3_winrate_6m": 0.5},
        {"team": "Delta", "seed": 4, "rank": 24, "rmr_points": 480, "rating": 0.98, "kd": 0.96, "recent_winrate_10": 0.3, "bo1_winrate_6m": 0.38, "bo3_winrate_6m": 0.4},
    ]


def profiles():
    return {
        "Alpha": {"prefer_top3": ["mirage", "inferno"], "ban_top3": ["nuke"], "map_winrates": {"mirage": 0.75, "inferno": 0.7, "ancient": 0.6}},
        "Bravo": {"prefer_top3": ["mirage", "anubis"], "ban_top3": ["vertigo"], "map_winrates": {"mirage": 0.45, "inferno": 0.5, "ancient": 0.42}},
        "Charlie": {"prefer_top3": ["inferno", "ancient"], "ban_top3": ["nuke"], "map_winrates": {"mirage": 0.5, "inferno": 0.55, "ancient": 0.52}},
        "Delta": {"prefer_top3": ["ancient"], "ban_top3": ["vertigo"], "map_winrates": {"mirage": 0.42, "inferno": 0.4, "ancient": 0.48}},
    }


class PickemTests(unittest.TestCase):
    def assertMovedTowardMarket(self, details):
        model_probability = details["model_probability_team1"]
        adjusted_probability = details["adjusted_probability_team1"]
        market_probability = details["market_probability_team1"]

        self.assertLessEqual(
            abs(adjusted_probability - market_probability),
            abs(model_probability - market_probability),
        )
        if market_probability >= model_probability:
            self.assertGreaterEqual(adjusted_probability, model_probability)
            self.assertLessEqual(adjusted_probability, market_probability)
        else:
            self.assertLessEqual(adjusted_probability, model_probability)
            self.assertGreaterEqual(adjusted_probability, market_probability)

    def test_model_driven_pickem_uses_trained_match_probabilities_for_swiss(self):
        from cs2pickem.pickem import model_driven_pickems

        report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=80,
            seed=5,
            top_k=8,
            epochs=5,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        self.assertEqual(report["trained_matches"], 8)
        self.assertEqual(report["simulations"], 80)
        self.assertEqual(report["model_hyperparameters"]["neural_network"]["epochs"], 5)
        self.assertEqual(report["model_hyperparameters"]["random_forest"]["min_leaf_samples"], 3)
        self.assertEqual(set(report["team_probabilities"]), {"Alpha", "Bravo", "Charlie", "Delta"})
        self.assertIn("pickems", report)
        self.assertEqual(len(report["pickems"]["3-0"]), 1)
        self.assertIn("pickem_details", report)
        self.assertEqual(report["pickem_details"]["3-0"][0]["team"], report["pickems"]["3-0"][0])
        self.assertEqual(report["pickem_details"]["3-0"][0]["category"], "3-0")
        self.assertIn("probability", report["pickem_details"]["3-0"][0])
        self.assertIn("selection_margin", report["pickem_details"]["3-0"][0])
        self.assertIn("selection_score", report["pickem_details"]["3-0"][0])
        self.assertIn("next_best_score", report["pickem_details"]["3-0"][0])
        self.assertIsNone(report["pickem_details"]["3-0"][0]["next_best_probability"])
        self.assertIn("pickem_risk_details", report)
        self.assertIn("upset_penalty_multiplier", report["pickem_risk_details"]["3-0"][0])
        self.assertIn("stage_adjustment", report["pickem_risk_details"]["3-0"][0])
        self.assertIn("sample_match_probabilities", report)
        self.assertIn("Alpha__Bravo__bo1", report["sample_match_probabilities"])
        self.assertTrue(0.0 <= report["sample_match_probabilities"]["Alpha__Bravo__bo1"] <= 1.0)
        self.assertIn("sample_match_details", report)
        details = report["sample_match_details"]["Alpha__Bravo__bo1"]
        self.assertEqual(details["model_weights"]["neural_network"], 0.0)
        self.assertEqual(set(details["model_probabilities_team1"]), {"logistic", "random_forest", "xgboost"})
        self.assertIn("adjusted_probability_team1", details)
        self.assertFalse(details["market_adjustment_applied"])
        self.assertAlmostEqual(details["adjusted_probability_team1"], details["model_probability_team1"])
        self.assertAlmostEqual(details["adjusted_probability_team1"], report["sample_match_probabilities"]["Alpha__Bravo__bo1"])
        self.assertAlmostEqual(sum(details["weighted_model_contributions_team1"].values()), details["model_probability_team1"])

    def test_model_driven_pickem_file_workflow(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.pickem import model_driven_pickems_file

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.csv")
            teams_path = os.path.join(tmpdir, "teams.csv")
            profiles_path = os.path.join(tmpdir, "profiles.json")
            write_matches_csv(history_path, history_rows())
            write_matches_csv(teams_path, team_rows())
            with open(profiles_path, "w", encoding="utf-8") as handle:
                json.dump(profiles(), handle)

            report = model_driven_pickems_file(
                history_path=history_path,
                teams_path=teams_path,
                reference_date="2026-05-31",
                profiles_path=profiles_path,
                simulations=40,
                seed=7,
                top_k=6,
            )

        self.assertEqual(report["teams"], 4)
        self.assertIn("pickems", report)

    def test_fixture_from_team_rows_carries_major_history_features(self):
        from cs2pickem.pickem import _fixture_from_team_rows

        fixture = _fixture_from_team_rows(
            {**team_rows()[0], "major_best_placement": 1},
            {**team_rows()[1], "major_best_placement": 12},
            best_of=3,
        )

        self.assertEqual(fixture["team1_major_best_placement"], 1)
        self.assertEqual(fixture["team2_major_best_placement"], 12)

    def test_model_driven_pickem_reports_stage_strategy(self):
        from cs2pickem.pickem import model_driven_pickems

        report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            stage="challengers",
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        self.assertEqual(report["stage_strategy"]["stage"], "challengers")
        self.assertIn("BO1", report["stage_strategy"]["focus"])

    def test_model_driven_pickem_applies_market_odds_adjustment_to_swiss_probabilities(self):
        from cs2pickem.pickem import model_driven_pickems

        odds_rows = [
            {**team_rows()[0], "odds": 10.0},
            {**team_rows()[1], "odds": 1.1},
            *team_rows()[2:],
        ]

        report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=odds_rows,
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        details = report["sample_match_details"]["Alpha__Bravo__bo1"]

        self.assertIn("adjusted_probability_team1", details)
        self.assertTrue(details["market_adjustment_applied"])
        self.assertLess(details["adjusted_probability_team1"], details["model_probability_team1"])
        self.assertAlmostEqual(
            report["sample_match_probabilities"]["Alpha__Bravo__bo1"],
            details["adjusted_probability_team1"],
        )

    def test_model_driven_pickem_uses_fixture_level_odds_for_swiss_matchups(self):
        from cs2pickem.pickem import model_driven_pickems

        report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            fixture_rows=[{"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo", "odds_team1": 10.0, "odds_team2": 1.1}],
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        details = report["sample_match_details"]["Alpha__Bravo__bo1"]

        self.assertTrue(details["market_adjustment_applied"])
        self.assertLess(details["adjusted_probability_team1"], details["model_probability_team1"])
        self.assertGreaterEqual(report["market_adjustment_summary"]["adjusted_matchups"], 1)
        self.assertIn("Alpha__Bravo__bo1", report["market_adjustment_summary"]["adjusted_matchup_keys"])

    def test_model_driven_pickem_uses_explicit_market_probability_but_only_reports_poll_proxy(self):
        from cs2pickem.pickem import model_driven_pickems

        explicit_report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            fixture_rows=[
                {
                    "date": "2026-06-01",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "market_probability_team1": 0.2,
                    "market_signal_source": "closing_consensus",
                }
            ],
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )
        proxy_report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            fixture_rows=[
                {
                    "date": "2026-06-01",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "hltv_poll_team1": 80,
                    "hltv_poll_team2": 20,
                    "market_proxy_source": "hltv_fan_poll_not_odds",
                }
            ],
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        explicit_details = explicit_report["sample_match_details"]["Alpha__Bravo__bo1"]
        proxy_details = proxy_report["sample_match_details"]["Alpha__Bravo__bo1"]

        self.assertTrue(explicit_details["market_adjustment_applied"])
        self.assertEqual(explicit_details["market_signal"]["basis"], "explicit_market_probability")
        self.assertEqual(explicit_details["market_signal"]["source"], "closing_consensus")
        self.assertMovedTowardMarket(explicit_details)
        self.assertFalse(proxy_details["market_adjustment_applied"])
        self.assertEqual(proxy_details["market_signal"]["basis"], "poll_proxy")
        self.assertTrue(proxy_details["market_signal"]["proxy"])
        self.assertAlmostEqual(proxy_details["adjusted_probability_team1"], proxy_details["model_probability_team1"])
        self.assertGreaterEqual(proxy_report["market_adjustment_summary"]["signal_counts"]["poll_proxy"], 1)
        self.assertGreaterEqual(proxy_report["market_adjustment_summary"]["proxy_signal_matchups"], 1)

    def test_model_driven_pickem_uses_preaveraged_real_odds_probability_without_decimal_odds(self):
        from cs2pickem.pickem import model_driven_pickems

        report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            fixture_rows=[
                {
                    "date": "2026-06-01",
                    "team1": "Alpha",
                    "team2": "Bravo",
                    "market_probability_team1": 0.2,
                    "market_signal_basis": "real_odds",
                    "market_signal_proxy": "False",
                    "market_signal_source": "odds_provider_average",
                }
            ],
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        details = report["sample_match_details"]["Alpha__Bravo__bo1"]

        self.assertTrue(details["market_adjustment_applied"])
        self.assertEqual(details["market_signal"]["basis"], "real_odds")
        self.assertEqual(details["market_signal"]["source"], "odds_provider_average")
        self.assertFalse(details["market_signal"]["proxy"])
        self.assertMovedTowardMarket(details)

    def test_model_driven_pickem_propagates_opening_market_strength_to_unpriced_swiss_matchups(self):
        from cs2pickem.pickem import model_driven_pickems

        report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            fixture_rows=[
                {"date": "2026-06-01", "team1": "Alpha", "team2": "Delta", "odds_team1": 1.2, "odds_team2": 4.6},
                {"date": "2026-06-01", "team1": "Bravo", "team2": "Charlie", "odds_team1": 2.9, "odds_team2": 1.45},
            ],
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        details = report["sample_match_details"]["Alpha__Bravo__bo1"]

        self.assertTrue(details["market_adjustment_applied"])
        self.assertEqual(details["market_adjustment_source"], "team_market_strength")
        self.assertGreater(details["adjusted_probability_team1"], details["model_probability_team1"])
        self.assertIn("Alpha__Bravo__bo1", report["market_adjustment_summary"]["adjusted_matchup_keys"])

    def test_model_driven_pickem_swaps_reversed_fixture_level_odds(self):
        from cs2pickem.pickem import model_driven_pickems

        report = model_driven_pickems(
            history_rows=history_rows(),
            team_rows=team_rows(),
            fixture_rows=[{"date": "2026-06-01", "team1": "Bravo", "team2": "Alpha", "odds_team1": 1.1, "odds_team2": 10.0}],
            reference_date="2026-05-31",
            profiles=profiles(),
            simulations=20,
            seed=5,
            top_k=6,
            epochs=3,
            slots={"3-0": 1, "advance": 2, "0-3": 1},
        )

        details = report["sample_match_details"]["Alpha__Bravo__bo1"]

        self.assertTrue(details["market_adjustment_applied"])
        self.assertLess(details["adjusted_probability_team1"], details["model_probability_team1"])

    def test_pickem_cli_accepts_fixture_level_odds(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.csv")
            teams_path = os.path.join(tmpdir, "teams.csv")
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            output_path = os.path.join(tmpdir, "pickem.json")
            write_matches_csv(history_path, history_rows())
            write_matches_csv(teams_path, team_rows())
            write_matches_csv(
                fixtures_path,
                [{"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo", "best_of": 1, "map": "unknown", "odds_team1": 10.0, "odds_team2": 1.1}],
            )
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "pickem",
                "--history",
                history_path,
                "--teams",
                teams_path,
                "--fixtures",
                fixtures_path,
                "--reference-date",
                "2026-05-31",
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
                self.assertEqual(main(), 0)
            finally:
                sys.argv = old_argv
            with open(output_path, encoding="utf-8") as handle:
                report = json.load(handle)

        details = report["sample_match_details"]["Alpha__Bravo__bo1"]
        self.assertTrue(details["market_adjustment_applied"])
        self.assertGreaterEqual(report["market_adjustment_summary"]["adjusted_matchups"], 1)


if __name__ == "__main__":
    unittest.main()
