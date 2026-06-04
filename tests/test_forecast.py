import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def history_rows():
    rows = []
    for index in range(8):
        alpha_side = index % 2 == 0
        team1 = "Alpha" if alpha_side else "Bravo"
        team2 = "Bravo" if alpha_side else "Alpha"
        alpha_won = index != 3
        winner = "Alpha" if alpha_won else "Bravo"
        team1_strong = (team1 == "Alpha" and alpha_won) or (team1 == "Bravo" and not alpha_won)
        rows.append(
            {
                "date": f"2026-05-{index + 1:02d}",
                "event": "IEM Cologne Qualifier",
                "event_tier": "S",
                "status": "completed",
                "team1": team1,
                "team2": team2,
                "winner": winner,
                "best_of": 1 if index % 2 == 0 else 3,
                "map": "mirage" if index < 5 else "inferno",
                "team1_rank": 4 if team1 == "Alpha" else 20,
                "team2_rank": 20 if team2 == "Bravo" else 4,
                "team1_rmr_points": 900 if team1 == "Alpha" else 520,
                "team2_rmr_points": 520 if team2 == "Bravo" else 900,
                "team1_recent_winrate_10": 0.8 if team1 == "Alpha" else 0.35,
                "team2_recent_winrate_10": 0.35 if team2 == "Bravo" else 0.8,
                "team1_bo1_winrate_6m": 0.78 if team1 == "Alpha" else 0.42,
                "team2_bo1_winrate_6m": 0.42 if team2 == "Bravo" else 0.78,
                "team1_bo3_winrate_6m": 0.75 if team1 == "Alpha" else 0.44,
                "team2_bo3_winrate_6m": 0.44 if team2 == "Bravo" else 0.75,
                "team1_map_winrate": 0.72 if team1 == "Alpha" else 0.44,
                "team2_map_winrate": 0.44 if team2 == "Bravo" else 0.72,
                "team1_rating": 1.18 if team1 == "Alpha" else 1.01,
                "team2_rating": 1.01 if team2 == "Bravo" else 1.18,
                "team1_kd": 1.16 if team1 == "Alpha" else 0.99,
                "team2_kd": 0.99 if team2 == "Bravo" else 1.16,
                "team1_opening_success": 0.56 if team1 == "Alpha" else 0.48,
                "team2_opening_success": 0.48 if team2 == "Bravo" else 0.56,
                "team1_clutch_winrate": 0.62 if team1 == "Alpha" else 0.45,
                "team2_clutch_winrate": 0.45 if team2 == "Bravo" else 0.62,
                "team1_star_rating": 1.32 if team1 == "Alpha" else 1.06,
                "team2_star_rating": 1.06 if team2 == "Bravo" else 1.32,
                "h2h_team1_winrate": 0.75 if team1 == "Alpha" else 0.25,
                "odds_team1": 1.55 if team1_strong else 2.25,
                "odds_team2": 2.25 if team1_strong else 1.55,
            }
        )
    return rows


class ForecastTests(unittest.TestCase):
    def test_forecast_fixtures_outputs_adjusted_pick_and_unknown_map_average(self):
        from cs2pickem.forecast import forecast_fixtures

        fixtures = [
            {
                "date": "2026-06-01",
                "event": "IEM Cologne Major",
                "event_tier": "S",
                "status": "scheduled",
                "team1": "Alpha",
                "team2": "Bravo",
                "best_of": 1,
                "map": "unknown",
                "team1_rank": 4,
                "team2_rank": 20,
                "team1_rmr_points": 900,
                "team2_rmr_points": 520,
                "team1_recent_winrate_10": 0.8,
                "team2_recent_winrate_10": 0.35,
                "team1_bo1_winrate_6m": 0.78,
                "team2_bo1_winrate_6m": 0.42,
                "team1_bo3_winrate_6m": 0.75,
                "team2_bo3_winrate_6m": 0.44,
                "team1_rating": 1.18,
                "team2_rating": 1.01,
                "team1_kd": 1.16,
                "team2_kd": 0.99,
                "team1_star_rating": 1.32,
                "team2_star_rating": 1.06,
                "team1_player_form_score": 0.12,
                "team2_player_form_score": 0.01,
                "team1_player_form_trend": 0.03,
                "team2_player_form_trend": -0.04,
                "team1_player_sample_confidence": 0.9,
                "team2_player_sample_confidence": 0.4,
                "team2_substitute_flag": 1,
                "odds_team1": 2.05,
                "odds_team2": 1.85,
            }
        ]
        profiles = {
            "Alpha": {"ban_top3": ["nuke"], "prefer_top3": ["mirage", "inferno", "ancient"], "map_winrates": {"mirage": 0.75, "inferno": 0.7, "ancient": 0.6}},
            "Bravo": {"ban_top3": ["vertigo"], "prefer_top3": ["mirage", "anubis", "inferno"], "map_winrates": {"mirage": 0.45, "inferno": 0.5, "ancient": 0.42, "anubis": 0.55}},
        }

        report = forecast_fixtures(history_rows(), fixtures, reference_date="2026-05-31", profiles=profiles, top_k=8, epochs=5)

        self.assertEqual(report["trained_matches"], 8)
        self.assertEqual(report["model_hyperparameters"]["neural_network"]["epochs"], 5)
        self.assertEqual(report["model_hyperparameters"]["xgboost"]["subsample"], 0.8)
        self.assertEqual(len(report["predictions"]), 1)
        prediction = report["predictions"][0]
        self.assertEqual(prediction["team1"], "Alpha")
        self.assertIn("candidate_maps", prediction)
        self.assertEqual(len(prediction["candidate_maps"]), 3)
        self.assertTrue(0.0 <= prediction["model_probability_team1"] <= 1.0)
        self.assertTrue(0.0 <= prediction["adjusted_probability_team1"] <= 1.0)
        self.assertTrue(prediction["market_adjustment_applied"])
        self.assertIn(prediction["pick"], {"Alpha", "Bravo", "avoid"})
        self.assertIn("confidence_margin", prediction)
        self.assertAlmostEqual(prediction["confidence_margin"], abs(prediction["adjusted_probability_team1"] - 0.5))
        self.assertEqual(prediction["low_confidence"], prediction["confidence_margin"] <= 0.02)
        self.assertEqual(report["decision_summary"]["fixtures"], 1)
        self.assertEqual(
            report["decision_summary"]["actionable_picks"] + report["decision_summary"]["low_confidence_avoids"],
            1,
        )
        self.assertEqual(prediction["model_weights"]["neural_network"], 0.0)
        self.assertEqual(set(prediction["model_probabilities_team1"]), {"logistic", "random_forest", "xgboost"})
        self.assertEqual(set(prediction["weighted_model_contributions_team1"]), set(prediction["model_probabilities_team1"]))
        self.assertAlmostEqual(sum(prediction["model_weights"].values()), 1.0)
        self.assertAlmostEqual(sum(prediction["weighted_model_contributions_team1"].values()), prediction["model_probability_team1"])
        self.assertEqual(report["feature_preparation"]["elo"]["basis"], "chronological_pre_match_online")
        self.assertGreater(prediction["team1_elo"], prediction["team2_elo"])
        self.assertEqual(prediction["player_form_summary"]["team1"]["score"], 0.12)
        self.assertEqual(prediction["player_form_summary"]["team2"]["sample_confidence"], 0.4)
        self.assertEqual(prediction["player_form_summary"]["team2"]["substitute_flag"], 1)
        self.assertAlmostEqual(prediction["player_form_summary"]["diff"]["score"], 0.11)
        self.assertAlmostEqual(prediction["player_form_summary"]["diff"]["trend"], 0.07)

    def test_forecast_fixtures_applies_custom_single_match_policy(self):
        from cs2pickem.forecast import forecast_fixtures

        fixtures = [
            {
                "date": "2026-06-01",
                "event": "IEM Cologne Major",
                "event_tier": "S",
                "status": "scheduled",
                "team1": "Alpha",
                "team2": "Bravo",
                "best_of": 1,
                "map": "mirage",
                "team1_rank": 4,
                "team2_rank": 20,
                "team1_player_form_score": -0.08,
                "team2_player_form_score": 0.02,
                "team1_player_sample_confidence": 0.8,
                "team2_player_sample_confidence": 0.8,
            }
        ]

        report = forecast_fixtures(
            history_rows(),
            fixtures,
            reference_date="2026-05-31",
            top_k=8,
            epochs=5,
            minimum_margin=0.5,
            avoid_player_form_counter_signal=True,
        )

        prediction = report["predictions"][0]
        self.assertEqual(prediction["pick"], "avoid")
        self.assertEqual(prediction["avoid_reason"], "low_confidence")
        self.assertTrue(prediction["low_confidence"])
        self.assertEqual(report["decision_policy"]["minimum_margin"], 0.5)
        self.assertTrue(report["decision_policy"]["avoid_player_form_counter_signal"])
        self.assertEqual(report["decision_summary"]["avoid_picks"], 1)

    def test_apply_forecast_policy_file_adds_fixture_player_form_without_retraining(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            forecast_path = os.path.join(tmpdir, "forecast.json")
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            output_path = os.path.join(tmpdir, "policy.json")
            with open(forecast_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "trained_matches": 8,
                        "predictions": [
                            {
                                "date": "2026-06-01",
                                "team1": "Alpha",
                                "team2": "Bravo",
                                "adjusted_probability_team1": 0.54,
                                "pick": "Alpha",
                            },
                            {
                                "date": "2026-06-01",
                                "team1": "Charlie",
                                "team2": "Delta",
                                "adjusted_probability_team1": 0.62,
                                "pick": "Charlie",
                            },
                            {
                                "date": "2026-06-01",
                                "team1": "Echo",
                                "team2": "Foxtrot",
                                "adjusted_probability_team1": 0.62,
                                "pick": "Echo",
                            },
                        ],
                    },
                    handle,
                )
            write_matches_csv(
                fixtures_path,
                [
                    {"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo"},
                    {
                        "date": "2026-06-01",
                        "team1": "Charlie",
                        "team2": "Delta",
                        "team1_player_form_score": -0.04,
                        "team2_player_form_score": 0.04,
                        "team1_player_sample_confidence": 0.7,
                        "team2_player_sample_confidence": 0.7,
                    },
                    {
                        "date": "2026-06-01",
                        "team1": "Echo",
                        "team2": "Foxtrot",
                        "team1_player_form_score": 0.05,
                        "team2_player_form_score": -0.01,
                        "team1_player_sample_confidence": 0.7,
                        "team2_player_sample_confidence": 0.7,
                    },
                ],
            )

            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "apply-forecast-policy",
                "--forecast-report",
                forecast_path,
                "--fixtures",
                fixtures_path,
                "--minimum-margin",
                "0.05",
                "--avoid-player-form-counter-signal",
                "--player-form-counter-min-confidence",
                "0.4",
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
        self.assertEqual(report["decision_policy"]["minimum_margin"], 0.05)
        self.assertEqual(report["decision_policy"]["player_form_counter_min_confidence"], 0.4)
        self.assertEqual(report["decision_summary"]["avoid_picks"], 2)
        self.assertEqual(report["decision_summary"]["low_confidence_avoids"], 1)
        self.assertEqual(report["decision_summary"]["player_form_counter_signal_avoids"], 1)
        self.assertEqual(report["predictions"][0]["avoid_reason"], "low_confidence")
        self.assertEqual(report["predictions"][1]["avoid_reason"], "player_form_counter_signal")
        self.assertEqual(report["predictions"][1]["player_form_summary"]["diff"]["score"], -0.08)
        self.assertEqual(report["predictions"][2]["pick"], "Echo")

    def test_match_predictor_applies_training_cutoff_elo_to_future_rows_without_elo_columns(self):
        from cs2pickem.predictor import MatchPredictor

        predictor = MatchPredictor.train(history_rows(), reference_date="2026-05-31", top_k=10, epochs=3)
        details = predictor.predict_probability_details(
            {
                "date": "2026-06-01",
                "event": "IEM Cologne Major",
                "event_tier": "S",
                "team1": "Alpha",
                "team2": "Bravo",
                "best_of": 1,
                "map": "mirage",
            }
        )

        self.assertEqual(details["feature_preparation"]["elo"]["basis"], "chronological_pre_match_online")
        self.assertGreater(details["team1_elo"], details["team2_elo"])

    def test_forecast_file_workflow_reads_csv_and_profiles_json(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.forecast import forecast_fixtures_file

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.csv")
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            profiles_path = os.path.join(tmpdir, "profiles.json")
            write_matches_csv(history_path, history_rows())
            write_matches_csv(fixtures_path, [{"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo", "best_of": 1, "map": "mirage", "odds_team1": 1.7, "odds_team2": 2.1}])
            with open(profiles_path, "w", encoding="utf-8") as handle:
                json.dump({"Alpha": {"map_winrates": {"mirage": 0.7}}, "Bravo": {"map_winrates": {"mirage": 0.45}}}, handle)

            report = forecast_fixtures_file(history_path, fixtures_path, reference_date="2026-05-31", profiles_path=profiles_path, top_k=6)

        self.assertEqual(report["fixtures"], 1)
        self.assertIn("adjusted_probability_team1", report["predictions"][0])

    def test_forecast_does_not_apply_market_adjustment_without_real_odds(self):
        from cs2pickem.forecast import forecast_fixtures

        fixtures = [
            {
                "date": "2026-06-01",
                "event": "IEM Cologne Major",
                "event_tier": "S",
                "status": "scheduled",
                "team1": "Alpha",
                "team2": "Bravo",
                "best_of": 1,
                "map": "mirage",
                "team1_rank": 4,
                "team2_rank": 20,
                "team1_rmr_points": 900,
                "team2_rmr_points": 520,
                "team1_recent_winrate_10": 0.8,
                "team2_recent_winrate_10": 0.35,
                "team1_bo1_winrate_6m": 0.78,
                "team2_bo1_winrate_6m": 0.42,
                "team1_bo3_winrate_6m": 0.75,
                "team2_bo3_winrate_6m": 0.44,
                "team1_rating": 1.18,
                "team2_rating": 1.01,
                "team1_kd": 1.16,
                "team2_kd": 0.99,
                "team1_star_rating": 1.32,
                "team2_star_rating": 1.06,
            }
        ]

        report = forecast_fixtures(history_rows(), fixtures, reference_date="2026-05-31", top_k=6, epochs=3)

        prediction = report["predictions"][0]
        self.assertFalse(prediction["market_adjustment_applied"])
        self.assertAlmostEqual(prediction["adjusted_probability_team1"], prediction["model_probability_team1"])

    def test_forecast_can_adjust_with_explicit_market_probability_but_not_poll_proxy(self):
        from cs2pickem.forecast import forecast_fixtures

        base_fixture = {
            "date": "2026-06-01",
            "event": "IEM Cologne Major",
            "event_tier": "S",
            "status": "scheduled",
            "team1": "Alpha",
            "team2": "Bravo",
            "best_of": 1,
            "map": "mirage",
            "team1_rank": 4,
            "team2_rank": 20,
            "team1_recent_winrate_10": 0.8,
            "team2_recent_winrate_10": 0.35,
        }
        report = forecast_fixtures(
            history_rows(),
            [
                {**base_fixture, "market_probability_team1": 0.8, "market_signal_source": "book_consensus"},
                {**base_fixture, "hltv_poll_team1": 80, "hltv_poll_team2": 20, "market_proxy_source": "hltv_fan_poll_not_odds"},
            ],
            reference_date="2026-05-31",
            top_k=6,
            epochs=3,
        )

        explicit, proxy = report["predictions"]
        self.assertTrue(explicit["market_adjustment_applied"])
        self.assertEqual(explicit["market_signal"]["basis"], "explicit_market_probability")
        self.assertFalse(explicit["market_signal"]["proxy"])
        self.assertGreater(explicit["adjusted_probability_team1"], explicit["model_probability_team1"])
        self.assertFalse(proxy["market_adjustment_applied"])
        self.assertEqual(proxy["market_signal"]["basis"], "poll_proxy")
        self.assertTrue(proxy["market_signal"]["proxy"])
        self.assertEqual(proxy["adjusted_probability_team1"], proxy["model_probability_team1"])

    def test_forecast_file_applies_bp_intel_before_prediction(self):
        from cs2pickem.data import write_matches_csv
        from cs2pickem.forecast import forecast_fixtures_file

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = os.path.join(tmpdir, "history.csv")
            fixtures_path = os.path.join(tmpdir, "fixtures.csv")
            bp_path = os.path.join(tmpdir, "bp.csv")
            write_matches_csv(history_path, history_rows())
            write_matches_csv(fixtures_path, [{"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo", "best_of": 1, "map": "unknown", "odds_team1": 1.7, "odds_team2": 2.1}])
            write_matches_csv(bp_path, [{"date": "2026-06-01", "team1": "Alpha", "team2": "Bravo", "map": "inferno", "source": "scrim-leak"}])

            report = forecast_fixtures_file(history_path, fixtures_path, reference_date="2026-05-31", bp_path=bp_path, top_k=6)

        prediction = report["predictions"][0]
        self.assertEqual(prediction["map"], "inferno")
        self.assertEqual(prediction["bp_source"], "scrim-leak")
        self.assertEqual(prediction["candidate_maps"], [])
        self.assertEqual(report["bp_report"]["map_overrides"], 1)


if __name__ == "__main__":
    unittest.main()
