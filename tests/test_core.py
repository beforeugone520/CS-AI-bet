import json
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def sample_matches():
    return [
        {
            "date": "2026-05-20",
            "event": "IEM Dallas",
            "event_tier": "S",
            "status": "completed",
            "team1": "Alpha",
            "team2": "Bravo",
            "winner": "Alpha",
            "best_of": 1,
            "map": "mirage",
            "team1_rank": 3,
            "team2_rank": 18,
            "team1_rmr_points": 950,
            "team2_rmr_points": 500,
            "team1_recent_winrate_10": 0.8,
            "team2_recent_winrate_10": 0.4,
            "team1_bo1_winrate_6m": 0.72,
            "team2_bo1_winrate_6m": 0.48,
            "team1_bo3_winrate_6m": 0.7,
            "team2_bo3_winrate_6m": 0.55,
            "team1_map_winrate": 0.74,
            "team2_map_winrate": 0.44,
            "team1_rating": 1.16,
            "team2_rating": 1.01,
            "team1_kd": 1.18,
            "team2_kd": 0.98,
            "team1_opening_success": 0.55,
            "team2_opening_success": 0.49,
            "team1_clutch_winrate": 0.63,
            "team2_clutch_winrate": 0.45,
            "team1_star_rating": 1.3,
            "team2_star_rating": 1.08,
            "h2h_team1_winrate": None,
            "odds_team1": 1.45,
            "odds_team2": 2.7,
            "swiss_round": 1,
            "team1_wins": 0,
            "team1_losses": 0,
            "team2_wins": 0,
            "team2_losses": 0,
        },
        {
            "date": "2026-05-21",
            "event": "IEM Dallas",
            "event_tier": "A",
            "status": "completed",
            "team1": "Charlie",
            "team2": "Delta",
            "winner": "Delta",
            "best_of": 3,
            "map": "inferno",
            "team1_rank": 25,
            "team2_rank": 8,
            "team1_rmr_points": 430,
            "team2_rmr_points": 860,
            "team1_recent_winrate_10": 0.45,
            "team2_recent_winrate_10": 0.75,
            "team1_bo1_winrate_6m": 0.5,
            "team2_bo1_winrate_6m": 0.64,
            "team1_bo3_winrate_6m": 0.46,
            "team2_bo3_winrate_6m": 0.78,
            "team1_map_winrate": 0.47,
            "team2_map_winrate": 0.69,
            "team1_rating": 1.0,
            "team2_rating": 1.17,
            "team1_kd": 0.99,
            "team2_kd": 1.16,
            "team1_opening_success": 0.48,
            "team2_opening_success": 0.56,
            "team1_clutch_winrate": 0.44,
            "team2_clutch_winrate": 0.61,
            "team1_star_rating": 1.05,
            "team2_star_rating": 1.29,
            "h2h_team1_winrate": 0.33,
            "odds_team1": 2.4,
            "odds_team2": 1.55,
            "swiss_round": 1,
            "team1_wins": 0,
            "team1_losses": 0,
            "team2_wins": 0,
            "team2_losses": 0,
        },
        {
            "date": "2026-01-03",
            "event": "Showmatch Cup",
            "event_tier": "C",
            "status": "completed",
            "team1": "Echo Academy",
            "team2": "Foxtrot",
            "winner": "Foxtrot",
            "best_of": 1,
            "map": "dust2",
            "team1_is_secondary": True,
            "team2_is_secondary": False,
            "team1_rating": 9.9,
            "team2_rating": 1.0,
            "team1_kd": 9.9,
            "team2_kd": 1.0,
        },
        {
            "date": "2026-05-22",
            "event": "IEM Dallas",
            "event_tier": "S",
            "status": "forfeit",
            "team1": "Golf",
            "team2": "Hotel",
            "winner": "Golf",
            "best_of": 1,
        },
    ]


class CoreWorkflowTests(unittest.TestCase):
    def test_clean_matches_filters_noise_and_fills_neutral_h2h(self):
        from cs2pickem.cleaning import clean_matches

        cleaned = clean_matches(sample_matches(), reference_date="2026-05-31")

        self.assertEqual([row["team1"] for row in cleaned], ["Alpha", "Charlie"])
        self.assertEqual(cleaned[0]["h2h_team1_winrate"], 0.5)
        self.assertNotIn("Echo Academy", {row["team1"] for row in cleaned})

    def test_clean_matches_filters_temporary_mix_teams(self):
        from cs2pickem.cleaning import clean_matches

        rows = [
            {
                **sample_matches()[0],
                "team1": "Alpha Mix",
                "team2": "Bravo",
                "winner": "Bravo",
                "team1_is_secondary": False,
            },
            {
                **sample_matches()[0],
                "team1": "Charlie",
                "team2": "Delta",
                "winner": "Charlie",
                "team1_is_temporary": True,
            },
            {
                **sample_matches()[0],
                "team1": "Echo",
                "team2": "Foxtrot",
                "winner": "Echo",
            },
        ]

        cleaned = clean_matches(rows, reference_date="2026-05-31")

        self.assertEqual([row["team1"] for row in cleaned], ["Echo"])

    def test_clean_matches_parses_string_secondary_and_temporary_flags(self):
        from cs2pickem.cleaning import clean_matches

        rows = [
            {
                **sample_matches()[0],
                "team1": "Alpha",
                "team2": "Bravo",
                "winner": "Alpha",
                "team1_is_secondary": "False",
                "team2_is_temporary": "0",
            },
            {
                **sample_matches()[0],
                "team1": "Charlie",
                "team2": "Delta",
                "winner": "Charlie",
                "team1_is_secondary": "true",
            },
            {
                **sample_matches()[0],
                "team1": "Echo",
                "team2": "Foxtrot",
                "winner": "Echo",
                "team2_is_temporary": "yes",
            },
        ]

        cleaned = clean_matches(rows, reference_date="2026-05-31")

        self.assertEqual([row["team1"] for row in cleaned], ["Alpha"])

    def test_clean_matches_filters_restart_status_variants(self):
        from cs2pickem.cleaning import clean_matches

        rows = [
            {
                **sample_matches()[0],
                "team1": "RestartedMatch",
                "team2": "Bravo",
                "winner": "RestartedMatch",
                "status": " restarted ",
            },
            {
                **sample_matches()[0],
                "team1": "RematchRequired",
                "team2": "Delta",
                "winner": "RematchRequired",
                "status": "restart",
            },
            {
                **sample_matches()[0],
                "team1": "CleanMatch",
                "team2": "Foxtrot",
                "winner": "CleanMatch",
                "status": "completed",
            },
        ]

        cleaned = clean_matches(rows, reference_date="2026-05-31")

        self.assertEqual([row["team1"] for row in cleaned], ["CleanMatch"])

    def test_clean_matches_accepts_whitespace_padded_event_tiers(self):
        from cs2pickem.cleaning import clean_matches

        rows = [
            {
                **sample_matches()[0],
                "team1": "Alpha",
                "team2": "Bravo",
                "winner": "Alpha",
                "event_tier": " s ",
            },
            {
                **sample_matches()[0],
                "team1": "Charlie",
                "team2": "Delta",
                "winner": "Charlie",
                "event_tier": "\tA\n",
            },
            {
                **sample_matches()[0],
                "team1": "Echo",
                "team2": "Foxtrot",
                "winner": "Echo",
                "event_tier": "B",
            },
        ]

        cleaned = clean_matches(rows, reference_date="2026-05-31")

        self.assertEqual([row["team1"] for row in cleaned], ["Alpha", "Charlie"])

    def test_clean_matches_filters_rating_and_kd_statistical_outliers(self):
        from cs2pickem.cleaning import clean_matches

        rows = [
            {
                **sample_matches()[0],
                "team1": f"Alpha{index}",
                "team2": f"Bravo{index}",
                "winner": f"Alpha{index}",
                "team1_rating": rating,
                "team1_kd": kd,
            }
            for index, (rating, kd) in enumerate(((1.0, 1.0), (1.02, 1.01), (0.98, 0.99), (3.8, 3.7)), start=1)
        ]

        cleaned = clean_matches(rows, reference_date="2026-05-31")

        self.assertEqual([row["team1"] for row in cleaned], ["Alpha1", "Alpha2", "Alpha3"])

    def test_feature_builder_outputs_model_ready_rows_and_labels(self):
        from cs2pickem.cleaning import clean_matches
        from cs2pickem.features import FeatureBuilder

        cleaned = clean_matches(sample_matches(), reference_date="2026-05-31")
        builder = FeatureBuilder()
        dataset = builder.fit_transform(cleaned)

        self.assertEqual(dataset.labels, [1, 0])
        self.assertIn("rank_diff", dataset.feature_names)
        self.assertIn("map_winrate_diff", dataset.feature_names)
        self.assertIn("is_bo1", dataset.feature_names)
        self.assertTrue(all(0.0 <= value <= 1.0 for row in dataset.rows for value in row))

    def test_time_split_and_folds_preserve_chronology(self):
        from cs2pickem.splitting import time_series_date_split, time_series_folds, time_series_split

        rows = [{"date": f"2026-05-{day:02d}", "value": day} for day in range(1, 11)]
        split = time_series_split(rows, train_ratio=0.6, validation_ratio=0.2)
        self.assertEqual([r["value"] for r in split.train], [1, 2, 3, 4, 5, 6])
        self.assertEqual([r["value"] for r in split.validation], [7, 8])
        self.assertEqual([r["value"] for r in split.test], [9, 10])

        compact_split = time_series_split(rows[:6], train_ratio=0.8, validation_ratio=0.1)
        self.assertEqual([r["value"] for r in compact_split.train], [1, 2, 3, 4])
        self.assertEqual([r["value"] for r in compact_split.validation], [5])
        self.assertEqual([r["value"] for r in compact_split.test], [6])

        date_split = time_series_date_split(rows, train_end_date="2026-05-04", validation_end_date="2026-05-07")
        self.assertEqual([r["value"] for r in date_split.train], [1, 2, 3, 4])
        self.assertEqual([r["value"] for r in date_split.validation], [5, 6, 7])
        self.assertEqual([r["value"] for r in date_split.test], [8, 9, 10])

        folds = list(time_series_folds(rows, folds=3))
        self.assertEqual(len(folds), 3)
        for train, validation in folds:
            self.assertLess(max(r["value"] for r in train), min(r["value"] for r in validation))

    def test_weighted_ensemble_trains_and_returns_probabilities(self):
        from cs2pickem.cleaning import clean_matches
        from cs2pickem.features import FeatureBuilder
        from cs2pickem.models import default_ensemble

        cleaned = clean_matches(sample_matches(), reference_date="2026-05-31")
        dataset = FeatureBuilder().fit_transform(cleaned)
        model = default_ensemble(seed=7, epochs=5)
        model.fit(dataset.rows, dataset.labels)
        probabilities = model.predict_proba(dataset.rows)
        component_probabilities = model.predict_components(dataset.rows)

        self.assertEqual(len(probabilities), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probabilities))
        self.assertAlmostEqual(sum(model.weights.values()), 1.0)
        self.assertEqual(set(component_probabilities), {"logistic", "random_forest", "xgboost", "neural_network"})
        self.assertTrue(all(len(values) == len(probabilities) for values in component_probabilities.values()))
        for row_index, probability in enumerate(probabilities):
            blended = sum(model.weights[name] * component_probabilities[name][row_index] for name in model.weights)
            self.assertAlmostEqual(probability, blended)

    def test_swiss_simulation_returns_pickem_relevant_distributions(self):
        from cs2pickem.swiss import TeamSeed, simulate_swiss

        teams = [TeamSeed("Alpha", 1), TeamSeed("Bravo", 2), TeamSeed("Charlie", 3), TeamSeed("Delta", 4)]

        def predictor(team_a, team_b, best_of, state):
            return 0.8 if team_a.seed < team_b.seed else 0.2

        result = simulate_swiss(teams, predictor, simulations=80, seed=3)

        self.assertEqual(set(result.team_probabilities), {"Alpha", "Bravo", "Charlie", "Delta"})
        self.assertGreater(result.team_probabilities["Alpha"]["advance"], 0.5)
        self.assertIn("3-0", result.team_probabilities["Alpha"])
        self.assertIn("0-3", result.team_probabilities["Delta"])

    def test_strategy_adjusts_odds_and_outputs_json_pickems(self):
        from cs2pickem.strategy import adjust_probability_with_market, choose_pickems, single_match_pick

        adjusted = adjust_probability_with_market(0.6, odds_team1=2.2, odds_team2=1.7)
        self.assertLess(adjusted, 0.6)
        self.assertEqual(single_match_pick(0.515, "Alpha", "Bravo"), "avoid")
        self.assertEqual(single_match_pick(0.52, "Alpha", "Bravo"), "avoid")
        self.assertEqual(single_match_pick(0.48, "Alpha", "Bravo"), "avoid")
        self.assertEqual(single_match_pick(0.521, "Alpha", "Bravo"), "Alpha")
        self.assertEqual(single_match_pick(0.479, "Alpha", "Bravo"), "Bravo")
        self.assertEqual(single_match_pick(0.63, "Alpha", "Bravo"), "Alpha")

        probabilities = {
            "Alpha": {"3-0": 0.42, "advance": 0.91, "0-3": 0.01, "eliminate": 0.09},
            "Bravo": {"3-0": 0.27, "advance": 0.78, "0-3": 0.03, "eliminate": 0.22},
            "Charlie": {"3-0": 0.12, "advance": 0.63, "0-3": 0.15, "eliminate": 0.37},
            "Delta": {"3-0": 0.03, "advance": 0.22, "0-3": 0.48, "eliminate": 0.78},
        }
        pickems = choose_pickems(probabilities, rankings={"Alpha": 1, "Bravo": 8, "Charlie": 24, "Delta": 31}, slots={"3-0": 1, "advance": 2, "0-3": 1})

        self.assertEqual(pickems["3-0"], ["Alpha"])
        self.assertEqual(pickems["0-3"], ["Delta"])
        json.dumps(pickems)

    def test_csv_training_and_team_simulation_workflows(self):
        import tempfile

        from cs2pickem.data import read_matches_csv, read_teams_csv
        from cs2pickem.pipeline import simulate_from_team_rows, train_evaluate

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            teams_path = os.path.join(tmpdir, "teams.csv")
            with open(matches_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "date,event,event_tier,status,team1,team2,winner,best_of,map,team1_rank,team2_rank,"
                    "team1_recent_winrate_10,team2_recent_winrate_10,team1_map_winrate,team2_map_winrate,"
                    "team1_rating,team2_rating,team1_kd,team2_kd,odds_team1,odds_team2\n"
                )
                for row in sample_matches()[:2]:
                    handle.write(
                        f"{row['date']},{row['event']},{row['event_tier']},{row['status']},{row['team1']},{row['team2']},"
                        f"{row['winner']},{row['best_of']},{row['map']},{row['team1_rank']},{row['team2_rank']},"
                        f"{row['team1_recent_winrate_10']},{row['team2_recent_winrate_10']},{row['team1_map_winrate']},"
                        f"{row['team2_map_winrate']},{row['team1_rating']},{row['team2_rating']},{row['team1_kd']},"
                        f"{row['team2_kd']},{row['odds_team1']},{row['odds_team2']}\n"
                    )
            with open(teams_path, "w", encoding="utf-8") as handle:
                handle.write("team,seed,strength\nAlpha,1,0.82\nBravo,2,0.62\nCharlie,3,0.48\nDelta,4,0.35\n")

            matches = read_matches_csv(matches_path)
            teams = read_teams_csv(teams_path)
            report = train_evaluate(matches, reference_date="2026-05-31", epochs=4, top_k=5)
            simulation = simulate_from_team_rows(teams, simulations=40, seed=9)

        self.assertEqual(matches[0]["best_of"], 1)
        self.assertEqual(teams[0]["seed"], 1)
        self.assertIn("metrics", report)
        self.assertIn("accuracy", report["metrics"])
        self.assertLessEqual(len(report["selected_feature_names"]), 5)
        self.assertIn("feature_importance", report)
        self.assertIn("pickems", simulation)


if __name__ == "__main__":
    unittest.main()
