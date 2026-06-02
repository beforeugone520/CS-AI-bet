import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def chronological_matches():
    rows = []
    teams = [("Alpha", "Bravo"), ("Charlie", "Delta"), ("Echo", "Foxtrot"), ("Golf", "Hotel")]
    for index in range(12):
        team1, team2 = teams[index % len(teams)]
        strong_team1 = index % 3 != 1
        rows.append(
            {
                "date": f"2026-05-{index + 1:02d}",
                "event": "IEM Cologne Qualifier",
                "event_tier": "S" if index % 2 == 0 else "A",
                "status": "completed",
                "team1": team1,
                "team2": team2,
                "winner": team1 if strong_team1 else team2,
                "best_of": 1 if index % 2 == 0 else 3,
                "map": "mirage" if index % 2 == 0 else "inferno",
                "team1_rank": 5 if strong_team1 else 24,
                "team2_rank": 24 if strong_team1 else 5,
                "team1_rmr_points": 900 if strong_team1 else 420,
                "team2_rmr_points": 420 if strong_team1 else 900,
                "team1_recent_winrate_10": 0.78 if strong_team1 else 0.42,
                "team2_recent_winrate_10": 0.42 if strong_team1 else 0.78,
                "team1_bo1_winrate_6m": 0.74 if strong_team1 else 0.45,
                "team2_bo1_winrate_6m": 0.45 if strong_team1 else 0.74,
                "team1_bo3_winrate_6m": 0.76 if strong_team1 else 0.44,
                "team2_bo3_winrate_6m": 0.44 if strong_team1 else 0.76,
                "team1_map_winrate": 0.7 if strong_team1 else 0.46,
                "team2_map_winrate": 0.46 if strong_team1 else 0.7,
                "team1_rating": 1.18 if strong_team1 else 1.0,
                "team2_rating": 1.0 if strong_team1 else 1.18,
                "team1_kd": 1.15 if strong_team1 else 0.98,
                "team2_kd": 0.98 if strong_team1 else 1.15,
                "team1_opening_success": 0.56 if strong_team1 else 0.47,
                "team2_opening_success": 0.47 if strong_team1 else 0.56,
                "team1_clutch_winrate": 0.61 if strong_team1 else 0.44,
                "team2_clutch_winrate": 0.44 if strong_team1 else 0.61,
                "team1_star_rating": 1.31 if strong_team1 else 1.05,
                "team2_star_rating": 1.05 if strong_team1 else 1.31,
                "h2h_team1_winrate": 0.6 if strong_team1 else 0.4,
                "odds_team1": 1.55 if strong_team1 else 2.45,
                "odds_team2": 2.45 if strong_team1 else 1.55,
                "swiss_round": (index % 5) + 1,
                "team1_wins": min(2, index % 3),
                "team1_losses": min(2, (index + 1) % 3),
                "team2_wins": min(2, (index + 1) % 3),
                "team2_losses": min(2, index % 3),
                "version_tag": "patch-a" if index < 6 else "patch-b",
            }
        )
    return rows


def six_month_matches():
    rows = []
    for index, month_day in enumerate(
        [
            "2026-01-05",
            "2026-01-20",
            "2026-02-10",
            "2026-03-05",
            "2026-03-20",
            "2026-04-05",
            "2026-04-20",
            "2026-05-05",
            "2026-05-15",
            "2026-05-25",
        ]
    ):
        strong_team1 = index % 2 == 0
        rows.append(
            {
                "date": month_day,
                "event": "IEM Cologne Qualifier",
                "event_tier": "S",
                "status": "completed",
                "team1": "Alpha" if strong_team1 else "Bravo",
                "team2": "Bravo" if strong_team1 else "Alpha",
                "winner": "Alpha",
                "best_of": 1 if index % 2 == 0 else 3,
                "map": "mirage",
                "team1_rank": 5 if strong_team1 else 20,
                "team2_rank": 20 if strong_team1 else 5,
                "team1_recent_winrate_10": 0.75 if strong_team1 else 0.45,
                "team2_recent_winrate_10": 0.45 if strong_team1 else 0.75,
                "team1_map_winrate": 0.7 if strong_team1 else 0.5,
                "team2_map_winrate": 0.5 if strong_team1 else 0.7,
                "team1_rating": 1.15 if strong_team1 else 1.02,
                "team2_rating": 1.02 if strong_team1 else 1.15,
                "team1_kd": 1.12 if strong_team1 else 0.98,
                "team2_kd": 0.98 if strong_team1 else 1.12,
                "odds_team1": 1.7 if strong_team1 else 2.1,
                "odds_team2": 2.1 if strong_team1 else 1.7,
            }
        )
    return rows


class TrainingEvaluationTests(unittest.TestCase):
    def test_train_evaluate_uses_time_split_cv_segments_and_model_comparison(self):
        from cs2pickem.pipeline import train_evaluate

        report = train_evaluate(
            chronological_matches(),
            reference_date="2026-05-31",
            epochs=4,
            top_k=6,
            cv_folds=3,
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        self.assertEqual(report["split_counts"], {"train": 7, "validation": 2, "test": 3})
        self.assertLess(report["leakage_guard"]["max_train_date"], report["leakage_guard"]["min_validation_date"])
        self.assertLess(report["leakage_guard"]["max_validation_date"], report["leakage_guard"]["min_test_date"])
        self.assertEqual(len(report["cv_metrics"]), 3)
        self.assertIn("BO1", report["segment_metrics"])
        self.assertIn("BO3", report["segment_metrics"])
        self.assertEqual(
            set(report["model_comparison"]),
            {"logistic", "random_forest", "xgboost", "neural_network", "ensemble"},
        )
        self.assertLessEqual(len(report["selected_feature_names"]), 6)
        self.assertEqual([row["date"] for row in report["probabilities"]], ["2026-05-10", "2026-05-11", "2026-05-12"])

    def test_train_evaluate_handles_sparse_cv_folds_without_feature_variance(self):
        from cs2pickem.pipeline import train_evaluate

        rows = chronological_matches()[:6]
        report = train_evaluate(rows, reference_date="2026-05-31", epochs=3, top_k=10, cv_folds=3, train_ratio=0.8, validation_ratio=0.1)

        self.assertEqual(report["cleaned_matches"], 6)
        self.assertIn("metrics", report)
        self.assertIn("cv_metrics", report)

    def test_train_evaluate_in_sample_report_keeps_split_metadata(self):
        from cs2pickem.pipeline import train_evaluate

        report = train_evaluate(
            chronological_matches()[:2],
            reference_date="2026-05-31",
            epochs=3,
            top_k=8,
            max_age_days=180,
        )

        self.assertEqual(report["max_age_days"], 180)
        self.assertEqual(report["split_strategy"], "in_sample")
        self.assertEqual(report["split_boundaries"], {"train_end_date": None, "validation_end_date": None})

    def test_train_evaluate_reports_imbalance_strategy_for_training_split(self):
        from cs2pickem.pipeline import train_evaluate

        report = train_evaluate(
            chronological_matches(),
            reference_date="2026-05-31",
            epochs=3,
            top_k=8,
            cv_folds=3,
            train_ratio=0.75,
            validation_ratio=0.1,
        )

        self.assertEqual(report["imbalance"]["strategy"], "smote_minority_oversample_plus_class_weight")
        self.assertGreater(report["imbalance"]["synthetic_rows"], 0)
        self.assertEqual(report["imbalance"]["balanced_counts"]["0"], report["imbalance"]["balanced_counts"]["1"])

    def test_train_evaluate_reports_validation_tuned_ensemble_weights(self):
        from cs2pickem.pipeline import train_evaluate

        report = train_evaluate(
            chronological_matches(),
            reference_date="2026-05-31",
            epochs=3,
            top_k=8,
            cv_folds=3,
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        expected_models = {"logistic", "random_forest", "xgboost", "neural_network"}
        self.assertEqual(set(report["ensemble_weights"]), expected_models)
        self.assertAlmostEqual(sum(report["ensemble_weights"].values()), 1.0)
        tuning = report["validation_tuned_ensemble_weights"]
        self.assertEqual(tuning["basis"], "validation_log_loss")
        self.assertEqual(tuning["validation_count"], 2)
        self.assertEqual(set(tuning["weights"]), expected_models)
        self.assertAlmostEqual(sum(tuning["weights"].values()), 1.0)
        self.assertTrue(all(value > 0 for value in tuning["weights"].values()))

    def test_train_evaluate_reports_objective_model_hyperparameters(self):
        from cs2pickem.pipeline import train_evaluate

        report = train_evaluate(
            chronological_matches(),
            reference_date="2026-05-31",
            epochs=7,
            top_k=8,
            cv_folds=3,
            train_ratio=0.6,
            validation_ratio=0.2,
        )

        hyperparameters = report["model_hyperparameters"]
        self.assertEqual(hyperparameters["logistic"]["learning_rate"], 0.08)
        self.assertEqual(hyperparameters["random_forest"]["trees"], 150)
        self.assertEqual(hyperparameters["random_forest"]["max_depth"], 8)
        self.assertEqual(hyperparameters["random_forest"]["min_leaf_samples"], 3)
        self.assertEqual(hyperparameters["xgboost"]["rounds"], 120)
        self.assertEqual(hyperparameters["xgboost"]["learning_rate"], 0.08)
        self.assertEqual(hyperparameters["xgboost"]["max_depth"], 6)
        self.assertEqual(hyperparameters["xgboost"]["subsample"], 0.8)
        self.assertEqual(hyperparameters["neural_network"]["hidden_layers"], [64, 32])
        self.assertEqual(hyperparameters["neural_network"]["epochs"], 7)

    def test_train_evaluate_can_keep_six_month_window_for_cologne_split(self):
        from cs2pickem.pipeline import train_evaluate

        default_window = train_evaluate(six_month_matches(), reference_date="2026-05-31", epochs=3, top_k=8)
        six_month_window = train_evaluate(
            six_month_matches(),
            reference_date="2026-05-31",
            epochs=3,
            top_k=8,
            max_age_days=180,
            train_ratio=0.8,
            validation_ratio=0.1,
        )

        self.assertLess(default_window["cleaned_matches"], 10)
        self.assertEqual(six_month_window["cleaned_matches"], 10)
        self.assertTrue(six_month_window["leakage_guard"]["max_train_date"].startswith("2026-05") or six_month_window["leakage_guard"]["max_train_date"].startswith("2026-04"))
        self.assertEqual(six_month_window["probabilities"][0]["date"], "2026-05-25")

    def test_train_evaluate_supports_explicit_cologne_calendar_boundaries(self):
        from cs2pickem.pipeline import train_evaluate

        report = train_evaluate(
            six_month_matches(),
            reference_date="2026-05-31",
            epochs=3,
            top_k=8,
            max_age_days=180,
            train_end_date="2026-04-30",
            validation_end_date="2026-05-15",
        )

        self.assertEqual(report["split_strategy"], "date_boundaries")
        self.assertEqual(report["split_boundaries"], {"train_end_date": "2026-04-30", "validation_end_date": "2026-05-15"})
        self.assertEqual(report["split_counts"], {"train": 7, "validation": 2, "test": 1})
        self.assertEqual(report["leakage_guard"]["max_train_date"], "2026-04-20")
        self.assertEqual(report["leakage_guard"]["min_validation_date"], "2026-05-05")
        self.assertEqual(report["leakage_guard"]["max_validation_date"], "2026-05-15")
        self.assertEqual(report["leakage_guard"]["min_test_date"], "2026-05-25")
        self.assertEqual([row["date"] for row in report["probabilities"]], ["2026-05-25"])

    def test_train_evaluate_requires_complete_calendar_boundaries(self):
        from cs2pickem.pipeline import train_evaluate

        with self.assertRaisesRegex(ValueError, "both train_end_date and validation_end_date"):
            train_evaluate(
                six_month_matches(),
                reference_date="2026-05-31",
                epochs=3,
                top_k=8,
                max_age_days=180,
                train_end_date="2026-04-30",
            )

    def test_train_evaluate_rejects_empty_calendar_split_buckets(self):
        from cs2pickem.pipeline import train_evaluate

        with self.assertRaisesRegex(ValueError, "date boundary split produced empty"):
            train_evaluate(
                chronological_matches()[:6],
                reference_date="2026-05-31",
                epochs=3,
                top_k=8,
                max_age_days=180,
                train_end_date="2026-04-30",
                validation_end_date="2026-05-15",
            )


if __name__ == "__main__":
    unittest.main()
