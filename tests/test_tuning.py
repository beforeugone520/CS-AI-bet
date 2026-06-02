import contextlib
import io
import json
import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tests.test_training_evaluation import chronological_matches


class MatchTuningTests(unittest.TestCase):
    def test_optimize_match_predictions_replays_holdout_and_selects_best_config(self):
        from cs2pickem.tuning import optimize_match_predictions

        report = optimize_match_predictions(
            chronological_matches(),
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 6, "epochs": 3, "weights": {"logistic": 1.0}},
                {"name": "forest_small", "top_k": 6, "epochs": 3, "weights": {"random_forest": 1.0}},
            ],
        )

        self.assertEqual(report["split_counts"], {"train": 7, "validation": 2, "test": 3})
        self.assertEqual(len(report["candidate_results"]), 2)
        self.assertIn(report["best_by_validation_accuracy"]["name"], {"logistic_small", "forest_small"})
        self.assertIn(report["best_by_test_accuracy"]["name"], {"logistic_small", "forest_small"})
        self.assertIn("accuracy_delta_vs_baseline", report["best_by_validation_accuracy"])
        self.assertEqual(len(report["test_predictions"]), 3)
        self.assertEqual(len(report["best_test_accuracy_predictions"]), 3)
        self.assertIn("predicted_winner", report["test_predictions"][0])
        self.assertEqual(report["feature_preparation"]["elo"]["basis"], "chronological_pre_match_online")
        for result in report["candidate_results"]:
            self.assertIn("rolling_validation", result)
            self.assertGreaterEqual(result["rolling_validation"]["folds"], 1)
            self.assertTrue(set(result["excluded_feature_names"]).issuperset({"team1_code", "team2_code", "event_code", "version_tag_code"}))
            self.assertFalse({"team1_code", "team2_code", "event_code", "version_tag_code"} & set(result["selected_feature_names"]))

    def test_optimize_match_predictions_reports_calibration_choice_and_market_fusion(self):
        from cs2pickem.tuning import optimize_match_predictions

        rows = chronological_matches()
        rows[-2]["odds_team1"] = ""
        rows[-2]["odds_team2"] = ""
        rows[-2]["market_probability_team1"] = 0.72
        rows[-2]["market_signal_source"] = "closing_consensus"
        rows[-1]["odds_team1"] = ""
        rows[-1]["odds_team2"] = ""
        rows[-1]["hltv_poll_team1"] = 25
        rows[-1]["hltv_poll_team2"] = 75
        rows[-1]["market_proxy_source"] = "hltv_fan_poll_not_odds"

        report = optimize_match_predictions(
            rows,
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 8, "epochs": 3, "weights": {"logistic": 1.0}},
            ],
            market_weight=0.35,
            probability_objective="log_loss",
        )

        result = report["candidate_results"][0]
        self.assertEqual(result["probability_selection"]["objective"], "log_loss")
        self.assertIn(result["probability_selection"]["selected_basis"], {"raw_model", "calibrated_model"})
        self.assertIn("raw_test_metrics", result["probability_selection"])
        self.assertIn("calibrated_test_metrics", result["probability_selection"])
        self.assertEqual(result["market_fusion"]["market_weight"], 0.35)
        self.assertEqual(result["market_fusion"]["test_rows_with_market"], 3)
        self.assertEqual(result["market_fusion"]["signal_counts"]["real_odds"], 1)
        self.assertEqual(result["market_fusion"]["signal_counts"]["explicit_market_probability"], 1)
        self.assertEqual(result["market_fusion"]["signal_counts"]["poll_proxy"], 1)
        self.assertEqual(result["market_fusion"]["proxy_rows"], 1)
        self.assertIn("market_only_test_metrics", result["market_fusion"])
        self.assertIn("fused_test_metrics", result["market_fusion"])
        self.assertIn("market_fusion", report["best_by_validation_log_loss"])

    def test_optimize_match_predictions_can_compare_with_and_without_elo(self):
        from cs2pickem.tuning import optimize_match_predictions

        report = optimize_match_predictions(
            chronological_matches(),
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            top_k_values=[8],
            epochs_values=[3],
            candidate_names=["logistic"],
            elo_modes=["with", "without"],
        )

        candidate_names = {result["name"] for result in report["candidate_results"]}
        self.assertEqual(candidate_names, {"logistic_with_elo_k8_e3", "logistic_without_elo_k8_e3"})
        by_name = {result["name"]: result for result in report["candidate_results"]}
        self.assertEqual(by_name["logistic_with_elo_k8_e3"]["feature_preparation"]["elo"]["basis"], "chronological_pre_match_online")
        self.assertEqual(by_name["logistic_without_elo_k8_e3"]["feature_preparation"]["elo"]["basis"], "not_applied")

    def test_optimize_matches_cli_reads_csv_and_writes_report(self):
        from cs2pickem.cli import main
        from cs2pickem.data import write_matches_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            matches_path = os.path.join(tmpdir, "matches.csv")
            output_path = os.path.join(tmpdir, "tuning-report.json")
            write_matches_csv(matches_path, chronological_matches())
            old_argv = sys.argv
            sys.argv = [
                "cs2pickem",
                "optimize-matches",
                "--matches",
                matches_path,
                "--reference-date",
                "2026-05-31",
                "--train-ratio",
                "0.6",
                "--validation-ratio",
                "0.2",
                "--max-age-days",
                "180",
                "--top-k-values",
                "6",
                "--epochs-values",
                "3",
                "--candidates",
                "logistic,random_forest",
                "--market-weight",
                "0.25",
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
        self.assertEqual(report["matches"], 12)
        self.assertEqual(len(report["candidate_results"]), 2)
        self.assertEqual(report["candidate_results"][0]["market_fusion"]["market_weight"], 0.25)


if __name__ == "__main__":
    unittest.main()
