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

    def test_market_fusion_defaults_to_legacy_method(self):
        from cs2pickem.tuning import optimize_match_predictions

        report = optimize_match_predictions(
            chronological_matches(),
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 8, "epochs": 3, "weights": {"logistic": 1.0}},
            ],
        )
        # Default skeleton: legacy linear blend, frozen model_weight prior reported.
        self.assertEqual(report["fusion_method"], "legacy")
        self.assertAlmostEqual(report["model_weight"], 0.35)
        fusion = report["candidate_results"][0]["market_fusion"]
        self.assertEqual(fusion["fusion_method"], "legacy")
        self.assertEqual(fusion["market_weight"], 0.30)
        self.assertAlmostEqual(fusion["model_weight"], 0.35)

    def test_market_fusion_logit_pool_is_opt_in_and_reported(self):
        from cs2pickem.tuning import optimize_match_predictions

        rows = chronological_matches()
        rows[-2]["odds_team1"] = ""
        rows[-2]["odds_team2"] = ""
        rows[-2]["market_probability_team1"] = 0.72
        rows[-1]["odds_team1"] = 1.50
        rows[-1]["odds_team2"] = 2.80

        report = optimize_match_predictions(
            rows,
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 8, "epochs": 3, "weights": {"logistic": 1.0}},
            ],
            fusion_method="logit_pool",
            model_weight=0.40,
        )
        self.assertEqual(report["fusion_method"], "logit_pool")
        self.assertAlmostEqual(report["model_weight"], 0.40)
        fusion = report["candidate_results"][0]["market_fusion"]
        self.assertEqual(fusion["fusion_method"], "logit_pool")
        self.assertAlmostEqual(fusion["model_weight"], 0.40)
        # Existing keys remain intact regardless of the fusion method.
        self.assertIn("fused_test_metrics", fusion)
        self.assertIn("market_only_test_metrics", fusion)

    def test_unknown_fusion_method_raises(self):
        from cs2pickem.tuning import optimize_match_predictions

        with self.assertRaises(ValueError):
            optimize_match_predictions(
                chronological_matches(),
                reference_date="2026-05-31",
                train_ratio=0.6,
                validation_ratio=0.2,
                max_age_days=180,
                candidate_configs=[
                    {"name": "logistic_small", "top_k": 8, "epochs": 3, "weights": {"logistic": 1.0}},
                ],
                fusion_method="bogus",
            )

    def test_default_grid_axes_keep_names_and_settings_byte_identical(self):
        """WF-2F wiring safety: the new probability/feature grid axes (calibration_method,
        fusion_method, model_weight, include_unverified) all default to the production value,
        so the default grid is byte-identical -- no name drift, no extra candidates, every
        candidate pinned to platt / legacy / model_weight=0.35 / unverified-off."""
        from cs2pickem.tuning import _candidate_grid

        grid = _candidate_grid([8], [3], ["logistic"])
        self.assertEqual([c["name"] for c in grid], ["logistic_k8_e3"])
        only = grid[0]
        self.assertEqual(only["calibration_methods"], ["platt"])
        self.assertEqual(only["fusion_method"], "legacy")
        self.assertAlmostEqual(only["model_weight"], 0.35)
        self.assertFalse(only["include_unverified_features"])

    def test_grid_axes_fan_out_with_disambiguating_suffixes(self):
        """Each new axis fans out the grid AND tags the variant so names never collide."""
        from cs2pickem.tuning import _candidate_grid

        cal = {c["name"] for c in _candidate_grid([8], [3], ["logistic"], calibration_method_modes=["platt", "beta"])}
        self.assertEqual(cal, {"logistic_platt_k8_e3", "logistic_beta_k8_e3"})

        fusion = {c["name"] for c in _candidate_grid([8], [3], ["logistic"], fusion_method_modes=["legacy_clip", "logit_pool"])}
        self.assertEqual(fusion, {"logistic_legacy_k8_e3", "logistic_logit_pool_k8_e3"})

        mw = {c["name"] for c in _candidate_grid([8], [3], ["logistic"], fusion_method_modes=["logit_pool"], model_weight_values=[0.35, 0.5])}
        self.assertEqual(mw, {"logistic_logit_pool_mw035_k8_e3", "logistic_logit_pool_mw05_k8_e3"})

        unv = {c["name"] for c in _candidate_grid([8], [3], ["logistic"], include_unverified_modes=[False, True])}
        self.assertEqual(unv, {"logistic_k8_e3", "logistic_unv_k8_e3"})

    def test_grid_calibration_method_axis_is_same_口径_single_method(self):
        """A pinned calibration method is honoured verbatim (no platt re-prepend) so the
        {platt vs beta vs temperature} A/B compares each calibrator in isolation."""
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
            calibration_method_modes=["platt", "beta"],
        )
        by = {c["name"]: c for c in report["candidate_results"]}
        self.assertEqual(set(by), {"logistic_platt_k8_e3", "logistic_beta_k8_e3"})
        # Each candidate uses EXACTLY its pinned method -- beta is not silently joined by platt.
        self.assertEqual(by["logistic_platt_k8_e3"]["calibration_methods"], ["platt"])
        self.assertEqual(by["logistic_beta_k8_e3"]["calibration_methods"], ["beta"])
        self.assertEqual(by["logistic_platt_k8_e3"]["calibration"]["basis"], "validation_platt_logistic")
        self.assertEqual(by["logistic_beta_k8_e3"]["calibration"]["basis"], "validation_beta")

    def test_grid_fusion_axis_drives_per_candidate_market_fusion(self):
        """fusion_method + model_weight are resolved per candidate and flow into the
        market_fusion block, enabling a same-口径 legacy-vs-logit_pool comparison."""
        from cs2pickem.tuning import optimize_match_predictions

        rows = chronological_matches()
        rows[-1]["odds_team1"] = 1.50
        rows[-1]["odds_team2"] = 2.80
        report = optimize_match_predictions(
            rows,
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            top_k_values=[8],
            epochs_values=[3],
            candidate_names=["logistic"],
            fusion_method_modes=["legacy_clip", "logit_pool"],
            model_weight_values=[0.4],
        )
        by = {c["name"]: c for c in report["candidate_results"]}
        self.assertEqual(by["logistic_legacy_k8_e3"]["market_fusion"]["fusion_method"], "legacy")
        self.assertEqual(by["logistic_logit_pool_k8_e3"]["market_fusion"]["fusion_method"], "logit_pool")
        self.assertAlmostEqual(by["logistic_logit_pool_k8_e3"]["market_fusion"]["model_weight"], 0.4)

    def test_grid_include_unverified_axis_opts_columns_into_candidate_pool(self):
        """include_unverified=True opts the gated columns into the selector's candidate pool;
        the off variant cannot select them. Folding it into the matrix cache key keeps the
        on/off candidates from sharing a feature matrix (anti train/serve skew within backtest)."""
        from cs2pickem.tuning import optimize_match_predictions

        rows = chronological_matches()
        # Give event_grade real variance so the unverified column is selectable when opted in.
        for index, row in enumerate(rows):
            row["event_grade"] = (index % 4) + 1
        report = optimize_match_predictions(
            rows,
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            top_k_values=[30],
            epochs_values=[3],
            candidate_names=["logistic"],
            include_unverified_modes=[False, True],
        )
        by = {c["name"]: c for c in report["candidate_results"]}
        self.assertFalse(by["logistic_k30_e3"]["include_unverified_features"])
        self.assertTrue(by["logistic_unv_k30_e3"]["include_unverified_features"])
        # The off candidate never exposes a gated column; the on candidate can select it.
        self.assertNotIn("event_grade_sum", by["logistic_k30_e3"]["selected_feature_names"])
        self.assertIn("event_grade_sum", by["logistic_unv_k30_e3"]["selected_feature_names"])

    def test_default_calibration_method_keeps_legacy_basis_labels(self):
        from cs2pickem.tuning import optimize_match_predictions

        report = optimize_match_predictions(
            chronological_matches(),
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 8, "epochs": 3, "weights": {"logistic": 1.0}},
            ],
        )
        self.assertEqual(report["calibration_methods"], ["platt"])
        result = report["candidate_results"][0]
        # Single-platt path keeps the historic raw_model / calibrated_model labels
        # and never advertises a multi-method block.
        self.assertIn(result["probability_selection"]["selected_basis"], {"raw_model", "calibrated_model"})
        self.assertNotIn("method_validation_metrics", result["probability_selection"])
        self.assertEqual(result["calibration"]["basis"], "validation_platt_logistic")

    def test_optimize_match_predictions_selects_calibration_method_on_validation(self):
        from cs2pickem.tuning import optimize_match_predictions

        report = optimize_match_predictions(
            chronological_matches(),
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 8, "epochs": 3, "weights": {"logistic": 1.0}},
            ],
            calibration_methods=["platt", "beta", "temperature"],
            probability_objective="log_loss",
        )
        self.assertEqual(report["calibration_methods"], ["platt", "beta", "temperature"])
        selection = report["candidate_results"][0]["probability_selection"]
        # Multi-method run reports every method's VALIDATION metrics and chooses
        # among {raw, platt, beta, temperature} -- never peeking at test labels.
        self.assertEqual(selection["selection_basis"], "validation_only")
        self.assertIn("method_validation_metrics", selection)
        self.assertEqual(
            set(selection["method_validation_metrics"]),
            {"raw", "platt", "beta", "temperature"},
        )
        self.assertIn(selection["selected_method"], {"raw", "platt", "beta", "temperature"})
        # The chosen method must be (weakly) the best on the validation objective.
        chosen = selection["selected_method"]
        chosen_ll = selection["method_validation_metrics"][chosen]["log_loss"]
        for metrics in selection["method_validation_metrics"].values():
            self.assertLessEqual(chosen_ll, metrics["log_loss"] + 1e-9)

    def test_calibration_method_choice_ignores_test_labels(self):
        from cs2pickem.tuning import _probability_selection, _metric_summary

        validation_labels = [1, 1, 0, 0] * 4
        validation_rows = [{} for _ in validation_labels]
        validation_raw = [0.55, 0.55, 0.45, 0.45] * 4
        test_labels = [1, 1, 0, 0] * 4
        test_rows = [{} for _ in test_labels]
        test_raw = [0.55] * 16

        # temperature is confidently correct on validation -> must be selected
        # regardless of how the test labels are arranged.
        temperature_val = [0.95, 0.95, 0.05, 0.05] * 4
        method_candidates = [
            {
                "method": "platt",
                "basis": "calibrated_platt",
                "validation_metrics": _metric_summary(validation_labels, [0.7, 0.7, 0.3, 0.3] * 4, validation_rows),
                "test_probabilities": [0.7] * 16,
            },
            {
                "method": "temperature",
                "basis": "calibrated_temperature",
                "validation_metrics": _metric_summary(validation_labels, temperature_val, validation_rows),
                "test_probabilities": [0.99] * 16,
            },
        ]

        first = _probability_selection(
            validation_labels, validation_raw, [0.7, 0.7, 0.3, 0.3] * 4, validation_rows,
            test_raw, [0.7] * 16, test_labels, test_rows,
            objective="log_loss", calibrated_available=True, method_candidates=method_candidates,
        )
        flipped = _probability_selection(
            validation_labels, validation_raw, [0.7, 0.7, 0.3, 0.3] * 4, validation_rows,
            test_raw, [0.7] * 16, [0, 0, 1, 1] * 4, test_rows,
            objective="log_loss", calibrated_available=True, method_candidates=method_candidates,
        )
        self.assertEqual(first["selected_method"], "temperature")
        self.assertEqual(first["selected_basis"], flipped["selected_basis"])
        self.assertEqual(first["selected_probabilities"], flipped["selected_probabilities"])

    def test_out_of_fold_selection_metric_is_not_in_sample(self):
        import random

        from cs2pickem.tuning import _out_of_fold_validation_probabilities

        # Pure noise: labels independent of probabilities. An IN-SAMPLE beta fit
        # (3 DOF) can drive its own validation log-loss down by over-fitting; the
        # expanding-window OUT-OF-FOLD predictions must NOT reproduce that
        # in-sample optimism (review red-line (b): held-out, not in-sample).
        rng = random.Random(2026)
        probabilities = [rng.uniform(0.1, 0.9) for _ in range(40)]
        labels = [1 if rng.random() < 0.5 else 0 for _ in range(40)]

        out_of_fold = _out_of_fold_validation_probabilities("beta", probabilities, labels)
        self.assertIsNotNone(out_of_fold)
        self.assertEqual(len(out_of_fold), len(probabilities))
        # The out-of-fold vector differs from a full-fit in-sample transform: the
        # earliest rows (before the first fold's train window) stay raw and the
        # held rows are predicted by a calibrator that never saw them.
        from cs2pickem.calibration import make_calibrator

        in_sample = make_calibrator("beta").fit(probabilities, labels).transform(probabilities)
        self.assertNotEqual(out_of_fold, in_sample)

    def test_out_of_fold_selection_falls_back_when_too_few_rows(self):
        from cs2pickem.tuning import _out_of_fold_validation_probabilities

        # Below the fold threshold there is no honest held-out split -> None, so
        # the caller uses the in-sample metric rather than fabricating folds.
        self.assertIsNone(
            _out_of_fold_validation_probabilities("beta", [0.4, 0.6, 0.5], [1, 0, 1])
        )

    def test_multi_method_selection_uses_out_of_fold_basis(self):
        from cs2pickem.tuning import optimize_match_predictions

        report = optimize_match_predictions(
            chronological_matches(),
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 8, "epochs": 3, "weights": {"logistic": 1.0}},
            ],
            calibration_methods=["platt", "beta", "temperature"],
            probability_objective="log_loss",
        )
        selection = report["candidate_results"][0]["probability_selection"]
        # Multi-method runs report every method's (held-out where possible)
        # validation metric and still never peek at test for the choice.
        self.assertEqual(selection["selection_basis"], "validation_only")
        self.assertIn("method_validation_metrics", selection)

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

    def test_probability_selection_follows_validation_not_test(self):
        from cs2pickem.tuning import _probability_selection

        # Build a case where calibrated clearly wins on VALIDATION (lower log_loss)
        # but raw clearly wins on TEST. The selected basis must track validation.
        validation_labels = [1, 1, 0, 0]
        validation_raw = [0.55, 0.55, 0.45, 0.45]  # weak, near 0.5
        validation_calibrated = [0.95, 0.95, 0.05, 0.05]  # confident + correct
        validation_rows = [{} for _ in validation_labels]

        test_labels = [1, 1, 0, 0]
        # On TEST, calibrated is confidently WRONG while raw is mild + correct.
        test_raw = [0.55, 0.55, 0.45, 0.45]
        test_calibrated = [0.05, 0.05, 0.95, 0.95]
        test_rows = [{} for _ in test_labels]

        selection = _probability_selection(
            validation_labels,
            validation_raw,
            validation_calibrated,
            validation_rows,
            test_raw,
            test_calibrated,
            test_labels,
            test_rows,
            objective="log_loss",
            calibrated_available=True,
        )
        # Validation prefers calibrated -> chosen, even though test would prefer raw.
        self.assertEqual(selection["selected_basis"], "calibrated_model")
        self.assertEqual(selection["selection_basis"], "validation_only")
        self.assertEqual(selection["selected_probabilities"], list(test_calibrated))

        # Permuting/replacing the test labels must NOT change the chosen basis,
        # proving test labels never participate in the selection.
        permuted = _probability_selection(
            validation_labels,
            validation_raw,
            validation_calibrated,
            validation_rows,
            test_raw,
            test_calibrated,
            [0, 0, 1, 1],  # flipped test labels
            test_rows,
            objective="log_loss",
            calibrated_available=True,
        )
        self.assertEqual(permuted["selected_basis"], selection["selected_basis"])
        self.assertEqual(permuted["selected_probabilities"], selection["selected_probabilities"])

    def test_report_tags_test_oracle_outputs_as_diagnostic_only(self):
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

        # Authoritative selection is validation-based and explicitly marked.
        self.assertEqual(report["selection_basis"], "validation")
        self.assertEqual(report["best_by_validation_accuracy"]["selection_basis"], "validation")
        self.assertEqual(report["best_by_validation_log_loss"]["selection_basis"], "validation")
        self.assertEqual(report["authoritative_best"]["name"], report["best_by_validation_accuracy"]["name"])
        # Test-oracle picks are tagged as diagnostic and must not be treated as a selection.
        self.assertEqual(
            report["best_by_test_accuracy"]["selection_basis"],
            "test_oracle_diagnostic_do_not_use_for_model_selection",
        )
        self.assertEqual(
            report["best_by_test_log_loss"]["selection_basis"],
            "test_oracle_diagnostic_do_not_use_for_model_selection",
        )
        # The reported test_predictions come from the validation-selected best.
        self.assertEqual(len(report["test_predictions"]), 3)

    def test_default_rating_mode_is_elo_and_behaviour_unchanged(self):
        """WF-2C: the rating_mode axis defaults to 'elo' so the existing Elo-only path is
        unchanged -- no glicko injection, no glicko columns selected, candidate names carry
        no rating suffix, and each candidate reports rating_mode='elo'."""
        from cs2pickem.tuning import optimize_match_predictions

        report = optimize_match_predictions(
            chronological_matches(),
            reference_date="2026-05-31",
            train_ratio=0.6,
            validation_ratio=0.2,
            max_age_days=180,
            candidate_configs=[
                {"name": "logistic_small", "top_k": 6, "epochs": 3, "weights": {"logistic": 1.0}},
            ],
        )
        result = report["candidate_results"][0]
        self.assertEqual(result["rating_mode"], "elo")
        # Glicko is NOT injected by default -> its columns are constant 0/700 -> never selected.
        self.assertNotIn("glicko_diff", result["selected_feature_names"])
        self.assertNotIn("glicko_rd_sum", result["selected_feature_names"])
        # Authoritative selection carries the rating_mode through for downstream visibility.
        self.assertEqual(report["authoritative_best"]["rating_mode"], "elo")

    def test_default_candidate_grid_has_no_rating_suffix(self):
        """The default grid (rating_modes unset) emits a single 'elo' mode with no name
        suffix, so existing candidate names are byte-identical."""
        from cs2pickem.tuning import _candidate_grid

        grid = _candidate_grid([8], [3], ["logistic"])
        self.assertTrue(all(c["rating_mode"] == "elo" for c in grid))
        self.assertTrue(all("glicko" not in c["name"] and "_elo" not in c["name"] for c in grid))

    def test_single_glicko_mode_grid_carries_disambiguating_suffix(self):
        """WF-2C review fix: a glicko-only single-mode grid must still suffix '_glicko' so
        its candidate names never collide with the byte-identical default elo names in
        reports (the internal rating_mode field disambiguated, but human-readable names did
        not). The default elo single-mode path stays unsuffixed."""
        from cs2pickem.tuning import _candidate_grid

        grid = _candidate_grid([8], [3], ["logistic"], rating_modes=["glicko"])
        self.assertTrue(all(c["rating_mode"] == "glicko" for c in grid))
        self.assertTrue(all(c["name"].startswith("logistic_glicko_") for c in grid))
        # And the default elo single-mode grid is unchanged (no suffix).
        elo_grid = _candidate_grid([8], [3], ["logistic"], rating_modes=["elo"])
        self.assertTrue(all(c["name"] == "logistic_k8_e3" for c in elo_grid))

    def test_rating_mode_normalization_and_config_default(self):
        from cs2pickem.tuning import (
            RATING_MODES,
            _DEFAULT_RATING_MODE,
            _config_rating_mode,
            _normalize_rating_mode,
        )

        self.assertEqual(_DEFAULT_RATING_MODE, "elo")
        self.assertEqual(set(RATING_MODES), {"elo", "glicko"})
        self.assertEqual(_normalize_rating_mode("elo"), "elo")
        self.assertEqual(_normalize_rating_mode("Glicko-2"), "glicko")
        self.assertEqual(_normalize_rating_mode("glicko2"), "glicko")
        # Missing / empty rating_mode in a config falls back to the default 'elo'.
        self.assertEqual(_config_rating_mode({}), "elo")
        self.assertEqual(_config_rating_mode({"rating_mode": ""}), "elo")
        self.assertEqual(_config_rating_mode({"rating_mode": "glicko"}), "glicko")
        with self.assertRaises(ValueError):
            _normalize_rating_mode("trueskill")

    def test_optimize_match_predictions_can_compare_elo_and_glicko_rating_modes(self):
        """WF-2C skeleton: the backtest can switch the rating source. Both candidates run;
        the glicko candidate injects Glicko-2 (so glicko_diff becomes a live candidate), the
        elo candidate does not. A/B significance is left to WF-2F -- here we only verify the
        switch is wired and reported."""
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
            rating_modes=["elo", "glicko"],
        )
        by_name = {result["name"]: result for result in report["candidate_results"]}
        self.assertEqual(set(by_name), {"logistic_elo_k8_e3", "logistic_glicko_k8_e3"})
        self.assertEqual(by_name["logistic_elo_k8_e3"]["rating_mode"], "elo")
        self.assertEqual(by_name["logistic_glicko_k8_e3"]["rating_mode"], "glicko")
        # The elo candidate never injects Glicko -> glicko_diff cannot be selected for it.
        self.assertNotIn("glicko_diff", by_name["logistic_elo_k8_e3"]["selected_feature_names"])

    def test_glicko_rating_mode_routes_through_leakage_free_injection(self):
        """Anti-leakage contract: selecting rating_mode='glicko' must route through the
        period-batched, pre-match Glicko-2 injection (same engine test_reliability locks),
        reported via the glicko feature-preparation basis -- never a same-day-result peek."""
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
            rating_modes=["glicko"],
        )
        result = report["candidate_results"][0]
        self.assertEqual(result["rating_mode"], "glicko")
        glicko_prep = result["feature_preparation"]["glicko"]
        # The glicko candidate is fed by the rolling pre-match snapshot engine, not 'not_applied'.
        self.assertEqual(glicko_prep["basis"], "chronological_pre_match_rolling")
        # Elo stays injected too (rating_mode swaps the *added* engine, not the Elo baseline).
        self.assertEqual(result["feature_preparation"]["elo"]["basis"], "chronological_pre_match_online")

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
