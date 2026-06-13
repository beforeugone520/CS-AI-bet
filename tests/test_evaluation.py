import math
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class BrierScoreTests(unittest.TestCase):
    def test_brier_score_is_mean_squared_error_of_probabilities(self):
        from cs2pickem.evaluation import brier_score

        self.assertAlmostEqual(brier_score([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]), 0.0)
        self.assertAlmostEqual(brier_score([1, 0], [0.5, 0.5]), 0.25)
        # ((0.8-1)^2 + (0.3-0)^2) / 2 = (0.04 + 0.09) / 2 = 0.065
        self.assertAlmostEqual(brier_score([1, 0], [0.8, 0.3]), 0.065)

    def test_brier_score_handles_empty(self):
        from cs2pickem.evaluation import brier_score

        self.assertEqual(brier_score([], []), 0.0)


class CalibrationTableTests(unittest.TestCase):
    def test_perfectly_calibrated_predictions_have_zero_ece(self):
        from cs2pickem.evaluation import calibration_table

        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        probabilities = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        table = calibration_table(labels, probabilities, bins=2)

        self.assertAlmostEqual(table["ece"], 0.0)
        self.assertEqual(len(table["bins"]), 2)

    def test_overconfident_constant_prediction_has_large_ece(self):
        from cs2pickem.evaluation import calibration_table

        # Predict 0.9 for everything but only half the outcomes happen.
        labels = [1, 0, 1, 0]
        probabilities = [0.9, 0.9, 0.9, 0.9]
        table = calibration_table(labels, probabilities, bins=2)

        # |mean_predicted 0.9 - observed 0.5| = 0.4
        self.assertGreater(table["ece"], 0.3)


class BrierDecompositionTests(unittest.TestCase):
    def test_decomposition_identity_holds(self):
        from cs2pickem.evaluation import brier_decomposition

        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        result = brier_decomposition(labels, probabilities, n_bins=10)
        identity = result["reliability"] - result["resolution"] + result["uncertainty"]
        self.assertAlmostEqual(identity, result["brier"], places=9)

    def test_perfectly_calibrated_separated_predictions(self):
        from cs2pickem.evaluation import brier_decomposition

        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        probabilities = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        result = brier_decomposition(labels, probabilities, n_bins=2)
        self.assertAlmostEqual(result["reliability"], 0.0, places=9)
        self.assertAlmostEqual(result["uncertainty"], 0.25, places=9)
        self.assertAlmostEqual(result["resolution"], 0.25, places=9)
        self.assertAlmostEqual(result["brier"], 0.0, places=9)
        # identity must hold here too
        identity = result["reliability"] - result["resolution"] + result["uncertainty"]
        self.assertAlmostEqual(identity, result["brier"], places=9)

    def test_constant_half_prediction_has_zero_resolution(self):
        from cs2pickem.evaluation import brier_decomposition

        labels = [1, 0, 1, 0]
        probabilities = [0.5, 0.5, 0.5, 0.5]
        result = brier_decomposition(labels, probabilities, n_bins=10)
        self.assertAlmostEqual(result["resolution"], 0.0, places=9)
        self.assertAlmostEqual(result["reliability"], 0.0, places=9)
        self.assertAlmostEqual(result["uncertainty"], 0.25, places=9)
        self.assertAlmostEqual(result["brier"], 0.25, places=9)

    def test_identity_is_exact_when_bins_separate_unique_probs(self):
        from cs2pickem.evaluation import brier_decomposition

        # With probabilities that are exactly 0 or 1, equal-width binning groups
        # all identical predictions together, so the identity is exact.
        labels = [1, 1, 0, 0, 1, 0]
        probabilities = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        result = brier_decomposition(labels, probabilities, n_bins=10)
        identity = result["reliability"] - result["resolution"] + result["uncertainty"]
        self.assertAlmostEqual(identity, result["brier"], places=12)

    def test_reconstructed_is_self_consistent_but_only_approximate_when_coarse(self):
        # With coarse bins a single bin holds several distinct predictions, so
        # the Murphy identity is only an approximation. `reconstructed` must
        # still equal reliability - resolution + uncertainty exactly (key is
        # self-consistent), but must differ from the true Brier score, locking
        # the documented "approximation, not identity" contract.
        from cs2pickem.evaluation import brier_decomposition

        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
        result = brier_decomposition(labels, probabilities, n_bins=2)
        self.assertAlmostEqual(
            result["reconstructed"],
            result["reliability"] - result["resolution"] + result["uncertainty"],
            places=12,
        )
        self.assertGreater(abs(result["reconstructed"] - result["brier"]), 0.0)

    def test_empty_input_returns_zeros(self):
        from cs2pickem.evaluation import brier_decomposition

        result = brier_decomposition([], [], n_bins=10)
        self.assertEqual(result["reliability"], 0.0)
        self.assertEqual(result["resolution"], 0.0)
        self.assertEqual(result["uncertainty"], 0.0)
        self.assertEqual(result["brier"], 0.0)

    def test_all_positive_labels_zero_uncertainty(self):
        from cs2pickem.evaluation import brier_decomposition

        labels = [1, 1, 1, 1]
        probabilities = [0.6, 0.7, 0.8, 0.9]
        result = brier_decomposition(labels, probabilities, n_bins=10)
        self.assertAlmostEqual(result["uncertainty"], 0.0, places=9)
        self.assertAlmostEqual(result["resolution"], 0.0, places=9)


class BrierSkillScoreTests(unittest.TestCase):
    def test_perfect_prediction_versus_climatology_is_one(self):
        from cs2pickem.evaluation import brier_skill_score

        labels = [0, 0, 1, 1]
        probabilities = [0.0, 0.0, 1.0, 1.0]
        self.assertAlmostEqual(brier_skill_score(labels, probabilities), 1.0, places=9)

    def test_model_equal_to_climatology_is_zero(self):
        from cs2pickem.evaluation import brier_skill_score

        labels = [0, 0, 1, 1]
        # climatology baseline = mean(labels) = 0.5
        probabilities = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(brier_skill_score(labels, probabilities), 0.0, places=9)

    def test_worse_than_climatology_is_negative(self):
        from cs2pickem.evaluation import brier_skill_score

        labels = [0, 0, 1, 1]
        # reversed predictions are worse than constant 0.5
        probabilities = [1.0, 1.0, 0.0, 0.0]
        self.assertLess(brier_skill_score(labels, probabilities), 0.0)

    def test_scalar_and_sequence_baseline_paths_agree(self):
        from cs2pickem.evaluation import brier_skill_score

        labels = [1, 0, 1, 0, 1]
        probabilities = [0.7, 0.4, 0.6, 0.3, 0.8]
        scalar = brier_skill_score(labels, probabilities, 0.5)
        sequence = brier_skill_score(labels, probabilities, [0.5] * len(labels))
        self.assertAlmostEqual(scalar, sequence, places=12)

    def test_zero_reference_brier_returns_zero(self):
        from cs2pickem.evaluation import brier_skill_score

        labels = [1, 1, 1]
        probabilities = [0.9, 0.9, 0.9]
        # baseline perfectly predicts -> BS_ref = 0 -> guarded to 0.0
        self.assertEqual(brier_skill_score(labels, probabilities, [1.0, 1.0, 1.0]), 0.0)

    def test_empty_input_returns_zero(self):
        from cs2pickem.evaluation import brier_skill_score

        self.assertEqual(brier_skill_score([], []), 0.0)


class ExpectedCalibrationErrorTests(unittest.TestCase):
    def test_equal_width_matches_calibration_table_ece(self):
        from cs2pickem.evaluation import calibration_table, expected_calibration_error

        labels = [1, 0, 1, 0, 1, 1, 0, 0]
        probabilities = [0.9, 0.8, 0.6, 0.55, 0.4, 0.2, 0.3, 0.1]
        table = calibration_table(labels, probabilities, bins=10)
        result = expected_calibration_error(labels, probabilities, bins=10, binning="equal_width")
        self.assertAlmostEqual(result["ece"], table["ece"], places=12)

    def test_perfectly_calibrated_has_zero_ece_and_mce(self):
        from cs2pickem.evaluation import expected_calibration_error

        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        probabilities = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        result = expected_calibration_error(labels, probabilities, bins=10, binning="equal_width")
        self.assertAlmostEqual(result["ece"], 0.0, places=12)
        self.assertAlmostEqual(result["mce"], 0.0, places=12)

    def test_overconfident_constant_ece_and_mce(self):
        from cs2pickem.evaluation import expected_calibration_error

        labels = [1, 0, 1, 0]
        probabilities = [0.9, 0.9, 0.9, 0.9]
        result = expected_calibration_error(labels, probabilities, bins=10, binning="equal_width")
        self.assertAlmostEqual(result["ece"], 0.4, places=9)
        self.assertAlmostEqual(result["mce"], 0.4, places=9)

    def test_equal_mass_bins_have_balanced_counts(self):
        from cs2pickem.evaluation import expected_calibration_error

        labels = [i % 2 for i in range(20)]
        probabilities = [i / 20.0 for i in range(20)]
        result = expected_calibration_error(labels, probabilities, bins=5, binning="equal_mass")
        counts = [b["count"] for b in result["bins"] if b["count"] > 0]
        self.assertGreaterEqual(min(counts), 1)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_equal_mass_differs_from_equal_width_on_skewed_input(self):
        from cs2pickem.evaluation import expected_calibration_error

        # heavily skewed toward high probabilities -> equal_width leaves empty bins,
        # equal_mass redistributes mass.
        labels = [1, 1, 0, 1, 1, 0, 1, 1]
        probabilities = [0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
        ew = expected_calibration_error(labels, probabilities, bins=5, binning="equal_width")
        em = expected_calibration_error(labels, probabilities, bins=5, binning="equal_mass")
        self.assertNotAlmostEqual(ew["ece"], em["ece"], places=6)
        # equal_mass should not produce a degenerate empty / infinite value
        self.assertTrue(math.isfinite(em["ece"]))
        self.assertTrue(math.isfinite(em["mce"]))

    def test_maximum_calibration_error_helper(self):
        from cs2pickem.evaluation import maximum_calibration_error

        labels = [1, 0, 1, 0]
        probabilities = [0.9, 0.9, 0.9, 0.9]
        self.assertAlmostEqual(maximum_calibration_error(labels, probabilities), 0.4, places=9)

    def test_empty_input_returns_zero(self):
        from cs2pickem.evaluation import expected_calibration_error

        result = expected_calibration_error([], [], bins=10)
        self.assertEqual(result["ece"], 0.0)
        self.assertEqual(result["mce"], 0.0)


class BootstrapMetricTests(unittest.TestCase):
    def test_deterministic_metric_has_degenerate_interval(self):
        from cs2pickem.evaluation import accuracy, bootstrap_metric

        labels = [1, 0, 1, 0]
        probabilities = [1.0, 0.0, 1.0, 0.0]
        result = bootstrap_metric(accuracy, labels, probabilities, n_boot=200, seed=0)
        self.assertAlmostEqual(result["point"], 1.0, places=9)
        self.assertAlmostEqual(result["lo"], 1.0, places=9)
        self.assertAlmostEqual(result["hi"], 1.0, places=9)
        # A deterministically perfect metric has zero bootstrap dispersion.
        self.assertAlmostEqual(result["std"], 0.0, places=9)

    def test_interval_brackets_point_estimate(self):
        from cs2pickem.evaluation import bootstrap_metric, brier_score

        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        probabilities = [0.7, 0.4, 0.6, 0.3, 0.8, 0.2, 0.55, 0.45]
        result = bootstrap_metric(brier_score, labels, probabilities, n_boot=500, seed=0)
        self.assertLessEqual(result["lo"], result["point"])
        self.assertLessEqual(result["point"], result["hi"])

    def test_reproducible_with_same_seed(self):
        from cs2pickem.evaluation import bootstrap_metric, brier_score

        labels = [1, 0, 1, 0, 1, 0]
        probabilities = [0.7, 0.4, 0.6, 0.3, 0.8, 0.2]
        a = bootstrap_metric(brier_score, labels, probabilities, n_boot=300, seed=7)
        b = bootstrap_metric(brier_score, labels, probabilities, n_boot=300, seed=7)
        self.assertEqual(a["lo"], b["lo"])
        self.assertEqual(a["hi"], b["hi"])

    def test_narrower_interval_with_lower_confidence(self):
        from cs2pickem.evaluation import bootstrap_metric, brier_score

        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        probabilities = [0.7, 0.4, 0.6, 0.3, 0.8, 0.2, 0.55, 0.45]
        wide = bootstrap_metric(brier_score, labels, probabilities, n_boot=500, alpha=0.05, seed=3)
        narrow = bootstrap_metric(brier_score, labels, probabilities, n_boot=500, alpha=0.5, seed=3)
        self.assertLessEqual(narrow["hi"] - narrow["lo"], wide["hi"] - wide["lo"])

    def test_groups_resample_keeps_clusters_together(self):
        from cs2pickem.evaluation import bootstrap_metric

        labels = [1, 1, 0, 0]
        # Distinct per-member probabilities act as identity tags so we can
        # recover which original index each resampled member came from.
        probs = [0.1, 0.2, 0.8, 0.9]
        groups = ["m1", "m1", "m2", "m2"]
        captured = []

        def spy(lab, prob):
            captured.append(list(prob))
            return 0.0

        bootstrap_metric(spy, labels, probs, n_boot=30, seed=2, groups=groups)
        self.assertTrue(captured)
        for prob_sample in captured:
            # Members of a cluster must enter/leave together: idx0 (0.1) and
            # idx1 (0.2) belong to m1, so they must appear the same number of
            # times; likewise idx2 (0.8) and idx3 (0.9) for m2. This rejects a
            # broken implementation that mixes members across groups (e.g.
            # [0.1, 0.8, 0.1, 0.8]) which an even-parity-only check would pass.
            count_0 = prob_sample.count(0.1)
            count_1 = prob_sample.count(0.2)
            count_2 = prob_sample.count(0.8)
            count_3 = prob_sample.count(0.9)
            self.assertEqual(count_0, count_1)
            self.assertEqual(count_2, count_3)
            # Every drawn member belongs to exactly one of the two clusters.
            self.assertEqual(count_0 + count_1 + count_2 + count_3, len(prob_sample))

    def test_empty_input_degenerate(self):
        from cs2pickem.evaluation import bootstrap_metric, brier_score

        result = bootstrap_metric(brier_score, [], [], n_boot=50, seed=0)
        self.assertEqual(result["point"], 0.0)
        self.assertEqual(result["lo"], result["point"])
        self.assertEqual(result["hi"], result["point"])


class PairedBootstrapCompareTests(unittest.TestCase):
    def test_a_strictly_better_accuracy(self):
        from cs2pickem.evaluation import accuracy, paired_bootstrap_compare

        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        a = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        result = paired_bootstrap_compare(
            labels, a, b, accuracy, n_boot=300, seed=0, lower_is_better=False
        )
        self.assertEqual(result["better"], "a")
        self.assertTrue(result["significant"])
        self.assertGreater(result["lo"], 0.0)

    def test_identical_probabilities_tie(self):
        from cs2pickem.evaluation import brier_score, paired_bootstrap_compare

        labels = [1, 0, 1, 0, 1, 0]
        probs = [0.7, 0.4, 0.6, 0.3, 0.8, 0.2]
        result = paired_bootstrap_compare(labels, probs, probs, brier_score, n_boot=300, seed=0)
        self.assertAlmostEqual(result["delta"], 0.0, places=12)
        self.assertLessEqual(result["lo"], 0.0)
        self.assertGreaterEqual(result["hi"], 0.0)
        self.assertEqual(result["better"], "tie")
        self.assertFalse(result["significant"])
        self.assertAlmostEqual(result["p_value"], 1.0, places=9)

    def test_lower_is_better_picks_smaller_loss(self):
        from cs2pickem.evaluation import log_loss, paired_bootstrap_compare

        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        good = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
        bad = [0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4]
        result = paired_bootstrap_compare(
            labels, good, bad, log_loss, n_boot=300, seed=0, lower_is_better=True
        )
        self.assertEqual(result["better"], "a")

    def test_pairing_uses_same_indices_for_a_and_b(self):
        from cs2pickem.evaluation import paired_bootstrap_compare

        labels = [1, 0, 1, 0]
        a = [0.1, 0.2, 0.3, 0.4]
        b = [0.6, 0.7, 0.8, 0.9]
        seen_a = []
        seen_b = []

        def spy(lab, prob):
            # detect whether this call is over A or B by value range
            if all(p <= 0.5 for p in prob):
                seen_a.append(tuple(prob))
            else:
                seen_b.append(tuple(prob))
            return 0.0

        paired_bootstrap_compare(labels, a, b, spy, n_boot=10, seed=4)
        # For paired bootstrap, each resample maps A and B through identical indices,
        # so the multiset of indices used for A equals that for B.
        # Translate sampled prob tuples back to index multisets.
        a_to_idx = {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3}
        b_to_idx = {0.6: 0, 0.7: 1, 0.8: 2, 0.9: 3}
        for sa, sb in zip(seen_a, seen_b):
            idx_a = sorted(a_to_idx[p] for p in sa)
            idx_b = sorted(b_to_idx[p] for p in sb)
            self.assertEqual(idx_a, idx_b)

    def test_p_value_in_unit_interval_consistent_with_significance(self):
        from cs2pickem.evaluation import brier_score, paired_bootstrap_compare

        labels = [1, 0, 1, 0, 1, 0]
        a = [0.7, 0.35, 0.65, 0.3, 0.75, 0.25]
        b = [0.6, 0.45, 0.55, 0.4, 0.62, 0.38]
        result = paired_bootstrap_compare(labels, a, b, brier_score, n_boot=400, seed=0)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)
        ci_contains_zero = result["lo"] <= 0.0 <= result["hi"]
        if ci_contains_zero:
            self.assertFalse(result["significant"])


class DieboldMarianoTests(unittest.TestCase):
    def test_identical_predictions_give_zero_stat(self):
        from cs2pickem.evaluation import diebold_mariano

        labels = [1, 0, 1, 0, 1, 0]
        probs = [0.7, 0.4, 0.6, 0.3, 0.8, 0.2]
        result = diebold_mariano(labels, probs, probs, loss="squared")
        self.assertAlmostEqual(result["dm_stat"], 0.0, places=9)
        self.assertAlmostEqual(result["p_value"], 1.0, places=9)
        self.assertEqual(result["better"], "tie")

    def test_systematically_better_a_is_significant(self):
        from cs2pickem.evaluation import diebold_mariano

        labels = [1, 1, 1, 1, 0, 0, 0, 0] * 3
        a = [0.92, 0.93, 0.94, 0.95, 0.05, 0.06, 0.07, 0.08] * 3
        b = [0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4] * 3
        result = diebold_mariano(labels, a, b, loss="squared")
        self.assertLess(result["dm_stat"], 0.0)  # a has lower loss
        self.assertLess(result["p_value"], 0.05)
        self.assertEqual(result["better"], "a")

    def test_harvey_correction_shrinks_statistic(self):
        from cs2pickem.evaluation import diebold_mariano

        labels = [1, 1, 1, 0, 0, 0]
        a = [0.8, 0.85, 0.9, 0.1, 0.15, 0.2]
        b = [0.7, 0.6, 0.65, 0.35, 0.3, 0.4]
        corrected = diebold_mariano(labels, a, b, loss="squared", harvey_correction=True)
        uncorrected = diebold_mariano(labels, a, b, loss="squared", harvey_correction=False)
        self.assertLess(abs(corrected["dm_stat"]), abs(uncorrected["dm_stat"]))
        # more conservative -> larger p_value
        self.assertGreaterEqual(corrected["p_value"], uncorrected["p_value"])

    def test_brier_and_log_loss_agree_on_direction(self):
        from cs2pickem.evaluation import diebold_mariano

        labels = [1, 1, 1, 1, 0, 0, 0, 0] * 2
        a = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1] * 2
        b = [0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4] * 2
        r_brier = diebold_mariano(labels, a, b, loss="brier")
        r_log = diebold_mariano(labels, a, b, loss="log_loss")
        self.assertEqual(r_brier["better"], "a")
        self.assertEqual(r_log["better"], "a")

    def test_single_sample_is_degenerate(self):
        from cs2pickem.evaluation import diebold_mariano

        result = diebold_mariano([1], [0.7], [0.5], loss="squared")
        self.assertEqual(result["dm_stat"], 0.0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["better"], "tie")

    def test_p_value_in_unit_interval(self):
        from cs2pickem.evaluation import diebold_mariano

        labels = [1, 0, 1, 1, 0, 0, 1, 0]
        a = [0.6, 0.4, 0.7, 0.8, 0.3, 0.35, 0.65, 0.45]
        b = [0.55, 0.5, 0.6, 0.7, 0.4, 0.45, 0.6, 0.5]
        result = diebold_mariano(labels, a, b, loss="log_loss")
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_pure_python_student_t_matches_known_values(self):
        # The project mandates a working pure-Python fallback, but CI always
        # has scipy installed so the hand-rolled regularized incomplete beta
        # never runs under the normal suite. Force the pure-Python branch and
        # assert against independently known two-sided Student-t tail values.
        import cs2pickem.evaluation as ev

        saved = ev._scipy_stats
        ev._scipy_stats = None
        try:
            # P(|T| > 2.0), df=10 = 0.073388 ; P(|T| > 7.27), df=5 = 7.6982e-4
            self.assertAlmostEqual(ev._student_t_sf(2.0, 10), 0.073388, places=6)
            self.assertAlmostEqual(ev._student_t_sf(7.27, 5), 0.00076982, places=8)
            # Symmetry: sign of t does not matter (two-sided).
            self.assertAlmostEqual(
                ev._student_t_sf(-2.0, 10), ev._student_t_sf(2.0, 10), places=12
            )
            # t=0 -> the whole mass is in the tails -> 1.0.
            self.assertAlmostEqual(ev._student_t_sf(0.0, 8), 1.0, places=9)
        finally:
            ev._scipy_stats = saved

    def test_pure_python_and_scipy_paths_agree(self):
        # When scipy is present both branches must produce the same DM p-value
        # (to floating tolerance), proving the fallback is a faithful drop-in.
        import cs2pickem.evaluation as ev

        if ev._scipy_stats is None:
            self.skipTest("scipy not installed; nothing to cross-check")
        labels = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        a = [0.6, 0.4, 0.7, 0.8, 0.3, 0.35, 0.65, 0.45, 0.7, 0.3]
        b = [0.55, 0.5, 0.6, 0.7, 0.4, 0.45, 0.6, 0.5, 0.62, 0.41]
        with_scipy = ev.diebold_mariano(labels, a, b, loss="squared")
        saved = ev._scipy_stats
        ev._scipy_stats = None
        try:
            without_scipy = ev.diebold_mariano(labels, a, b, loss="squared")
        finally:
            ev._scipy_stats = saved
        self.assertAlmostEqual(with_scipy["dm_stat"], without_scipy["dm_stat"], places=12)
        self.assertAlmostEqual(with_scipy["p_value"], without_scipy["p_value"], places=9)

    def test_multi_lag_newey_west_differs_from_single_lag(self):
        # h=1 collapses range(1, h) to empty, so the multi-lag autocovariance
        # accumulation never runs in the rest of the suite. Exercise h=2 on a
        # series with autocorrelated loss differentials and assert the long-run
        # variance path actually changes the statistic and p-value (the Harvey
        # factor's h*(h-1)/n term also only becomes non-zero when h>1). A modest
        # signal is used so both p-values stay numerically distinguishable.
        from cs2pickem.evaluation import diebold_mariano

        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        a = [0.62, 0.40, 0.63, 0.39, 0.61, 0.41, 0.64, 0.38, 0.62, 0.40, 0.63, 0.39]
        b = [0.55, 0.45, 0.56, 0.44, 0.54, 0.46, 0.57, 0.43, 0.55, 0.45, 0.56, 0.44]
        h1 = diebold_mariano(labels, a, b, loss="squared", h=1)
        h2 = diebold_mariano(labels, a, b, loss="squared", h=2)
        self.assertEqual(h2["h"], 2)
        # Loss differential mean is independent of the lag truncation.
        self.assertAlmostEqual(h1["mean_loss_diff"], h2["mean_loss_diff"], places=12)
        # The multi-lag Newey-West long-run-variance path materially changes the
        # statistic (>1% relative shift), proving range(1, h) actually ran and
        # the Harvey h*(h-1)/n term kicked in. (p-values here are astronomically
        # small, so we assert on the stable statistic, not their absolute gap.)
        rel_shift = abs(h2["dm_stat"] - h1["dm_stat"]) / abs(h1["dm_stat"])
        self.assertGreater(rel_shift, 0.01)
        # Both p-values remain valid probabilities.
        for r in (h1, h2):
            self.assertGreaterEqual(r["p_value"], 0.0)
            self.assertLessEqual(r["p_value"], 1.0)
        # Direction (A lower loss) is preserved across lag choices.
        self.assertEqual(h1["better"], "a")
        self.assertEqual(h2["better"], "a")


class ProbabilityCalibratorTests(unittest.TestCase):
    def test_platt_calibrator_shifts_systematically_low_probabilities_up(self):
        from cs2pickem.calibration import ProbabilityCalibrator
        from cs2pickem.evaluation import log_loss

        labels = [0, 0, 1, 1]
        probabilities = [0.2, 0.2, 0.2, 0.2]

        calibrator = ProbabilityCalibrator(epochs=160, learning_rate=0.4).fit(probabilities, labels)
        calibrated = calibrator.transform(probabilities)

        self.assertGreater(calibrated[0], 0.35)
        self.assertLess(log_loss(labels, calibrated), log_loss(labels, probabilities))
        self.assertEqual(calibrator.report()["basis"], "platt_logistic")


if __name__ == "__main__":
    unittest.main()
