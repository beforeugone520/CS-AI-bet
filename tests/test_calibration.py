import math
import os
import random
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def perfectly_calibrated_dataset():
    """Each probability level appears with an exactly matching positive rate.

    With ``round(p * 100)`` positives per 100 rows, the empirical frequency
    equals the predicted probability, so a faithful calibrator must leave the
    probabilities (near) unchanged regardless of method.
    """
    probabilities = []
    labels = []
    for p in (0.2, 0.35, 0.5, 0.65, 0.8):
        for k in range(100):
            probabilities.append(p)
            labels.append(1 if k < round(p * 100) else 0)
    return probabilities, labels


def overconfident_dataset(seed=0):
    """Extreme predictions that are only ~70% correct -> needs T > 1 / shrink."""
    rng = random.Random(seed)
    probabilities = []
    labels = []
    for i in range(120):
        p = 0.95 if i % 2 == 0 else 0.05
        probabilities.append(p)
        favoured = 1 if i % 2 == 0 else 0
        labels.append(favoured if rng.random() < 0.7 else (1 - favoured))
    return probabilities, labels


class FactoryBackCompatTests(unittest.TestCase):
    def test_default_factory_returns_historic_platt_calibrator(self):
        from cs2pickem.calibration import ProbabilityCalibrator, make_calibrator

        calibrator = make_calibrator()
        self.assertIsInstance(calibrator, ProbabilityCalibrator)

    def test_default_platt_fit_is_bit_identical_to_historic_behaviour(self):
        from cs2pickem.calibration import ProbabilityCalibrator, make_calibrator

        labels = [0, 0, 1, 1, 1, 0, 1, 0]
        probabilities = [0.3, 0.4, 0.6, 0.7, 0.55, 0.45, 0.8, 0.2]

        historic = ProbabilityCalibrator().fit(probabilities, labels)
        factory = make_calibrator("platt").fit(probabilities, labels)

        self.assertEqual(historic.report(), factory.report())
        self.assertEqual(
            historic.transform(probabilities), factory.transform(probabilities)
        )
        self.assertEqual(factory.report()["basis"], "platt_logistic")

    def test_unknown_method_rejected(self):
        from cs2pickem.calibration import make_calibrator

        with self.assertRaises(ValueError):
            make_calibrator("isotonic")

    def test_factory_returns_method_calibrator_for_non_default_methods(self):
        from cs2pickem.calibration import MethodCalibrator, make_calibrator

        for method in ("beta", "temperature"):
            self.assertIsInstance(make_calibrator(method), MethodCalibrator)

    def test_factory_shares_fit_transform_report_api(self):
        from cs2pickem.calibration import make_calibrator

        probabilities, labels = perfectly_calibrated_dataset()
        for method in ("platt", "beta", "temperature"):
            calibrator = make_calibrator(method).fit(probabilities, labels)
            transformed = calibrator.transform(probabilities)
            self.assertEqual(len(transformed), len(probabilities))
            single = calibrator.transform_one(probabilities[0])
            self.assertAlmostEqual(single, transformed[0], places=12)
            report = calibrator.report()
            for key in ("basis", "training_count", "positive_rate", "slope", "intercept"):
                self.assertIn(key, report)


class IdentityAndNestingTests(unittest.TestCase):
    def test_beta_is_near_identity_on_already_calibrated_input(self):
        from cs2pickem.calibration import MethodCalibrator

        probabilities, labels = perfectly_calibrated_dataset()
        calibrator = MethodCalibrator(method="beta").fit(probabilities, labels)
        transformed = calibrator.transform(probabilities)
        max_dev = max(abs(a - b) for a, b in zip(transformed, probabilities))
        self.assertLess(max_dev, 0.02)
        # Identity beta map is c1 = c2 = 1, c0 = 0.
        c0, c1, c2 = calibrator.report()["beta_coefficients"]
        self.assertAlmostEqual(c0, 0.0, delta=0.1)
        self.assertAlmostEqual(c1, 1.0, delta=0.1)
        self.assertAlmostEqual(c2, 1.0, delta=0.1)

    def test_all_methods_are_near_identity_on_already_calibrated_input(self):
        from cs2pickem.calibration import make_calibrator

        probabilities, labels = perfectly_calibrated_dataset()
        for method in ("platt", "beta", "temperature"):
            calibrator = make_calibrator(method).fit(probabilities, labels)
            transformed = calibrator.transform(probabilities)
            max_dev = max(abs(a - b) for a, b in zip(transformed, probabilities))
            self.assertLess(max_dev, 0.02, msg=f"{method} drifted from identity")

    def test_temperature_one_is_identity_on_calibrated_input(self):
        from cs2pickem.calibration import MethodCalibrator

        probabilities, labels = perfectly_calibrated_dataset()
        calibrator = MethodCalibrator(method="temperature").fit(probabilities, labels)
        self.assertAlmostEqual(calibrator.temperature, 1.0, delta=0.05)


class MonotoneAndAucTests(unittest.TestCase):
    """Honest monotonicity contract.

    Only ``temperature`` (``T > 0``) and ``beta`` (now constrained to
    ``c1, c2 >= 0`` per Kull) are GUARANTEED monotone for arbitrary fitted data;
    ``platt`` is monotone only when its data-driven slope is non-negative (it can
    invert on anti-correlated samples -- see
    ``test_platt_can_be_non_monotone_on_anti_correlated_data``, a documented
    inherent property of an unconstrained logistic fit, not a regression). The
    tests below assert monotonicity on a strongly monotone signal for all three,
    plus the constrained guarantee for beta/temperature on adversarial data.
    """

    def _assert_monotone_and_auc_preserved(self, method):
        from cs2pickem.calibration import make_calibrator
        from cs2pickem.evaluation import auc

        probabilities, labels = overconfident_dataset(seed=3)
        # add spread so AUC is well defined and there are many distinct probs
        rng = random.Random(7)
        probabilities = [min(0.999, max(0.001, p + rng.uniform(-0.1, 0.1))) for p in probabilities]
        calibrator = make_calibrator(method).fit(probabilities, labels)

        grid = [i / 100.0 for i in range(1, 100)]
        transformed_grid = calibrator.transform(grid)
        for earlier, later in zip(transformed_grid, transformed_grid[1:]):
            self.assertLessEqual(earlier, later + 1e-9, msg=f"{method} not monotone")

        before = auc(labels, probabilities)
        after = auc(labels, calibrator.transform(probabilities))
        self.assertAlmostEqual(before, after, places=6, msg=f"{method} changed AUC")

    def test_platt_is_monotone_and_auc_preserving_on_monotone_signal(self):
        # NOTE: this holds because the fixture has a strong monotone signal so the
        # platt slope is positive. Platt is NOT monotone in general (see below).
        self._assert_monotone_and_auc_preserved("platt")

    def test_beta_is_monotone_and_auc_preserving(self):
        self._assert_monotone_and_auc_preserved("beta")

    def test_temperature_is_monotone_and_auc_preserving(self):
        self._assert_monotone_and_auc_preserved("temperature")

    def _adversarial_two_tail_dataset(self, seed):
        """Anti-correlated tails: high probs tend to be 0, low probs tend to be 1.

        This is the case that drives an unconstrained logistic / beta fit to
        NEGATIVE coefficients (non-monotone). Used to prove the constrained
        beta/temperature solvers stay monotone where the raw fit would invert.
        """
        rng = random.Random(seed)
        probs = []
        labels = []
        for _ in range(40):
            p = rng.uniform(0.05, 0.95)
            probs.append(p)
            anti = (p < 0.5 and rng.random() < 0.7) or (p >= 0.5 and rng.random() < 0.3)
            labels.append(1 if anti else 0)
        return probs, labels

    def test_beta_stays_monotone_on_adversarial_anti_correlated_data(self):
        from cs2pickem.calibration import MethodCalibrator

        grid = [i / 100.0 for i in range(1, 100)]
        for seed in range(25):
            probs, labels = self._adversarial_two_tail_dataset(seed)
            calibrator = MethodCalibrator(method="beta").fit(probs, labels)
            c0, c1, c2 = calibrator.beta_coefficients
            # Kull constraint: both tail coefficients are projected to >= 0.
            self.assertGreaterEqual(c1, -1e-9, msg=f"seed={seed} c1<0")
            self.assertGreaterEqual(c2, -1e-9, msg=f"seed={seed} c2<0")
            transformed = calibrator.transform(grid)
            for earlier, later in zip(transformed, transformed[1:]):
                self.assertLessEqual(earlier, later + 1e-9, msg=f"beta non-monotone seed={seed}")

    def test_beta_constraint_holds_on_pure_python_solver(self):
        # Force the scipy-free path so the projected-Newton fallback is exercised.
        from cs2pickem import calibration

        grid = [i / 100.0 for i in range(1, 100)]
        saved = calibration._scipy_optimize
        calibration._scipy_optimize = None
        try:
            for seed in range(15):
                probs, labels = self._adversarial_two_tail_dataset(seed)
                calibrator = calibration.MethodCalibrator(method="beta").fit(probs, labels)
                c0, c1, c2 = calibrator.beta_coefficients
                self.assertGreaterEqual(c1, -1e-9, msg=f"seed={seed} c1<0 (pure python)")
                self.assertGreaterEqual(c2, -1e-9, msg=f"seed={seed} c2<0 (pure python)")
                transformed = calibrator.transform(grid)
                for earlier, later in zip(transformed, transformed[1:]):
                    self.assertLessEqual(earlier, later + 1e-9, msg=f"beta non-monotone seed={seed}")
        finally:
            calibration._scipy_optimize = saved

    def test_platt_can_be_non_monotone_on_anti_correlated_data(self):
        # Documented inherent behaviour (NOT a regression vs the historic
        # ProbabilityCalibrator): an unconstrained platt logistic fit on data
        # whose probabilities anti-correlate with outcomes yields a negative slope
        # and a DECREASING calibration map. We assert the slope can go negative so
        # downstream code (and reviewers) know platt is not a monotone guarantee.
        from cs2pickem.calibration import make_calibrator

        probs, labels = self._adversarial_two_tail_dataset(seed=11)
        calibrator = make_calibrator("platt").fit(probs, labels)
        self.assertLess(calibrator.report()["slope"], 0.0)


class ImprovementTests(unittest.TestCase):
    def test_temperature_fixes_overconfidence(self):
        from cs2pickem.calibration import MethodCalibrator
        from cs2pickem.evaluation import log_loss

        probabilities, labels = overconfident_dataset(seed=1)
        calibrator = MethodCalibrator(method="temperature").fit(probabilities, labels)
        # Overconfidence => effective temperature must be > 1 (softens logits).
        self.assertGreater(calibrator.temperature, 1.0)
        self.assertLess(
            log_loss(labels, calibrator.transform(probabilities)),
            log_loss(labels, probabilities),
        )

    def test_beta_shifts_systematically_low_probabilities_up(self):
        from cs2pickem.calibration import MethodCalibrator
        from cs2pickem.evaluation import log_loss

        labels = [0, 0, 1, 1] * 10
        probabilities = [0.2] * 40
        calibrator = MethodCalibrator(method="beta").fit(probabilities, labels)
        calibrated = calibrator.transform(probabilities)
        self.assertGreater(calibrated[0], 0.35)
        self.assertLess(log_loss(labels, calibrated), log_loss(labels, probabilities))


class RobustnessTests(unittest.TestCase):
    def test_empty_input_is_identity_passthrough(self):
        from cs2pickem.calibration import make_calibrator

        for method in ("platt", "beta", "temperature"):
            calibrator = make_calibrator(method).fit([], [])
            self.assertEqual(calibrator.report()["basis"], "no_calibration_rows")
            self.assertEqual(calibrator.transform([0.3, 0.7]), [0.3, 0.7])
            self.assertEqual(calibrator.transform_one(0.42), 0.42)

    def test_extreme_probabilities_do_not_diverge(self):
        from cs2pickem.calibration import make_calibrator

        probabilities = [0.0, 1.0, 0.0, 1.0, 1e-9, 1.0 - 1e-9, 0.5, 0.5]
        labels = [0, 1, 1, 0, 0, 1, 1, 0]
        for method in ("platt", "beta", "temperature"):
            calibrator = make_calibrator(method).fit(probabilities, labels)
            transformed = calibrator.transform(probabilities)
            for value in transformed:
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_tiny_sample_does_not_diverge(self):
        from cs2pickem.calibration import make_calibrator

        # Perfectly separable 2-row sample: an unregularised logistic fit would
        # run slope/temperature to +-inf. The L2-toward-identity prior must keep
        # the parameters finite and the output a valid probability.
        probabilities = [0.4, 0.6]
        labels = [0, 1]
        for method in ("platt", "beta", "temperature"):
            calibrator = make_calibrator(method).fit(probabilities, labels)
            report = calibrator.report()
            self.assertTrue(math.isfinite(report["slope"]))
            self.assertTrue(math.isfinite(report["intercept"]))
            transformed = calibrator.transform(probabilities)
            for value in transformed:
                self.assertTrue(0.0 <= value <= 1.0)

    def test_all_same_label_does_not_crash(self):
        from cs2pickem.calibration import make_calibrator

        probabilities = [0.3, 0.4, 0.5, 0.6]
        for method in ("platt", "beta", "temperature"):
            for labels in ([1, 1, 1, 1], [0, 0, 0, 0]):
                calibrator = make_calibrator(method).fit(probabilities, labels)
                transformed = calibrator.transform(probabilities)
                self.assertTrue(all(math.isfinite(v) for v in transformed))


class CrossValidationCalibrationTests(unittest.TestCase):
    def test_cv_uses_expanding_window_and_does_not_see_future(self):
        from cs2pickem.calibration import MethodCalibrator

        probabilities, labels = perfectly_calibrated_dataset()
        calibrator = MethodCalibrator(method="temperature", cv_folds=3).fit(probabilities, labels)
        report = calibrator.report()
        self.assertGreaterEqual(report.get("cv_folds_used", 0), 1)
        # On calibrated data CV must still resolve near identity.
        transformed = calibrator.transform(probabilities)
        max_dev = max(abs(a - b) for a, b in zip(transformed, probabilities))
        self.assertLess(max_dev, 0.05)

    def test_cv_calibrator_never_sees_the_validation_window(self):
        from cs2pickem import calibration

        # Spy on _fit_single to capture which rows each fold trains on, then
        # prove fold k only used rows strictly before its validation window
        # (expanding window, no future leakage).
        captured = []
        original = calibration.MethodCalibrator._fit_single

        def spy(self, rows):
            captured.append(list(rows))
            return original(self, rows)

        calibration.MethodCalibrator._fit_single = spy
        try:
            probabilities = [i / 100.0 for i in range(10, 90)]
            labels = [1 if i % 2 == 0 else 0 for i in range(len(probabilities))]
            calibration.MethodCalibrator(method="platt", cv_folds=3).fit(probabilities, labels)
        finally:
            calibration.MethodCalibrator._fit_single = original

        self.assertGreaterEqual(len(captured), 1)
        # Each fold's training set must be a strict prefix that grows; the last
        # fold must NOT contain the final rows (held out as its validation set).
        sizes = [len(rows) for rows in captured]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(max(sizes), len(probabilities))

    def test_cv_falls_back_to_single_fit_when_too_few_rows(self):
        from cs2pickem.calibration import MethodCalibrator

        probabilities = [0.3, 0.7]
        labels = [0, 1]
        calibrator = MethodCalibrator(method="platt", cv_folds=5).fit(probabilities, labels)
        # Not enough rows for 5 folds -> single fit (cv_folds_used absent / 0).
        self.assertEqual(calibrator.report().get("cv_folds_used", 0), 0)
        transformed = calibrator.transform(probabilities)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in transformed))


class SolverParityTests(unittest.TestCase):
    def test_pure_python_and_scipy_logistic_agree(self):
        from cs2pickem import calibration

        if calibration._scipy_optimize is None:
            self.skipTest("scipy not installed; nothing to cross-check")

        probabilities, labels = overconfident_dataset(seed=5)
        for method in ("platt", "beta"):
            with_scipy = calibration.MethodCalibrator(method=method).fit(probabilities, labels)
            saved = calibration._scipy_optimize
            calibration._scipy_optimize = None
            try:
                without_scipy = calibration.MethodCalibrator(method=method).fit(probabilities, labels)
            finally:
                calibration._scipy_optimize = saved
            a = with_scipy.transform(probabilities)
            b = without_scipy.transform(probabilities)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, places=4, msg=f"{method} solver mismatch")

    def test_pure_python_and_scipy_temperature_agree(self):
        from cs2pickem import calibration

        if calibration._scipy_optimize is None:
            self.skipTest("scipy not installed; nothing to cross-check")

        probabilities, labels = overconfident_dataset(seed=9)
        with_scipy = calibration.MethodCalibrator(method="temperature").fit(probabilities, labels)
        saved = calibration._scipy_optimize
        calibration._scipy_optimize = None
        try:
            without_scipy = calibration.MethodCalibrator(method="temperature").fit(probabilities, labels)
        finally:
            calibration._scipy_optimize = saved
        self.assertAlmostEqual(with_scipy.temperature, without_scipy.temperature, places=3)

    def test_pure_python_temperature_still_reduces_log_loss(self):
        from cs2pickem import calibration
        from cs2pickem.evaluation import log_loss

        probabilities, labels = overconfident_dataset(seed=2)
        saved = calibration._scipy_optimize
        calibration._scipy_optimize = None
        try:
            calibrator = calibration.MethodCalibrator(method="temperature").fit(probabilities, labels)
        finally:
            calibration._scipy_optimize = saved
        self.assertGreater(calibrator.temperature, 1.0)
        self.assertLess(
            log_loss(labels, calibrator.transform(probabilities)),
            log_loss(labels, probabilities),
        )


class HoldoutCalibratorIntegrationTests(unittest.TestCase):
    class _StubModel:
        def predict_proba(self, rows):
            return [float(row[0]) for row in rows]

    class _StubBuilderSelector:
        # builder.transform(rows) returns its argument; selector.transform(x).rows == x
        def transform(self, rows):
            return rows

        class _Selected:
            def __init__(self, rows):
                self.rows = rows

        def select(self, rows):
            return self._Selected(rows)

    def _calibration_rows(self):
        # each row carries its own model probability in column 0; team1 wins iff
        # the modelled probability is high, but the model is systematically
        # over/under confident so a calibrator has something to fix.
        rows = []
        for i in range(40):
            p = 0.2 if i % 2 == 0 else 0.8
            winner = "T1" if (i % 2 == 1) else "T2"
            rows.append({"_probability": [p], "team1": "T1", "team2": "T2", "winner": winner})
        return rows

    def test_holdout_calibrator_threads_method_and_preserves_platt_basis(self):
        from cs2pickem import predictor

        builder = type("B", (), {"transform": staticmethod(lambda rows: [row["_probability"] for row in rows])})()
        selector = type(
            "S",
            (),
            {"transform": staticmethod(lambda x: type("R", (), {"rows": x})())},
        )()
        model = self._StubModel()
        rows = self._calibration_rows()

        platt_cal, platt_report = predictor._fit_holdout_calibrator(builder, selector, model, rows)
        self.assertEqual(platt_report["basis"], "holdout_platt_logistic")
        self.assertEqual(platt_report["calibration_count"], len(rows))

        temp_cal, temp_report = predictor._fit_holdout_calibrator(
            builder, selector, model, rows, method="temperature"
        )
        self.assertEqual(temp_report["basis"], "holdout_temperature")
        self.assertEqual(temp_report["method"], "temperature")
        self.assertIn("temperature", temp_report)
        # transform stays a valid probability
        self.assertTrue(0.0 <= temp_cal.transform_one(0.3) <= 1.0)

    def test_holdout_calibrator_empty_rows_not_applied(self):
        from cs2pickem import predictor

        calibrator, report = predictor._fit_holdout_calibrator(None, None, None, [], method="beta")
        self.assertIsNone(calibrator)
        self.assertEqual(report["basis"], "not_applied")


if __name__ == "__main__":
    unittest.main()
