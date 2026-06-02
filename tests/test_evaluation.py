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
