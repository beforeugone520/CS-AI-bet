import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


class ImbalanceTests(unittest.TestCase):
    def test_rebalance_training_data_creates_synthetic_minority_rows_and_class_weights(self):
        from cs2pickem.imbalance import rebalance_training_data

        rows = [[0.0], [0.1], [0.2], [0.3], [0.8], [1.0]]
        labels = [0, 0, 0, 0, 1, 1]
        result = rebalance_training_data(rows, labels)

        self.assertEqual(result.report["strategy"], "smote_minority_oversample_plus_class_weight")
        self.assertEqual(result.report["original_counts"], {"0": 4, "1": 2})
        self.assertEqual(result.report["balanced_counts"], {"0": 4, "1": 4})
        self.assertEqual(result.report["synthetic_rows"], 2)
        self.assertEqual(result.labels.count(0), result.labels.count(1))
        self.assertAlmostEqual(result.report["class_weights"]["0"], 0.75)
        self.assertAlmostEqual(result.report["class_weights"]["1"], 1.5)
        self.assertAlmostEqual(result.sample_weights[-1], 1.5)
        self.assertTrue(all(0.8 <= row[0] <= 1.0 for row in result.rows[-2:]))

    def test_logistic_model_uses_sample_weights_for_weighted_prior(self):
        from cs2pickem.models import LogisticRegressionGD

        model = LogisticRegressionGD(epochs=0).fit([[], []], [0, 1], sample_weights=[1.0, 9.0])

        self.assertGreater(model.predict_proba([[]])[0], 0.85)


if __name__ == "__main__":
    unittest.main()
