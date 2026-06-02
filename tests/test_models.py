import os
import sys
from unittest import mock
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _and_rows(repeats=12):
    rows = []
    labels = []
    for _ in range(repeats):
        rows.extend(([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]))
        labels.extend((0, 0, 0, 1))
    return rows, labels


class _FakeEstimator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, rows, labels, sample_weight=None):
        self.rows = rows
        self.labels = labels
        self.sample_weight = sample_weight
        return self

    def predict_proba(self, rows):
        return [[0.25, 0.75] for _ in rows]


class _FakeModule:
    LogisticRegression = _FakeEstimator
    RandomForestClassifier = _FakeEstimator
    XGBClassifier = _FakeEstimator
    MLPClassifier = _FakeEstimator
    StandardScaler = _FakeEstimator

    @staticmethod
    def make_pipeline(*steps):
        return steps[-1]


class ModelHyperparameterTests(unittest.TestCase):
    def test_default_ensemble_can_force_pure_python_backend(self):
        from cs2pickem.models import default_ensemble

        model = default_ensemble(seed=7, epochs=2, prefer_accelerated=False, n_jobs=2)

        self.assertEqual(model.n_jobs, 2)
        self.assertEqual(
            model.component_backends,
            {
                "logistic": "pure_python",
                "random_forest": "pure_python",
                "xgboost": "pure_python",
                "neural_network": "pure_python",
            },
        )

    def test_default_ensemble_exposes_backend_metadata_for_acceleration(self):
        from cs2pickem.models import default_ensemble

        model = default_ensemble(seed=7, epochs=2)

        self.assertEqual(set(model.component_backends), {"logistic", "random_forest", "xgboost", "neural_network"})
        self.assertTrue(
            all(
                backend in {"pure_python", "sklearn", "xgboost"}
                for backend in model.component_backends.values()
            )
        )
        self.assertTrue(model.accelerated_requested)

    def test_default_ensemble_uses_no_nn_weights_after_historical_holdout(self):
        from cs2pickem.models import default_ensemble

        model = default_ensemble(seed=7, epochs=2)

        self.assertEqual(model.weights["neural_network"], 0.0)
        self.assertAlmostEqual(model.weights["logistic"], 0.20 / 0.85)
        self.assertAlmostEqual(model.weights["random_forest"], 0.30 / 0.85)
        self.assertAlmostEqual(model.weights["xgboost"], 0.35 / 0.85)

    def test_default_ensemble_uses_accelerated_backends_when_imports_are_available(self):
        from cs2pickem import models

        def fake_import(module_name):
            if module_name in {"sklearn.linear_model", "sklearn.ensemble", "sklearn.neural_network", "sklearn.pipeline", "sklearn.preprocessing", "xgboost"}:
                return _FakeModule
            return None

        with mock.patch.object(models, "_optional_import", side_effect=fake_import), mock.patch.dict(os.environ, {"CS2PICKEM_ACCELERATED_MLP": "1"}):
            model = models.default_ensemble(seed=7, epochs=2, n_jobs=1)

        self.assertEqual(
            model.component_backends,
            {
                "logistic": "sklearn",
                "random_forest": "sklearn",
                "xgboost": "xgboost",
                "neural_network": "sklearn",
            },
        )
        model.fit([[0.0], [1.0]], [0, 1], sample_weights=[1.0, 2.0])
        self.assertEqual(model.predict_components([[1.0]])["xgboost"], [0.75])

    def test_default_ensemble_falls_back_when_accelerated_import_raises_runtime_error(self):
        from cs2pickem import models

        def fake_import(module_name):
            if module_name == "xgboost":
                raise RuntimeError("missing libomp")
            return None

        with mock.patch.object(models.importlib, "import_module", side_effect=fake_import):
            model = models.default_ensemble(seed=7, epochs=2)

        self.assertEqual(model.component_backends["xgboost"], "pure_python")

    def test_random_forest_uses_depth_to_model_feature_interactions(self):
        from cs2pickem.models import StumpForestModel

        rows, labels = _and_rows()
        shallow = StumpForestModel(trees=80, max_depth=1, min_leaf_samples=3, seed=11).fit(rows, labels)
        deep = StumpForestModel(trees=80, max_depth=2, min_leaf_samples=3, seed=11).fit(rows, labels)

        shallow_positive = shallow.predict_proba([[1.0, 1.0]])[0]
        shallow_partial = shallow.predict_proba([[1.0, 0.0]])[0]
        deep_positive = deep.predict_proba([[1.0, 1.0]])[0]
        deep_partial = deep.predict_proba([[1.0, 0.0]])[0]

        self.assertGreater(deep_positive - deep_partial, shallow_positive - shallow_partial + 0.25)
        self.assertGreater(deep_positive, 0.75)
        self.assertLess(deep_partial, 0.25)

    def test_random_forest_min_leaf_samples_blocks_tiny_splits(self):
        from cs2pickem.models import StumpForestModel

        rows = [[0.0], [1.0], [2.0], [3.0]]
        labels = [0, 0, 1, 1]

        model = StumpForestModel(trees=6, max_depth=3, min_leaf_samples=3, seed=3).fit(rows, labels)

        self.assertTrue(all(tree.feature_index is None for tree in model.forest))

    def test_neural_network_stays_calibrated_with_large_magnitude_features(self):
        import random

        from cs2pickem.models import NeuralStyleModel

        rng = random.Random(0)
        rows, labels = [], []
        for _ in range(200):
            signal = rng.random()
            nuisance_code = float(rng.randint(0, 300))  # large label-encoded feature like team1_code
            rows.append([signal, nuisance_code])
            labels.append(1 if signal > 0.5 else 0)

        model = NeuralStyleModel(epochs=50, seed=3).fit(rows, labels)
        probabilities = model.predict_proba(rows)

        # Bug regression: with a large-magnitude feature the model used to collapse to
        # ~0 for every row, so the mean prediction was pinned near an extreme (~0.01)
        # with no discrimination. A healthy model keeps a balanced mean on balanced labels.
        mean_probability = sum(probabilities) / len(probabilities)
        self.assertGreater(mean_probability, 0.3)
        self.assertLess(mean_probability, 0.7)

        # It must still discriminate: positives score higher than negatives on average.
        positive = [p for p, label in zip(probabilities, labels) if label == 1]
        negative = [p for p, label in zip(probabilities, labels) if label == 0]
        self.assertGreater(sum(positive) / len(positive), sum(negative) / len(negative) + 0.15)

    def test_boosting_uses_depth_and_row_subsample(self):
        from cs2pickem.models import BoostingStumpModel

        rows, labels = _and_rows()
        model = BoostingStumpModel(rounds=6, learning_rate=0.3, max_depth=2, subsample=0.5, seed=5).fit(rows, labels)

        probabilities = model.predict_proba([[1.0, 1.0], [1.0, 0.0]])

        self.assertEqual(len(model.trees), 6)
        self.assertTrue(all(count == len(rows) // 2 for count in model.round_sample_counts))
        self.assertTrue(any(tree.depth > 1 for tree in model.trees))
        self.assertGreater(probabilities[0], probabilities[1] + 0.2)


if __name__ == "__main__":
    unittest.main()
