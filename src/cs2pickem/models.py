from __future__ import annotations

import importlib
import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_MODEL_WEIGHTS = {"logistic": 0.20, "random_forest": 0.30, "xgboost": 0.35, "neural_network": 0.15}


def model_hyperparameters(epochs: int = 50) -> Dict[str, object]:
    return {
        "logistic": {
            "learning_rate": 0.08,
            "epochs": epochs,
        },
        "random_forest": {
            "trees": 150,
            "max_depth": 8,
            "min_leaf_samples": 3,
        },
        "xgboost": {
            "rounds": 120,
            "learning_rate": 0.08,
            "max_depth": 6,
            "subsample": 0.8,
        },
        "neural_network": {
            "hidden_layers": [64, 32],
            "activation": "relu",
            "epochs": epochs,
        },
    }


def default_ensemble(
    seed: int = 13,
    epochs: int = 50,
    weights: Dict[str, float] | None = None,
    prefer_accelerated: bool = True,
    n_jobs: int | None = None,
) -> "WeightedEnsembleModel":
    models, component_backends = _model_components(seed=seed, epochs=epochs, prefer_accelerated=prefer_accelerated, n_jobs=n_jobs)
    return WeightedEnsembleModel(
        models=models,
        weights=weights or DEFAULT_MODEL_WEIGHTS,
        component_backends=component_backends,
        accelerated_requested=prefer_accelerated,
        n_jobs=n_jobs,
    )


def _model_components(seed: int, epochs: int, prefer_accelerated: bool, n_jobs: int | None) -> tuple[Dict[str, object], Dict[str, str]]:
    fallback = {
        "logistic": lambda: LogisticRegressionGD(epochs=epochs, learning_rate=0.08),
        "random_forest": lambda: StumpForestModel(trees=150, max_depth=8, min_leaf_samples=3, seed=seed),
        "xgboost": lambda: BoostingStumpModel(rounds=120, learning_rate=0.08, max_depth=6, subsample=0.8, seed=seed),
        "neural_network": lambda: NeuralStyleModel(hidden_layers=(64, 32), epochs=epochs, seed=seed),
    }
    if not prefer_accelerated:
        return {name: factory() for name, factory in fallback.items()}, {name: "pure_python" for name in fallback}

    models: Dict[str, object] = {}
    backends: Dict[str, str] = {}
    for name, factory in fallback.items():
        accelerated = _accelerated_model(name=name, seed=seed, epochs=epochs, n_jobs=n_jobs)
        if accelerated is None:
            models[name] = factory()
            backends[name] = "pure_python"
        else:
            model, backend = accelerated
            models[name] = model
            backends[name] = backend
    return models, backends


def _accelerated_model(name: str, seed: int, epochs: int, n_jobs: int | None) -> tuple[object, str] | None:
    if name == "logistic":
        linear_model = _optional_import("sklearn.linear_model")
        if linear_model is None:
            return None
        estimator = linear_model.LogisticRegression(max_iter=max(100, epochs * 20), solver="lbfgs", random_state=seed)
        return SklearnProbabilityModel(estimator), "sklearn"

    if name == "random_forest":
        ensemble = _optional_import("sklearn.ensemble")
        if ensemble is None:
            return None
        estimator = ensemble.RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=_estimator_n_jobs(n_jobs),
        )
        return SklearnProbabilityModel(estimator), "sklearn"

    if name == "xgboost":
        xgboost = _optional_import("xgboost")
        if xgboost is None:
            return None
        estimator = xgboost.XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=_estimator_n_jobs(n_jobs),
        )
        return SklearnProbabilityModel(estimator), "xgboost"

    if name == "neural_network":
        if os.environ.get("CS2PICKEM_ACCELERATED_MLP") != "1":
            return None
        neural_network = _optional_import("sklearn.neural_network")
        pipeline = _optional_import("sklearn.pipeline")
        preprocessing = _optional_import("sklearn.preprocessing")
        if neural_network is None or pipeline is None or preprocessing is None:
            return None
        estimator = pipeline.make_pipeline(
            preprocessing.StandardScaler(),
            neural_network.MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=max(100, epochs * 20),
                random_state=seed,
            ),
        )
        return SklearnProbabilityModel(estimator), "sklearn"

    return None


@dataclass
class LogisticRegressionGD:
    epochs: int = 50
    learning_rate: float = 0.08

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "LogisticRegressionGD":
        sample_weights = _sample_weights(labels, sample_weights)
        width = len(rows[0]) if rows else 0
        self.weights = [0.0] * width
        self.bias = _safe_logit(_weighted_mean(labels, sample_weights))
        for _ in range(self.epochs):
            for row, label, weight in zip(rows, labels, sample_weights):
                pred = _sigmoid(self.bias + _dot(self.weights, row))
                error = (pred - label) * weight
                self.bias -= self.learning_rate * error
                for index, value in enumerate(row):
                    self.weights[index] -= self.learning_rate * error * value
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> List[float]:
        return [_sigmoid(self.bias + _dot(self.weights, row)) for row in rows]


@dataclass
class ProbabilityTree:
    prediction: float
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "ProbabilityTree | None" = None
    right: "ProbabilityTree | None" = None

    def predict(self, row: Sequence[float]) -> float:
        if self.feature_index is None or self.threshold is None or self.left is None or self.right is None:
            return self.prediction
        return self.left.predict(row) if row[self.feature_index] <= self.threshold else self.right.predict(row)


@dataclass
class StumpForestModel:
    trees: int = 150
    max_depth: int = 8
    min_leaf_samples: int = 3
    seed: int = 13

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "StumpForestModel":
        rng = random.Random(self.seed)
        sample_weights = _sample_weights(labels, sample_weights)
        self.prior = _weighted_mean(labels, sample_weights)
        self.forest: List[ProbabilityTree] = []
        if not rows or len(rows[0]) == 0:
            self.stumps = self.forest
            return self
        for _ in range(max(1, self.trees)):
            sample_indexes = _weighted_sample_indexes(rng, len(rows), sample_weights)
            self.forest.append(
                _build_probability_tree(
                    rows,
                    labels,
                    sample_weights,
                    sample_indexes,
                    depth=0,
                    max_depth=max(1, self.max_depth),
                    min_leaf_samples=max(1, self.min_leaf_samples),
                    default=self.prior,
                )
            )
        self.stumps = self.forest
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> List[float]:
        if not getattr(self, "forest", None):
            return [self.prior] * len(rows)
        return [_clip(sum(tree.predict(row) for tree in self.forest) / len(self.forest)) for row in rows]


@dataclass
class ResidualTree:
    prediction: float
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "ResidualTree | None" = None
    right: "ResidualTree | None" = None

    def value(self, row: Sequence[float]) -> float:
        if self.feature_index is None or self.threshold is None or self.left is None or self.right is None:
            return self.prediction
        return self.left.value(row) if row[self.feature_index] <= self.threshold else self.right.value(row)


@dataclass
class BoostingStumpModel:
    rounds: int = 120
    learning_rate: float = 0.08
    max_depth: int = 6
    subsample: float = 0.8
    seed: int = 13

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "BoostingStumpModel":
        rng = random.Random(self.seed)
        sample_weights = _sample_weights(labels, sample_weights)
        self.base = _safe_logit(_weighted_mean(labels, sample_weights))
        self.trees: List[ResidualTree] = []
        self.round_sample_counts: List[int] = []
        if not rows or len(rows[0]) == 0:
            self.stumps = self.trees
            return self
        scores = [self.base for _ in rows]
        for _ in range(max(1, self.rounds)):
            probabilities = [_sigmoid(score) for score in scores]
            residuals = [label - probability for label, probability in zip(labels, probabilities)]
            sample_indexes = _subsample_indexes(rng, len(rows), self.subsample)
            self.round_sample_counts.append(len(sample_indexes))
            tree = _build_residual_tree(
                rows,
                residuals,
                sample_weights,
                sample_indexes,
                depth=0,
                max_depth=max(1, self.max_depth),
            )
            self.trees.append(tree)
            for index, row in enumerate(rows):
                scores[index] += self.learning_rate * tree.value(row)
        self.stumps = self.trees
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> List[float]:
        output = []
        for row in rows:
            score = self.base + sum(self.learning_rate * tree.value(row) for tree in self.trees)
            output.append(_sigmoid(score))
        return output


@dataclass
class NeuralStyleModel:
    hidden_layers: tuple[int, int] = (64, 32)
    epochs: int = 50
    seed: int = 13

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "NeuralStyleModel":
        self._projection_seed = self.seed
        if not rows or len(rows[0]) == 0:
            self._means = []
            self._stds = []
            self.output = LogisticRegressionGD(epochs=self.epochs, learning_rate=0.05).fit([[] for _ in labels], labels, sample_weights=sample_weights)
            return self
        self._means, self._stds = _standardization_stats(rows)
        projected = [self._project(self._standardize(row)) for row in rows]
        self.output = LogisticRegressionGD(epochs=self.epochs, learning_rate=0.05).fit(projected, labels, sample_weights=sample_weights)
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> List[float]:
        return self.output.predict_proba([self._project(self._standardize(row)) for row in rows])

    def _standardize(self, row: Sequence[float]) -> List[float]:
        # Center/scale features so large label-encoded columns can't dominate the
        # random projection and saturate the downstream sigmoid.
        if not getattr(self, "_means", None):
            return list(row)
        return [(value - mean) / std for value, mean, std in zip(row, self._means, self._stds)]

    def _project(self, row: Sequence[float]) -> List[float]:
        rng = random.Random(self._projection_seed + len(row))
        first_width = min(self.hidden_layers[0], 12)
        second_width = min(self.hidden_layers[1], 8)
        # Scale each layer by 1/sqrt(fan_in) (He/Xavier-style) so activations stay O(1).
        scale_one = 1.0 / math.sqrt(len(row)) if row else 1.0
        hidden_one = []
        for _ in range(first_width):
            weights = [rng.uniform(-1.0, 1.0) for _ in row]
            hidden_one.append(max(0.0, _dot(weights, row) * scale_one))
        scale_two = 1.0 / math.sqrt(len(hidden_one)) if hidden_one else 1.0
        hidden_two = []
        for _ in range(second_width):
            weights = [rng.uniform(-1.0, 1.0) for _ in hidden_one]
            hidden_two.append(max(0.0, _dot(weights, hidden_one) * scale_two))
        return list(row) + hidden_one + hidden_two


class ConstantProbabilityModel:
    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "ConstantProbabilityModel":
        self.probability = _weighted_mean(labels, _sample_weights(labels, sample_weights))
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> List[float]:
        return [_clip(self.probability) for _ in rows]


class SklearnProbabilityModel:
    def __init__(self, estimator: object) -> None:
        self.estimator = estimator
        self.constant_model: ConstantProbabilityModel | None = None

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "SklearnProbabilityModel":
        if not rows or len(rows[0]) == 0 or len(set(labels)) < 2:
            self.constant_model = ConstantProbabilityModel().fit(rows, labels, sample_weights=sample_weights)
            return self
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.estimator.fit(rows, labels, sample_weight=sample_weights)
            except (TypeError, ValueError):
                self.estimator.fit(rows, labels)
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> List[float]:
        if not rows:
            return []
        if self.constant_model is not None:
            return self.constant_model.predict_proba(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            probabilities = self.estimator.predict_proba(rows)
        return [_clip(float(row[1])) for row in probabilities]


class WeightedEnsembleModel:
    def __init__(
        self,
        models: Dict[str, object],
        weights: Dict[str, float],
        component_backends: Dict[str, str] | None = None,
        accelerated_requested: bool = False,
        n_jobs: int | None = None,
    ) -> None:
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        self.models = models
        self.weights = {name: value / total for name, value in weights.items()}
        self.component_backends = component_backends or {name: "unknown" for name in models}
        self.accelerated_requested = accelerated_requested
        self.n_jobs = _parallel_n_jobs(n_jobs)

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "WeightedEnsembleModel":
        active_models = [(name, model) for name, model in self.models.items() if self.weights.get(name, 0.0) > 0]
        parallel = _joblib_parallel()
        if parallel is not None and self.n_jobs not in (0, 1) and len(active_models) > 1:
            Parallel, delayed = parallel
            fitted = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_fit_component)(name, model, rows, labels, sample_weights)
                for name, model in active_models
            )
            for name, model in fitted:
                self.models[name] = model
            return self

        for name, model in active_models:
            model.fit(rows, labels, sample_weights=sample_weights)
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> List[float]:
        blended = [0.0] * len(rows)
        for name, probabilities in self.predict_components(rows).items():
            weight = self.weights.get(name, 0.0)
            for index, probability in enumerate(probabilities):
                blended[index] += weight * probability
        return [_clip(value) for value in blended]

    def predict_components(self, rows: Sequence[Sequence[float]]) -> Dict[str, List[float]]:
        components: Dict[str, List[float]] = {}
        for name, model in self.models.items():
            if self.weights.get(name, 0.0) == 0:
                continue
            components[name] = [_clip(value) for value in model.predict_proba(rows)]
        return components


def _fit_component(
    name: str,
    model: object,
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None,
) -> tuple[str, object]:
    model.fit(rows, labels, sample_weights=sample_weights)
    return name, model


def _optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _joblib_parallel() -> tuple[object, object] | None:
    joblib = _optional_import("joblib")
    if joblib is None:
        return None
    return joblib.Parallel, joblib.delayed


def _parallel_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is not None:
        return int(n_jobs)
    raw_value = os.environ.get("CS2PICKEM_MODEL_JOBS", "-1")
    try:
        parsed = int(raw_value)
    except ValueError:
        return -1
    return parsed if parsed != 0 else 1


def _estimator_n_jobs(n_jobs: int | None) -> int:
    if n_jobs in (None, 0, 1):
        return 1
    return 1


def _build_probability_tree(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    sample_weights: Sequence[float],
    indexes: Sequence[int],
    depth: int,
    max_depth: int,
    min_leaf_samples: int,
    default: float,
) -> ProbabilityTree:
    prediction = _weighted_mean([labels[index] for index in indexes], [sample_weights[index] for index in indexes], default)
    if depth >= max_depth or len(indexes) < min_leaf_samples * 2:
        return ProbabilityTree(prediction=prediction, depth=depth)
    split = _best_probability_split(rows, labels, sample_weights, indexes, min_leaf_samples)
    if split is None:
        return ProbabilityTree(prediction=prediction, depth=depth)
    feature_index, threshold, left_indexes, right_indexes = split
    left = _build_probability_tree(rows, labels, sample_weights, left_indexes, depth + 1, max_depth, min_leaf_samples, prediction)
    right = _build_probability_tree(rows, labels, sample_weights, right_indexes, depth + 1, max_depth, min_leaf_samples, prediction)
    return ProbabilityTree(
        prediction=prediction,
        depth=max(left.depth, right.depth),
        feature_index=feature_index,
        threshold=threshold,
        left=left,
        right=right,
    )


def _best_probability_split(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    sample_weights: Sequence[float],
    indexes: Sequence[int],
    min_leaf_samples: int,
) -> tuple[int, float, List[int], List[int]] | None:
    current_loss = _weighted_gini(labels, sample_weights, indexes)
    best_loss = current_loss
    best_split: tuple[int, float, List[int], List[int]] | None = None
    width = len(rows[0])
    for feature_index in range(width):
        values = [rows[index][feature_index] for index in indexes]
        for threshold in _candidate_thresholds(values):
            left_indexes, right_indexes = _split_indexes(rows, indexes, feature_index, threshold)
            if len(left_indexes) < min_leaf_samples or len(right_indexes) < min_leaf_samples:
                continue
            total_weight = sum(sample_weights[index] for index in indexes)
            left_weight = sum(sample_weights[index] for index in left_indexes)
            right_weight = sum(sample_weights[index] for index in right_indexes)
            if total_weight <= 0:
                continue
            loss = (
                left_weight * _weighted_gini(labels, sample_weights, left_indexes)
                + right_weight * _weighted_gini(labels, sample_weights, right_indexes)
            ) / total_weight
            if loss < best_loss - 1e-12:
                best_loss = loss
                best_split = (feature_index, threshold, left_indexes, right_indexes)
    return best_split


def _build_residual_tree(
    rows: Sequence[Sequence[float]],
    residuals: Sequence[float],
    sample_weights: Sequence[float],
    indexes: Sequence[int],
    depth: int,
    max_depth: int,
) -> ResidualTree:
    prediction = _weighted_mean([residuals[index] for index in indexes], [sample_weights[index] for index in indexes], 0.0)
    if depth >= max_depth or len(indexes) < 2:
        return ResidualTree(prediction=prediction, depth=depth)
    split = _best_residual_split(rows, residuals, sample_weights, indexes)
    if split is None:
        return ResidualTree(prediction=prediction, depth=depth)
    feature_index, threshold, left_indexes, right_indexes = split
    left = _build_residual_tree(rows, residuals, sample_weights, left_indexes, depth + 1, max_depth)
    right = _build_residual_tree(rows, residuals, sample_weights, right_indexes, depth + 1, max_depth)
    return ResidualTree(
        prediction=prediction,
        depth=max(left.depth, right.depth),
        feature_index=feature_index,
        threshold=threshold,
        left=left,
        right=right,
    )


def _best_residual_split(
    rows: Sequence[Sequence[float]],
    residuals: Sequence[float],
    sample_weights: Sequence[float],
    indexes: Sequence[int],
) -> tuple[int, float, List[int], List[int]] | None:
    current_loss = _weighted_squared_error(residuals, sample_weights, indexes)
    best_loss = current_loss
    best_split: tuple[int, float, List[int], List[int]] | None = None
    width = len(rows[0])
    for feature_index in range(width):
        values = [rows[index][feature_index] for index in indexes]
        for threshold in _candidate_thresholds(values):
            left_indexes, right_indexes = _split_indexes(rows, indexes, feature_index, threshold)
            if not left_indexes or not right_indexes:
                continue
            loss = _weighted_squared_error(residuals, sample_weights, left_indexes) + _weighted_squared_error(residuals, sample_weights, right_indexes)
            if loss < best_loss - 1e-12:
                best_loss = loss
                best_split = (feature_index, threshold, left_indexes, right_indexes)
    return best_split


def _split_indexes(
    rows: Sequence[Sequence[float]],
    indexes: Sequence[int],
    feature_index: int,
    threshold: float,
) -> tuple[List[int], List[int]]:
    left_indexes: List[int] = []
    right_indexes: List[int] = []
    for index in indexes:
        if rows[index][feature_index] <= threshold:
            left_indexes.append(index)
        else:
            right_indexes.append(index)
    return left_indexes, right_indexes


def _candidate_thresholds(values: Sequence[float]) -> List[float]:
    unique_values = sorted(set(values))
    if len(unique_values) <= 1:
        return []
    thresholds = [(left + right) / 2.0 for left, right in zip(unique_values, unique_values[1:])]
    if len(thresholds) <= 16:
        return thresholds
    step = (len(thresholds) - 1) / 15
    return sorted({thresholds[round(index * step)] for index in range(16)})


def _weighted_gini(labels: Sequence[int], sample_weights: Sequence[float], indexes: Sequence[int]) -> float:
    total_weight = sum(sample_weights[index] for index in indexes)
    if total_weight <= 0:
        return 0.0
    positive_weight = sum(sample_weights[index] for index in indexes if labels[index] == 1)
    positive = positive_weight / total_weight
    negative = 1.0 - positive
    return 1.0 - positive**2 - negative**2


def _weighted_squared_error(values: Sequence[float], sample_weights: Sequence[float], indexes: Sequence[int]) -> float:
    if not indexes:
        return 0.0
    weights = [sample_weights[index] for index in indexes]
    mean = _weighted_mean([values[index] for index in indexes], weights, 0.0)
    return sum(sample_weights[index] * (values[index] - mean) ** 2 for index in indexes)


def _subsample_indexes(rng: random.Random, size: int, subsample: float) -> List[int]:
    indexes = list(range(size))
    if not indexes:
        return []
    fraction = _clip(subsample, 0.0, 1.0)
    count = max(1, int(size * fraction)) if fraction < 1.0 else size
    if count >= size:
        return indexes
    return rng.sample(indexes, count)


def _dot(weights: Sequence[float], row: Sequence[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, row))


def _standardization_stats(rows: Sequence[Sequence[float]]) -> tuple[List[float], List[float]]:
    """Per-column mean and std (std floored to 1.0 to avoid divide-by-zero / amplifying constants)."""
    width = len(rows[0])
    n = len(rows)
    means = [sum(row[i] for row in rows) / n for i in range(width)]
    stds = []
    for i in range(width):
        variance = sum((row[i] - means[i]) ** 2 for row in rows) / n
        std = math.sqrt(variance)
        stds.append(std if std > 1e-9 else 1.0)
    return means, stds


def _sigmoid(value: float) -> float:
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _safe_logit(probability: float) -> float:
    probability = _clip(probability, 0.001, 0.999)
    return math.log(probability / (1.0 - probability))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def _mean(values: Iterable[float], default: float = 0.5) -> float:
    values = list(values)
    return sum(values) / len(values) if values else default


def _weighted_mean(values: Iterable[float], weights: Sequence[float], default: float = 0.5) -> float:
    values = list(values)
    weights = list(weights)
    total = sum(weights)
    if not values or total <= 0:
        return default
    return sum(float(value) * weight for value, weight in zip(values, weights)) / total


def _sample_weights(labels: Sequence[int], sample_weights: Sequence[float] | None) -> List[float]:
    if sample_weights is None:
        return [1.0] * len(labels)
    if len(sample_weights) != len(labels):
        raise ValueError("sample_weights length must match labels")
    return [float(weight) for weight in sample_weights]


def _weighted_sample_indexes(rng: random.Random, size: int, weights: Sequence[float]) -> List[int]:
    indexes = list(range(size))
    if not indexes:
        return []
    if sum(weights) <= 0:
        return [rng.randrange(size) for _ in indexes]
    return rng.choices(indexes, weights=weights, k=size)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.5
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
