from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass
class ProbabilityCalibrator:
    """Platt-style logistic calibration for already-produced probabilities."""

    epochs: int = 120
    learning_rate: float = 0.25
    l2: float = 0.001

    def fit(
        self,
        probabilities: Sequence[float],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "ProbabilityCalibrator":
        pairs = [
            (_safe_logit(probability), 1 if label else 0, float(weight))
            for probability, label, weight in zip(probabilities, labels, sample_weights or [1.0] * len(labels))
        ]
        self.training_count = len(pairs)
        self.positive_rate = (sum(label for _, label, _ in pairs) / len(pairs)) if pairs else 0.0
        self.slope = 1.0
        self.intercept = 0.0
        if not pairs:
            self.basis = "no_calibration_rows"
            return self

        total_weight = sum(weight for _, _, weight in pairs) or 1.0
        for _ in range(max(1, self.epochs)):
            slope_gradient = self.l2 * (self.slope - 1.0)
            intercept_gradient = 0.0
            for feature, label, weight in pairs:
                prediction = _sigmoid(self.slope * feature + self.intercept)
                error = (prediction - label) * weight
                slope_gradient += error * feature / total_weight
                intercept_gradient += error / total_weight
            self.slope -= self.learning_rate * slope_gradient
            self.intercept -= self.learning_rate * intercept_gradient
        self.basis = "platt_logistic"
        return self

    def transform(self, probabilities: Sequence[float]) -> list[float]:
        if getattr(self, "basis", "no_calibration_rows") == "no_calibration_rows":
            return [_clip(probability) for probability in probabilities]
        return [_clip(_sigmoid(self.slope * _safe_logit(probability) + self.intercept)) for probability in probabilities]

    def transform_one(self, probability: float) -> float:
        return self.transform([probability])[0]

    def report(self) -> Dict[str, object]:
        return {
            "basis": getattr(self, "basis", "no_calibration_rows"),
            "training_count": getattr(self, "training_count", 0),
            "positive_rate": getattr(self, "positive_rate", 0.0),
            "slope": getattr(self, "slope", 1.0),
            "intercept": getattr(self, "intercept", 0.0),
        }


def _safe_logit(probability: float) -> float:
    probability = min(0.999999, max(0.000001, float(probability)))
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
