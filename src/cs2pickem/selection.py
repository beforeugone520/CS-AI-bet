from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class SelectedDataset:
    rows: List[List[float]]
    feature_names: List[str]
    selected_indexes: List[int]


class FeatureSelector:
    """Variance, correlation, and importance filtering for model-ready matrices."""

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.8,
        top_k: int = 25,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self.importance_scores: Dict[str, float] = {}
        self.selected_indexes: List[int] = []
        self.selected_feature_names: List[str] = []

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[int], feature_names: Sequence[str]) -> "FeatureSelector":
        if not rows:
            self.selected_indexes = []
            self.selected_feature_names = []
            self.importance_scores = {}
            return self

        surviving = self._variance_filter(rows)
        surviving = self._correlation_filter(rows, surviving)
        label_values = [float(label) for label in labels]
        self.importance_scores = {
            feature_names[index]: abs(_pearson(_column(rows, index), label_values))
            for index in surviving
        }
        ordered = sorted(surviving, key=lambda index: (-self.importance_scores[feature_names[index]], index))
        self.selected_indexes = ordered[: self.top_k]
        self.selected_feature_names = [feature_names[index] for index in self.selected_indexes]
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> SelectedDataset:
        return SelectedDataset(
            rows=[[row[index] for index in self.selected_indexes] for row in rows],
            feature_names=list(self.selected_feature_names),
            selected_indexes=list(self.selected_indexes),
        )

    def fit_transform(self, rows: Sequence[Sequence[float]], labels: Sequence[int], feature_names: Sequence[str]) -> SelectedDataset:
        self.fit(rows, labels, feature_names)
        return self.transform(rows)

    def _variance_filter(self, rows: Sequence[Sequence[float]]) -> List[int]:
        width = len(rows[0])
        return [
            index
            for index in range(width)
            if _variance(_column(rows, index)) >= self.variance_threshold
        ]

    def _correlation_filter(self, rows: Sequence[Sequence[float]], indexes: Sequence[int]) -> List[int]:
        kept: List[int] = []
        for index in indexes:
            values = _column(rows, index)
            if all(abs(_pearson(values, _column(rows, kept_index))) <= self.correlation_threshold for kept_index in kept):
                kept.append(index)
        return kept


def _column(rows: Sequence[Sequence[float]], index: int) -> List[float]:
    return [float(row[index]) for row in rows]


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = sum(values) / len(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    denominator_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    denominator = denominator_left * denominator_right
    return numerator / denominator if denominator else 0.0
