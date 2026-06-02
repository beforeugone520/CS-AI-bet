from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class RebalancedTrainingData:
    rows: List[List[float]]
    labels: List[int]
    sample_weights: List[float]
    report: Dict[str, object]


def rebalance_training_data(rows: Sequence[Sequence[float]], labels: Sequence[int]) -> RebalancedTrainingData:
    """Apply deterministic SMOTE-like minority oversampling plus class weights."""

    original_rows = [list(row) for row in rows]
    original_labels = [int(label) for label in labels]
    counts = Counter(original_labels)
    class_weights = _class_weights(counts)

    balanced_rows = [list(row) for row in original_rows]
    balanced_labels = list(original_labels)
    synthetic_rows = 0
    target_count = max(counts.values()) if counts else 0

    for label in sorted(counts):
        label_rows = [row for row, row_label in zip(original_rows, original_labels) if row_label == label]
        needed = target_count - counts[label]
        for index in range(needed):
            balanced_rows.append(_synthetic_row(label_rows, index, needed))
            balanced_labels.append(label)
            synthetic_rows += 1

    balanced_counts = Counter(balanced_labels)
    sample_weights = [class_weights.get(label, 1.0) for label in balanced_labels]
    strategy = "smote_minority_oversample_plus_class_weight" if synthetic_rows or len(set(class_weights.values())) > 1 else "class_weight"
    report: Dict[str, object] = {
        "strategy": strategy,
        "original_counts": _string_counts(counts),
        "balanced_counts": _string_counts(balanced_counts),
        "synthetic_rows": synthetic_rows,
        "class_weights": {str(label): weight for label, weight in sorted(class_weights.items())},
    }
    return RebalancedTrainingData(balanced_rows, balanced_labels, sample_weights, report)


def _class_weights(counts: Counter) -> Dict[int, float]:
    if not counts:
        return {}
    total = sum(counts.values())
    classes = len(counts)
    return {int(label): total / (classes * count) for label, count in counts.items() if count}


def _string_counts(counts: Counter) -> Dict[str, int]:
    return {str(label): count for label, count in sorted(counts.items())}


def _synthetic_row(rows: List[List[float]], index: int, needed: int) -> List[float]:
    if not rows:
        return []
    if len(rows) == 1:
        return list(rows[0])
    left = rows[index % len(rows)]
    right = rows[(index + 1) % len(rows)]
    alpha = (index + 1) / (needed + 1)
    return [float(a) + (float(b) - float(a)) * alpha for a, b in zip(left, right)]
