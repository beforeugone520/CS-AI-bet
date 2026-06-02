from __future__ import annotations

import math
from typing import Iterable, Sequence


def accuracy(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for label, probability in zip(labels, probabilities) if int(probability >= threshold) == label)
    return correct / len(labels)


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for label, probability in zip(labels, probabilities):
        probability = min(0.999999, max(0.000001, probability))
        total += -(label * math.log(probability) + (1 - label) * math.log(1 - probability))
    return total / len(labels)


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean squared error between predicted probabilities and outcomes (lower is better)."""
    if not labels:
        return 0.0
    return sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)


def calibration_table(labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> dict:
    """Reliability bins plus expected calibration error (ECE).

    Each bin reports its predicted-probability range, the count, the mean predicted
    probability, and the observed positive frequency. ECE is the count-weighted mean
    gap between predicted and observed across non-empty bins.
    """
    buckets: list = [[] for _ in range(bins)]
    for label, probability in zip(labels, probabilities):
        probability = min(1.0, max(0.0, probability))
        index = min(bins - 1, int(probability * bins))
        buckets[index].append((label, probability))
    total = len(labels)
    rows = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        low, high = index / bins, (index + 1) / bins
        count = len(bucket)
        if count:
            mean_predicted = sum(p for _, p in bucket) / count
            observed = sum(l for l, _ in bucket) / count
            if total:
                ece += (count / total) * abs(mean_predicted - observed)
        else:
            mean_predicted = observed = 0.0
        rows.append({
            "range": [round(low, 3), round(high, 3)],
            "count": count,
            "mean_predicted": mean_predicted,
            "observed_frequency": observed,
        })
    return {"bins": rows, "ece": ece}


def auc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    positives = [(probability, label) for label, probability in zip(labels, probabilities) if label == 1]
    negatives = [(probability, label) for label, probability in zip(labels, probabilities) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive, _ in positives:
        for negative, _ in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def profit_loss(labels: Sequence[int], probabilities: Sequence[float], decimal_odds: Iterable[float], stake: float = 1.0) -> float:
    total = 0.0
    for label, probability, odds in zip(labels, probabilities, decimal_odds):
        pick = int(probability >= 0.5)
        total += stake * (float(odds) - 1.0) if pick == label else -stake
    return total
