from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Tuple


@dataclass
class TimeSplit:
    train: List[Dict[str, Any]]
    validation: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


def time_series_split(
    rows: Iterable[Dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> TimeSplit:
    ordered = sorted(rows, key=lambda row: row["date"])
    count = len(ordered)
    train_end = int(count * train_ratio)
    validation_size = int(count * validation_ratio)
    if count >= 3 and validation_ratio > 0:
        train_end = max(1, min(train_end, count - 2))
        validation_size = max(1, validation_size)
        validation_end = min(train_end + validation_size, count - 1)
    else:
        validation_end = train_end + validation_size
    return TimeSplit(
        train=ordered[:train_end],
        validation=ordered[train_end:validation_end],
        test=ordered[validation_end:],
    )


def time_series_date_split(
    rows: Iterable[Dict[str, Any]],
    train_end_date: str,
    validation_end_date: str,
) -> TimeSplit:
    ordered = sorted(rows, key=lambda row: row["date"])
    train = [row for row in ordered if str(row["date"])[:10] <= train_end_date]
    validation = [row for row in ordered if train_end_date < str(row["date"])[:10] <= validation_end_date]
    test = [row for row in ordered if str(row["date"])[:10] > validation_end_date]
    return TimeSplit(train=train, validation=validation, test=test)


def time_series_folds(rows: Iterable[Dict[str, Any]], folds: int = 5) -> Iterator[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    ordered = sorted(rows, key=lambda row: row["date"])
    if folds < 1:
        raise ValueError("folds must be >= 1")
    validation_size = max(1, len(ordered) // (folds + 1))
    for fold_index in range(1, folds + 1):
        train_end = validation_size * fold_index
        validation_end = train_end + validation_size
        if validation_end > len(ordered):
            break
        yield ordered[:train_end], ordered[train_end:validation_end]
