from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from .cleaning import parse_date


@dataclass
class TimeSplit:
    train: List[Dict[str, Any]]
    validation: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


def time_series_split(
    rows: Iterable[Dict[str, Any]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    *,
    gap_train_validation: int = 0,
    gap_validation_test: int = 0,
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
    if gap_train_validation <= 0 and gap_validation_test <= 0:
        return TimeSplit(
            train=ordered[:train_end],
            validation=ordered[train_end:validation_end],
            test=ordered[validation_end:],
        )
    # Optional purge/embargo gaps. A time buffer of `gap_train_validation` rows
    # is inserted between train and validation, and `gap_validation_test` rows
    # between validation and test; rows landing in a gap are excluded from every
    # split. This only places a temporal buffer at the boundaries to reduce
    # near-duplicate adjacent samples -- it does NOT (and need not) alter the
    # rolling features that were already computed causally upstream, which only
    # ever look at the past. The validation window keeps its intended size
    # (it slides forward by the gap) instead of being absorbed by the gap, so a
    # train-validation buffer never silently shrinks the held-out test segment.
    gap_tv = max(0, int(gap_train_validation))
    gap_vt = max(0, int(gap_validation_test))
    validation_size = max(0, validation_end - train_end)
    validation_start = train_end + gap_tv
    validation_stop = validation_start + validation_size
    test_start = validation_stop + gap_vt
    if test_start > count or validation_start > count:
        raise ValueError("gap too large for dataset size")
    return TimeSplit(
        train=ordered[:train_end],
        validation=ordered[validation_start:validation_stop],
        test=ordered[test_start:],
    )


def time_series_date_split(
    rows: Iterable[Dict[str, Any]],
    train_end_date: str,
    validation_end_date: str,
    *,
    embargo_days: int = 0,
) -> TimeSplit:
    ordered = sorted(rows, key=lambda row: row["date"])
    train = [row for row in ordered if str(row["date"])[:10] <= train_end_date]
    validation = [row for row in ordered if train_end_date < str(row["date"])[:10] <= validation_end_date]
    test = [row for row in ordered if str(row["date"])[:10] > validation_end_date]
    if embargo_days and embargo_days > 0:
        # Drop validation rows that fall within `embargo_days` of the training
        # boundary so a calendar gap separates train from validation.
        train_boundary = parse_date(train_end_date)
        cutoff = train_boundary + timedelta(days=int(embargo_days))
        validation = [row for row in validation if parse_date(str(row["date"])[:10]) > cutoff]
    return TimeSplit(train=train, validation=validation, test=test)


def time_series_folds(
    rows: Iterable[Dict[str, Any]],
    folds: int = 5,
    *,
    embargo: int = 0,
    purge: int = 0,
) -> Iterator[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    ordered = sorted(rows, key=lambda row: row["date"])
    if folds < 1:
        raise ValueError("folds must be >= 1")
    embargo = max(0, int(embargo))
    purge = max(0, int(purge))
    validation_size = max(1, len(ordered) // (folds + 1))
    for fold_index in range(1, folds + 1):
        train_end = validation_size * fold_index
        if embargo <= 0 and purge <= 0:
            validation_end = train_end + validation_size
            if validation_end > len(ordered):
                break
            yield ordered[:train_end], ordered[train_end:validation_end]
            continue
        # Walk-forward with a gap: validation starts `embargo` rows after the
        # training window, and the last `purge` rows of the training window are
        # dropped. Rows in [train_end - purge, train_end + embargo) are excluded
        # from both train and validation. This buffers the boundary to reduce
        # near-duplicate adjacent samples; it does not change the causally
        # computed rolling features (those only look at the past and need no
        # row-level purge), so treat the gap as a temporal buffer, not a claim
        # that all cross-boundary information has been removed.
        validation_start = train_end + embargo
        validation_end = validation_start + validation_size
        if validation_end > len(ordered):
            break
        train_cut = max(0, train_end - purge)
        train_fold = ordered[:train_cut]
        validation_fold = ordered[validation_start:validation_end]
        if not train_fold or not validation_fold:
            continue
        yield train_fold, validation_fold
