import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _rows(n: int):
    return [{"date": f"2026-05-{day:02d}", "value": day} for day in range(1, n + 1)]


class TimeSeriesFoldsEmbargoTests(unittest.TestCase):
    def test_default_args_match_legacy_folds_exactly(self):
        # Regression guard: with default embargo/purge the folds must be
        # byte-for-byte identical to the historical (gap-free) behaviour.
        from cs2pickem.splitting import time_series_folds

        rows = _rows(20)
        legacy = []
        ordered = sorted(rows, key=lambda r: r["date"])
        validation_size = max(1, len(ordered) // (3 + 1))
        for fold_index in range(1, 4):
            train_end = validation_size * fold_index
            validation_end = train_end + validation_size
            if validation_end > len(ordered):
                break
            legacy.append((ordered[:train_end], ordered[train_end:validation_end]))

        produced = list(time_series_folds(rows, folds=3))
        self.assertEqual(len(produced), len(legacy))
        for (lt, lv), (pt, pv) in zip(legacy, produced):
            self.assertEqual([r["value"] for r in pt], [r["value"] for r in lt])
            self.assertEqual([r["value"] for r in pv], [r["value"] for r in lv])

    def test_embargo_creates_gap_between_train_and_validation(self):
        from cs2pickem.splitting import time_series_folds

        rows = _rows(20)
        embargo = 2
        folds = list(time_series_folds(rows, folds=3, embargo=embargo))
        self.assertTrue(folds)
        for train, validation in folds:
            self.assertTrue(train)
            self.assertTrue(validation)
            max_train = max(r["value"] for r in train)
            min_validation = min(r["value"] for r in validation)
            # There must be at least `embargo` dropped rows between the last
            # training sample and the first validation sample.
            self.assertGreaterEqual(min_validation - max_train, embargo + 1)

    def test_purge_drops_tail_of_train_window(self):
        from cs2pickem.splitting import time_series_folds

        rows = _rows(20)
        baseline = list(time_series_folds(rows, folds=3))
        purged = list(time_series_folds(rows, folds=3, purge=2))
        self.assertEqual(len(baseline), len(purged))
        for (bt, _bv), (pt, _pv) in zip(baseline, purged):
            self.assertEqual(len(pt), len(bt) - 2)
            # The purged train is a strict prefix of the baseline train.
            self.assertEqual(
                [r["value"] for r in pt],
                [r["value"] for r in bt][: len(pt)],
            )

    def test_embargo_keeps_chronology_and_skips_degenerate_folds(self):
        from cs2pickem.splitting import time_series_folds

        rows = _rows(12)
        folds = list(time_series_folds(rows, folds=3, embargo=1, purge=1))
        for train, validation in folds:
            if train and validation:
                self.assertLess(
                    max(r["value"] for r in train),
                    min(r["value"] for r in validation),
                )


class TimeSeriesSplitGapTests(unittest.TestCase):
    def test_default_gap_matches_legacy_split(self):
        from cs2pickem.splitting import time_series_split

        rows = _rows(10)
        split = time_series_split(rows, train_ratio=0.6, validation_ratio=0.2)
        self.assertEqual([r["value"] for r in split.train], [1, 2, 3, 4, 5, 6])
        self.assertEqual([r["value"] for r in split.validation], [7, 8])
        self.assertEqual([r["value"] for r in split.test], [9, 10])

    def test_gap_train_validation_creates_embargo(self):
        from cs2pickem.splitting import time_series_split

        rows = _rows(10)
        split = time_series_split(
            rows, train_ratio=0.6, validation_ratio=0.2, gap_train_validation=1
        )
        max_train = max(r["value"] for r in split.train)
        min_validation = min(r["value"] for r in split.validation)
        self.assertGreaterEqual(min_validation - max_train, 2)

    def test_train_validation_gap_preserves_window_and_loses_no_rows(self):
        # Regression guard for the old `max(validation_end, validation_start)`
        # bug: a train<->validation embargo used to *shrink* the validation
        # window and make boundary rows vanish (neither in val, test, nor gap).
        # The fix keeps the validation window full-size and accounts for every
        # row: only exactly `gap_train_validation` rows are buffered out.
        from cs2pickem.splitting import time_series_split

        rows = _rows(10)
        baseline = time_series_split(rows, train_ratio=0.6, validation_ratio=0.2)
        gapped = time_series_split(
            rows,
            train_ratio=0.6,
            validation_ratio=0.2,
            gap_train_validation=1,
            gap_validation_test=0,
        )
        # The validation window keeps its full size (slides forward, not shrunk).
        self.assertEqual(len(gapped.validation), len(baseline.validation))
        # Exactly `gap_train_validation` rows are buffered out; none vanish.
        kept = len(gapped.train) + len(gapped.validation) + len(gapped.test)
        self.assertEqual(kept, len(rows) - 1)
        # The only excluded row is the buffer row at the train/val boundary.
        kept_values = (
            {r["value"] for r in gapped.train}
            | {r["value"] for r in gapped.validation}
            | {r["value"] for r in gapped.test}
        )
        dropped = {r["value"] for r in rows} - kept_values
        self.assertEqual(len(dropped), 1)
        # The dropped row sits in the gap immediately after training ends.
        self.assertEqual(dropped, {max(r["value"] for r in gapped.train) + 1})

    def test_gap_too_large_raises_instead_of_silently_emptying(self):
        from cs2pickem.splitting import time_series_split

        rows = _rows(10)
        with self.assertRaises(ValueError):
            time_series_split(
                rows,
                train_ratio=0.6,
                validation_ratio=0.2,
                gap_train_validation=5,
            )


class TimeSeriesDateSplitEmbargoTests(unittest.TestCase):
    def test_default_embargo_days_matches_legacy(self):
        from cs2pickem.splitting import time_series_date_split

        rows = _rows(10)
        split = time_series_date_split(
            rows, train_end_date="2026-05-04", validation_end_date="2026-05-07"
        )
        self.assertEqual([r["value"] for r in split.train], [1, 2, 3, 4])
        self.assertEqual([r["value"] for r in split.validation], [5, 6, 7])
        self.assertEqual([r["value"] for r in split.test], [8, 9, 10])

    def test_embargo_days_drops_rows_near_boundary(self):
        from cs2pickem.splitting import time_series_date_split

        rows = _rows(10)
        split = time_series_date_split(
            rows,
            train_end_date="2026-05-04",
            validation_end_date="2026-05-07",
            embargo_days=1,
        )
        # The 5/5 row falls within 1 day of the train boundary and is dropped
        # from validation, leaving a gap between train and validation.
        self.assertEqual([r["value"] for r in split.train], [1, 2, 3, 4])
        self.assertNotIn(5, [r["value"] for r in split.validation])
        max_train = max(r["value"] for r in split.train)
        min_validation = min(r["value"] for r in split.validation)
        self.assertGreater(min_validation - max_train, 1)


if __name__ == "__main__":
    unittest.main()
