from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, List, Sequence, Tuple

try:  # optional; only used to sharpen Diebold-Mariano tail probabilities
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _scipy_stats = None


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


# ---------------------------------------------------------------------------
# Statistical rigour layer: Brier decomposition, skill score, ECE/MCE with
# selectable binning, bootstrap confidence intervals, paired bootstrap
# comparison, and the Diebold-Mariano test with HLN small-sample correction.
#
# Every routine has a pure-Python implementation (standard library only).
# numpy / scipy are used *only* to accelerate or sharpen results when present
# and are never required; all random processes are seedable for reproducibility.
# ---------------------------------------------------------------------------


def brier_decomposition(
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> dict:
    """Murphy (1973) three-component decomposition of the Brier score.

    ``Brier = reliability - resolution + uncertainty`` where

    * reliability  (calibration, lower is better) = (1/n) Σ n_k (f_k - o_k)^2
    * resolution   (discrimination, higher is better) = (1/n) Σ n_k (o_k - ō)^2
    * uncertainty  (irreducible base-rate variance) = ō (1 - ō)

    with K equal-width bins on the predicted probability, ``f_k`` the mean
    prediction in bin ``k``, ``o_k`` the observed positive rate in bin ``k``,
    and ``ō`` the global positive rate.  The identity is exact whenever every
    prediction inside a bin is identical (e.g. bins finer than the number of
    distinct predictions) and an equal-width-binning approximation otherwise.

    Returns a dict with ``reliability``, ``resolution``, ``uncertainty``,
    ``brier`` (true mean squared error), ``reconstructed``
    (= reliability - resolution + uncertainty) and the per-bin ``bins`` rows.
    """
    n = len(labels)
    if n == 0:
        return {
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
            "brier": 0.0,
            "reconstructed": 0.0,
            "bins": [],
        }

    paired = list(zip(labels, probabilities))
    o_bar = sum(label for label, _ in paired) / n
    uncertainty = o_bar * (1.0 - o_bar)

    buckets: List[List[Tuple[int, float]]] = [[] for _ in range(n_bins)]
    for label, probability in paired:
        clipped = min(1.0, max(0.0, probability))
        index = min(n_bins - 1, int(clipped * n_bins))
        buckets[index].append((int(label), clipped))

    reliability = 0.0
    resolution = 0.0
    rows = []
    for index, bucket in enumerate(buckets):
        low, high = index / n_bins, (index + 1) / n_bins
        count = len(bucket)
        if count:
            f_k = sum(p for _, p in bucket) / count
            o_k = sum(l for l, _ in bucket) / count
            reliability += count * (f_k - o_k) ** 2
            resolution += count * (o_k - o_bar) ** 2
        else:
            f_k = o_k = 0.0
        rows.append(
            {
                "range": [round(low, 3), round(high, 3)],
                "count": count,
                "mean_predicted": f_k,
                "observed_frequency": o_k,
            }
        )
    reliability /= n
    resolution /= n

    brier = brier_score(labels, probabilities)
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": brier,
        "reconstructed": reliability - resolution + uncertainty,
        "bins": rows,
    }


def brier_skill_score(
    labels: Sequence[int],
    probabilities: Sequence[float],
    baseline_probabilities: "Sequence[float] | float | None" = None,
) -> float:
    """Brier skill score ``1 - BS_model / BS_reference``.

    ``> 0`` beats the reference, ``= 0`` matches it, ``< 0`` is worse.  When
    ``baseline_probabilities`` is ``None`` the reference is the sample
    climatology (constant equal to the positive rate).  A scalar applies a
    constant baseline (e.g. ``0.5``); a sequence supplies a per-sample baseline
    (e.g. market-implied probabilities).  A degenerate reference (``BS_ref==0``)
    returns ``0.0`` instead of dividing by zero.
    """
    n = len(labels)
    if n == 0:
        return 0.0

    bs_model = brier_score(labels, probabilities)

    if baseline_probabilities is None:
        p0 = sum(labels) / n
        reference = [p0] * n
    elif isinstance(baseline_probabilities, (int, float)):
        reference = [float(baseline_probabilities)] * n
    else:
        reference = list(baseline_probabilities)

    bs_ref = brier_score(labels, reference)
    if bs_ref == 0.0:
        return 0.0
    return 1.0 - bs_model / bs_ref


def _equal_width_bins(
    paired: Sequence[Tuple[int, float]], bins: int
) -> List[List[Tuple[int, float]]]:
    buckets: List[List[Tuple[int, float]]] = [[] for _ in range(bins)]
    for label, probability in paired:
        clipped = min(1.0, max(0.0, probability))
        index = min(bins - 1, int(clipped * bins))
        buckets[index].append((int(label), clipped))
    return buckets


def _equal_mass_bins(
    paired: Sequence[Tuple[int, float]], bins: int
) -> List[List[Tuple[int, float]]]:
    ordered = sorted(((min(1.0, max(0.0, p)), int(l)) for l, p in paired))
    n = len(ordered)
    buckets: List[List[Tuple[int, float]]] = [[] for _ in range(bins)]
    for position, (probability, label) in enumerate(ordered):
        index = min(bins - 1, position * bins // n)
        buckets[index].append((label, probability))
    return buckets


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    bins: int = 10,
    binning: str = "equal_width",
) -> dict:
    """Expected and maximum calibration error with selectable binning.

    ``binning='equal_width'`` reproduces the :func:`calibration_table` ECE
    exactly (fixed ``[k/bins, (k+1)/bins)`` edges).  ``binning='equal_mass'``
    (a.k.a. quantile / adaptive) splits the sorted predictions into bins with
    near-equal sample counts, which avoids empty bins on skewed predictions.

    ECE is the count-weighted mean per-bin gap ``|f_k - o_k|``; MCE is the
    largest per-bin gap over non-empty bins.  Returns ``{'ece', 'mce',
    'binning', 'bins'}`` (bins rows are structurally compatible with
    :func:`calibration_table`).
    """
    total = len(labels)
    if total == 0:
        return {"ece": 0.0, "mce": 0.0, "binning": binning, "bins": []}

    paired = list(zip(labels, probabilities))
    if binning in ("equal_width", "uniform"):
        buckets = _equal_width_bins(paired, bins)
    elif binning in ("equal_mass", "quantile", "adaptive"):
        buckets = _equal_mass_bins(paired, bins)
    else:
        raise ValueError(f"unknown binning strategy: {binning!r}")

    ece = 0.0
    mce = 0.0
    rows = []
    for bucket in buckets:
        count = len(bucket)
        if count:
            mean_predicted = sum(p for _, p in bucket) / count
            observed = sum(l for l, _ in bucket) / count
            gap = abs(mean_predicted - observed)
            ece += (count / total) * gap
            if gap > mce:
                mce = gap
        else:
            mean_predicted = observed = 0.0
        rows.append(
            {
                "count": count,
                "mean_predicted": mean_predicted,
                "observed_frequency": observed,
            }
        )
    return {"ece": ece, "mce": mce, "binning": binning, "bins": rows}


def maximum_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    bins: int = 10,
    binning: str = "equal_width",
) -> float:
    """Largest per-bin calibration gap (MCE). Convenience wrapper over ECE."""
    return expected_calibration_error(labels, probabilities, bins=bins, binning=binning)["mce"]


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' convention)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    rank = fraction * (n - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def _resample_index_sets(
    n: int,
    n_boot: int,
    rng: random.Random,
    groups: "Sequence[Any] | None",
) -> List[List[int]]:
    """Produce ``n_boot`` resampled index lists (sample- or cluster-level)."""
    if groups is None:
        return [[rng.randrange(n) for _ in range(n)] for _ in range(n_boot)]

    group_members: dict = {}
    order: List[Any] = []
    for idx, key in enumerate(groups):
        if key not in group_members:
            group_members[key] = []
            order.append(key)
        group_members[key].append(idx)
    unique = order
    g = len(unique)
    resamples = []
    for _ in range(n_boot):
        chosen = [unique[rng.randrange(g)] for _ in range(g)]
        indices: List[int] = []
        for key in chosen:
            indices.extend(group_members[key])
        resamples.append(indices)
    return resamples


def bootstrap_metric(
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: "Sequence[Any] | None" = None,
) -> dict:
    """Percentile bootstrap confidence interval for any per-sample metric.

    Resamples match-level (or, when ``groups`` is given, whole clusters /
    matches) with replacement ``n_boot`` times, recomputing ``metric_fn`` each
    time.  Returns ``{'point', 'lo', 'hi', 'confidence', 'n_boot', 'std'}``.
    The point estimate is ``metric_fn`` on the original data; ``lo``/``hi`` are
    the ``alpha/2`` and ``1 - alpha/2`` percentiles of the bootstrap
    distribution.  ``seed`` makes the procedure fully reproducible.
    """
    n = len(labels)
    point = float(metric_fn(labels, probabilities)) if n else float(metric_fn([], []))
    if n == 0 or n_boot <= 0:
        return {
            "point": point,
            "lo": point,
            "hi": point,
            "confidence": 1.0 - alpha,
            "n_boot": 0,
            "std": 0.0,
        }

    rng = random.Random(seed)
    index_sets = _resample_index_sets(n, n_boot, rng, groups)
    labels_list = list(labels)
    probs_list = list(probabilities)

    stats: List[float] = []
    for indices in index_sets:
        boot_labels = [labels_list[i] for i in indices]
        boot_probs = [probs_list[i] for i in indices]
        stats.append(float(metric_fn(boot_labels, boot_probs)))

    stats.sort()
    lo = _percentile(stats, alpha / 2.0)
    hi = _percentile(stats, 1.0 - alpha / 2.0)
    mean = sum(stats) / len(stats)
    variance = sum((s - mean) ** 2 for s in stats) / len(stats)
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "confidence": 1.0 - alpha,
        "n_boot": len(stats),
        "std": math.sqrt(variance),
    }


def paired_bootstrap_compare(
    labels: Sequence[int],
    probabilities_a: Sequence[float],
    probabilities_b: Sequence[float],
    metric_fn: Callable[[Sequence[int], Sequence[float]], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    lower_is_better: bool = True,
    groups: "Sequence[Any] | None" = None,
) -> dict:
    """Paired bootstrap comparison of two probability sets on the same labels.

    Each resample applies *identical* indices to A and B (paired), computing
    ``delta_b = metric(A) - metric(B)``.  Returns ``{'delta', 'lo', 'hi',
    'p_value', 'significant', 'better', 'confidence'}``.  The point ``delta`` is
    on the original data; the CI is the percentile interval of the bootstrap
    delta distribution.  The two-sided bootstrap ``p_value`` is
    ``2 * min(P(delta_b <= 0), P(delta_b >= 0))`` clipped to ``[0, 1]``.
    Significance is "CI excludes 0"; ``better`` respects ``lower_is_better``
    (a tie when the CI contains 0).
    """
    n = len(labels)
    metric_a = float(metric_fn(labels, probabilities_a)) if n else float(metric_fn([], []))
    metric_b = float(metric_fn(labels, probabilities_b)) if n else float(metric_fn([], []))
    delta = metric_a - metric_b

    if n == 0 or n_boot <= 0:
        return {
            "delta": delta,
            "lo": delta,
            "hi": delta,
            "p_value": 1.0,
            "significant": False,
            "better": "tie",
            "confidence": 1.0 - alpha,
        }

    rng = random.Random(seed)
    index_sets = _resample_index_sets(n, n_boot, rng, groups)
    labels_list = list(labels)
    a_list = list(probabilities_a)
    b_list = list(probabilities_b)

    deltas: List[float] = []
    for indices in index_sets:
        boot_labels = [labels_list[i] for i in indices]
        boot_a = [a_list[i] for i in indices]
        boot_b = [b_list[i] for i in indices]
        deltas.append(float(metric_fn(boot_labels, boot_a)) - float(metric_fn(boot_labels, boot_b)))

    deltas.sort()
    lo = _percentile(deltas, alpha / 2.0)
    hi = _percentile(deltas, 1.0 - alpha / 2.0)

    m = len(deltas)
    frac_le = sum(1 for d in deltas if d <= 0.0) / m
    frac_ge = sum(1 for d in deltas if d >= 0.0) / m
    p_value = min(1.0, max(0.0, 2.0 * min(frac_le, frac_ge)))

    significant = not (lo <= 0.0 <= hi)
    if not significant:
        better = "tie"
    else:
        a_better = delta < 0.0 if lower_is_better else delta > 0.0
        better = "a" if a_better else "b"

    return {
        "delta": delta,
        "lo": lo,
        "hi": hi,
        "p_value": p_value,
        "significant": significant,
        "better": better,
        "confidence": 1.0 - alpha,
    }


def _pointwise_loss(label: int, probability: float, loss: str) -> float:
    if loss in ("squared", "brier"):
        return (probability - label) ** 2
    if loss == "absolute":
        return abs(probability - label)
    if loss == "log_loss":
        p = min(0.999999, max(0.000001, probability))
        return -(label * math.log(p) + (1 - label) * math.log(1 - p))
    raise ValueError(f"unknown loss: {loss!r}")


def _student_t_sf(t: float, df: float) -> float:
    """Two-sided Student-t survival helper returns P(|T| > |t|).

    Uses scipy when available; otherwise a pure-Python regularized incomplete
    beta evaluation of the t-distribution CDF (Lentz continued fraction).
    """
    t = abs(t)
    if _scipy_stats is not None:
        return float(2.0 * _scipy_stats.t.sf(t, df))

    # Pure-Python: P(|T|>t) = I_x(df/2, 1/2) with x = df/(df + t^2).
    x = df / (df + t * t)
    return _regularized_incomplete_beta(x, df / 2.0, 0.5)


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(x, a, b)
    return 1.0 - (math.exp(math.log(1.0 - x) * b + math.log(x) * a - lbeta) / b) * _beta_cf(1.0 - x, b, a)


def _beta_cf(x: float, a: float, b: float) -> float:
    tiny = 1e-30
    max_iter = 300
    eps = 1e-12
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for i in range(1, max_iter + 1):
        m = i
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def diebold_mariano(
    labels: Sequence[int],
    probabilities_a: Sequence[float],
    probabilities_b: Sequence[float],
    loss: str = "squared",
    h: int = 1,
    harvey_correction: bool = True,
) -> dict:
    """Diebold-Mariano test for equal predictive accuracy of two forecasts.

    Loss differential ``d_t = L(a_t, y_t) - L(b_t, y_t)`` with ``loss`` in
    ``{'squared'/'brier', 'absolute', 'log_loss'}`` (log_loss uses the same
    clip as :func:`log_loss`).  The long-run variance uses the Newey-West
    estimator with ``h - 1`` lags; the Harvey-Leybourne-Newbold (1997)
    small-sample correction scales the statistic by
    ``sqrt((n + 1 - 2h + h(h-1)/n) / n)`` and refers it to a Student-t with
    ``n - 1`` degrees of freedom.  Returns ``{'dm_stat', 'p_value',
    'mean_loss_diff', 'n', 'h', 'corrected', 'better', 'loss'}``.

    ``dm_stat < 0`` means forecast A has the lower loss (``better == 'a'``).
    Degenerate cases (``n < 2`` or non-positive variance) return a tie.
    """
    diffs = [
        _pointwise_loss(int(label), prob_a, loss) - _pointwise_loss(int(label), prob_b, loss)
        for label, prob_a, prob_b in zip(labels, probabilities_a, probabilities_b)
    ]
    n = len(diffs)
    degenerate = {
        "dm_stat": 0.0,
        "p_value": 1.0,
        "mean_loss_diff": (sum(diffs) / n) if n else 0.0,
        "n": n,
        "h": h,
        "corrected": harvey_correction,
        "better": "tie",
        "loss": loss,
    }
    if n < 2:
        return degenerate

    d_bar = sum(diffs) / n
    centered = [d - d_bar for d in diffs]
    gamma0 = sum(c * c for c in centered) / n
    long_run = gamma0
    for k in range(1, h):
        if k >= n:
            break
        gamma_k = sum(centered[t] * centered[t - k] for t in range(k, n)) / n
        long_run += 2.0 * (1.0 - k / h) * gamma_k

    if long_run <= 0.0:
        return degenerate

    dm = d_bar / math.sqrt(long_run / n)

    if harvey_correction:
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor <= 0.0:
            return degenerate
        dm *= math.sqrt(factor)

    p_value = min(1.0, max(0.0, _student_t_sf(dm, n - 1)))

    if abs(dm) < 1e-12:
        better = "tie"
    elif dm < 0.0:
        better = "a"
    else:
        better = "b"

    return {
        "dm_stat": dm,
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "n": n,
        "h": h,
        "corrected": harvey_correction,
        "better": better,
        "loss": loss,
    }
