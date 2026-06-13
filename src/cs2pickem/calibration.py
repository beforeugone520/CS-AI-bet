from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

try:  # optional; only used to sharpen / accelerate the convex calibration fit
    from scipy import optimize as _scipy_optimize  # type: ignore
except Exception:  # pragma: no cover - exercised only when scipy is absent
    _scipy_optimize = None


@dataclass
class ProbabilityCalibrator:
    """Platt-style logistic calibration for already-produced probabilities.

    This is the historic default implementation and is preserved verbatim so
    that ``method='platt'`` stays bit-for-bit identical to the behaviour the
    rest of the pipeline (and the locked tests) already depend on. The
    hand-written fixed-step gradient descent is intentionally unchanged; the
    richer scipy/Newton solvers live on :class:`MethodCalibrator` and are only
    reached through :func:`make_calibrator` for the non-default methods.
    """

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


# ---------------------------------------------------------------------------
# Multi-method calibration (platt / beta / temperature).
#
# ``MethodCalibrator`` shares the _safe_logit / _sigmoid / _clip primitives with
# the historic ``ProbabilityCalibrator`` and emits the SAME report() shape
# (basis, training_count, positive_rate, slope, intercept) plus method-specific
# additive keys (``method``, ``temperature``, ``beta_coefficients``) so existing
# readers keep working. The convex weighted-NLL fit uses scipy.optimize when
# present and a self-contained Newton/IRLS loop (standard library only) as a
# faithful drop-in fallback, selected purely by feature-detection -- not a new
# dependency. An optional expanding-window time-series CV averages per-fold
# parameters to cut small-sample variance while never letting the calibrator see
# the held-out/test rows.
# ---------------------------------------------------------------------------


_VALID_METHODS = ("platt", "beta", "temperature")


@dataclass
class MethodCalibrator:
    """Convex post-hoc probability calibration with selectable method.

    ``method``:

    * ``platt``       -- ``sigmoid(a * logit(p) + b)``; two free params.
    * ``beta``        -- Kull/Silva-Filho beta calibration with features
      ``[ln(p), -ln(1 - p)]`` => ``sigmoid(c0 + c1 ln(p) + c2 (-ln(1 - p)))``.
      The identity map ``(c0=0, c1=1, c2=1)`` is reachable so it degrades to a
      no-op; it nests Platt yet adds one DOF to bend the two tails
      asymmetrically.
    * ``temperature`` -- single-parameter scaling ``sigmoid(logit(p) / T)`` with
      ``T > 0`` (``T = 1`` is identity); the minimum-variance choice for small
      single-event calibration sets.

    All fits minimise the L2-toward-identity-regularised weighted negative
    log-likelihood (slope/temperature shrink toward 1, biases toward 0).
    """

    method: str = "platt"
    l2: float = 0.001
    cv_folds: int = 0
    max_iter: int = 100

    # fitted parameters -------------------------------------------------------
    slope: float = field(default=1.0, init=False)
    intercept: float = field(default=0.0, init=False)
    temperature: float = field(default=1.0, init=False)
    beta_coefficients: Tuple[float, float, float] = field(default=(0.0, 1.0, 1.0), init=False)
    training_count: int = field(default=0, init=False)
    positive_rate: float = field(default=0.0, init=False)
    basis: str = field(default="no_calibration_rows", init=False)
    cv_folds_used: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        method = str(self.method).strip().lower()
        if method not in _VALID_METHODS:
            raise ValueError(f"unknown calibration method: {self.method!r}")
        self.method = method

    # -- fitting --------------------------------------------------------------
    def fit(
        self,
        probabilities: Sequence[float],
        labels: Sequence[int],
        sample_weights: Sequence[float] | None = None,
    ) -> "MethodCalibrator":
        rows = _prepared_rows(probabilities, labels, sample_weights)
        self.training_count = len(rows)
        self.positive_rate = (sum(label for _, label, _ in rows) / len(rows)) if rows else 0.0
        self.cv_folds_used = 0
        if not rows:
            self._set_identity()
            self.basis = "no_calibration_rows"
            return self

        params = None
        if self.cv_folds and self.cv_folds >= 2:
            params = self._fit_cv(rows)
        if params is None:
            params = self._fit_single(rows)
        self._assign(params)
        self.basis = f"{self.method}_calibration"
        return self

    def _fit_cv(self, rows: List[Tuple[float, int, float]]) -> "_Params | None":
        """Expanding-window time-series CV: average per-fold fitted params.

        Fold ``k`` fits the calibrator ONLY on rows strictly before its
        validation window (expanding training side); we average the per-fold
        parameter vectors for a lower-variance final calibrator. The rows are
        consumed in their given (chronological) order. When there are too few
        rows for ``cv_folds`` folds we signal a fall back to the single fit.
        """
        folds = int(self.cv_folds)
        n = len(rows)
        if n < folds + 1:
            return None
        validation_size = max(1, n // (folds + 1))
        fold_params: List[_Params] = []
        for fold_index in range(1, folds + 1):
            train_end = validation_size * fold_index
            validation_end = train_end + validation_size
            if validation_end > n:
                break
            train_rows = rows[:train_end]
            if not train_rows:
                continue
            fold_params.append(self._fit_single(train_rows))
        if not fold_params:
            return None
        self.cv_folds_used = len(fold_params)
        return _average_params(fold_params)

    def _fit_single(self, rows: List[Tuple[float, int, float]]) -> "_Params":
        if self.method == "temperature":
            return self._fit_temperature(rows)
        if self.method == "beta":
            return self._fit_beta(rows)
        return self._fit_platt(rows)

    # -- per-method solvers ---------------------------------------------------
    def _fit_platt(self, rows: List[Tuple[float, int, float]]) -> "_Params":
        # design: [logit(p), 1]; prior pulls slope->1, intercept->0.
        design = [(logit, 1.0) for logit, _, _ in rows]
        coef = _fit_logistic(design, rows, l2=self.l2, prior=(1.0, 0.0), max_iter=self.max_iter)
        return _Params(slope=coef[0], intercept=coef[1])

    def _fit_beta(self, rows: List[Tuple[float, int, float]]) -> "_Params":
        # features = [ln(p), -ln(1 - p)] with a bias; identity = (0, 1, 1).
        design = []
        for logit, _, _ in rows:
            probability = _sigmoid(logit)
            ln_p = math.log(min(0.999999, max(0.000001, probability)))
            ln_1mp = math.log(min(0.999999, max(0.000001, 1.0 - probability)))
            design.append((1.0, ln_p, -ln_1mp))
        # Kull/Silva-Filho beta calibration is monotone iff the two tail
        # coefficients ``c1, c2 >= 0`` (the original derivation constrains
        # ``a, b >= 0``). We enforce that here so the calibrated map never
        # reorders samples / corrupts AUC; the bias ``c0`` stays free.
        coef = _fit_logistic(
            design,
            rows,
            l2=self.l2,
            prior=(0.0, 1.0, 1.0),
            max_iter=self.max_iter,
            nonneg=(False, True, True),
        )
        return _Params(beta=(coef[0], coef[1], coef[2]))

    def _fit_temperature(self, rows: List[Tuple[float, int, float]]) -> "_Params":
        temperature = _fit_temperature_scalar(rows, l2=self.l2, max_iter=self.max_iter)
        return _Params(temperature=temperature)

    # -- transforms -----------------------------------------------------------
    def transform(self, probabilities: Sequence[float]) -> List[float]:
        if self.basis == "no_calibration_rows":
            return [_clip(probability) for probability in probabilities]
        return [self.transform_one(probability) for probability in probabilities]

    def transform_one(self, probability: float) -> float:
        if self.basis == "no_calibration_rows":
            return _clip(probability)
        logit = _safe_logit(probability)
        if self.method == "temperature":
            return _clip(_sigmoid(logit / self.temperature))
        if self.method == "beta":
            c0, c1, c2 = self.beta_coefficients
            prob = _sigmoid(logit)
            ln_p = math.log(min(0.999999, max(0.000001, prob)))
            ln_1mp = math.log(min(0.999999, max(0.000001, 1.0 - prob)))
            return _clip(_sigmoid(c0 + c1 * ln_p + c2 * (-ln_1mp)))
        return _clip(_sigmoid(self.slope * logit + self.intercept))

    def report(self) -> Dict[str, object]:
        report: Dict[str, object] = {
            "basis": self.basis,
            "training_count": self.training_count,
            "positive_rate": self.positive_rate,
            "slope": self.slope,
            "intercept": self.intercept,
            "method": self.method,
        }
        if self.method == "temperature":
            report["temperature"] = self.temperature
        if self.method == "beta":
            report["beta_coefficients"] = list(self.beta_coefficients)
        if self.cv_folds_used:
            report["cv_folds_used"] = self.cv_folds_used
        return report

    # -- internal helpers -----------------------------------------------------
    def _set_identity(self) -> None:
        self.slope = 1.0
        self.intercept = 0.0
        self.temperature = 1.0
        self.beta_coefficients = (0.0, 1.0, 1.0)

    def _assign(self, params: "_Params") -> None:
        if self.method == "temperature":
            self.temperature = params.temperature
            # Report an equivalent platt-style slope (1/T) so generic readers of
            # the slope/intercept keys still see the effective scaling.
            self.slope = 1.0 / params.temperature if params.temperature else 1.0
            self.intercept = 0.0
        elif self.method == "beta":
            self.beta_coefficients = params.beta
            self.slope = params.beta[1] + params.beta[2]
            self.intercept = params.beta[0]
        else:
            self.slope = params.slope
            self.intercept = params.intercept


@dataclass
class _Params:
    slope: float = 1.0
    intercept: float = 0.0
    temperature: float = 1.0
    beta: Tuple[float, float, float] = (0.0, 1.0, 1.0)


def _average_params(fold_params: Sequence["_Params"]) -> "_Params":
    count = len(fold_params)
    return _Params(
        slope=sum(p.slope for p in fold_params) / count,
        intercept=sum(p.intercept for p in fold_params) / count,
        temperature=sum(p.temperature for p in fold_params) / count,
        beta=(
            sum(p.beta[0] for p in fold_params) / count,
            sum(p.beta[1] for p in fold_params) / count,
            sum(p.beta[2] for p in fold_params) / count,
        ),
    )


def make_calibrator(
    method: str = "platt",
    *,
    l2: float = 0.001,
    cv_folds: int = 0,
):
    """Factory returning a calibrator with the shared fit/transform/report API.

    ``method='platt'`` with the default (no CV) returns the historic
    :class:`ProbabilityCalibrator`, keeping default behaviour byte-identical and
    the ``platt_logistic`` basis string intact. Any non-default method (or CV)
    returns a :class:`MethodCalibrator`. Both objects expose the same
    ``fit`` / ``transform`` / ``transform_one`` / ``report`` surface.
    """
    method = str(method).strip().lower()
    if method not in _VALID_METHODS:
        raise ValueError(f"unknown calibration method: {method!r}")
    if method == "platt" and not cv_folds:
        return ProbabilityCalibrator(l2=l2)
    return MethodCalibrator(method=method, l2=l2, cv_folds=cv_folds)


# ---------------------------------------------------------------------------
# Convex solvers: weighted logistic NLL (multi-param) and scalar temperature.
# scipy.optimize when available, else a damped Newton / scalar Newton fallback.
# ---------------------------------------------------------------------------


def _prepared_rows(
    probabilities: Sequence[float],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None,
) -> List[Tuple[float, int, float]]:
    weights = sample_weights if sample_weights is not None else [1.0] * len(list(labels))
    rows: List[Tuple[float, int, float]] = []
    for probability, label, weight in zip(probabilities, labels, weights):
        rows.append((_safe_logit(probability), 1 if label else 0, max(0.0, float(weight))))
    return rows


def _weighted_logistic_nll(
    coef: Sequence[float],
    design: Sequence[Sequence[float]],
    rows: Sequence[Tuple[float, int, float]],
    l2: float,
    prior: Sequence[float],
) -> float:
    total = 0.0
    for features, (_, label, weight) in zip(design, rows):
        z = sum(c * x for c, x in zip(coef, features))
        # numerically stable -log sigmoid for the chosen label
        if label:
            total += weight * _softplus(-z)
        else:
            total += weight * _softplus(z)
    for c, p in zip(coef, prior):
        total += 0.5 * l2 * (c - p) ** 2
    return total


def _fit_logistic(
    design: List[Sequence[float]],
    rows: List[Tuple[float, int, float]],
    l2: float,
    prior: Sequence[float],
    max_iter: int,
    nonneg: Sequence[bool] | None = None,
) -> List[float]:
    """Weighted-NLL logistic fit with optional per-coefficient ``>= 0`` bounds.

    ``nonneg[j] = True`` constrains coefficient ``j`` to be non-negative (used by
    beta calibration so the calibrated map stays monotone, per Kull). scipy's
    L-BFGS-B honours the bounds directly; the pure-Python Newton fallback
    projects onto the feasible box each step. With ``nonneg=None`` the behaviour
    is the unconstrained fit (back-compat for platt).
    """
    if _scipy_optimize is not None:
        try:
            bounds = None
            if nonneg is not None:
                bounds = [(0.0, None) if flag else (None, None) for flag in nonneg]
            result = _scipy_optimize.minimize(
                _weighted_logistic_nll,
                x0=list(prior),
                args=(design, rows, l2, prior),
                jac=lambda coef: _logistic_gradient(coef, design, rows, l2, prior),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": max(50, max_iter * 5)},
            )
            coef = [float(value) for value in result.x]
            if all(math.isfinite(value) for value in coef):
                return _project_nonneg(coef, nonneg)
        except Exception:  # pragma: no cover - defensive, falls back to Newton
            pass
    return _newton_logistic(design, rows, l2, prior, max_iter, nonneg=nonneg)


def _project_nonneg(coef: List[float], nonneg: Sequence[bool] | None) -> List[float]:
    if nonneg is None:
        return coef
    return [max(0.0, value) if flag else value for value, flag in zip(coef, nonneg)]


def _logistic_gradient(
    coef: Sequence[float],
    design: Sequence[Sequence[float]],
    rows: Sequence[Tuple[float, int, float]],
    l2: float,
    prior: Sequence[float],
) -> List[float]:
    dim = len(prior)
    grad = [l2 * (coef[j] - prior[j]) for j in range(dim)]
    for features, (_, label, weight) in zip(design, rows):
        z = sum(c * x for c, x in zip(coef, features))
        residual = (_sigmoid(z) - label) * weight
        for j in range(dim):
            grad[j] += residual * features[j]
    return grad


def _newton_logistic(
    design: List[Sequence[float]],
    rows: List[Tuple[float, int, float]],
    l2: float,
    prior: Sequence[float],
    max_iter: int,
    nonneg: Sequence[bool] | None = None,
) -> List[float]:
    """Pure-Python damped Newton/IRLS on the weighted logistic NLL.

    Analytic gradient and Hessian of the L2-toward-prior penalised weighted
    negative log-likelihood; a small backtracking line search keeps it stable on
    tiny / separable samples so the fit converges without scipy. When ``nonneg``
    is supplied each accepted iterate is projected onto the feasible box
    ``coef[j] >= 0`` (projected Newton), matching the scipy L-BFGS-B bounds so the
    pure-Python path produces the same constrained (monotone) beta fit.
    """
    dim = len(prior)
    coef = _project_nonneg(list(prior), nonneg)
    prev_loss = _weighted_logistic_nll(coef, design, rows, l2, prior)
    for _ in range(max(1, max_iter)):
        grad = [l2 * (coef[j] - prior[j]) for j in range(dim)]
        hess = [[l2 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        for features, (_, label, weight) in zip(design, rows):
            z = sum(c * x for c, x in zip(coef, features))
            mu = _sigmoid(z)
            residual = (mu - label) * weight
            w = max(1e-9, mu * (1.0 - mu)) * weight
            for i in range(dim):
                grad[i] += residual * features[i]
                for j in range(dim):
                    hess[i][j] += w * features[i] * features[j]
        step = _solve_linear(hess, grad)
        if step is None:
            break
        # Backtracking line search to guarantee descent on the convex objective.
        alpha = 1.0
        improved = False
        for _ in range(20):
            candidate = _project_nonneg([coef[j] - alpha * step[j] for j in range(dim)], nonneg)
            loss = _weighted_logistic_nll(candidate, design, rows, l2, prior)
            if loss <= prev_loss + 1e-12:
                coef = candidate
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
        if abs(prev_loss - loss) <= 1e-10 * (1.0 + abs(prev_loss)):
            prev_loss = loss
            break
        prev_loss = loss
    return coef


def _solve_linear(matrix: List[List[float]], vector: List[float]) -> List[float] | None:
    """Solve ``matrix @ x = vector`` via Gaussian elimination with pivoting."""
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            return None
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def _temperature_nll(t: float, rows: Sequence[Tuple[float, int, float]], l2: float) -> float:
    # parametrise by t = 1/T so the objective is convex in t (logit scaling).
    total = 0.0
    for logit, label, weight in rows:
        z = t * logit
        if label:
            total += weight * _softplus(-z)
        else:
            total += weight * _softplus(z)
    total += 0.5 * l2 * (t - 1.0) ** 2
    return total


def _fit_temperature_scalar(
    rows: Sequence[Tuple[float, int, float]],
    l2: float,
    max_iter: int,
) -> float:
    """Fit temperature ``T`` (returns ``T = 1/t`` with ``t`` the logit scale).

    The NLL is convex in ``t = 1/T``; scipy's bounded scalar minimiser is used
    when present, otherwise a damped scalar Newton with backtracking. ``T`` is
    clamped to a sane positive range so the transform stays well defined.
    """
    if _scipy_optimize is not None:
        try:
            result = _scipy_optimize.minimize_scalar(
                _temperature_nll,
                bounds=(1e-3, 1e3),
                args=(rows, l2),
                method="bounded",
                options={"maxiter": max(50, max_iter * 5)},
            )
            t = float(result.x)
            if math.isfinite(t) and t > 0.0:
                return 1.0 / t
        except Exception:  # pragma: no cover - defensive
            pass
    t = _newton_temperature(rows, l2, max_iter)
    return 1.0 / t if t > 0.0 else 1.0


def _newton_temperature(
    rows: Sequence[Tuple[float, int, float]],
    l2: float,
    max_iter: int,
) -> float:
    t = 1.0
    prev_loss = _temperature_nll(t, rows, l2)
    for _ in range(max(1, max_iter)):
        grad = l2 * (t - 1.0)
        hess = l2
        for logit, label, weight in rows:
            z = t * logit
            mu = _sigmoid(z)
            grad += (mu - label) * weight * logit
            hess += max(1e-9, mu * (1.0 - mu)) * weight * logit * logit
        if hess <= 1e-12:
            break
        step = grad / hess
        alpha = 1.0
        improved = False
        for _ in range(30):
            candidate = t - alpha * step
            if candidate <= 1e-3 or candidate >= 1e3:
                alpha *= 0.5
                continue
            loss = _temperature_nll(candidate, rows, l2)
            if loss <= prev_loss + 1e-12:
                t = candidate
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break
        if abs(prev_loss - loss) <= 1e-10 * (1.0 + abs(prev_loss)):
            prev_loss = loss
            break
        prev_loss = loss
    return max(1e-3, min(1e3, t))


def _safe_logit(probability: float) -> float:
    probability = min(0.999999, max(0.000001, float(probability)))
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _softplus(value: float) -> float:
    # numerically stable log(1 + exp(value))
    if value > 60:
        return value
    if value < -60:
        return 0.0
    return math.log1p(math.exp(value))


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
