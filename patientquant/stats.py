"""Statistical primitives written from scratch.

The sandbox has no SciPy and PyPI is unreachable, so every distribution the
book needs is implemented here from its definition.  This is not an
inconvenience to apologise for: the book's thesis is that the reader should be
able to audit every line of the machinery, and a normal CDF that is nine
lines long can be audited by anyone.

Accuracy is checked against known values in ``tests/test_stats.py``:

``erf`` and ``erfc``
    The Maclaurin series of the integral below the point where the series
    starts to lose digits, and the continued fraction for the complement
    above it; both to about 1e-15.

``norm_cdf``
    From ``erfc``, so that the upper tail keeps its precision.

``norm_ppf``
    Acklam's rational approximation, refined by one Halley step using
    ``norm_cdf``; absolute error below 1e-12 over the whole range.

``t_cdf`` and ``t_ppf``
    Derived from the incomplete beta function, which is itself computed from
    the continued fraction in Lentz's algorithm.
"""

from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
TWO_OVER_SQRTPI = 2.0 / math.sqrt(math.pi)


def _erf_series(x: float) -> float:
    """erf(x) from its Maclaurin series, for |x| below the crossover.

        erf(x) = (2/sqrt(pi)) * sum_n (-1)^n x^(2n+1) / (n! (2n+1))

    The terms peak near n = x^2/2, so the series is used only while the
    largest term is small enough that the cancellation in the alternating
    sum costs less than a digit or two.
    """
    term = x
    total = x
    for n in range(1, 200):
        term *= -x * x / n
        total += term / (2 * n + 1)
        if abs(term) < 1e-19 * max(abs(total), 1e-300):
            break
    return TWO_OVER_SQRTPI * total


def _gammq(a: float, x: float) -> float:
    """The regularized upper incomplete gamma function Q(a, x).

    Below the crossover the series for P(a, x) is summed and subtracted
    from one; above it, the continued fraction for Q(a, x) is evaluated by
    Lentz's algorithm.  Both forms are the standard ones.
    """
    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(600):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-17:
                break
        p = math.exp(-x + a * math.log(x) - math.lgamma(a)) * total
        return 1.0 - p

    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 600):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _erfc_cf(x: float) -> float:
    """erfc(x) from the incomplete gamma, for x above the crossover.

        erfc(x) = Q(1/2, x^2)

    which is the standard identity between the complementary error
    function and the upper incomplete gamma function.
    """
    return _gammq(0.5, x * x)


def _erf_scalar(x: float) -> float:
    if x == 0.0:
        return 0.0
    sign = 1.0 if x > 0 else -1.0
    ax = abs(x)
    if ax < 2.0:
        return sign * _erf_series(ax)
    return sign * (1.0 - _erfc_cf(ax))


def erf(x):
    """The error function, erf(x) = (2/sqrt(pi)) * integral_0^x exp(-t^2) dt.

    The Maclaurin series is used below |x| = 2 and the continued fraction
    for the complement above it, which between them hold the error below
    1e-15 over the whole line.
    """
    array = np.asarray(x, dtype=float)
    if array.ndim == 0:
        return np.float64(_erf_scalar(float(array)))
    return np.array([_erf_scalar(float(v)) for v in array.ravel()],
                    dtype=float).reshape(array.shape)


def erfc(x):
    """The complementary error function, 1 - erf(x)."""
    array = np.asarray(x, dtype=float)
    if array.ndim == 0:
        value = float(array)
        if value > 2.0:
            return np.float64(_erfc_cf(value))
        if value < -2.0:
            return np.float64(2.0 - _erfc_cf(-value))
        return np.float64(1.0 - _erf_scalar(value))
    flat = array.ravel()
    out = np.empty(flat.shape, dtype=float)
    for index, value in enumerate(flat):
        out[index] = float(erfc(value))
    return out.reshape(array.shape)


def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    array = np.asarray(x, dtype=float)
    return 0.5 * erfc(-array / SQRT2)


def norm_sf(x):
    """Standard normal survival function."""
    return 0.5 * erfc(np.asarray(x, dtype=float) / SQRT2)


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQRT2PI


_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]

_P_LOW = 0.02425


def norm_ppf(p):
    """Standard normal quantile function (Acklam), refined by one Halley step."""
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("norm_ppf requires 0 < p < 1")

    # Work on the smaller of the two tails, so that the rational
    # approximation is always evaluated where it is accurate, and mirror
    # afterwards.  Acklam's three branches are not interchangeable: the
    # tail branch carries a different polynomial and the opposite sign
    # from the central one.
    flip = p < 0.5
    small = np.where(flip, p, 1.0 - p)

    lower = np.empty_like(small)
    central = small > _P_LOW

    if np.any(central):
        # The central region: a rational approximation in r = p - 1/2.
        r = small[central] - 0.5
        r2 = r * r
        num = (((((_A[0] * r2 + _A[1]) * r2 + _A[2]) * r2 + _A[3]) * r2
                + _A[4]) * r2 + _A[5]) * r
        den = (((((_B[0] * r2 + _B[1]) * r2 + _B[2]) * r2 + _B[3]) * r2
                + _B[4]) * r2 + 1.0)
        lower[central] = num / den

    if np.any(~central):
        # The lower tail: a rational approximation in sqrt(-2 ln p),
        # with the sign the tail requires.
        x = np.sqrt(-2.0 * np.log(small[~central]))
        num = (((((_C[0] * x + _C[1]) * x + _C[2]) * x + _C[3]) * x
                + _C[4]) * x + _C[5])
        den = ((((_D[0] * x + _D[1]) * x + _D[2]) * x + _D[3]) * x + 1.0)
        lower[~central] = num / den

    out = np.where(flip, lower, -lower)

    # One Halley refinement against the CDF.
    err = norm_cdf(out) - p
    dens = norm_pdf(out)
    out = out - err / dens
    return out


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-14) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student-t cumulative distribution function."""
    t = float(t)
    df = float(df)
    x = df / (df + t * t)
    tail = 0.5 * betainc(0.5 * df, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_pdf(t: float, df: float) -> float:
    t = float(t)
    df = float(df)
    return (math.gamma(0.5 * (df + 1.0)) / (math.sqrt(df * math.pi) * math.gamma(0.5 * df))
            * (1.0 + t * t / df) ** (-0.5 * (df + 1.0)))


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile function by bisection (robust and adequate)."""
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
