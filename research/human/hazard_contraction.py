"""Check the hazard-curvature account of contraction, and the tail counterexample.

Section 6 of the draft claims heavier tails contract less and bounded support
contracts more. The referee's counterexample says that ordering is not invariant to
the share configuration. The proposed replacement is a theorem about the curvature of
the log hazard rather than about tails:

    lowest draw wins; h = f / Fbar is the hazard; if log h is concave then the
    full-field log odds exceed the pairwise log odds, so removing contestants
    contracts the favourite's advantage.

The mechanism is a covariance inequality. With A(x) = f(x-a_i) Fbar(x-a_j),
B(x) = f(x-a_j) Fbar(x-a_i) and H(x) the survival product of the other contestants,

    A(x)/B(x) = h(x-a_i)/h(x-a_j),

which is decreasing in x when log h is concave and a_i < a_j. H is decreasing too, so
the two are comonotone and

    int A H / int B H  >=  int A / int B,

the left side being the full-field odds and the right the pairwise odds.

Three things are checked numerically here, none of them taken on trust:

  1. that log h is concave for the Gaussian, and affine for the minimum-Gumbel, so
     the Gumbel case gives exact equality and recovers the axiom;
  2. the identity (log h)'' = -Var(Z | Z > x) for the standard normal;
  3. whether the tail ordering in the draft's Table 4 survives recalibration to a
     lopsided share vector, which is the referee's counterexample.

Usage:  python hazard_contraction.py
"""
import sys

import numpy as np
from scipy import integrate, optimize, stats

FIELDS = {
    "near-uniform 5": [0.24, 0.22, 0.20, 0.18, 0.16],
    "moderate 3": [0.60, 0.25, 0.15],
    "lopsided 3": [0.88, 0.07, 0.05],
    "lopsided 5": [0.70, 0.12, 0.08, 0.06, 0.04],
}


def laws():
    """Unit-variance, zero-mean noise laws. Lowest draw wins throughout."""
    out = {}
    out["Gaussian"] = stats.norm(0, 1)
    # minimum-Gumbel: negate a standard (maximum) Gumbel so small values win the
    # way the axiom requires. sd of Gumbel is pi/sqrt(6).
    s = np.sqrt(6) / np.pi
    out["Gumbel (min)"] = _Shift(stats.gumbel_r(loc=0, scale=s), flip=True)
    for df in (3, 8):
        out[f"t({df})"] = _Shift(stats.t(df), scale=1 / np.sqrt(df / (df - 2)))
    out["uniform"] = stats.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3))
    a = 4.0
    d = a / np.sqrt(1 + a * a)
    sd = np.sqrt(1 - 2 * d * d / np.pi)
    out["skew-normal(4)"] = _Shift(stats.skewnorm(a), shift=-d * np.sqrt(2 / np.pi),
                                   scale=1 / sd)
    return out


class _Shift:
    """Affine transform of a frozen scipy law, standardised or reflected."""

    def __init__(self, rv, shift=0.0, scale=1.0, flip=False):
        self.rv, self.shift, self.scale, self.flip = rv, shift, scale, flip

    def _to(self, x):
        y = x / self.scale if not self.flip else -x
        return y - self.shift if not self.flip else y

    def cdf(self, x):
        if self.flip:
            return self.rv.sf(-np.asarray(x))
        return self.rv.cdf(np.asarray(x) / self.scale + self.shift)

    def pdf(self, x):
        if self.flip:
            return self.rv.pdf(-np.asarray(x))
        return self.rv.pdf(np.asarray(x) / self.scale + self.shift) / self.scale


def win_prob(law, a, i, lo=-40, hi=40, n=40001):
    """P(item i has the lowest draw) with locations a."""
    x = np.linspace(lo, hi, n)
    f = law.pdf(x - a[i])
    surv = np.ones_like(x)
    for k in range(len(a)):
        if k != i:
            surv = surv * (1.0 - law.cdf(x - a[k]))
    return float(np.trapezoid(f * surv, x))


def calibrate(law, target, lo=-40, hi=40):
    """Locations reproducing the target winning probabilities, a_0 fixed at 0."""
    K = len(target)

    def resid(z):
        a = np.concatenate([[0.0], z])
        w = np.array([win_prob(law, a, i, lo, hi) for i in range(K)])
        w = w / w.sum()
        return np.log(w[1:]) - np.log(np.array(target[1:]) / target[0] * w[0])

    sol = optimize.fsolve(resid, np.linspace(0.3, 1.5, K - 1), full_output=True)
    z = sol[0]
    a = np.concatenate([[0.0], z])
    w = np.array([win_prob(law, a, i, lo, hi) for i in range(K)])
    return a, w / w.sum()


def pairwise(law, ai, aj, lo=-40, hi=40, n=40001):
    """P(X_i < X_j) for a two-item field."""
    x = np.linspace(lo, hi, n)
    return float(np.trapezoid(law.pdf(x - ai) * (1.0 - law.cdf(x - aj)), x))


def lam(law, target):
    """Contraction slope: regress -delta on the full-field log odds, through 0."""
    a, w = calibrate(law, target)
    num = den = 0.0
    K = len(target)
    for i in range(K):
        for j in range(i + 1, K):
            hi_, lo_ = (i, j) if w[i] >= w[j] else (j, i)
            lo_odds = np.log(w[hi_] / w[lo_])
            q = pairwise(law, a[hi_], a[lo_])
            q = min(max(q, 1e-9), 1 - 1e-9)
            d = np.log(q / (1 - q)) - lo_odds
            num += lo_odds * (-d)
            den += lo_odds ** 2
    return num / den if den > 0 else float("nan")


def log_hazard_curvature():
    print("1. curvature of log hazard, lowest-draw-wins orientation")
    print("   negative means concave, which the proposition needs\n")
    x = np.linspace(-3, 3, 25)
    for name, law in laws().items():
        h = law.pdf(x) / np.maximum(1.0 - law.cdf(x), 1e-300)
        lh = np.log(np.maximum(h, 1e-300))
        d2 = np.gradient(np.gradient(lh, x), x)
        inner = slice(3, -3)
        mx = float(d2[inner].max())
        flag = "affine" if abs(mx) < 5e-3 and abs(d2[inner].min()) < 5e-3 else (
            "concave" if mx < 1e-6 else "NOT concave")
        print(f"   {name:<16} max (log h)'' = {mx:+.4f}   {flag}")
    print("\n2. identity (log h)'' = -Var(Z | Z > x) for the standard normal")
    for xv in (-1.0, 0.0, 1.0, 2.0):
        eps = 1e-4
        f = lambda t: np.log(stats.norm.pdf(t) / stats.norm.sf(t))
        d2 = (f(xv + eps) - 2 * f(xv) + f(xv - eps)) / eps ** 2
        m1 = integrate.quad(lambda t: t * stats.norm.pdf(t), xv, 40)[0] / stats.norm.sf(xv)
        m2 = integrate.quad(lambda t: t * t * stats.norm.pdf(t), xv, 40)[0] / stats.norm.sf(xv)
        var = m2 - m1 * m1
        print(f"   x={xv:+.1f}   (log h)'' = {d2:+.6f}   -Var(Z|Z>x) = {-var:+.6f}")


def tail_table():
    print("\n3. contraction by noise law and share configuration")
    print("   each law recalibrated to the same shares, so the comparison is")
    print("   at fixed observables rather than fixed locations\n")
    ls = laws()
    print(f"   {'field':<18}" + "".join(f"{k:>16}" for k in ls))
    for fname, target in FIELDS.items():
        row = []
        for lname, law in ls.items():
            try:
                row.append(lam(law, target))
            except Exception:
                row.append(float("nan"))
        print(f"   {fname:<18}" + "".join(f"{v:>16.4f}" for v in row))
    print("\n   if the Gaussian column is not always below the t columns, the")
    print("   draft's tail ordering is configuration-specific and must go.")


if __name__ == "__main__":
    log_hazard_curvature()
    tail_table()
