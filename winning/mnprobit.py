"""Multinomial probit estimation: the model statsmodels does not have
and sklearn cannot express.

Built on winning.likelihood (exact factor-conditional likelihood with
analytic score). Covariance parameterization: rank-r factor loadings
with a zero reference row and a strictly-lower-triangular free block,
unit idiosyncratic variance. At J = 4 alternatives and r = 2 this covers
every positive-definite differenced covariance up to scale with the
same degree-of-freedom count as the differenced-Cholesky
parameterization used by R's mlogit (see r/mlogitfast, whose Fishing
fit this module reproduces cross-language). The rank defaults to
min(2, J - 2); a rank above J - 2 is refused, because with unit
idiosyncratic variances r >= J - 1 leaves the utility scale free and
coefficients and loadings are not identified (#201).

    fit = MNProbit(X, choice).fit()          # X: (T, J, p) covariates
    clf = MNProbitClassifier().fit(X, y)     # sklearn-style
    P = clf.predict_proba(X)
"""

from __future__ import annotations

import numbers
import warnings
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from .likelihood import (check_choice, choice_loglik_and_score,
                         _choice_logprob_terms)


def max_identified_rank(J):
    """Largest factor rank the unit-idiosyncratic parameterization
    identifies at J alternatives.

    With D fixed to one and a zero reference row the free loadings number
    r*J - r*(r+1)/2, while the differenced covariance has J*(J-1)/2 - 1
    shape degrees of freedom. At r >= J - 1 the loading block is full
    rank and D = 1 no longer fixes the scale: for J = 3, r = 2 there is a
    V' whose contrast covariance is exactly 4x that of V, so (2 beta, V')
    prices every choice like (beta, V) -- an exact ridge (#201).
    """
    return max(int(J) - 2, 0)


def _as_count(x, name):
    """A finite, non-negative whole number. int() floored 1.9 to 1 and
    fitted a different model than the one asked for (#479), and SciPy
    read maxiter=nan as zero iterations with success=True (#560)."""
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise ValueError(f"{name} must be a whole number; got {x!r}")
    if not np.isfinite(x) or float(x) != int(x) or x < 0:
        raise ValueError(
            f"{name} must be a finite non-negative whole number; got {x!r}")
    return int(x)


def _check_rank(r, J):
    rmax = max_identified_rank(J)
    r = min(2, rmax) if r is None else _as_count(r, "factor rank r")
    if not 0 <= r <= rmax:
        raise ValueError(
            f"factor rank r = {r} is not identified at J = {J} "
            f"alternatives: with unit idiosyncratic variances the utility "
            f"scale is fixed only for r <= J - 2 = {rmax} (#201)")
    return r


def _check_contrast_rank(X, names, tol=1e-10):
    """Refuse mean-design columns that within-choice-set contrasts do
    not identify. A choice model sees beta only through utility
    DIFFERENCES between the alternatives of one observation, so a
    covariate common to all alternatives, or one duplicating a generated
    intercept, lies on an exact likelihood ridge: the fitter reported an
    arbitrary point on it as a converged estimate whose split depended
    on the covariate's units (#581). Columns are taken left to right and
    a column is aliased when its contrast residual on the earlier kept
    columns is below tol of its own contrast norm (Gram scale)."""
    T, J, p = X.shape
    if p == 0 or J < 2:
        return
    Dc = (X[:, 1:, :] - X[:, :1, :]).reshape(-1, p)
    G = Dc.T @ Dc
    kept, aliased = [], []
    for k in range(p):
        gkk = G[k, k]
        if gkk > 0.0 and kept:
            S = np.array(kept)
            g = G[S, k]
            coef = np.linalg.lstsq(G[np.ix_(S, S)], g, rcond=None)[0]
            resid = gkk - g @ coef
        else:
            resid = gkk
        if gkk > 0.0 and resid > tol * gkk:
            kept.append(k)
        else:
            aliased.append(k)
    if aliased:
        raise ValueError(
            "mean-design column(s) "
            + ", ".join(names[k] for k in aliased)
            + " are not identified from within-choice-set contrasts: each "
            "is constant across the alternatives of every observation or "
            "a linear combination of earlier columns (the generated "
            "intercepts come first). The likelihood is flat along them, "
            "so any reported coefficient would be arbitrary; drop them "
            "(#581)")


def _fill_positions(J, r):
    pos = []
    for col in range(r):
        for row in range(col + 1, J):
            pos.append((row, col))
    return pos


class MNProbit:
    """Exact multinomial probit MLE for alternative-specific covariates.

    Parameters
    ----------
    X : (T, J, p) array of alternative-specific covariates.
    choice : (T,) chosen alternative indices in 0..J-1.
    intercepts : add J-1 alternative intercepts (reference = 0).
    r : factor rank, default min(2, J - 2) (r = 2 spans the full
        identified covariance at J=4). A rank above J - 2 is not
        identified and raises ValueError (#201).
    """

    def __init__(self, X, choice, intercepts=True, r=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(
                f"X must be (observations, alternatives, covariates); "
                f"got {X.ndim} dimensions")
        self.T, self.J, p = X.shape
        # An empty design has an empty-sum likelihood with zero score,
        # so the optimiser stopped at once and reported the arbitrary
        # initial coefficients as a converged fit (#591).
        if self.T == 0:
            raise ValueError("at least one observation is required")
        self.choice = check_choice(choice, self.T, self.J)
        self.r = _check_rank(r, self.J)
        if intercepts:
            Z = np.zeros((self.T, self.J, self.J - 1))
            for j in range(1, self.J):
                Z[:, j, j - 1] = 1.0
            X = np.concatenate([Z, X], axis=2)
        self.X = X
        self.p = X.shape[2]
        # Whether the constructor GENERATED the alternative intercept
        # columns, and how many covariates the caller supplied. Without
        # them predict_proba could only guess from the column count, and
        # the guess is wrong exactly when wrong_width + (J - 1) happens
        # to equal p (#261; the julia port had the same defect, #195).
        self.intercepts = bool(intercepts)
        self.p_raw = int(p)
        self.pos = _fill_positions(self.J, self.r)

    def _unpack(self, theta):
        beta = theta[:self.p]
        V = np.zeros((self.J, self.r))
        for k, (row, col) in enumerate(self.pos):
            V[row, col] = theta[self.p + k]
        return beta, V

    def _negloglik_grad(self, theta):
        beta, V = self._unpack(theta)
        mu = self.X @ beta
        ll, dmu, dV = choice_loglik_and_score(mu, V, self.choice)
        gbeta = np.einsum("tj,tjp->p", dmu, self.X)
        gw = np.array([dV[row, col] for (row, col) in self.pos])
        return -ll, -np.concatenate([gbeta, gw])

    def fit(self, maxiter=400, polish=False):
        """Fit by BFGS with the analytic score; the reported likelihood
        comes from an independent stabilized referee (two Sobol
        scrambles at 2^15), not from the optimizer's own landscape,
        which at sharp loadings carries ~1-nat quadrature noise.

        polish=True continues optimization on a denser node set. Use
        with care: on the Fishing benchmark the unrestricted-covariance
        likelihood is BOUNDARY-SEEKING (loadings run to ||v|| ~ 1e3+
        with the true likelihood still rising, verified at 2^16-2^18
        across scrambles), so polishing chases a ridge with no interior
        maximum -- a known multinomial-probit pathology that GHK's
        simulation noise accidentally regularizes. The boundary_ flag
        reports detection either way."""
        maxiter = _as_count(maxiter, "maxiter")
        names = ([f"intercept[{j}]" for j in range(1, self.J)]
                 if self.intercepts else []) + [
                     f"X[:, :, {k}]" for k in range(self.p_raw)]
        _check_contrast_rank(self.X, names)
        theta0 = np.concatenate([np.zeros(self.p),
                                 np.full(len(self.pos), 0.1)])
        if theta0.size == 0:
            # A parameter-free model (no covariates, no intercepts,
            # r = 0) is the equal-choice null: nothing to optimise, and
            # BFGS died taking the max of an empty gradient (#495).
            res = SimpleNamespace(x=theta0, success=True)
            polish = False
        else:
            res = minimize(self._negloglik_grad, theta0, jac=True,
                           method="BFGS",
                           options={"maxiter": maxiter, "gtol": 1e-6})
        if polish:
            import winning.likelihood as _L
            from scipy.stats import qmc
            from scipy.special import ndtri
            n = 2 ** 13
            u = qmc.Sobol(self.r + 1, scramble=True, seed=3).random(n)
            F = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
            W = np.full(n, 1.0 / n)
            orig = _L.nodes_for_likelihood
            _L.nodes_for_likelihood =                 lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
            try:
                res = minimize(self._negloglik_grad, res.x, jac=True,
                               method="BFGS",
                               options={"maxiter": 100, "gtol": 1e-6})
            finally:
                _L.nodes_for_likelihood = orig
        self.params_, self.V_ = self._unpack(res.x)
        self.converged_ = bool(res.success)
        self.theta_ = res.x
        self.boundary_ = bool(
            np.sqrt((self.V_ ** 2).sum(axis=1)).max() > 50.0)
        # With generated intercepts, an alternative nobody chose has no
        # finite MLE: lowering its intercept (or raising every other one,
        # for the reference) raises the likelihood without limit, and
        # BFGS stopped only when the score fell under gtol, at a point
        # set by the tolerance -- reported as converged (#496).
        self.never_chosen_ = (
            np.flatnonzero(np.bincount(self.choice, minlength=self.J) == 0)
            if self.intercepts else np.array([], dtype=int))
        if self.never_chosen_.size:
            self.converged_ = False
            self.boundary_ = True
            warnings.warn(
                f"alternative(s) {self.never_chosen_.tolist()} are never "
                "chosen, so their intercepts have no finite MLE (complete "
                "separation); the reported fit is a tolerance-dependent "
                "point on the way to -inf, flagged converged_=False, "
                "boundary_=True. Drop those alternatives or fit without "
                "intercepts.", RuntimeWarning, stacklevel=2)
        # referee likelihood: independent scrambles, reported with se
        import winning.likelihood as _L
        from scipy.stats import qmc as _qmc
        from scipy.special import ndtri as _ndtri
        mu = self.X @ self.params_
        vals = []
        for seed in (101, 102):
            n = 2 ** 15
            u = _qmc.Sobol(self.r + 1, scramble=True, seed=seed).random(n)
            F = _ndtri(np.clip(u, 1e-12, 1 - 1e-12))
            W = np.full(n, 1.0 / n)
            orig = _L.nodes_for_likelihood
            _L.nodes_for_likelihood =                 lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
            try:
                vals.append(choice_loglik_and_score(
                    mu, self.V_, self.choice)[0])
            finally:
                _L.nodes_for_likelihood = orig
        self.loglik_ = float(np.mean(vals))
        self.loglik_se_ = float(abs(vals[0] - vals[1]) / 2)
        return self

    def predict_proba(self, X=None):
        """Choice probabilities per observation, by per-alternative
        conditional-product integrals under the fitted parameters."""
        X = self.X if X is None else np.asarray(X, dtype=float)
        if X is not self.X:
            # New data must carry the design the model was FITTED on.
            # This guessed whether to prepend generated intercept columns
            # from the COLUMN COUNT alone, so new data with the wrong
            # number of features were silently REINTERPRETED rather than
            # refused -- and the guess is wrong exactly when the wrong
            # width plus J-1 generated columns equals the fitted width.
            # Three covariates fitted without intercepts, handed ONE,
            # gained two synthetic intercepts and returned a plausible
            # probability row (#261).
            if X.ndim != 3:
                raise ValueError(
                    f"X must be (observations, alternatives, covariates); "
                    f"got {X.ndim} dimensions")
            if X.shape[1] != self.J:
                raise ValueError(
                    f"X has {X.shape[1]} alternatives; the model was "
                    f"fitted on {self.J}")
            nc = X.shape[2]
            if nc == self.p:
                pass                      # already the fitted design
            elif self.intercepts and nc == self.p_raw:
                Z = np.zeros((X.shape[0], self.J, self.J - 1))
                for j in range(1, self.J):
                    Z[:, j, j - 1] = 1.0
                X = np.concatenate([Z, X], axis=2)
            else:
                want = (f"{self.p_raw} or {self.p}" if self.intercepts
                        else f"{self.p}")
                raise ValueError(
                    f"X has {nc} covariate columns; this model was fitted "
                    f"on {self.p_raw} covariates"
                    + (" plus generated intercepts" if self.intercepts
                       else " and no generated intercepts")
                    + f", so pass {want}")
        mu = X @ self.params_
        T, J = mu.shape
        P = np.empty((T, J))
        for k in range(J):
            P[:, k] = _prob_of(mu, self.V_, k)
        return P / P.sum(axis=1, keepdims=True)


def _prob_of(mu, V, k):
    """P(alternative k wins) for each row of mu (vectorized).

    The likelihood's own per-observation map (_choice_logprob_terms), so
    prediction prices a model exactly as fitting does: the same gauge-
    invariant dispatch (#213), the same continuous blend across the
    node-family switch (#420), the J = 2 closed form, and log_ndtr
    instead of the 1e-300 probability floor this copy still carried."""
    T = mu.shape[0]
    lp, _, _ = _choice_logprob_terms(mu, V, choice=np.full(T, int(k)))
    return np.exp(lp)


class MNProbitClassifier:
    """sklearn-style interface: fit(X, y) / predict_proba(X) / score.

    X is (T, J, p) alternative-specific covariates (documented departure
    from sklearn's 2-D X: choice models need per-alternative features);
    y is (T,) integer choices.
    """

    def __init__(self, r=None, intercepts=True):
        self.r = r
        self.intercepts = intercepts

    def fit(self, X, y):
        self.model_ = MNProbit(X, y, intercepts=self.intercepts,
                               r=self.r).fit()
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        P = self.predict_proba(X)
        # the fit's label contract: one label in 0..J-1 per row of X. A
        # bare index scored a short y on a prefix of X and read -1 as
        # the last alternative (#492).
        y = check_choice(y, P.shape[0], P.shape[1])
        return float(np.mean(np.log(np.maximum(
            P[np.arange(len(y)), y], 1e-300))))
