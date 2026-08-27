"""Multinomial probit estimation: the model statsmodels does not have
and sklearn cannot express.

Built on winning.likelihood (exact factor-conditional likelihood with
analytic score). Covariance parameterization: rank-r factor loadings
with a zero reference row and a strictly-lower-triangular free block,
unit idiosyncratic variance. At J alternatives and r = 2 this covers
every positive-definite differenced covariance up to scale with the
same degree-of-freedom count as the differenced-Cholesky
parameterization used by R's mlogit (see r/mlogitfast, whose Fishing
fit this module reproduces cross-language).

    fit = MNProbit(X, choice).fit()          # X: (T, J, p) covariates
    clf = MNProbitClassifier().fit(X, y)     # sklearn-style
    P = clf.predict_proba(X)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .likelihood import choice_loglik_and_score, nodes_for_likelihood


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
    r : factor rank (r = 2 spans the full identified covariance at J=4).
    """

    def __init__(self, X, choice, intercepts=True, r=2):
        X = np.asarray(X, dtype=float)
        self.T, self.J, p = X.shape
        self.choice = np.asarray(choice)
        self.r = int(r)
        if intercepts:
            Z = np.zeros((self.T, self.J, self.J - 1))
            for j in range(1, self.J):
                Z[:, j, j - 1] = 1.0
            X = np.concatenate([Z, X], axis=2)
        self.X = X
        self.p = X.shape[2]
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
        theta0 = np.concatenate([np.zeros(self.p),
                                 np.full(len(self.pos), 0.1)])
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
        if X.shape[2] != self.p:
            Z = np.zeros((X.shape[0], self.J, self.J - 1))
            for j in range(1, self.J):
                Z[:, j, j - 1] = 1.0
            X = np.concatenate([Z, X], axis=2)
        mu = X @ self.params_
        T, J = mu.shape
        P = np.empty((T, J))
        for k in range(J):
            P[:, k] = _prob_of(mu, self.V_, k)
        return P / P.sum(axis=1, keepdims=True)


def _prob_of(mu, V, k):
    """P(alternative k wins) for each row of mu (vectorized)."""
    from scipy.special import ndtr
    T, J = mu.shape
    r = V.shape[1]
    sharp = float(np.max(np.sqrt((V ** 2).sum(axis=1))))
    F, W = nodes_for_likelihood(r, 7, 7, sharp)
    Fq, zq = F[:, :r], F[:, r]
    Vf = Fq @ V.T
    rivals = [j for j in range(J) if j != k]
    acc = np.zeros((T, len(W)))
    for j in rivals:
        shift = Vf[:, k] - Vf[:, j] + zq
        a = (mu[:, k] - mu[:, j])[:, None] + shift[None, :]
        acc += np.log(np.maximum(ndtr(a), 1e-300))
    m = acc.max(axis=1)
    return np.exp(m) * (np.exp(acc - m[:, None]) @ W)


class MNProbitClassifier:
    """sklearn-style interface: fit(X, y) / predict_proba(X) / score.

    X is (T, J, p) alternative-specific covariates (documented departure
    from sklearn's 2-D X: choice models need per-alternative features);
    y is (T,) integer choices.
    """

    def __init__(self, r=2, intercepts=True):
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
        return float(np.mean(np.log(np.maximum(
            P[np.arange(len(y)), np.asarray(y)], 1e-300))))
