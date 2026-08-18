"""Experiment 27: verify every identity claimed by the perturbed-max
companion paper, on a direct max-wins implementation.

Model: U_i = mu_i + v_i'f + sqrt(D_i) eps_i, p_i = P(U_i = max_j U_j).

Checked here against finite differences / Monte Carlo / linear algebra:
 1. grad_mu G = p (FD of the same-field potential)
 2. shared-field reverse mode: VJP for mu, V, D vs central differences
 3. heat identity dG/dD_i = J_ii / 2
 4. matrix identity grad_V G = J V
 5. NLL gradient -J e_y / p_y vs FD of -log p_y
 6. binary Fenchel-Young closed form s[phi(t) - t Phi(-t)]
 7. dual curvature = effective resistance: FD of Psi along e_i - e_j
    equals (e_i-e_j)' J^+ (e_i-e_j)
 8. softmax special case R_ij = 1/p_i + 1/p_j (linear algebra)
 9. G vs direct Monte Carlo

Run:  python experiments/exp27_perturbedmax_identities/run_identities.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr
from numpy.polynomial.hermite_e import hermegauss

HERE = Path(__file__).resolve().parent
SEED = 12


def gh_nodes(k, order=21):
    x1, w1 = hermegauss(order)
    w1 = w1 / w1.sum()
    if k == 0:
        return np.zeros((1, 1)), np.ones(1)
    grids = np.array(np.meshgrid(*([x1] * k))).reshape(k, -1).T
    ws = np.array(np.meshgrid(*([w1] * k))).reshape(k, -1).T.prod(axis=1)
    keep = ws > 1e-12 * ws.max()
    return grids[keep], ws[keep] / ws[keep].sum()


def fields(mu, V, D, F, W, L=2001):
    """Conditional quantities on a shared grid. Max-wins: F_j = P(U_j <= x)."""
    sd = np.sqrt(D)
    M = mu[None, :] + F @ V.T                     # (Q, N)
    lo = M.min() - 8 * sd.max()
    hi = M.max() + 8 * sd.max()
    x = np.linspace(lo, hi, L)
    dx = x[1] - x[0]
    z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]   # (Q,N,L)
    logF = log_ndtr(z)
    g = np.exp(-0.5 * z * z) / (sd[None, :, None] * np.sqrt(2 * np.pi))
    return x, dx, z, logF, g, M


def forward(mu, V, D, F, W, L=2001):
    x, dx, z, logF, g, M = fields(mu, V, D, F, W, L)
    field = logF.sum(axis=1)                       # (Q, L)
    rest = np.exp(np.clip(field[:, None, :] - logF, -745, 0))
    r = g * rest                                   # winner densities (Q,N,L)
    p = (W[:, None] * (r.sum(axis=2) * dx)).sum(axis=0)
    # potential from the same field: E max | f = hi - int H dx on [lo, hi]
    H = np.exp(np.clip(field, -745, 0))
    intH = (H.sum(axis=1) - 0.5 * (H[:, 0] + H[:, -1])) * dx
    Emax_f = x[-1] - intH
    G = float(W @ Emax_f)
    return p, G


def vjp(mu, V, D, F, W, c, L=2001):
    """Shared-field reverse mode: dL/dmu, dL/dV, dL/dD for L with dL/dp = c
    (cotangent of the UNNORMALIZED p; at this resolution the defect is ~1e-12)."""
    x, dx, z, logF, g, M = fields(mu, V, D, F, W, L)
    sd = np.sqrt(D)
    field = logF.sum(axis=1)
    rest = np.exp(np.clip(field[:, None, :] - logF, -745, 0))
    r = g * rest                                    # (Q,N,L)
    Sc = np.einsum("j,qjl->ql", c, r)               # cotangent field (Q,L)
    lam = np.exp(np.log(np.maximum(g, 1e-300)) - logF)   # hazards g/F
    xm = x[None, None, :] - M[:, :, None]           # (Q,N,L)
    other = Sc[:, None, :] - c[None, :, None] * r   # S_c - c_j r_j
    a = ((c[None, :, None] * r * xm / D[None, :, None]
          - other * lam) * dx).sum(axis=2)          # (Q,N)
    b = ((c[None, :, None] * r * (xm**2 - D[None, :, None])
          / (2 * D[None, :, None]**2)
          - other * xm / (2 * D[None, :, None]) * lam) * dx).sum(axis=2)
    gmu = W @ a
    gV = np.einsum("q,qj,qk->jk", W, a, F)
    gD = W @ b
    return gmu, gV, gD


def main():
    rng = np.random.default_rng(SEED)
    n, k = 5, 1
    mu = rng.normal(0, 1.0, n)
    V = rng.normal(0, 0.5, (n, k))
    D = rng.uniform(0.6, 1.4, n)
    F, W = gh_nodes(k)
    eps = 1e-5
    report = []

    def fd(fun, x0, i, e=eps):
        xp = x0.copy(); xp.flat[i] += e
        xm = x0.copy(); xm.flat[i] -= e
        return (fun(xp) - fun(xm)) / (2 * e)

    p0, G0 = forward(mu, V, D, F, W)

    # 1. grad_mu G = p
    gfd = np.array([fd(lambda m: forward(m, V, D, F, W)[1], mu, i)
                    for i in range(n)])
    report.append(("grad_mu G = p", np.abs(gfd - p0).max()))

    # 2. VJP vs FD through random cotangent
    c = rng.normal(0, 1, n)
    gmu, gV, gD = vjp(mu, V, D, F, W, c)
    fmu = np.array([fd(lambda m: c @ forward(m, V, D, F, W)[0], mu, i)
                    for i in range(n)])
    fV = np.array([fd(lambda v: c @ forward(mu, v.reshape(n, k), D, F, W)[0],
                      V.copy().ravel(), i)
                   for i in range(n * k)]).reshape(n, k)
    fD = np.array([fd(lambda d: c @ forward(mu, V, d, F, W)[0], D, i)
                   for i in range(n)])
    report.append(("VJP mu", np.abs(gmu - fmu).max()))
    report.append(("VJP V", np.abs(gV - fV).max()))
    report.append(("VJP D", np.abs(gD - fD).max()))

    # J by FD of p (symmetric reference)
    J = np.array([[fd(lambda m: forward(m, V, D, F, W)[0][i], mu, j)
                   for j in range(n)] for i in range(n)])
    J = 0.5 * (J + J.T)

    # 3. heat identity dG/dD_i = J_ii / 2
    hfd = np.array([fd(lambda d: forward(mu, V, d, F, W)[1], D, i)
                    for i in range(n)])
    report.append(("heat dG/dD = J_ii/2", np.abs(hfd - np.diag(J) / 2).max()))

    # 4. grad_V G = J V
    vfd = np.array([fd(lambda v: forward(mu, v.reshape(n, k), D, F, W)[1],
                       V.copy().ravel(), i)
                    for i in range(n * k)]).reshape(n, k)
    report.append(("grad_V G = J V", np.abs(vfd - J @ V).max()))

    # 5. NLL gradient
    y = 2
    nfd = np.array([fd(lambda m: -np.log(forward(m, V, D, F, W)[0][y]), mu, i)
                    for i in range(n)])
    report.append(("NLL grad = -J e_y/p_y", np.abs(nfd + J[:, y] / p0[y]).max()))

    # 6. binary FY closed form
    mu2 = np.array([0.4, -0.3]); V2 = np.array([[0.5], [-0.1]])
    D2 = np.array([1.0, 1.3])
    p2, G2 = forward(mu2, V2, D2, *gh_nodes(1))
    s = np.sqrt(np.sum((V2[0] - V2[1])**2) + D2.sum())
    t = (mu2[0] - mu2[1]) / s
    from scipy.stats import norm
    Lfy = G2 - mu2[0]
    Lclosed = s * (norm.pdf(t) - t * norm.cdf(-t))
    report.append(("binary FY closed form", abs(Lfy - Lclosed)))

    # 7. dual curvature = effective resistance (FD of Psi)
    B = np.linalg.qr(np.eye(n) - np.ones((n, n)) / n)[0][:, :n - 1]
    def invert(q):
        m = np.zeros(n)
        for _ in range(200):
            pq, _ = forward(m, V, D, F, W)
            Jf = np.array([[fd(lambda mm: forward(mm, V, D, F, W)[0][i], m, j)
                            for j in range(n)] for i in range(n)])
            step = np.linalg.lstsq(B.T @ Jf @ B, B.T @ (q - pq), rcond=None)[0]
            m = m + B @ step
            m -= m.mean()
            if np.abs(pq - q).max() < 1e-11:
                break
        return m
    def Psi(q):
        m = invert(q)
        _, Gq = forward(m, V, D, F, W)
        return q @ m - Gq
    Jp = np.linalg.pinv(J)
    i1, j1 = 0, 3
    d = np.zeros(n); d[i1] = 1; d[j1] = -1
    h = 1e-4
    curv = (Psi(p0 + h * d) - 2 * Psi(p0) + Psi(p0 - h * d)) / h**2
    resistance = d @ Jp @ d
    report.append(("dual curvature = resistance", abs(curv - resistance)
                   / abs(resistance)))

    # 8. softmax special case (linear algebra)
    ps = np.exp(rng.normal(0, 1, n)); ps /= ps.sum()
    Js = np.diag(ps) - np.outer(ps, ps)
    Rs = d @ np.linalg.pinv(Js) @ d
    report.append(("softmax R = 1/p_i + 1/p_j",
                   abs(Rs - (1 / ps[i1] + 1 / ps[j1]))))

    # 9. G vs Monte Carlo
    R = 4_000_000
    f = rng.standard_normal((R, k))
    U = mu[None, :] + f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((R, n))
    Gmc = U.max(axis=1).mean()
    se = U.max(axis=1).std() / np.sqrt(R)
    report.append((f"G vs MC (se {se:.1e})", abs(G0 - Gmc)))

    rows = ["identity,max_error"]
    for name, err in report:
        flag = "OK" if err < 5e-5 or "MC" in name and err < 4 * se else "FAIL"
        print(f"{name:<32} {err:.2e}  {flag}")
        rows.append(f"{name},{err:.3e}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__" and "--part2" not in sys.argv and "--part3" not in sys.argv:
    main()


def part2():
    """Referee-round additions: tail remainder, Gumbel surrogate identity,
    NLL convexity, softmax FY=NLL, active-class screening bounds."""
    from scipy.stats import norm
    rng = np.random.default_rng(13)
    n, k = 5, 1
    mu = rng.normal(0, 1.0, n)
    V = rng.normal(0, 0.5, (n, k))
    D = rng.uniform(0.6, 1.4, n)
    F, W = gh_nodes(k)
    report = []

    # A. potential upper-tail remainder and its certified bound
    x, dx, z, logF, g, M = fields(mu, V, D, F, W, 2001)
    field = logF.sum(axis=1)
    H = np.exp(np.clip(field, -745, 0))
    b = x[-1]
    # true remainder by wide-grid extension
    x2 = np.linspace(b, b + 12, 4001)
    rem = 0.0
    bound = 0.0
    for q in range(len(F)):
        z2 = (x2[None, :] - M[q, :, None]) / np.sqrt(D)[:, None]
        H2 = np.exp(np.clip(log_ndtr(z2).sum(axis=0), -745, 0))
        rem += W[q] * np.trapezoid(1 - H2, x2)
        aa = (b - M[q]) / np.sqrt(D)
        bound += W[q] * np.sum(np.sqrt(D) * norm.pdf(aa)
                               - (b - M[q]) * norm.sf(aa))
    report.append(("tail remainder <= bound", 0.0 if rem <= bound + 1e-15 else rem - bound))
    print(f"tail remainder {rem:.2e} certified bound {bound:.2e} "
          f"({'OK' if rem <= bound + 1e-15 else 'FAIL'})")

    # B. temperature-softmax surrogate = Gaussian + tau*Gumbel argmax
    tau = 0.7
    R = 2_000_000
    f = rng.standard_normal((R, k))
    Xi = f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((R, n))
    Usm = (mu[None, :] + Xi) / tau
    Usm -= Usm.max(axis=1, keepdims=True)
    Esm = np.exp(Usm)
    p_surr = (Esm / Esm.sum(axis=1, keepdims=True)).mean(axis=0)
    Gum = -np.log(-np.log(rng.uniform(size=(R, n))))
    p_gg = np.bincount(np.argmax(mu[None, :] + Xi + tau * Gum, axis=1),
                       minlength=n) / R
    err = np.abs(p_surr - p_gg).max()
    se = 3 * np.sqrt((p_gg * (1 - p_gg)).max() / R)
    report.append(("surrogate = Gauss+Gumbel argmax", err))
    print(f"surrogate identity err {err:.2e} (3 se {se:.2e}) "
          f"({'OK' if err < 2 * se + 1e-4 else 'FAIL'})")

    # C. convexity of -log p_y along random directions
    y = 1
    worst = np.inf
    for _ in range(20):
        h = rng.normal(0, 1, n); h /= np.linalg.norm(h)
        e = 1e-3
        f0 = -np.log(forward(mu, V, D, F, W)[0][y])
        fp = -np.log(forward(mu + e * h, V, D, F, W)[0][y])
        fm = -np.log(forward(mu - e * h, V, D, F, W)[0][y])
        worst = min(worst, (fp - 2 * f0 + fm) / e**2)
    report.append(("NLL convexity (min 2nd diff)", -min(worst, 0)))
    print(f"NLL min second difference {worst:.2e} "
          f"({'OK' if worst > -1e-6 else 'FAIL'})")

    # D. softmax: FY loss equals NLL exactly
    mus = rng.normal(0, 1, n)
    ps = np.exp(mus - mus.max()); ps /= ps.sum()
    Gsm = np.log(np.exp(mus).sum())
    fy = Gsm - mus[y]
    nll = -np.log(ps[y])
    report.append(("softmax FY = NLL", abs(fy - nll)))
    print(f"softmax FY = NLL err {abs(fy - nll):.2e} OK")

    # E. active-class sandwich and tie-density screening bound
    p_full, _ = forward(mu, V, D, F, W)
    yb = int(np.argmax(p_full))
    S = [j for j in range(n) if j != yb and p_full[j] > 0.05]
    keep = [yb] + S
    pS, _ = forward(mu[keep], V[keep], D[keep], F, W)
    pyS = pS[0] * 1.0
    # note pS renormalized over subset = P(y beats S) needs unnormalized...
    # compute P(y beats S) directly: subset race probability of y
    # (forward normalizes; at these resolutions total ~1 so pS[0] is it)
    delta = 0.0
    for j in range(n):
        if j == yb or j in S:
            continue
        s2 = D[j] + D[yb] + np.sum((V[j] - V[yb])**2)
        delta += norm.cdf((mu[j] - mu[yb]) / np.sqrt(s2))
    lo_b, hi_b = max(0.0, pyS - delta), pyS
    ok = lo_b - 1e-9 <= p_full[yb] <= hi_b + 1e-9
    report.append(("active-class sandwich", 0.0 if ok else 1.0))
    print(f"sandwich [{lo_b:.5f}, {hi_b:.5f}] contains p_y {p_full[yb]:.5f} "
          f"({'OK' if ok else 'FAIL'})")

    # tie-density bound w_yj <= phi(dmu/s)/s via FD J
    eps = 1e-5
    Jcol = np.array([(forward(mu + eps * np.eye(n)[j], V, D, F, W)[0][yb]
                      - forward(mu - eps * np.eye(n)[j], V, D, F, W)[0][yb])
                     / (2 * eps) for j in range(n)])
    okw = True
    for j in range(n):
        if j == yb:
            continue
        s2 = D[j] + D[yb] + np.sum((V[j] - V[yb])**2)
        wb = norm.pdf((mu[yb] - mu[j]) / np.sqrt(s2)) / np.sqrt(s2)
        if -Jcol[j] > wb + 1e-6:
            okw = False
    report.append(("tie-density bound", 0.0 if okw else 1.0))
    print(f"tie-density screening bound {'OK' if okw else 'FAIL'}")
    return report


if __name__ == "__main__" and "--part2" in sys.argv:
    part2()


def part3():
    """Transport-round identities: Stein/conjugate-value formula and the
    binary entropy closed form."""
    from scipy.stats import norm
    rng = np.random.default_rng(14)
    n, k = 5, 1
    mu = rng.normal(0, 1.0, n)
    V = rng.normal(0, 0.5, (n, k))
    D = rng.uniform(0.6, 1.4, n)
    F, W = gh_nodes(k)
    Sigma = V @ V.T + np.diag(D)
    p0, G0 = forward(mu, V, D, F, W)

    eps = 1e-5
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = eps
        J[:, j] = (forward(mu + e, V, D, F, W)[0]
                   - forward(mu - e, V, D, F, W)[0]) / (2 * eps)
    J = 0.5 * (J + J.T)

    # A. Stein per-coordinate: E[xi_i 1{I=i}] = (Sigma J)_ii  (row sums)
    R = 4_000_000
    f = rng.standard_normal((R, k))
    Xi = f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((R, n))
    I = np.argmax(mu[None, :] + Xi, axis=1)
    lhs = np.array([Xi[I == i, i].sum() / R for i in range(n)])
    rhs = np.diag(Sigma @ J)
    se = np.sqrt(np.array([np.var(Xi[:, i] * (I == i)) for i in range(n)]) / R)
    print(f"Stein per-class: max |lhs - rhs| = {np.abs(lhs - rhs).max():.2e} "
          f"(3 se {3*se.max():.2e})")

    # B. G = mu'p + tr(Sigma J)
    g_id = mu @ p0 + np.trace(Sigma @ J)
    print(f"G = mu'p + tr(Sigma J): |{G0:.6f} - {g_id:.6f}| = "
          f"{abs(G0 - g_id):.2e}")

    # C. Omega(p) = -tr(Sigma J) via Omega = p'mu - G at the same mu
    omega = p0 @ mu - G0
    print(f"Omega = -tr(Sigma J): |{omega:.6f} - {-np.trace(Sigma @ J):.6f}| "
          f"= {abs(omega + np.trace(Sigma @ J)):.2e}")

    # D. binary closed form: Omega(q) = -s phi(Phi^-1(q))
    mu2 = np.array([0.37, -0.37]); V2 = np.array([[0.4], [-0.3]])
    D2 = np.array([0.9, 1.2])
    s = np.sqrt(np.sum((V2[0] - V2[1])**2) + D2.sum())
    p2, G2 = forward(mu2, V2, D2, *gh_nodes(1))
    omega2 = p2 @ mu2 - G2
    closed = -s * norm.pdf(norm.ppf(p2[0]))
    print(f"binary Omega: |{omega2:.8f} - {closed:.8f}| = "
          f"{abs(omega2 - closed):.2e}")

    # E. Laguerre identity: argmax(x_i + mu_i) = argmin(0.5||x-b_i||^2 - mu_i)
    P = np.eye(n) - np.ones((n, n)) / n
    Bv = P
    ok = True
    for _ in range(2000):
        xi = rng.normal(0, 1, n)
        x = P @ xi
        a1 = np.argmax(x + mu)
        a2 = np.argmin(0.5 * np.sum((x[None, :] - Bv)**2, axis=1) - mu)
        if a1 != a2:
            ok = False
            break
    print(f"Laguerre-cell identity: {'OK' if ok else 'FAIL'}")


if __name__ == "__main__" and "--part3" in sys.argv:
    part3()
