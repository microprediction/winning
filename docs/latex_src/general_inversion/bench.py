"""Seeded benchmarks behind the tables in 'A General Contest Inversion
Algorithm'. Run on one laptop; wall clock varies, ratios are stable.

  python bench.py ghk     # the against-GHK table (n = 10, 50, 200)
  python bench.py law     # GHK cost-law points (n = 200, 500, 1000, R=1000)
  python bench.py scale   # lattice at n = 1e4, 1e5, 1e6
  python bench.py alt     # the against-the-field table: Genz (scipy),
                          # frequency at matched time, Mendell-Elston,
                          # Clark-type moment matching (n = 10, 30)
"""
import sys
import time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtr, ndtri
from winning.factor.races import race_probabilities
import fastrace

def tail_band_err(p, ref, lo=1e-4, hi=1e-3):
    """Max abs log-error over reference probabilities in [lo, hi]: a
    log-odds-style tail metric the bulk TV number cannot see."""
    band = (ref >= lo) & (ref <= hi)
    if not band.any():
        return float("nan")
    return float(np.max(np.abs(np.log(np.maximum(p[band], 1e-300))
                               - np.log(ref[band]))))


mode = sys.argv[1] if len(sys.argv) > 1 else "ghk"
rng = np.random.default_rng(4 if mode != "scale" else 1)

if mode == "ghk":
    for n in (10, 50, 200):
        mu = rng.normal(size=n)
        V = rng.normal(size=(n, 2)) * 0.4
        D = 0.5 + rng.random(n)
        L = np.linalg.cholesky(V @ V.T + np.diag(D))
        z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=1)
                          .random_base2(20), 1e-12, 1 - 1e-12)).T
        ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                          minlength=n) / z.shape[1]
        t0 = time.time()
        p = race_probabilities(mu, V=V, D=D, points=257)
        line = (f"n={n:4d}  race {1e3*(time.time()-t0):7.1f} ms "
                f"TV {0.5*np.abs(p-ref).sum():.2e} "
                f"tail {tail_band_err(p, ref):.2f}")
        for R in (1000, 10000):
            t0 = time.time()
            g = np.asarray(fastrace.ghk_all_shares(-mu, V, D, R, 7))
            t_g = time.time() - t0
            g = g / g.sum()
            line += (f"  | GHK R={R}: {1e3*t_g:8.1f} ms "
                     f"TV {0.5*np.abs(g-ref).sum():.2e} "
                     f"tail {tail_band_err(g, ref):.2f}")
        print(line, flush=True)
elif mode == "alt":
    from scipy.stats import multivariate_normal, norm

    def diff_moments(mu, Sig, i):
        """Moments of d = (X_j - X_i)_{j != i}; p_i = P(d >= 0) (min wins)."""
        n = len(mu)
        A = np.eye(n)[np.arange(n) != i]
        A[:, i] = -1.0
        return A @ mu, A @ Sig @ A.T

    def genz_vector(mu, Sig, budget_s=600.0):
        # P(d >= 0) = P(d - m >= -m) = cdf of the centered law at +m
        p, t0 = np.zeros(len(mu)), time.time()
        for i in range(len(mu)):
            m, S = diff_moments(mu, Sig, i)
            p[i] = multivariate_normal(mean=np.zeros(len(m)), cov=S,
                                       allow_singular=True).cdf(m)
            if time.time() - t0 > budget_s:
                return None, time.time() - t0, i + 1
        return p / p.sum(), time.time() - t0, len(mu)

    def mendell_elston(m, S):
        """P(d >= 0) by sequential conditioning (Mendell-Elston 1974)."""
        m, S = m.copy(), S.copy()
        logp = 0.0
        for _ in range(len(m)):
            s = np.sqrt(max(S[0, 0], 1e-300))
            a = m[0] / s
            q = max(ndtr(a), 1e-300)
            logp += np.log(q)
            lam = np.exp(norm.logpdf(a) - np.log(q))
            delta = lam * (lam + a)
            if len(m) == 1:
                break
            c = S[0, 1:] / S[0, 0]
            m = m[1:] + c * (s * lam)
            S = S[1:, 1:] - np.outer(S[0, 1:], S[0, 1:]) / S[0, 0] * delta
        return np.exp(logp)

    def me_vector(mu, Sig):
        p = np.array([mendell_elston(*diff_moments(mu, Sig, i))
                      for i in range(len(mu))])
        return p / p.sum()

    def clark_vector(mu, Sig):
        """p_i ~ P(X_i < min_{j!=i} X_j) with Clark (1961) min moments,
        tracking the running min's covariance with X_i."""
        n = len(mu)
        sd = np.sqrt(np.diag(Sig))
        p = np.zeros(n)
        for i in range(n):
            rest = [j for j in range(n) if j != i]
            j0 = rest[0]
            mM, vM = mu[j0], Sig[j0, j0]
            cMi = Sig[j0, i]
            cM = {j: Sig[j0, j] for j in rest[1:]}
            for j in rest[1:]:
                a2 = vM + Sig[j, j] - 2 * cM[j]
                a = np.sqrt(max(a2, 1e-12))
                al = (Sig[j, j] ** 0 * (mu[j] - mM)) / a
                Phi, phi = ndtr(al), np.exp(norm.logpdf(al))
                # min(M, X_j): use min(x,y) = -max(-x,-y), Clark on maxima
                m_new = mM * Phi + mu[j] * (1 - Phi) - a * phi
                e2 = ((vM + mM ** 2) * Phi + (Sig[j, j] + mu[j] ** 2)
                      * (1 - Phi) - (mM + mu[j]) * a * phi)
                v_new = max(e2 - m_new ** 2, 1e-12)
                cMi = cMi * Phi + Sig[j, i] * (1 - Phi)
                for k in rest:
                    if k in cM and k != j:
                        cM[k] = cM[k] * Phi + Sig[j, k] * (1 - Phi)
                mM, vM = m_new, v_new
            s2 = max(Sig[i, i] + vM - 2 * cMi, 1e-12)
            p[i] = ndtr((mM - mu[i]) / np.sqrt(s2))
        return p / p.sum()

    def tail_band_err(p, ref, lo=1e-4, hi=1e-3):
        band = (ref >= lo) & (ref <= hi)
        if not band.any():
            return float("nan")
        return float(np.max(np.abs(np.log(np.maximum(p[band], 1e-300))
                                   - np.log(ref[band]))))

    for n in (10, 30):
        mu = rng.normal(size=n)
        V = rng.normal(size=(n, 2)) * 0.4
        D = 0.5 + rng.random(n)
        Sig = V @ V.T + np.diag(D)
        L = np.linalg.cholesky(Sig)
        z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=1)
                          .random_base2(20), 1e-12, 1 - 1e-12)).T
        ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                          minlength=n) / z.shape[1]

        t0 = time.time()
        p = race_probabilities(mu, V=V, D=D, points=257)
        t_race = time.time() - t0
        print(f"n={n}: race {1e3*t_race:8.1f} ms  "
              f"TV {0.5*np.abs(p-ref).sum():.2e}  "
              f"tail {tail_band_err(p, ref):.3f}", flush=True)

        g, t_g, done = genz_vector(mu, Sig)
        if g is None:
            print(f"       genz  budget hit after {done}/{n} probs "
                  f"({t_g:.1f} s)", flush=True)
        else:
            print(f"       genz {1e3*t_g:8.1f} ms  "
                  f"TV {0.5*np.abs(g-ref).sum():.2e}  "
                  f"tail {tail_band_err(g, ref):.3f}", flush=True)

        # frequency at wall-clock matched to the race, and at 10x
        for mult, tag in ((1.0, "freq(=t)"), (10.0, "freq(10t)")):
            t0 = time.time()
            draws = 0
            counts = np.zeros(n)
            while time.time() - t0 < mult * t_race:
                zz = rng.standard_normal((n, 4096))
                counts += np.bincount(
                    np.argmin(mu[:, None] + L @ zz, axis=0), minlength=n)
                draws += 4096
            f = counts / counts.sum()
            nz = int(np.sum((f == 0) & (ref > 0)))
            print(f"       {tag} {draws:9,d} draws  "
                  f"TV {0.5*np.abs(f-ref).sum():.2e}  "
                  f"tail {tail_band_err(f, ref):.3f}  zeros {nz}",
                  flush=True)

        t0 = time.time()
        m_ = me_vector(mu, Sig)
        print(f"       ME   {1e3*(time.time()-t0):8.1f} ms  "
              f"TV {0.5*np.abs(m_-ref).sum():.2e}  "
              f"tail {tail_band_err(m_, ref):.3f}", flush=True)

        t0 = time.time()
        c_ = clark_vector(mu, Sig)
        print(f"       Clark{1e3*(time.time()-t0):8.1f} ms  "
              f"TV {0.5*np.abs(c_-ref).sum():.2e}  "
              f"tail {tail_band_err(c_, ref):.3f}", flush=True)
elif mode == "law":
    for n in (200, 500, 1000):
        mu = rng.normal(size=n)
        V = rng.normal(size=(n, 2)) * 0.4
        D = 0.5 + rng.random(n)
        t0 = time.time()
        fastrace.ghk_all_shares(-mu, V, D, 1000, 7)
        print(f"n={n:5d}  {time.time()-t0:8.2f} s", flush=True)
elif mode == "scale":
    for n in (10_000, 100_000, 1_000_000):
        mu = rng.normal(size=n)
        V = rng.normal(size=(n, 1)) * 0.4
        D = 0.5 + rng.random(n)
        t0 = time.time()
        p = race_probabilities(mu, V=V, D=D, points=257)
        print(f"n={n:9,d}  {time.time()-t0:8.2f} s  (sum {p.sum():.6f})",
              flush=True)
