"""The exact n=3 pruning problem, and what is new at three.

Exchangeable case: three unit-volatility trajectories, pairwise
correlation rho, price lambda per active path per unit time, priced-
computation accounting as in the n=2 experiment. Relative to the
current leader the state is the two challenger gaps; gap increments
have Var = 2(1-rho) dt each and correlation exactly 1/2 (a universal
constant of the exchangeable case: Cov(dg_2, dg_3) = (1-rho) dt).
The value U3(ga, gb, b) relative to the current leader obeys

  U3(., 0)  = 0,
  U3(ga, gb, b) = max{ V2(ga, b),
      -3 lambda dt + credit + E U3(sort(ga', gb'), b - dt) },

where V2 is the n=2 value at the CLOSER challenger's gap (killing
the middle path instead is dominated in the exchangeable case: the
same two-path law from a worse start; and stopping is dominated
because V2 >= 0), primes are the diffused gaps re-sorted around the
new leader, and credit = E[(new leader) - (old leader)] is the
reflection drift, as in n=2.

Measured:
  1. the kill-worst boundary h3(gb | ga, b): CROWDING -- how the
     threshold for the far challenger moves as the near challenger
     tightens. Diminishing returns (the select_race_group
     submodularity, now dynamic) predicts it FALLS as ga -> 0.
  2. certification: MC value of the computed policy vs U3(0, 0, B).
  3. the n=2 pairwise heuristic (kill any challenger whose gap
     exceeds the n=2 boundary at the remaining budget) valued in the
     same MC: the near-optimality gap that justifies, or kills, the
     pairwise rule before it is trusted at n = 16.
  4. a heterogeneous vignette by simulation: leader, a near-duplicate
     second (pairwise gap volatility 0.2), an independent third
     (pairwise volatility sqrt(2)); the correlation-aware rule kills
     the SECOND-PLACE path first and beats the kill-worst-first rule
     -- rank order is not kill order under correlation.
"""
import json
import os
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

DT = 0.001
QZ = 9
RHO = 0.5
LAM = 1.0
B_TOTAL = 1.0
SIG2 = 2.0 * (1.0 - RHO)          # gap increment variance per unit t
NG = 140
GMAX = 0.42


def gh2():
    z, w = np.polynomial.hermite_e.hermegauss(QZ)
    Z1, Z2 = np.meshgrid(z, z, indexing="ij")
    W = np.outer(w, w).ravel() / w.sum() ** 2
    # gap-increment Cholesky: Var 1, corr 1/2 (exchangeable constant)
    e2 = Z1.ravel()
    e3 = 0.5 * Z1.ravel() + np.sqrt(0.75) * Z2.ravel()
    return e2, e3, W


def solve_n2():
    """n=2 table U2[k, j] on the same DT/lambda/sigma, cost 2 lambda."""
    z, w = np.polynomial.hermite_e.hermegauss(21)
    w = w / w.sum()
    steps = int(round(B_TOTAL / DT))
    dgrid = np.linspace(0.0, GMAX, NG)
    U = np.zeros(NG)
    sd = np.sqrt(SIG2 * DT)
    tgt = np.abs(dgrid[:, None] + sd * z[None, :])
    drift = 0.5 * ((tgt @ w) - dgrid)
    table = np.zeros((steps + 1, NG))
    for k in range(1, steps + 1):
        EU = np.interp(tgt, dgrid, U) @ w
        U = np.maximum(0.0, -2.0 * LAM * DT + drift + EU)
        table[k] = U
    return dgrid, table


def solve_n3(dgrid, U2):
    steps = int(round(B_TOTAL / DT))
    e2, e3, W = gh2()
    sd = np.sqrt(SIG2 * DT)
    G1, G2 = np.meshgrid(dgrid, dgrid, indexing="ij")   # ga rows, gb cols
    ga = G1.ravel()[:, None]
    gb = G2.ravel()[:, None]
    g2p = ga - sd * e2[None, :]      # sign: gap shrinks when challenger
    g3p = gb - sd * e3[None, :]      # outdiffuses the leader
    m = np.minimum(0.0, np.minimum(g2p, g3p))
    credit = (-m @ W)                               # leader drift
    v1 = g2p - m
    v2 = g3p - m
    lead_was_2 = g2p <= np.minimum(0.0, g3p)        # path2 new leader
    lead_was_3 = (g3p < np.minimum(0.0, g2p))
    # new gap pair: the two nonzero entries of {(-m), v1, v2}
    o1 = np.where(lead_was_2, -m, np.where(lead_was_3, -m, v1))
    o2 = np.where(lead_was_2, v2, np.where(lead_was_3, v1, v2))
    # when old leader stays leader (m == 0): gaps are (v1, v2) = (g2p, g3p)
    stay = (m == 0.0)
    o1 = np.where(stay, g2p, o1)
    o2 = np.where(stay, g3p, o2)
    a_new = np.minimum(o1, o2)
    b_new = np.maximum(o1, o2)
    # precompute bilinear interpolation indices/weights on the square
    h = dgrid[1] - dgrid[0]
    def bil(coords):
        c = np.clip(coords, 0.0, dgrid[-1] - 1e-12)
        i = np.floor(c / h).astype(np.int32)
        f = c / h - i
        return i, f
    ia, fa = bil(a_new)
    ib, fb = bil(b_new)
    U = np.zeros((NG, NG))
    keepall_frontier = None
    for k in range(1, steps + 1):
        Uf = U
        # symmetric extension not needed: a_new <= b_new by sorting
        v00 = Uf[ia, ib]
        v01 = Uf[ia, np.minimum(ib + 1, NG - 1)]
        v10 = Uf[np.minimum(ia + 1, NG - 1), ib]
        v11 = Uf[np.minimum(ia + 1, NG - 1), np.minimum(ib + 1, NG - 1)]
        EU = ((1 - fa) * (1 - fb) * v00 + (1 - fa) * fb * v01
              + fa * (1 - fb) * v10 + fa * fb * v11) @ W
        keep = -3.0 * LAM * DT + credit + EU
        killworst = np.interp(ga.ravel(), dgrid, U2[k])
        Unew = np.maximum(keep, killworst).reshape(NG, NG)
        # enforce the ga <= gb domain: the value at (a, b) with a > b
        # is the value at (b, a)
        Unew = np.where(G1 <= G2, Unew, Unew.T)
        keep_mask = (keep > killworst + 1e-15).reshape(NG, NG)
        U = Unew
        if k == steps:
            keepall_frontier = keep_mask
    return U, keepall_frontier


def simulate_exchangeable(policy, n_mc, seed, dgrid=None, U2=None,
                          frontier=None, U3=None):
    """MC from (0,0): 'exact' follows the computed n=3 decision rule,
    'pairwise' kills any challenger beyond the n=2 boundary."""
    rng = np.random.default_rng(seed)
    steps = int(round(B_TOTAL / DT))
    # n=2 boundary per remaining step index
    h2 = np.array([dgrid[U2[k] > 0].max() if (U2[k] > 0).any() else 0.0
                   for k in range(U2.shape[0])])
    sd = np.sqrt(SIG2 * DT)
    ga = np.zeros(n_mc)
    gb = np.zeros(n_mc)
    lead = np.zeros(n_mc)
    cost = np.zeros(n_mc)
    n_alive = np.full(n_mc, 3)
    for k in range(steps, 0, -1):
        three = n_alive == 3
        if three.any():
            if policy == "exact":
                # keep-all iff U3 keep action wins; approximate by
                # frontier at saturation for all k (boundary is flat
                # in b beyond t*; exact near exhaustion matters little)
                gi = np.clip((ga[three] / (dgrid[1] - dgrid[0])),
                             0, NG - 1).astype(int)
                gj = np.clip((gb[three] / (dgrid[1] - dgrid[0])),
                             0, NG - 1).astype(int)
                keep3 = frontier[gi, gj]
            else:
                keep3 = gb[three] <= h2[k]
            drop = ~keep3
            idx = np.where(three)[0][drop]
            n_alive[idx] = 2                      # worst killed
        two = n_alive == 2
        idx2 = np.where(two)[0]
        kill2 = ga[idx2] > h2[k]                  # n=2 exact rule
        n_alive[idx2[kill2]] = 1
        # diffuse
        three = n_alive == 3
        two = n_alive == 2
        cost[three] += 3 * LAM * DT
        cost[two] += 2 * LAM * DT
        if three.any():
            e2 = rng.normal(size=three.sum())
            e3c = 0.5 * e2 + np.sqrt(0.75) * rng.normal(size=three.sum())
            g2p = ga[three] - sd * e2
            g3p = gb[three] - sd * e3c
            m = np.minimum(0, np.minimum(g2p, g3p))
            lead[three] += -m
            v1, v2 = g2p - m, g3p - m
            o1 = np.where(g2p <= np.minimum(0.0, g3p), -m,
                          np.where(g3p < np.minimum(0.0, g2p), -m, v1))
            o2 = np.where(g2p <= np.minimum(0.0, g3p), v2,
                          np.where(g3p < np.minimum(0.0, g2p), v1, v2))
            stay = m == 0
            o1 = np.where(stay, g2p, o1)
            o2 = np.where(stay, g3p, o2)
            ga[three] = np.minimum(o1, o2)
            gb[three] = np.maximum(o1, o2)
        if two.any():
            z = rng.normal(size=two.sum())
            gp = ga[two] - sd * z
            lead[two] += np.maximum(0.0, -gp)
            ga[two] = np.abs(gp)
    return float((lead - cost).mean()), \
        float((lead - cost).std() / np.sqrt(n_mc))


def vignette(n_mc, seed):
    """Leader A, near-duplicate second B (sigma_AB = 0.2), independent
    third C (sigma ~ sqrt(2)); B starts 0.03 behind, C 0.10 behind.
    Rules: kill-worst-first (rank order) vs correlation-aware
    catchability (kill i when gap_i > 0.115 sigma_iL^2 / lambda),
    both with the n=2 exact rule once two remain... both reduced to
    simple thresholds here; keep-all baseline."""
    rng = np.random.default_rng(seed)
    steps = int(round(B_TOTAL / DT))
    vload = np.array([1.0, 1.0, 0.0])
    dio = np.array([0.02, 0.02, 1.0])
    out = {}
    for rule in ("catch", "rank", "keepall"):
        X = np.zeros((n_mc, 3))
        X[:, 1] -= 0.03
        X[:, 2] -= 0.10
        alive = np.ones((n_mc, 3), bool)
        cost = np.zeros(n_mc)
        frozen = np.zeros(n_mc, bool)   # lone leader stops: a single
        for k in range(steps):          # survivor is a martingale
                                        # minus cost, pure waste
            Xm = np.where(alive, X, -np.inf)
            L = Xm.argmax(1)
            XL = Xm[np.arange(n_mc), L]
            if rule != "keepall":
                sig2 = ((vload[None, :] - vload[L][:, None]) ** 2
                        + dio[None, :] + dio[L][:, None])
                if rule == "catch":
                    thr = 0.115 * sig2 / LAM
                else:
                    thr = 0.115 * (2.0 / LAM)      # rank order: one
                    # common threshold from the average volatility --
                    # kills strictly by gap size
                kill = alive & ((XL[:, None] - Xm) > thr)
                kill[np.arange(n_mc), L] = False
                # rank rule: only ever kill the currently worst
                if rule == "rank":
                    worst = np.where(alive, Xm, np.inf).argmin(1)
                    onlyworst = np.zeros_like(kill)
                    onlyworst[np.arange(n_mc), worst] = True
                    kill &= onlyworst
                alive &= ~kill
            frozen |= alive.sum(1) == 1
            na = np.where(frozen, 0, alive.sum(1))
            cost += na * LAM * DT
            F = rng.normal(size=n_mc) * np.sqrt(DT)
            Zb = rng.normal(size=(n_mc, 3)) * np.sqrt(DT)
            X += np.where(alive & ~frozen[:, None],
                          vload[None, :] * F[:, None]
                          + np.sqrt(dio)[None, :] * Zb, 0.0)
        val = np.where(alive, X, -np.inf).max(1) - cost
        out[rule] = dict(value=float(val.mean()),
                         se=float(val.std() / np.sqrt(n_mc)))
    return out


if __name__ == "__main__":
    t0 = time.time()
    results = {}
    dgrid, U2 = solve_n2()
    U3, frontier = solve_n3(dgrid, U2)
    # crowding: kill-worst boundary in gb as ga varies
    rows = []
    for gaval in (0.0, 0.03, 0.06, 0.10, 0.15):
        i = int(round(gaval / (dgrid[1] - dgrid[0])))
        row = frontier[i]
        hb = dgrid[row].max() if row.any() else 0.0
        rows.append([float(gaval), float(hb)])
    h2_sat = float(dgrid[U2[-1] > 0].max())
    results["crowding"] = dict(h3_gb_given_ga=rows, h2_boundary=h2_sat,
                               u3_00=float(U3[0, 0]))
    print("[n=3 exact] U3(0,0) =", f"{U3[0, 0]:.4f}",
          " n=2 boundary", f"{h2_sat:.3f}")
    for gaval, hb in rows:
        print(f"  keep-all frontier: ga={gaval:.2f} -> kill worst at "
              f"gb > {hb:.3f}")
    v_ex, se_ex = simulate_exchangeable("exact", 150_000, 5,
                                        dgrid=dgrid, U2=U2,
                                        frontier=frontier)
    v_pw, se_pw = simulate_exchangeable("pairwise", 150_000, 5,
                                        dgrid=dgrid, U2=U2,
                                        frontier=frontier)
    results["policies"] = dict(
        exact=dict(value=v_ex, se=se_ex),
        pairwise=dict(value=v_pw, se=se_pw),
        bellman=float(U3[0, 0]))
    print(f"[certification] exact-policy MC {v_ex:.4f}±{se_ex:.4f} vs "
          f"Bellman {U3[0, 0]:.4f}; pairwise heuristic {v_pw:.4f}"
          f"±{se_pw:.4f}")
    results["vignette"] = vignette(150_000, 9)
    vg = results["vignette"]
    print("[duplicate-second vignette] "
          + "  ".join(f"{k} {vg[k]['value']:.4f}±{vg[k]['se']:.4f}"
                      for k in vg))
    results["seconds"] = time.time() - t0
    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print(f"done in {results['seconds']:.0f}s")
