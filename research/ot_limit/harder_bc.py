"""Parts (B) and (C) of the gap-closing programme; part (A) lives in harder.py.

(B) The noisy transfer function lambda_k(sigma). Driving the eigenvector
    directly with a paired +-a design under common random numbers is far
    better conditioned than finite-differencing a Jacobian column. The
    amplitude is chosen k-adaptively, a_k = 0.1/k^2, so every k sits at the
    same point a_k k^2 = 0.1 of the nonlinear scale and only sigma varies.

(C) Nonuniform site density. Finite volume: the facet conductance is
    k_i = phi(0)/(2 Delta_i), so k_i Delta_i = phi(0)/2 is CONSTANT and the
    sampling density cancels. Prediction: rho = 1 - sqrt(pi/2) mu'' whatever
    the density.
"""

from __future__ import annotations

import numpy as np

import circle_spectral as C

C2 = np.sqrt(np.pi / 2)


def part_B(N=256, M=1_500_000, chunk=2000, seed=11):
    print("(B) noisy transfer function lambda_k(sigma) / (c_2 k^2)")
    th, V = C.geometry(N)
    ks = (1, 2, 4, 8, 16, 32)
    amps = {k: 0.1 / k ** 2 for k in ks}
    drive = {k: amps[k] * np.cos(k * th) for k in ks}
    sigmas = (0.0, 0.0316, 0.0707, 0.1414, 0.2828, 0.5657)
    print(f"   {'sigma':>7} " + " ".join(f"{'k=' + str(k):>8}" for k in ks))
    table = {}
    for sig in sigmas:
        rng = np.random.default_rng(seed)
        cp = {k: np.zeros(N) for k in ks}
        cm = {k: np.zeros(N) for k in ks}
        done = 0
        while done < M:
            m = min(chunk, M - done)
            base = rng.standard_normal((m, 2)) @ V.T
            if sig > 0:
                base = base + sig * rng.standard_normal((m, N))
            for k in ks:
                cp[k] += np.bincount((base + drive[k]).argmax(1), minlength=N)
                cm[k] += np.bincount((base - drive[k]).argmax(1), minlength=N)
            done += m
        row = []
        for k in ks:
            diff = N * (cp[k] - cm[k]) / (2 * done)
            row.append(C.cos_amp(diff, th, k) / amps[k] / (C2 * k * k))
        table[sig] = np.array(row)
        print(f"   {sig:7.4f} " + " ".join(f"{v:8.4f}" for v in row))

    base0 = table[0.0]
    print("\n   attenuation A(sigma,k) = gain(sigma)/gain(0), collapse quality")
    best = None
    for fn, name in ((lambda s, k: s * k * k, "sigma k^2"),
                     (lambda s, k: s * k, "sigma k"),
                     (lambda s, k: s * s * k * k, "sigma^2 k^2"),
                     (lambda s, k: s * k ** 1.5, "sigma k^1.5")):
        xs, ys = [], []
        for s in sigmas:
            if s == 0:
                continue
            for i, k in enumerate(ks):
                v = table[s][i] / base0[i]
                if 0.03 < v < 0.97:
                    xs.append(fn(s, k))
                    ys.append(v)
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 8:
            print(f"      {name:>12}: too few points")
            continue
        c = np.polyfit(np.log(xs), np.log(ys), 3)
        sc = float((np.log(ys) - np.polyval(c, np.log(xs))).std())
        print(f"      {name:>12}: n={len(xs):3d}   residual scatter {sc:.4f}")
        if best is None or sc < best[1]:
            best = (name, sc)
    print(f"   -> best collapse variable: {best[0]} (scatter {best[1]:.4f})")
    return table


def nonuniform_sites(N, amp):
    u = (np.arange(N) + 0.5) / N
    g = np.linspace(0, 2 * np.pi, 400001)
    return np.interp(u, (g + amp * np.sin(g)) / (2 * np.pi), g)


def part_C(N=1536, a=0.01, draws=6_000_000):
    print("\n(C) is the continuum operator independent of sampling density?")
    print(f"   {'k':>3} {'dens amp':>9} {'min/max gap':>12} {'gain':>9} "
          f"{'c_2 k^2':>9} {'ratio':>8}")
    out = {}
    for k in (2, 3):
        for amp in (0.0, 0.4, 0.8):
            th = nonuniform_sites(N, amp)
            V = np.column_stack([np.cos(th), np.sin(th)])
            mu = a * np.cos(k * th)
            rng = np.random.default_rng(2)
            cnt, done = np.zeros(N), 0
            while done < draws:
                m = min(4000, draws - done)
                cnt += np.bincount(
                    (mu + rng.standard_normal((m, 2)) @ V.T).argmax(1),
                    minlength=N)
                done += m
            gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
            arc = (gaps + np.roll(gaps, 1)) / 2
            wq = arc / (2 * np.pi)
            rho = (cnt / done) / wq
            gain = 2 * np.sum((rho - 1) * np.cos(k * th) * wq) / a
            out[(k, amp)] = gain
            print(f"   {k:3d} {amp:9.2f} {gaps.min()/gaps.max():12.3f} "
                  f"{gain:9.3f} {C2*k*k:9.3f} {gain/(C2*k*k):8.4f}")
    for k in (2, 3):
        v = [out[(k, a_)] for a_ in (0.0, 0.4, 0.8)]
        print(f"   k={k}: spread across densities = "
              f"{(max(v)-min(v))/np.mean(v):.2e} (relative)")


if __name__ == "__main__":
    part_B()
    part_C()
