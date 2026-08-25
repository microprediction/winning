"""The transform on a circle: spectral properties of the inverse-extrema map.

    Y(theta) = mu(theta) + F1 cos theta + F2 sin theta + sigma eps(theta)
    p(theta) = P(theta = argmax Y),      mu(theta) = sum_k a_k cos(k theta)

This is exactly the r = 2 factor race with sites v(theta) = (cos, sin) on the
unit circle, so all of soft_laguerre.py applies.

THE PREDICTION BEING TESTED.  Write F in polars, F = R(cos Theta, sin Theta).
Then F1 cos th + F2 sin th = R cos(th - Theta), so at sigma = 0 the winner is

    phi*(Theta, R) = argmax_phi [ mu(phi) + R cos(phi - Theta) ].

Stationarity mu'(phi) = R sin(phi - Theta) gives, to first order in mu,
phi* = Theta + mu'(Theta)/R.  Theta is uniform, so pushing the density through
this map,

    p(phi) = (1/2pi) / (1 + mu''(Theta)/R)  ~  (1/2pi)(1 - mu''(phi)/R),

and averaging over R (Rayleigh, E[1/R] = sqrt(pi/2)):

    p(theta) ~ (1/2pi) ( 1 - sqrt(pi/2) mu''(theta) )
             = (1/2pi) ( 1 + sqrt(pi/2) sum_k k^2 a_k cos(k theta) ).

So THE FORWARD MAP DIFFERENTIATES TWICE.  Harmonic k of mu appears in the
winner density amplified by k^2, with a constant sqrt(pi/2) = 1.2533.

That inverts the usual inverse-problem story: the forward map amplifies high
frequencies, so its inverse damps noise like 1/k^2, and the map is
ill-conditioned at LOW frequency, not high.  Two structural caveats:

  k = 0 is the null direction (adding a constant to mu changes nothing);
  k = 1 is the factor-aligned mode -- a1 cos theta merely shifts the mean of
        F1, so it is identified, but it lives in the noise geometry itself.

sigma > 0 should act as a low-pass filter and eventually kill the k^2 gain.
"""

from __future__ import annotations

import numpy as np

SQRT_PI_2 = np.sqrt(np.pi / 2)


def geometry(N):
    th = 2 * np.pi * np.arange(N) / N
    return th, np.column_stack([np.cos(th), np.sin(th)])


def shares_mc(mu, V, sigma, draws, seed=0, chunk=4000):
    """P(i = argmax) by Monte Carlo; sigma = 0 needs no per-site noise."""
    rng = np.random.default_rng(seed)
    N = len(mu)
    counts = np.zeros(N)
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        Y = mu + rng.standard_normal((m, 2)) @ V.T
        if sigma > 0:
            Y = Y + sigma * rng.standard_normal((m, N))
        counts += np.bincount(Y.argmax(1), minlength=N)
        done += m
    return counts / draws


def modulation(p):
    """m_n with p_n = (1/N)(1 + m_n)."""
    return len(p) * np.asarray(p) - 1.0


def cos_amp(m, th, k):
    return 2.0 * np.mean(m * np.cos(k * th))


def gain_sweep(N=1024, amp=0.02, ks=(1, 2, 3, 4, 6, 8, 12, 16, 24),
               sigmas=(0.0, 0.02, 0.05, 0.1, 0.2), draws=400_000, seed=1):
    th, V = geometry(N)
    print(f"gain of harmonic k: measured m_k/amp, vs prediction "
          f"sqrt(pi/2) k^2   (N={N}, amp={amp}, draws={draws:,})")
    header = "    k  " + "".join(f"{'sig=' + str(s):>12}" for s in sigmas) \
             + f"{'pred':>10}"
    print(header)
    table = {}
    for k in ks:
        mu = amp * np.cos(k * th)
        row = []
        for si, sg in enumerate(sigmas):
            p = shares_mc(mu, V, sg, draws, seed=seed + 100 * si + k)
            row.append(cos_amp(modulation(p), th, k) / amp)
        table[k] = row
        pred = SQRT_PI_2 * k**2
        print(f"  {k:3d}  " + "".join(f"{g:12.3f}" for g in row)
              + f"{pred:10.3f}")
    print("\n  ratio measured/predicted at sigma = 0:")
    print("   " + "  ".join(f"k={k}:{table[k][0]/(SQRT_PI_2*k**2):.3f}"
                            for k in ks))
    return table


def amplitude_linearity(N=1024, k=3, amps=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
                        draws=400_000, seed=7):
    th, V = geometry(N)
    print(f"\nlinearity in amplitude at k={k}, sigma=0 "
          f"(prediction {SQRT_PI_2*k**2:.3f}, independent of amp):")
    for a in amps:
        p = shares_mc(a * np.cos(k * th), V, 0.0, draws, seed=seed)
        print(f"   amp={a:6.3f}  gain={cos_amp(modulation(p), th, k)/a:8.3f}")


def two_harmonic_recovery(N=2048, a=0.03, b=0.02, draws=2_000_000, seed=3):
    """Recover (a, b) from winner frequencies alone, via the k^2 law."""
    th, V = geometry(N)
    mu = a * np.cos(th) + b * np.cos(2 * th)
    p = shares_mc(mu, V, 0.0, draws, seed=seed)
    m = modulation(p)
    est = {k: cos_amp(m, th, k) / (SQRT_PI_2 * k**2) for k in (1, 2, 3, 4)}
    print(f"\ntwo-harmonic recovery from {draws:,} winner observations only:")
    print(f"   true   a1={a:.4f}  a2={b:.4f}   a3=0  a4=0")
    print(f"   recov  a1={est[1]:.4f}  a2={est[2]:.4f}   "
          f"a3={est[3]:+.4f}  a4={est[4]:+.4f}")
    return est


if __name__ == "__main__":
    gain_sweep()
    amplitude_linearity()
    two_harmonic_recovery()
