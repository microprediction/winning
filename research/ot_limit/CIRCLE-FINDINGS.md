# The transform on a circle: the inverse-extrema map differentiates twice

Run 2026-08-19, `circle_spectral.py`. Setting:

    Y(theta) = mu(theta) + F1 cos theta + F2 sin theta + sigma eps(theta)
    p(theta) = P(theta = argmax Y),   mu(theta) = sum_k a_k cos(k theta)

This is the r = 2 factor race with sites on the unit circle, N sites.

## The law

**Forward map, to linear order:**

    p(theta) = (1/2pi) ( 1 - sqrt(pi/2) * mu''(theta) )

i.e. harmonic k of mu appears in the winner density amplified by
**sqrt(pi/2) k^2**, with sqrt(pi/2) = 1.25331.

Measured gains (modulation per unit a_k, N = 1024, amp = 0.02, sigma = 0,
4e5 draws) against the prediction:

| k | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| measured/predicted | 1.11 | 0.97 | 0.95 | 0.92 | 0.83 | 0.71 |

k >= 4 falls off because the linear-response condition is violated, not
because the law is wrong -- see the collapse below. k = 1 is the anomalous
factor-aligned mode.

## Two independent derivations agree to six decimals

**Probabilistic.** Write F = R(cos Theta, sin Theta); then the factor term is
R cos(theta - Theta). At sigma = 0 the winner solves mu'(phi) = R sin(phi -
Theta), so phi* = Theta + mu'(Theta)/R to first order. Theta is uniform, so
pushing its density through that map gives p = (1/2pi)/(1 + mu''/R), and
averaging over R (Rayleigh, E[1/R] = sqrt(pi/2)) gives the law.

**Geometric.** For sites on a circle the power diagram's adjacency graph is a
**cycle**, so the Jacobian -- which is the weighted graph Laplacian of the
Laguerre diagram (verified separately to 5.6e-17, see FINDINGS.md) -- has
eigenvalues lambda_k = 2 k_e (1 - cos(2 pi k/N)) with edge weight
k_e = phi(0)/2 / (2 sin(pi/N)). The implied gain N*lambda_k versus
sqrt(pi/2) k^2:

| N | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| 256 | 0.99998 | 0.99982 | 0.99922 | 0.99682 | 0.98724 |
| 4096 | 1.000000 | 0.999999 | 0.999997 | 0.999988 | 0.999950 |

Agreement to six decimals, improving as N grows (the gap is the O((k/N)^2)
discretization of the cycle Laplacian).

**So "the Jacobian is a graph Laplacian" and "the forward map is a second
derivative" are the same statement.** On the circle the k^2 is literally the
Laplacian's eigenvalue spectrum. The OT correspondence and the spectral
conditioning are one fact, not two.

> **Superseded in part by NONLINEAR-FINDINGS.md (same day).** The exact
> nonlinear winner density is now known in closed form on the circle, the
> falsified candidate operator was missing only the *exposure threshold*
> A(theta), and the universal G below is a derived consequence of that formula
> rather than an empirical curve. Everything in this file remains correct; it is
> the linearization.

## Regime of validity: a universal collapse in |mu''|

The gain is `sqrt(pi/2) k^2 * G(a_k k^2)` for a universal G with G(0) = 1.
Three different (amplitude, k) pairs at equal a k^2 give the same ratio:

| a k^2 | 0.32 | 1.28 | 5.12 | 20.5 |
|---|---|---|---|---|
| ratio (amp 0.005) | 0.921 | 0.704 | -- | -- |
| ratio (amp 0.02) | 0.922 | 0.706 | 0.288 | -- |
| ratio (amp 0.08) | 0.946 | 0.709 | 0.289 | 0.077 |

The collapsing variable is a_k k^2, which is exactly the amplitude of mu''.
That the nonlinear regime is also governed by mu'' is further evidence that
mu'' is the right object.

Large-x behaviour: x G(x) -> about 1.5, so the *modulation* m_k = gain * a_k
saturates near 1.9 regardless of k or amplitude. The ceiling is ~2, the value
for a winner density fully concentrated on the k maxima of mu -- so
saturation is the winner locking onto local maxima instead of tracking Theta.

## sigma is a low-pass filter

Gains at increasing sigma (N = 1024, amp = 0.02):

| k | sigma=0 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| 2 | 4.88 | 4.84 | 4.47 | 4.38 | 3.71 |
| 4 | 18.5 | 16.4 | 14.5 | 12.1 | 9.1 |
| 8 | 56.7 | 44.4 | 32.9 | 22.0 | 13.0 |
| 16 | 92.5 | 70.5 | 44.0 | 25.2 | 13.3 |

Idiosyncratic noise damps high harmonics, competing directly with the k^2
amplification and setting a resolution limit for the inverse map.

## Recovery from winners alone

mu = 0.03 cos(theta) + 0.02 cos(2 theta), N = 2048, 2e6 winner observations,
nothing else:

    true    a1 = 0.0300   a2 = 0.0200   a3 = 0        a4 = 0
    recovered a1 = 0.0297  a2 = 0.0198  a3 = +0.0000  a4 = +0.0001

About 1% error on the real coefficients, and the harmonics that are absent
come back absent.

## Why this matters: the conditioning runs the wrong way round

The forward map **amplifies** high frequencies (k^2), so the inverse **damps**
noise like 1/k^2. High harmonics of the latent field are recovered *more*
accurately than low ones; the map is ill-conditioned at LOW frequency. That is
the reverse of the usual inverse-problem situation and the reverse of the
compressed-sensing intuition, where fine detail is the fragile part.

For the physical "inverse extrema" programme this says: from repeated winner
locations you can recover the fine structure of the latent landscape, but the
broad shape is what you cannot see. The k = 0 mode is formally unidentifiable
(the **1** null direction) and low k is nearly so.

## Caveats

- k = 1 is the factor-aligned mode: a1 cos(theta) merely shifts the mean of
  F1. It is identified but sits inside the noise geometry, and it measured
  1.11-1.18 times the prediction rather than 1.0. It needs its own treatment.
- All of the above is sigma-small and amplitude-small; the saturation regime
  is characterized empirically here, not derived.
- Monte Carlo throughout (4e5-2e6 draws); the ratios carry roughly 1-3%
  noise, which is why the collapse table is more convincing than any single
  entry.
- Only cos harmonics were driven. Sin harmonics should behave identically by
  rotational symmetry, untested.

## Next

1. Derive G analytically, or at least its 1/x tail.
2. Repeat with the deterministic transform instead of Monte Carlo, to get the
   ratios to 1e-4 and pin the k=1 anomaly.
3. The Fisher-information calculation (notes/optimal-transport-connection.md
   section 7) is now much more interesting: the information about a_k should
   scale like k^4 / p, so the Cramer-Rao bound on a_k should *improve* with k.
   That is the rigorous version of this note.
4. ~~Higher r: on the sphere with r = 3 the same argument should give the
   Laplace-Beltrami spectrum l(l+1) instead of k^2.~~ **Done and confirmed
   for r = 2, 3, 4, 5: see SPHERE-FINDINGS.md.** The general law is
   gain = c_r l(l + r - 2) with c_r = E[1/R]; this circle result is its
   r = 2 member.
