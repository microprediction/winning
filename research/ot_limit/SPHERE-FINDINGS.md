# The photo-finish operator is Laplace-Beltrami on the sphere

Run 2026-08-19, `sphere_spectral.py`. Companion to CIRCLE-FINDINGS.md, which
established the r = 2 case.

## The law, confirmed

Put possible winners densely on S^{r-1} in factor space and race
U(v) = mu(v) + F'v. The predicted forward gain of a zonal harmonic of degree
l is

    g(l, r) = c_r * l(l + r - 2),      c_r = E[1/R] = 2^{-1/2} Gamma((r-1)/2) / Gamma(r/2)

with l(l + r - 2) the Laplace-Beltrami eigenvalue on S^{r-1}. Measured against
prediction, N = 30000 sites, amplitude 0.01, 60k paired draws:

| r | c_r | l=1 | l=2 | l=3 | l=4 |
|---|---|---|---|---|---|
| 2 | 1.25331 | 1.006 | 0.990 | 0.976 | 0.959 |
| 3 | 0.79788 | 1.007 | 1.005 | 0.993 | 1.003 |
| 4 | 0.62666 | 0.986 | 1.003 | 0.999 | 0.988 |
| 5 | 0.53192 | 0.991 | 0.954 | 0.969 | -- |

(entries are measured/predicted). r = 3 and r = 4 land within 1.5%
everywhere; r = 5 within 5%, consistent with S^4 being covered by only
30000^{1/4} ~ 13 points per dimension.

**The r = 2 falloff is not error -- it is the nonlinear law, quantitatively.**
At a = 0.01 and l = 4 the circle's collapse variable is a*l^2 = 0.16, where
the universal G measured in CIRCLE-FINDINGS.md gives about 0.96. Measured:
0.959. The linear spectral law and the nonlinear saturation law agree on the
same data point.

**The nonlinear onset is r-dependent.** At r = 3, l = 4 the analogous variable
a*l(l+1) = 0.2 shows no falloff at all (ratio 1.003). Consistent with the fold
picture: folding needs R to fall below the curvature scale of mu, and typical
R grows like sqrt(r), so higher factor rank delays the fold.

## It is not a finite-N artifact

Discretization check at r = 3, l = 2 (prediction 4.7873):

| N | 4000 | 10000 | 30000 | 80000 |
|---|---|---|---|---|
| ratio | 0.998 | 0.990 | 1.004 | 0.998 |

Flat across a factor of 20 in N. The continuum limit is reached early; N = 4000
already suffices at low degree.

## Method note

Sites are exactly uniform (normalized Gaussians), and the test field is the
zonal Gegenbauer polynomial C_l^{(r-2)/2}(v.e), an exact eigenfunction
(Legendre at r = 3, Chebyshev-U at r = 4, Chebyshev-T at r = 2). Gains use a
paired +a / -a design with common random numbers. This matters: at finite N
the dominant artifact is local site-density fluctuation -- a site in a sparse
patch wins more often -- which is identical in both arms and cancels exactly
in the difference, along with every even order in a. Without the pairing the
density noise would have swamped the measurement.

## What this settles

The "photo-finish graph -> outcome manifold" passage is real and general, not
a special property of the circle. In every factor rank tested, the Jacobian's
continuum limit is c_r times the Laplace-Beltrami operator of the sphere on
which the competitors live, and the inverse transform is its Green operator
with spectrum 1/(c_r l(l + r - 2)).

Practical consequence, unchanged across r: the forward map amplifies high
harmonic degree, the inverse damps sampling noise like 1/l^2, and the badly
conditioned directions are the LOW degrees -- the broad shape of the latent
field, not its fine structure. Since r here is exactly the factor rank the
fast transform is certified for (2 to 5), the whole family is reachable.

## Not yet done (queued)

1. **Soft spectrum.** Preliminary look at the existing circle data is
   discouraging: normalizing the sigma > 0 gains by their sigma = 0 values,
   the attenuation does *not* collapse cleanly against sigma*k^2 at large
   argument (at sigma k^2 = 12.8 the attenuations are 0.48 and 0.23 for
   sigma = 0.05 and 0.2). But that data was taken at amplitude 0.02, where
   the nonlinear factor is already active and confounds it. Needs a clean
   small-amplitude run with the deterministic transform before concluding
   anything.
2. **Fold law**: test onset against max sqrt(mu'^2 + mu''^2) rather than
   a k^2, which the r-dependence above already suggests is the better
   variable.
3. **Residual order**: whether the correction after the linear term is a^2 or
   a^2 log a (the Rayleigh density's mass near R = 0 could make higher orders
   singular).
