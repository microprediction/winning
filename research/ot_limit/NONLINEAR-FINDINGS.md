# The exact nonlinear extremal transform, and where global competition enters

Notes 2026-08-19, from a parallel thread, verified here in `nonlinear_circle.py`.
This supersedes the "candidate operator" that CIRCLE-FINDINGS.md reported as
falsified: the missing ingredient was the **exposure threshold**.

## The formula, and it is exact

For U(v) = mu(v) + F'v on S^{r-1} with F ~ N(0, I_r), the density of the
global argmax is

    p_mu(v) = (2 pi)^{-r/2} e^{-|grad_S mu(v)|^2 / 2}
              * INT_{A(v)}^{inf} e^{-a^2/2} det( a I - Hess_S mu(v) ) da,

    A(v) = sup_{w != v} [ mu(w) - mu(v) - grad_S mu(v) . w ] / (1 - v.w).

Derivation sketch: if v wins, tangential stationarity grad_S mu(v) + P_v F = 0
pins the tangential part of F, leaving only the radial part a = F.v free, so
F = a v - grad_S mu(v). The Jacobian of (v, a) -> F is det(aI - Hess mu), the
Gaussian weight is exp(-(a^2 + |grad mu|^2)/2), and v beats every competitor
exactly when a >= A(v). The integrand is the Gaussian Minkowski / Monge-Ampere
surface density with support function h = t - mu; the formula is that measure
integrated over the nested family of Wulff shapes.

**A(v) is the exposure threshold**: how hard the factor must push radially
toward v before v beats not just its neighbours but everything on the sphere.
Locally A(v) >= the curvature; globally it also sees distant peaks.

### On the circle it is elementary

r = 2 makes both integrals elementary:

    rho(th) = 2 pi p(th) = e^{-mu'(th)^2/2} [ e^{-A(th)^2/2}
                                              - sqrt(2pi) mu''(th) Phibar(A(th)) ]

Setting A == 0 gives Phibar(0) = 1/2 and recovers exactly the candidate
`e^{-mu'^2/2}(1 - sqrt(pi/2) mu'')` that this repo falsified. **So the entire
second-order error was the exposure threshold**, not the structure.

## Verification

**Analytic anchor at k = 1.** For mu = a cos(theta) the sup defining A is
constant in the shift: numerator = a cos(th)(cos d - 1), denominator = 1 - cos d,
so **A(th) = -a cos(th)** exactly. Substituting, and using
e^{-a^2 sin^2/2} = e^{-a^2/2} e^{c^2/2} with c = a cos th, the formula collapses
algebraically to

    rho = e^{-a^2/2} [ 1 + c sqrt(2pi) e^{c^2/2} Phi(c) ],

which is the **projected normal** -- the angular density of N((a,0), I_2), the
independently known exact answer, since a cos(theta) merely shifts F1 by a.
Numerically: max|A - (-a cos)| ~ 1e-11 and max|formula - projected normal| ~
1e-16 at a = 0.2, 0.7, 1.5.

**Monte Carlo, general (a, k), 3e6 races.** Harmonic gains, exact formula vs MC:

| a | k | a k^2 | gain (formula) | gain (MC) | linear | G |
|---|---|---|---|---|---|---|
| 0.05 | 1 | 0.05 | 1.253 | 1.263 | 1.253 | 1.000 |
| 1.00 | 1 | 1.00 | 1.114 | 1.115 | 1.253 | 0.889 |
| 0.15 | 3 | 1.35 | 7.899 | 7.900 | 11.280 | 0.700 |
| 0.02 | 8 | 1.28 | 56.642 | 56.626 | 80.212 | 0.706 |
| 0.05 | 8 | 3.20 | 34.025 | 33.980 | 80.212 | 0.424 |

Three to four significant figures, deep into the nonlinear regime. The formula
also self-normalizes (mean rho = 1.000000 at moderate amplitude), which is a
structural check nothing forced.

## The universal G is now derived, not measured

CIRCLE-FINDINGS.md reported an empirical universal function G(a k^2) with
G(0.32) ~ 0.92, G(1.28) ~ 0.71, G(5.12) ~ 0.29. Computing gain/linear-gain
straight from the exact formula:

| a k^2 | 0.32 | 1.28 | 5.12 | 20.5 |
|---|---|---|---|---|
| formula | .9222 / .9244 / .9335 | .7057 / .7062 / .7100 | .2912 / .2895 | .0883 |
| earlier MC | .921 / .922 / .946 | .704 / .706 / .709 | .288 / .289 | .077 |

(three entries per cell = three different (a, k) pairs at the same a k^2).

So the empirical collapse was real and is now explained. Note the formula also
shows the collapse is **excellent but not exact** (.9222 vs .9335 at the same
a k^2), because A depends on the whole function, not on mu'' alone. That is a
prediction the earlier noisy MC could not have resolved.

## Where global competition enters: order epsilon^r

Put mu = eps f. Because the sup defining A is positively homogeneous,
A_{eps f} = eps A_f for eps > 0. Split the radial integral:

    INT_{eps A}^{inf} = INT_0^{inf} - INT_0^{eps A}.

The first piece expands in local invariants: det(aI - eps H) = sum_j (-eps)^j
e_j(H) a^{d-j} with d = r - 1, and since e_1(H) = tr H = Laplace_S f, the
leading correction is -c_r eps Laplace_S f -- the linear law, recovered.

The second piece, substituting a = eps s, is

    eps^{d+1} INT_0^{A_f} e^{-eps^2 s^2/2} det(sI - H) ds,   d + 1 = r.

**So the global exposure functional first appears at order eps^r, where r is
the factor rank.** Orders eps^1 ... eps^{r-1} are pure local differential
geometry of f; only at eps^r does "can this direction actually beat everything
else on the sphere" enter.

### Why, in one sentence

Write F = R n. For a typical draw R = O(1), a competitor at fixed angular
distance loses R(1 - cos) = O(1) while the whole ability field is only O(eps) --
it cannot possibly win, so only the local neighbourhood of n matters. Global
competition needs R = O(eps), and for an r-dimensional Gaussian
P(R <~ eps) ~ eps^r because the small ball has volume eps^r. The local
expansion is really an expansion in eps/R, and eps^r is the measure of the
boundary layer where it breaks down.

Consequences by dimension: on the circle (r = 2) global exposure hits at second
order, which is why the circle leaves the linear regime so quickly and why its
first nonlinear correction is already nonlocal. On S^2 (r = 3) the first two
orders are local; on S^3 the first three are.

### Second order for r >= 3 is still local

With m_j = E[R^{-j}] = 2^{-j/2} Gamma((r-j)/2)/Gamma(r/2) (so m_1 = c_r and
m_2 = 1/(r-2), finite only for r > 2):

    p/p_0 = 1 - c_r eps Laplace_S f
              + eps^2 [ e_2(Hess f)/(r-2) - |grad f|^2 / 2 ] + O(eps^3),

with e_2(H) = ((tr H)^2 - ||H||_F^2)/2. **Untested.** This is the cleanest
next numerical target: it predicts specific quadratic spherical-harmonic mode
coupling with no free constants, so feeding a single Y_lm and reading off the
generated degrees is a zero-fitted-parameter test.

## Putting the idiosyncratic noise back

**The lifting makes D > 0 a hard race again.** With Z = (F, eps_1..eps_N) and
a_i = (v_i, sigma_i e_i), U_i = mu_i + a_i'Z, so the soft cell in F-space is the
epsilon-fiber measure of a hard Laguerre cell upstairs. Verified elsewhere in
this repo: J_sigma is an exact graph Laplacian at every sigma.

**D > 0 gives a genuinely analytic calculus.** p_i(mu) = INT_{C_i} phi_Sigma(y - mu) dy
is a Gaussian convolution of a polyhedral indicator, so with Sigma nonsingular
all derivatives exist and are Hermite moments: with q = f'Sigma^{-1}f and
Z_f = f'Sigma^{-1}X / sqrt(q),

    d^n/d eps^n p_i(eps f) |_0 = q^{n/2} E[ 1{X in C_i} He_n(Z_f) ].

So positive noise **removes** the one-sided eps^r nonsmoothness of the hard
problem. The trade is that locality is lost: J_sigma is dense, so the continuum
operator is nonlocal rather than -Laplace_S.

**A continuum trap worth recording.** You cannot take N -> infinity on the
sphere with independent noise at fixed sigma: max_i eps_i ~ sigma sqrt(2 log N)
eventually swamps the bounded geometric score, and the limit is a different
extreme-value problem entirely. Three defensible models: (i) finite N with
D > 0, which is the actual algorithm; (ii) dense sphere with sigma_N -> 0, a
controlled deformation of the hard theory; (iii) a *smooth correlated* Gaussian
residual field eta(v), where the exact hard formula can be reused inside an
expectation over eta.

**The transfer function is the right target.** By rotational symmetry (Funk-Hecke),
an isotropic design with equal sigma must have spherical harmonics as
eigenfunctions of the linearized operator, so

    L_sigma Y_lm = lambda_l(sigma) Y_lm,   lambda_l(0) = c_r l(l + r - 2).

Positive sigma should bend lambda_l over at high l instead of letting it grow
like l^2 -- that is the low-pass, as the symbol of an integrable nonlocal
kernel INT K(delta)[1 - cos(l delta)] d delta saturates. **Measuring
lambda_l(sigma) by projecting the Jacobian at uniformity onto harmonics is the
single most informative remaining experiment**, and it supersedes the earlier
attempt to collapse attenuation against sigma k^2 with noisy amplitude-confounded
data.

## Novelty, honestly

The determinant e^{-(|grad h|^2 + h^2)/2} det(Hess h + h I) is the standard
Gaussian surface-area density in Gaussian Minkowski theory, and first variations
of Wulff shapes are classical. What I have not found stated is the packaging as
an **argmax-density perturbation theorem** with the exposure threshold made
explicit, and specifically the eps^r local-to-global result. That statement is
simple enough that a convex geometer might call it a corollary -- but it is a
sharp, searchable claim rather than a vague one, which the earlier framing was
not.

The associated statistical law -- Fisher information per winner observation
I_l = c_r^2 [l(l + r - 2)]^2, hence sd(coefficient) >~ 1/(sqrt(M) c_r l(l+r-2))
-- remains the piece least likely to be owned by the OT or convex-geometry
literatures, since neither has a reason to state it.
