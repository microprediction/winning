# The Atlas connection: tie densities are collision local times

A short note. The winning paper's intro defers this "for another day";
this is the sketch of that day. Nothing here is verified against the
literature yet — every citation carries [U] until read at source.

## The static object

The paper's Proposition (photo-finish flux): for a race with locations
mu and smooth joint density,

    w_ij  =  dp_i / dmu_j  =  int phi_ij(t, t) Pr{ X_k > t for all k != i,j | X_i = X_j = t } dt,

the probability density that i and j dead-heat AHEAD of the field. A
symmetric, nonnegative edge weight; the Jacobian is the graph Laplacian
built from these conductances.

## The dynamic object

Take n particles X_i(t) = mu_i + sigma W_i(t) (independent Brownian
motions; correlated versions via the same factor conditioning as the
package). At any horizon T the time-T marginals are exactly a Gaussian
race with D_i = sigma^2 T, so the static machinery prices
p_i(T) = Pr{ X_i(T) = min } and its Jacobian at every horizon.

Now watch the boundary instead of the marginal. The event "i and j are
the leading pair and tied" is the particle system touching the set
{ X_i = X_j <= X_k }. For semimartingales, time spent straddling such a
set is measured by a LOCAL TIME: Tanaka / the occupation-times formula
give, for the difference Y = X_i - X_j (variance rate 2 sigma^2),

    E[ Lambda_T^{ij,lead} ]  =  lim_{eps->0} (1/2eps) E int_0^T 1{ |Y_s| < eps, both lead } d<Y>_s
                             =  2 sigma^2 int_0^T phi_ij-at-tie(s) Pr{ field above | tie at s } ds
                             =  2 sigma^2 int_0^T w_ij(s) ds,

where w_ij(s) is the static tie density of the time-s race. In words:

    the Jacobian entry at horizon s is (up to 2 sigma^2) the RATE at
    which the leading-pair collision local time accrues, and the
    integrated local time is the integral of the tie densities across
    horizons.

So the boundary flux is not an analogy to local time; for Brownian
races it IS the local-time density, and the lattice computes it — for
all n(n-1)/2 pairs implicitly, and any single pair explicitly — at
O(nL) per horizon. Claim to prove carefully: the eps-limit exchange and
the restriction "both lead" (the indicator is discontinuous on the
lower-dimensional triple-tie set, which has measure zero for
nondegenerate diffusions; triple collisions are exactly the degeneracy
the Atlas literature worries about [U: Ichiba–Karatzas on triple
collisions]).

## Why the Atlas literature should care

Rank-based (Atlas) models [U: Fernholz 2002; Banner–Fernholz–Karatzas,
Ann. Appl. Probab. 2005] write the dynamics of the k-th RANKED particle
with drift and diffusion assigned by rank; the semimartingale
decomposition of the ranked processes involves precisely the collision
local times Lambda^{(k,k+1)} between adjacent ranks — they are the
leakage terms, and in stochastic portfolio theory the size effect and
the decay of the largest market weight are expressed through them.
Market weights are a choice-probability vector (the winning paper's HRP
remark is the same bridge), and rank-crossing local times are what our
static machinery evaluates one horizon at a time.

Two directions of trade:

1. Computation flowing to Atlas: collision local times are analytically
   hard beyond small n or special symmetric cases [U]. The shared-field
   lattice evaluates E-local-time densities for the LEADING pair (and,
   with the removal counterfactual, for lower ranks: delete the leaders
   and re-race) at O(nL) per horizon, for n in the millions. That looks
   new for that literature, and is checkable against simulated ranked
   Brownian systems — an exact-vs-Monte-Carlo experiment of the kind
   this repo already knows how to referee.

2. Structure flowing back to us: the Atlas decomposition suggests what
   the DYNAMIC generalization of the inversion is: given observed
   occupation/turnover of ranks (how often leadership changes hands),
   recover the drifts. The static inversion is the T-marginal shadow of
   that problem.

## Scope honesty

- The identity above is for the leading pair. Adjacent-rank local
  times at rank k need the "top-k field" variant of the cavity (delete
  the k-1 leaders and re-race); cost and accuracy unmeasured.
- Correlated (factor) versions: time-T marginals stay in the grammar,
  so the static evaluation generalizes, but the local-time identity
  needs <Y> adjusted for the factor covariance — mechanical, unwritten.
- Nothing here yet says anything about stationary distributions of gaps
  [U: Pal–Pitman], which is where much of the Atlas literature lives.
  Our contribution would be finite-horizon and computational.
