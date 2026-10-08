// The general race: min-wins, normal/gumbel bases, winner-bulk lattice,
// adaptive factor quadrature. Port of winning/factor/races.py.
import { TINY, ndtr, logndtr, npdf, hermiteNodes, mean, checkOpts, OPT_HINTS, asLoadings, asIdio, gaugeCenter, firstPrimes, asFactorNodes, asWeights, asAbilities, asFiniteVector, asIterations, asTolerance, asCount } from "./core.mjs";

const EULER = 0.5772156649015329;

export const BASES = {
  normal: z => {
    // the upper tail directly: 1 - ndtr(z) cancels to 0 past ~8.3 sd,
    // and a 20-sd longshot came out ~92x too unlikely (#96)
    const S = Math.max(ndtr(-z), 1e-300);
    const f = npdf(z);
    return [S, f, -z * f];
  },
  gumbel: z => {
    const c = Math.PI / Math.sqrt(6);
    const u = Math.min(z * c - EULER, 30);
    const eu = Math.exp(u);
    const S = Math.max(Math.exp(-eu), 1e-300);
    const f = c * eu * S;
    return [S, f, c * c * eu * S * (1 - eu)];
  },
  logistic(z) {
    const c = Math.PI / Math.sqrt(3);
    const u = Math.min(Math.max(c * z, -700), 700);
    const S = 1 / (1 + Math.exp(u));
    const f = c * S * (1 - S);
    return [Math.max(S, 1e-300), f, -c * f * (1 - 2 * S)];
  },
  laplace(z) {
    const b = 1 / Math.sqrt(2);
    const f = Math.exp(-Math.abs(z) / b) / (2 * b);
    const S = z < 0 ? 1 - 0.5 * Math.exp(z / b) : 0.5 * Math.exp(-z / b);
    return [Math.max(S, 1e-300), f, -Math.sign(z) * f / b];
  },
};
/* A base's central scale in standardized units, as python's _resolution
   (#385): a unit-variance law can still concentrate on a much narrower
   centre, and spacing measured in sd alone left that centre between
   lattice points -- a 0.1-scale mixture core missed the race by 3.75
   points at 257 with no warning (#600). Declared as fn.resolution, the
   default 1. A value that is present but not a finite positive number is
   refused rather than read as 1: it is metadata the caller meant. */
export function baseResolution(fn) {
  const r = fn == null ? undefined : fn.resolution;
  if (r === undefined) return 1;
  if (typeof r !== "number" || !Number.isFinite(r) || !(r > 0))
    throw new Error(`base.resolution must be a finite positive number; got ${String(r)}`);
  return r;
}

export const SPANS = { normal: [8, 8], gumbel: [22, 8], logistic: [16, 16], laplace: [18, 18] };

export function requireWholeRule(F, W, where = "F/W") {
  if ((F == null) !== (W == null))
    throw new Error(
      `${where}: a caller factor rule needs BOTH F (the nodes) and W (their ` +
      `weights); got only ${F == null ? "W" : "F"}. Either half alone used ` +
      "to be replaced by the automatic Gaussian rule without a word, which " +
      "prices a different race. Pass both, or neither.");
}

function haltonRule(r, Q) {
  const F = [], W = new Array(Q).fill(1 / Q);
  // generated, not tabulated: a 24-entry table made a valid rank-25 V
  // answer all-NaN, silently (#233)
  const primes = firstPrimes(r);
  for (let idx = 0; idx < Q; idx++) {
    const node = [];
    for (let dim = 0; dim < r; dim++) {
      const b = primes[dim];
      let i = idx + 21, f = 1 / b, h = 0;
      while (i > 0) { h += f * (i % b); i = Math.floor(i / b); f /= b; }
      node.push(invNormalRational(Math.min(Math.max(h, 1e-12), 1 - 1e-12)));
    }
    F.push(node);
  }
  return { F, W };
}

/* The factor rule a race integrates over, chosen in ONE place.

   raceJacobian reimplemented only the Gauss-Hermite branch of this, so
   on a sharp rank-one field the forward used a 114-node midpoint-quantile
   rule and the Jacobian a 114-node Hermite rule: two approximations of
   the same law, and the "derivative" sat 0.058 from finite differences
   of the forward it claims to differentiate, at every lattice size
   (#325). The forward and the Jacobian now both call this.

   `V` must already be shape-normalised and gauge-centred, `D` validated.
   A caller rule is validated, never replaced (#290). */
export function factorRule(V, D, F = null, W = null) {
  requireWholeRule(F, W);
  const n = V.length, r = V[0].length;
  if (F != null) {
    // the caller's nodes go through the same door as V and D: every node
    // carries exactly the loadings' rank, and there is one weight per
    // node (#290)
    const Fv = asFactorNodes(F, r, "F");
    return { F: Fv, W: asWeights(W, Fv.length, "W") };
  }
  // adaptive order: sharpness rule identical to python/R. The statistic
  // is the pairwise-safe bound sqrt(2) * max_i |(PV)_i| / sqrt(D_i), on
  // the CENTERED rows: what decides a race is loading DIFFERENCES, and
  // the raw row norm both misses a sharp pair and depends on the gauge.
  let sharp = 0;
  for (let i = 0; i < n; i++) {
    const nv = Math.sqrt(V[i].reduce((a, b) => a + b * b, 0));
    sharp = Math.max(sharp, nv / Math.sqrt(Math.max(D[i], 1e-300)));
  }
  sharp *= Math.SQRT2;
  // per-rank (Gauss-Hermite order cap, sharpness past which even that
  // order loses to the low-discrepancy family); see GH_RULE in the
  // python reference for the measurements behind each number
  const cap = r === 1 ? 201 : r === 2 ? 41 : r === 3 ? 31 : 15;
  const sharpMax = r === 2 ? 3.75 : r === 3 ? 4.75 : 3.0;
  if (r >= 2 && sharp > sharpMax) {
    // escalate the FAMILY, not the order (matching python/R)
    return haltonRule(r, 8192);
  }
  if (r === 1 && Math.ceil(8 * sharp) > 80) {
    // rank-1 extreme sharpness (matching python/R): equal-weight
    // midpoint-quantile grid scaled with sharpness replaces GH
    const Q = Math.min(Math.ceil(8 * sharp), 4001);
    const Fq = [];
    for (let q = 0; q < Q; q++) Fq.push([invNormalRational((q + 0.5) / Q)]);
    return { F: Fq, W: new Array(Q).fill(1 / Q) };
  }
  if (Math.pow(cap, r) > 100000) {
    // high-rank tensor footgun (matching python/R): Halton fallback
    return haltonRule(r, 8192);
  }
  const Q = Math.min(Math.max(Math.ceil(8 * sharp), 15), cap);
  return hermiteNodes(r, Q);
}

function setup(mu, V, D, F, W, base) {
  // mu was the one argument nobody checked: D goes through asIdio, V
  // through asLoadings, W through asWeights, and the abilities
  // themselves went straight to the lattice. A NaN or inf there came
  // back as NaN probabilities, and an EMPTY field came back as an
  // empty answer rather than a refusal. R and julia already refused
  // both, which is how the cross-port divergence scan found it.
  // asAbilities also turns a Float64Array into a plain Array: its .map
  // coerces nested arrays to NaN, so the default loadings below broke
  // on a typed mu (#334).
  mu = asAbilities(mu);
  const n = mu.length;
  D = asIdio(D, n);        // the companion of asLoadings, #254
  // the shape contract at the door, as python's _setup does it: a
  // scalar, a length-n vector, (n, rank) and (rank, n) are the same
  // race, and a ragged V raises instead of being truncated to the first
  // row's width and answering NaN (#232)
  V = asLoadings(V, n);
  // a factor rule is a PAIR: nodes and their weights. Either half alone
  // used to be discarded in favour of the automatic rule, so a caller's
  // F (or a stale W) vanished and the default Gaussian race came back
  // with no indication (#290).
  requireWholeRule(F, W);
  if (!V) {
    V = mu.map(() => [0]);
    F = [[0]]; W = [1];
  } else {
    // Gauge-fix the loadings, as python, R, julia and the standalone
    // copy all do. A common loading column c adds the same c'f to every
    // performance and cannot move an argmin, so the centered V prices
    // the IDENTICAL race -- but only the centered one makes the node
    // family, the node order and the lattice window invariant under
    // V -> V + 1c'. Uncentered, adding 1 to every loading moved a
    // priced share by 0.0141 (#303). #139 fixed the standalone copy and
    // this newer API was left behind.
    V = gaugeCenter(V);
    ({ F, W } = factorRule(V, D, F, W));
  }
  const fn = typeof base === "function" ? base : BASES[base];
  if (typeof fn !== "function")
    throw new Error(`unknown base ${JSON.stringify(base)}; known: ${Object.keys(BASES).join(", ")}, or a function`);
  baseResolution(fn);          // malformed metadata fails here, not mid-grid
  // a callable base may declare its own span, as python's
  // getattr(base, "span", (12, 12)) (#106)
  const span = typeof base === "function"
    ? (Array.isArray(base.span) ? base.span : [12, 12])
    : (SPANS[base] || [12, 12]);
  return { mu, V, D, F, W, fn, left: span[0], right: span[1] };
}

function condMeans(mu, V, F) {
  // M[q][i] = mu_i + V_i . F_q
  return F.map(fq => mu.map((m, i) => {
    let s = m;
    for (let r = 0; r < fq.length; r++) s += V[i][r] * fq[r];
    return s;
  }));
}


export function invNormalRational(p) {
  // Acklam rational approximation, adequate for node placement
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969,
             138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887,
             66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
             -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143,
             3.75440866190742];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  if (p > 1 - pl) return -invNormalRational(1 - p);
  const q = p - 0.5, r2 = q * q;
  return (((((a[0]*r2+a[1])*r2+a[2])*r2+a[3])*r2+a[4])*r2+a[5])*q /
         (((((b[0]*r2+b[1])*r2+b[2])*r2+b[3])*r2+b[4])*r2+1);
}

/* Lattice over the winner distribution's bulk -- port of python's
 * races._bulk_window: the envelope uses the CALLER'S base survival `fn`
 * (normal when absent), both edges are bracketed before bisection, delta
 * is relaxed by factors of 100 (to at most 1e-4, with a warning) when the
 * requested quantiles will not fit the point budget, and a base's
 * declared `span` widens the pad. The hard-coded normal survival missed a
 * custom Student-t(3) race's first share by 1.2e-4 at any point count
 * (#106), and a Student-t(3) race stayed 1.05e-3 TV off at 4001 points
 * (#380). */
const RELAXED_WARNED = new Set();
function bulkWindow(Mall, sd, points, delta, fn = null) {
  const n = sd.length;
  const S = fn
    ? z => Math.max(fn(z)[0], 1e-300)
    : z => Math.max(ndtr(-z), 1e-300);     // no 1 - ndtr cancellation (#450)
  const muLo = new Array(n).fill(Infinity), muHi = new Array(n).fill(-Infinity);
  for (const row of Mall) for (let i = 0; i < n; i++) {
    if (row[i] < muLo[i]) muLo[i] = row[i];
    if (row[i] > muHi[i]) muHi[i] = row[i];
  }
  const smax = Math.max(...sd), smin = Math.min(...sd);
  const G = (x, mus) => {
    let ls = 0;
    for (let i = 0; i < n; i++) ls += Math.log(S((x - mus[i]) / sd[i]));
    return 1 - Math.exp(ls);
  };
  let warned = false;
  const bracket = (x0, step0, ok, sign) => {
    let step = step0;
    for (let it = 0; it < 60; it++) {
      if (ok(x0)) return x0;
      x0 += sign * step; step *= 2;
    }
    if (!warned && typeof console !== "undefined") {
      warned = true;
      console.warn("bulk window could not bracket the requested quantile " +
                   "after 60 doublings; the window is truncated rather " +
                   "than quantile-exact.");
    }
    return x0;
  };
  let pad = 2 * smax;
  if (fn && Array.isArray(fn.span)) pad = Math.max(pad, 0.25 * Math.max(...fn.span) * smax);
  const windowAt = d => {
    const step0 = Math.max(9 * smax, 1e-12);
    const lo0 = bracket(Math.min(...muLo) - 9 * smax, step0, x => G(x, muLo) <= d, -1);
    const hi0 = bracket(Math.max(...muHi) + 9 * smax, step0, x => G(x, muHi) >= 1 - d, +1);
    let a = lo0, b = hi0;
    for (let it = 0; it < 80; it++) {
      const m = 0.5 * (a + b);
      if (G(m, muLo) < d) a = m; else b = m;
    }
    const xlo = a;
    a = xlo; b = hi0;
    for (let it = 0; it < 80; it++) {
      const m = 0.5 * (a + b);
      if (G(m, muHi) < 1 - d) a = m; else b = m;
    }
    return [xlo - pad, b + pad];
  };
  // half the tightest sd TIMES the base's central scale (#600)
  const budget = 0.5 * smin * baseResolution(fn) * Math.max(points - 1, 1);
  let d = delta;
  let [lo, hi] = windowAt(d);
  while (hi - lo > budget && d < 1e-4) {
    d = Math.min(d * 100, 1e-4);
    [lo, hi] = windowAt(d);
  }
  // once per (budget, achieved delta): an inverse calls this every sweep,
  // and python's warnings registry deduplicates the same way
  const key = `${points}|${d}`;
  if (d > delta && typeof console !== "undefined" && !RELAXED_WARNED.has(key) &&
      RELAXED_WARNED.add(key))
    console.warn(
      `bulk window relaxed delta from ${delta.toExponential(0)} to ` +
      `${d.toExponential(0)}: this base's tail puts the requested quantile ` +
      `further out than ${points} points can resolve, so the window is ` +
      `${(hi - lo).toPrecision(3)} units wide at the relaxed delta and ` +
      "exact there. Raise points= to tighten it.");
  const out = new Array(points);
  const step = (hi - lo) / (points - 1);
  for (let t = 0; t < points; t++) out[t] = lo + t * step;
  return out;
}

/* Each race API declares its OWN keys: one shared union let each accept
   the other's options and ignore them (#186). See checkOpts in core.mjs. */
const FORWARD_OPTS = new Set([
  "V", "D", "F", "W", "base", "points", "returnSlopes", "window", "delta",
  "structure", "qa", "qf",
]);
const INVERSE_OPTS = new Set([
  "V", "D", "F", "W", "base", "points", "structure", "qa", "qf",
  "nIter", "tol", "targetFloor", "returnInfo",
]);

/* The lattice the forward map integrates on, as one function.
 *
 * raceJacobian built its OWN grid -- a plain span window with no
 * adaptive placement and no refinement -- so it differentiated a
 * different lattice than raceProbabilities computed on, and the two
 * stopped agreeing exactly where the lattice is coarse relative to the
 * field. On a four-runner race whose variances span 4005x, at 257
 * points, the analytic jacobian differed from finite differences of its
 * own forward by 1.0e-3 where python -- which shares its grid through
 * `forward_grid` -- was 1.7e-8. Both converge by 1025 points, which is
 * why it went unnoticed (#212).
 */
export function forwardGrid(Mall, sd, st, points, win = "bulk",
                            delta = 1e-12) {
  let x;
  if (win === "bulk") {
    x = bulkWindow(Mall, sd, points, delta, st.fn || null);
  } else {
    let mn = Infinity, mx = -Infinity;
    for (const row of Mall) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const smax = Math.max(...sd);
    x = new Array(points);
    const lo = mn - st.left * smax, hi = mx + st.right * smax;
    for (let t = 0; t < points; t++) x[t] = lo + t * (hi - lo) / (points - 1);
  }
  let dx = x[1] - x[0];
  // extreme-sharpness lattice refinement (matching python/R)
  const smin = Math.min(...sd);
  let vmax = 0;
  for (const row of st.V) vmax = Math.max(vmax, Math.sqrt(row.reduce((a, b) => a + b * b, 0)));
  if (vmax / Math.max(smin, 1e-300) > 25 && dx > 0.5 * smin) {
    const span = x[x.length - 1] - x[0];
    const need = Math.ceil(span / (0.5 * smin)) + 1;
    const pts2 = Math.min(need, 8193);
    if (pts2 > x.length) {
      const x0 = x[0];
      x = new Array(pts2);
      for (let t = 0; t < pts2; t++) x[t] = x0 + t * span / (pts2 - 1);
      dx = x[1] - x[0];
    }
  }
  // A correctly bracketed window can still be unusable: a narrow-centred
  // base (or a polynomial tail) leaves the spacing wider than the base's
  // central scale, and the result is a plausible normalized vector for an
  // under-resolved integral. Say so, as python's _refine_grid (#600).
  const res = baseResolution(st.fn || null);
  if (dx > 0.5 * smin * res && typeof console !== "undefined") {
    const key = `${x.length}|${smin.toPrecision(3)}|${res}`;
    if (!SPACING_WARNED.has(key) && SPACING_WARNED.add(key))
      console.warn(
        `lattice spacing ${dx.toPrecision(3)} exceeds half the smallest ` +
        `performance sd times the base's central scale (${smin.toPrecision(3)} ` +
        `x ${res}): the lattice cannot resolve this base's centre. Raise ` +
        "points=, raise delta=, or declare a span on the base.");
  }
  return { x, dx };
}
const SPACING_WARNED = new Set();

/* The general race's factor node rule for loadings V at variances D --
   gauge-centred V and its (F, W) -- for callers that mix their own
   conditional kernels over the factor law (topk.mjs, #340). */
export function _factorNodeRule(V, D) {
  const n = asLoadings(V, V.length).length;
  const st = setup(new Array(n).fill(0), V, D, null, null, "normal");
  return { V: st.V, F: st.F, W: st.W };
}

/* Independent and Factor grammars ARE the V/D race, so they are priced
   (and inverted) as one, keeping the caller's F/W. The dispatcher
   rebuilt the call from s.V and s.D alone, so with structure:
   Factor(V, D) the forward discarded a caller's factor rule while
   raceJacobian used it -- the structured Jacobian sat 0.20 from finite
   differences of the structured forward (#209). The other grammars have
   no factor rule to take, so an F/W there is refused rather than
   ignored. */
export function collapseStructure(structure, V, D, F, W, where) {
  if (!structure) return { structure: null, V, D };
  if (V != null || D != null)
    throw new Error(
      `${where}: give the covariance once -- structure= or V=/D=, not both`);
  if (structure.kind === "Independent")
    return { structure: null, V: null, D: structure.D };
  if (structure.kind === "Factor")
    return { structure: null, V: structure.V, D: structure.D };
  if (F != null || W != null)
    throw new Error(
      `${where}: F/W are a factor rule for the V/D (Factor) race; the ` +
      `${structure.kind} grammar has its own quadrature (qa=, qf=) and ` +
      "would ignore them");
  return { structure, V, D };
}

export function raceProbabilities(mu, opts = {}) {
  const { V = null, D = null, F = null, W = null, base = "normal",
          points = 257, returnSlopes = false, window: win = "bulk",
          delta = 1e-12, structure = null, qa = 9, qf = 15 } = opts;
  checkOpts(opts, FORWARD_OPTS, "raceProbabilities", OPT_HINTS);
  const c = collapseStructure(structure, V, D, F, W, "raceProbabilities");
  if (c.structure) {
    return dispatchProbabilities(mu, c.structure, { base, points, qa, qf, returnSlopes, window: win, delta });
  }
  const st = setup(mu, c.V, c.D, F, W, base);
  const n = st.mu.length;
  const sd = st.D.map(Math.sqrt);
  const Mall = condMeans(st.mu, st.V, st.F);
  const { x, dx } = forwardGrid(Mall, sd, st, points, win, delta);
  const p = new Array(n).fill(0);
  const slope = new Array(n).fill(0);
  const logS = new Array(n), fArr = new Array(n), fpArr = new Array(n);
  for (let q = 0; q < st.F.length; q++) {
    const Mq = Mall[q], wq = st.W[q];
    const L = new Array(x.length).fill(0);
    for (let i = 0; i < n; i++) {
      const li = new Array(x.length), fi = new Array(x.length), fpi = new Array(x.length);
      for (let t = 0; t < x.length; t++) {
        const z = (x[t] - Mq[i]) / sd[i];
        const [S, f, fp] = st.fn(z);
        li[t] = Math.log(S);
        fi[t] = f / sd[i];
        fpi[t] = fp;
        L[t] += li[t];
      }
      logS[i] = li; fArr[i] = fi; fpArr[i] = fpi;
    }
    for (let i = 0; i < n; i++) {
      let si = 0, sl = 0;
      const li = logS[i], fi = fArr[i], fpi = fpArr[i];
      const sd2 = sd[i] * sd[i];
      for (let t = 0; t < x.length; t++) {
        const e = Math.min(Math.max(L[t] - li[t], -745), 0);
        const rest = Math.exp(e);
        si += fi[t] * rest;
        sl += -fpi[t] / sd2 * rest;
      }
      p[i] += wq * si * dx;
      slope[i] += wq * sl * dx;
    }
  }
  const total = p.reduce((a, b) => a + b, 0);
  const pn = p.map(v => v / total);
  if (returnSlopes) return { p: pn, slopes: slope.map(v => v / total) };
  return pn;
}

/* One exit for every inversion path, as python's _inverse_return: warn
   on non-convergence unless the caller asked for the diagnostics, and
   report the iteration actually reached. Without it a starved solve came
   back silently, and `iterations` was always the requested budget --
   60 for a target the warm start already solved (#354). */
function inverseReturn(mu, converged, maxLogResidual, iterations, floored,
                       tol, returnInfo) {
  if (!converged && !returnInfo && typeof console !== "undefined")
    console.warn(
      `abilitiesFromRace did not converge: max |log residual| ` +
      `${maxLogResidual.toExponential(2)} after ${iterations} iterations ` +
      `(tol ${tol.toExponential(0)}). Pass returnInfo: true for the ` +
      "residual and iteration count instead of this warning.");
  return returnInfo
    ? { mu, converged, maxLogResidual, iterations, floored }
    : mu;
}

/* The target contract, matching python: a zero share has no finite
   inverse, so it RAISES unless the caller floors deliberately, and the
   floored entries are reported. Applied BEFORE any structure dispatch:
   the hierarchical grammars clamped a zero to 1e-300 and returned
   abilities near +/-230 as an inverse, a negative entry was quietly
   projected to zero, and targetFloor/returnInfo were dropped on the way
   through the dispatcher (#387). */
function validatedRaceTarget(pTarget, targetFloor) {
  let target = asFiniteVector(pTarget, "target", "probability");
  const n = target.length;
  let floored = new Array(n).fill(false);
  if (targetFloor != null) {
    if (!(typeof targetFloor === "number" && targetFloor > 0 && Number.isFinite(targetFloor)))
      throw new Error("targetFloor must be positive");
    floored = target.map(v => v < targetFloor);
    target = target.map(v => Math.max(v, targetFloor));
  } else if (target.some(v => v <= 0)) {
    throw new Error(
      "all target probabilities must be positive: a zero share has no " +
      "finite inverse (the supremum is approached as that contrast " +
      "diverges). Pass targetFloor to floor small entries deliberately " +
      "and read the result as a one-sided bound on the floored " +
      "contrasts, or supply a pseudocount upstream.");
  }
  // a target is a law up to a positive factor: when its SUM overflows
  // (entries all finite, e.g. [4e307, 2e307, 1e307, 1e307]) rescale by
  // the max first, as python does since #300. Conditional, so ordinary
  // inputs stay bit-identical (#326).
  let s = target.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(s)) {
    const mx = Math.max(...target);
    target = target.map(v => v / mx);
    s = target.reduce((a, b) => a + b, 0);
  }
  return { target: target.map(v => v / s), floored };
}

export function abilitiesFromRace(pTarget, opts = {}) {
  const { nIter = 60, tol = 1e-8, structure = null, V = null, D = null,
          F = null, W = null, base = "normal", points = 257,
          targetFloor = null, returnInfo = false, qa = 9, qf = 15 } = opts;
  checkOpts(opts, INVERSE_OPTS, "abilitiesFromRace", OPT_HINTS);
  asIterations(nIter, "abilitiesFromRace");
  asTolerance(tol, "abilitiesFromRace");
  const { target, floored } = validatedRaceTarget(pTarget, targetFloor);
  const n = target.length;
  const c = collapseStructure(structure, V, D, F, W, "abilitiesFromRace");
  if (c.structure) {
    // the caller's base, window and delta travel too, so the hierarchical
    // kernels can refuse what they would otherwise ignore (#89)
    const r = dispatchAbilities(target, c.structure,
      { ...opts, points, qa, qf, tol });
    return inverseReturn(r.mu, r.converged, r.maxLogResidual, r.iterations,
                         floored, tol, returnInfo);
  }
  return solveRace(target, floored, { V: c.V, D: c.D, F, W, base, points,
                                      nIter, tol, returnInfo });
}

function solveRace(target, floored, { V, D, F, W, base, points, nIter, tol,
                                      returnInfo }) {
  const n = target.length;
  requireWholeRule(F, W, "abilitiesFromRace");
  const logt = target.map(Math.log);
  const lm = mean(logt);
  // the field's contrast scale (matching python/R): median idiosyncratic
  // variance plus the mean factor variance under the represented nodes
  const Dn = asIdio(D, n);        // the inverse has its own copy (#254)
  const Vn = V != null ? asLoadings(V, n) : Array.from({ length: n }, () => [0]);
  const r = Vn[0].length;
  const Vc = gaugeCenter(Vn);
  let CovF = Array.from({ length: r }, (_, a) => Array.from({ length: r }, (_, b) => (a === b ? 1 : 0)));
  let Fm = new Array(r).fill(0);
  if (V != null && F != null) {
    // the same door the forward pass uses. The inverse consumed F and W
    // raw, so a W shorter than F left `Wq[q]` undefined past its end and
    // every moment below came out NaN -- surfacing, once mu was checked
    // for finiteness, as "mu[0] = NaN" rather than as the bad weights
    // the caller actually passed.
    const Fv = asFactorNodes(F, r, "F");
    const Wq = asWeights(W, Fv.length, "W");
    Fm = Array.from({ length: r }, (_, c) => Fv.reduce((acc, f, q) => acc + Wq[q] * f[c], 0));
    CovF = Array.from({ length: r }, (_, a) => Array.from({ length: r }, (_, b) =>
      Fv.reduce((acc, f, q) => acc + Wq[q] * (f[a] - Fm[a]) * (f[b] - Fm[b]), 0)));
  }
  const sigV = (i, j) => Vc[i].reduce((acc, va, a) => acc + va * CovF[a].reduce((acc2, cab, b) => acc2 + cab * Vc[j][b], 0), 0);
  const med = (arr) => { const z = arr.slice().sort((a, b) => a - b); const h = Math.floor(z.length / 2); return z.length % 2 ? z[h] : 0.5 * (z[h - 1] + z[h]); };
  const scale = Math.sqrt(med(Dn) + mean(Dn.map((_, i) => sigV(i, i))));
  let muStart = null;
  if (n === 2 && base === "normal") {
    // a pair is one Gaussian contrast: closed form (matching python/R),
    // including the rule's weighted MEAN, (v1 - v0) . E[F], which this
    // dropped -- a translated Hermite rule missed by 8.2 points (#374)
    const sdD = Math.sqrt(Math.max(sigV(0, 0) + sigV(1, 1) - 2 * sigV(0, 1) + Dn[0] + Dn[1], 1e-300));
    const shift = Vc[1].reduce((acc, v, c) => acc + (v - Vc[0][c]) * Fm[c], 0);
    // invert the SMALLER share: for [1, 1e-16] the normalized first
    // share rounds to exactly 1 and invNormalRational(1) is NaN, which
    // was certified as converged while [1e-16, 1] was finite (#412)
    const gap = (target[0] <= target[1]
      ? sdD * invNormalRational(target[0])
      : -sdD * invNormalRational(target[1])) - shift;
    const pair = [-0.5 * gap, 0.5 * gap];
    const ok = pair.every(Number.isFinite);
    if (F == null || !ok)
      return inverseReturn(pair, ok, ok ? 0 : Infinity, 0, floored, tol, returnInfo);
    // A caller's rule need not be Gaussian -- a centred two-point law has
    // the right mean and variance and a different pair map (12.2 points)
    // -- and the forward integrates the rule itself, so the closed form
    // is only a START, certified by the forward it claims to invert.
    const ph = raceProbabilities(pair, { V, D, F, W, base, points });
    const r0 = Math.max(...ph.map((v, i) => Math.abs(Math.log(Math.max(v, 1e-300)) - logt[i])));
    if (r0 < tol) return inverseReturn(pair, true, r0, 0, floored, tol, returnInfo);
    muStart = pair;
  }
  let mu = muStart || logt.map(v => -(v - lm) / 2 * scale);
  // damping: a pair, or two runners holding nearly all the mass, two-cycles
  // undamped; the sweeps then adapt to the contraction they observe
  // (matching python's _jacobi_sweeps)
  const top2 = n > 2 ? target.slice().sort((a, b) => b - a).slice(0, 2).reduce((a, b) => a + b, 0) : 1;
  let alpha = (n === 2 || top2 > 0.8) ? 0.7 : 1.0;
  // alphaBase: the Richardson value, persistent (NB: `base` is the density
  // argument). penalty: caution after a sweep that failed to contract, a
  // transient, restored on the next good sweep. One number for both can
  // only ratchet down (matching python, #178).
  let alphaBase = alpha;
  let penalty = 1;
  let prev = null;
  let prevStep = null;
  let iters = 0;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    const { p: praw, slopes: sl } = raceProbabilities(mu, { V, D, F, W, base, points, returnSlopes: true });
    const phat = praw.map(v => Math.max(v, 1e-300));
    let resid = phat.map((v, i) => Math.log(v) - logt[i]);
    let dlogp = sl.map((v, i) => Math.min(v / phat[i], -1e-6));
    let rmax = Math.max(...resid.map(Math.abs));
    let rrms = Math.sqrt(mean(resid.map(v => v * v)));
    if (rmax < tol) break;
    if (prev && rmax >= prev.rmax && rrms >= prev.rrms) {
      if (penalty > 0.1) {
        penalty = Math.max(0.5 * penalty, 0.1);
        ({ mu, resid, dlogp, rmax, rrms } = prev);
        prevStep = null;
      }
    } else if (prev && penalty < 1) {
      penalty = Math.min(1, penalty / 0.75);
    }
    prev = { mu, resid, dlogp, rmax, rrms };
    alpha = alphaBase * penalty;
    // residual-proportional step cap in the field's scale
    let step = resid.map((v, i) => {
      const lim = Math.min(2, 10 * Math.abs(v)) * scale;
      return Math.min(Math.max(alpha * v / dlogp[i], -lim), lim);
    });
    const sm = mean(step);
    step = step.map(v => v - sm);
    let extrapolated = false;
    if (prevStep) {
      const na = Math.sqrt(prevStep.reduce((a, b) => a + b * b, 0));
      const nb = Math.sqrt(step.reduce((a, b) => a + b * b, 0));
      if (na > 0) {
        const dot = step.reduce((a, b, i) => a + b * prevStep[i], 0);
        const rho = dot / (na * na);
        const cosn = nb > 0 ? dot / (na * nb) : 0;
        const ratio = nb / na;
        if (rho < 0) {
          const lam = 1 - (1 - rho) / alpha;
          alphaBase = Math.min(Math.max(2 / (2 - lam), 0.1), 1);
        } else if (cosn > 0.999 && ratio > 0.5 && ratio < 0.999 && rmax > 1e3 * tol) {
          // collinear steps decaying geometrically: sum the tail (Aitken)
          mu = mu.map((m, i) => m - step[i] / (1 - ratio));
          prevStep = null;
          extrapolated = true;
        }
      }
    }
    if (!extrapolated) {
      prevStep = step;
      mu = mu.map((m, i) => m - step[i]);
    }
  }
  // one more forward pass to report the residual actually achieved,
  // rather than the one from before the last step -- and to decide
  // whether to warn when the caller did not ask for the diagnostics
  const pf = raceProbabilities(mu, { V, D, F, W, base, points });
  const resid = pf.map((v, i) => Math.log(Math.max(v, 1e-300)) - logt[i]);
  const maxLogResidual = Math.max(...resid.map(Math.abs));
  return inverseReturn(mu, maxLogResidual < tol, maxLogResidual, iters,
                       floored, tol, returnInfo);
}

// filled in by structures.mjs to avoid a cycle
export let dispatchProbabilities = () => { throw new Error("import structures.mjs first"); };
export let dispatchAbilities = () => { throw new Error("import structures.mjs first"); };
export function _setDispatch(dp, da) { dispatchProbabilities = dp; dispatchAbilities = da; }
