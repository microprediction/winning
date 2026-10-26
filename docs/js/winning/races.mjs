// The general race: min-wins, normal/gumbel bases, winner-bulk lattice,
// adaptive factor quadrature. Port of winning/factor/races.py.
import { TINY, ndtr, logndtr, npdf, hermiteNodes, mean, checkOpts, OPT_HINTS, asLoadings, asIdio, gaugeCenter, firstPrimes, asFactorNodes, asWeights, asFactorLaw, asAbilities, asFiniteVector, asIterations, asTolerance, asCount, aitkenTrial, applyFloor, canonicalBasis, asPoints, kinkLattice } from "./core.mjs";

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
    const S0 = Math.exp(-eu);
    // density and slope from the unfloored survival (#502)
    const f = c * eu * S0;
    return [Math.max(S0, 1e-300), f, c * c * eu * S0 * (1 - eu)];
  },
  logistic(z) {
    const c = Math.PI / Math.sqrt(3);
    const u = c * z;
    // reflected, from e = exp(-|u|) <= 1, as python: f = c S (1 - S)
    // rounded 1 - S to zero once S hit 1, zeroing the left-tail density
    // (a runner 25 sd behind priced 7.3e-19 for 3.0e-18, #467)
    const e = Math.exp(-Math.abs(u));
    const S = u > 0 ? e / (1 + e) : 1 / (1 + e);
    const f = c * e / ((1 + e) * (1 + e));
    return [Math.max(S, 1e-300), f, -c * f * (1 - 2 * S)];
  },
  laplace(z) {
    const b = 1 / Math.sqrt(2);
    const f = Math.exp(-Math.abs(z) / b) / (2 * b);
    const S = z < 0 ? 1 - 0.5 * Math.exp(z / b) : 0.5 * Math.exp(-z / b);
    return [Math.max(S, 1e-300), f, -Math.sign(z) * f / b];
  },
};
// the density is not analytic at z = 0: the inverses take their slopes on
// the split lattice there (kinkLattice, #267); the forward is unchanged
BASES.laplace.kinks = [0];

/* Exact log survivals of the named bases, used by the integrator only
   where S was floored at 1e-300 (#502): the floor guards log(0), and
   summed as mass it plateaued a rare winner at ~1e-300 (normal gap 53:
   1.0e-300 for 1.105e-307; Laplace gap 500: 1.0e-300 for 1.43e-305).
   Results above that range are unchanged bit for bit. */
BASES.normal.logSf = z => logndtr(-z);
BASES.gumbel.logSf = z =>
  -Math.exp(Math.min(z * Math.PI / Math.sqrt(6) - EULER, 30));
BASES.logistic.logSf = z => {
  const u = Math.PI / Math.sqrt(3) * z;
  return -Math.max(u, 0) - Math.log1p(Math.exp(-Math.abs(u)));
};
/* The asymmetric min-Gumbel's lower-tail CDF, F = -expm1(-e^u): the count
   programs (top-k, rank) reflect the symmetric bases' survival for F, and
   for Gumbel fell back to 1 - S, which is 0 once S rounds to one -- a
   representable last place of 2.0e-39 (mu = [-35, 0, 0]) came back as an
   exact 0 at any point count (#710). */
BASES.gumbel.cdf = z =>
  -Math.expm1(-Math.exp(Math.min(z * (Math.PI / Math.sqrt(6)) - EULER, 30)));
BASES.laplace.logSf = z => {
  const b = 1 / Math.sqrt(2);
  return z < 0 ? Math.log1p(-0.5 * Math.exp(z / b)) : Math.log(0.5) - z / b;
};
/* Optional tail attributes of a CALLABLE base, as python's cdf / log_cdf /
   log_sf (#511 follow-up): fn.cdf(z) = 1 - S(z) accurate in the lower
   tail, fn.logCdf(z) its log (used when cdf is absent), fn.logSf(z) the
   log survival where S underflows. The named bases carry logSf and the
   count program reflects the symmetric ones; a wrapper has no identity to
   key that on, so it may carry the tails itself. Each is checked once per
   base on a probe of z in [-6, 6]: a function, finite, monotone, and
   within 1e-6 of the base's own survival. */
const NAMED_BASES = new Set(Object.values(BASES));
const TAILS_CHECKED = new WeakMap();
const TAIL_ATTRS = ["cdf", "logCdf", "logSf"];
export function checkTails(fn) {
  if (fn == null || NAMED_BASES.has(fn)) return;
  const attrs = TAIL_ATTRS.map(a => fn[a]);
  if (attrs.every(a => a === undefined)) return;
  const hit = TAILS_CHECKED.get(fn);
  if (hit && hit.every((a, j) => a === attrs[j])) return;
  const zs = Array.from({ length: 25 }, (_, t) => -6 + 0.5 * t);
  TAIL_ATTRS.forEach((name, j) => {
    const a = attrs[j];
    if (a === undefined) return;
    if (typeof a !== "function")
      throw new Error(`base.${name} must be a function z -> number; got ${typeof a}`);
    const v = zs.map(z => a(z));
    if (!v.every(Number.isFinite))
      throw new Error(`base.${name} must be finite on z in [-6, 6]`);
    let ok = true;
    for (let t = 0; t < v.length; t++) {
      const d = t ? v[t] - v[t - 1] : 0;
      if (name === "logSf") ok = ok && v[t] <= 1e-12 && d <= 1e-12;
      else if (name === "logCdf") ok = ok && v[t] <= 1e-12 && d >= -1e-12;
      else ok = ok && v[t] >= 0 && v[t] <= 1 && d >= -1e-12;
    }
    if (!ok)
      throw new Error(`base.${name} is not a valid ${name} on z in [-6, 6]: it must be ` +
        (name === "logSf" ? "non-increasing" : "non-decreasing") +
        (name === "cdf" ? " and in [0, 1]" : " and <= 0"));
    let gap = 0;
    for (let t = 0; t < v.length; t++) {
      const S = fn(zs[t])[0];
      const g = name === "logSf" ? Math.abs(Math.exp(v[t]) - S)
        : Math.abs((name === "logCdf" ? Math.exp(v[t]) : v[t]) - (1 - S));
      gap = Math.max(gap, g);
    }
    if (!(gap <= 1e-6))
      throw new Error(`base.${name} disagrees with the base's own survival S by ` +
        `${gap.toExponential(2)} on z in [-6, 6]; it must describe the same law ` +
        "(min-wins, mean zero, unit variance)");
  });
  TAILS_CHECKED.set(fn, attrs);
}

/* z -> F(z) from a callable base's cdf or logCdf, or null (the named
   symmetric bases reflect instead; Gumbel carries its own cdf, #710). */
export function lowerCdf(fn) {
  if (fn == null) return null;
  if (NAMED_BASES.has(fn)) return typeof fn.cdf === "function" ? fn.cdf : null;
  checkTails(fn);
  if (typeof fn.cdf === "function") return fn.cdf;
  if (typeof fn.logCdf === "function")
    return z => Math.min(Math.exp(Math.max(fn.logCdf(z), -1e300) || -1e300), 1);
  return null;
}

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
export function factorRule(V, D, F = null, W = null, maxNodes = Infinity) {
  requireWholeRule(F, W);
  const r = V[0].length;
  if (F != null) {
    // the caller's nodes go through the same door as V and D: every node
    // carries exactly the loadings' rank, and there is one weight per
    // node (#290)
    // paired, with null nodes dropped (#416)
    return asFactorLaw(F, W, r, "F", "W");
  }
  // The low-discrepancy rules are built for the CANONICAL loadings
  // C = V Q (null directions dropped, principal axes, cube-sum signs) and
  // handed back in V's own coordinates as Q f'. V f = C f' is the same
  // law, so every caller keeps using V, but the Halton placement now
  // depends on V V' alone: padding two active columns to positions 68,
  // 69 of 80 moved a share by 0.078 through the high-dimensional Halton
  // projections, and a rank-12 column reversal moved one by 0.0053
  // (#130). Wherever python and R would build a Gauss-Hermite tensor in
  // V's own axes, that exact rule is kept (it is symmetric under the
  // permutations and sign flips that matter there, and the ports stay
  // in parity); only a low-discrepancy family goes canonical, with the
  // canonical rank picking it.
  if (ruleFamily(V, D) === "hermite") {
    const raw = canonicalRule(V, D, maxNodes);
    if (raw.hermite) return raw;
  }
  const { C, Q } = canonicalBasis(V);
  const rc = C[0].length;
  if (rc === 0) return { F: [new Array(r).fill(0)], W: [1] };
  const rule = canonicalRule(C, D, maxNodes);
  const Fo = rule.F.map(f => Q.map(row => {
    let a = 0;
    for (let k = 0; k < rc; k++) a += row[k] * f[k];
    return a;
  }));
  return { F: Fo, W: rule.W };
}

function midpointRule(Q) {
  const F = [];
  for (let q = 0; q < Q; q++) F.push([invNormalRational((q + 0.5) / Q)]);
  return { F, W: new Array(Q).fill(1 / Q) };
}

/* The rank-one Gauss-Hermite/midpoint handover was a hard switch at
   ceil(8 sharp) > 80, where the two rules disagree: a 1e-7 change in one
   runner's sd moved a top-2 membership by 31bp (#783). Across (lo, hi]
   the rule is the weighted union (1 - w) GH(8 lo) + w midpoint(8 hi + 1),
   with w a C1 smoothstep in the sharpness, as python (_R1_BLEND_LO/HI,
   which says why the band sits on the Gauss-Hermite side) and R/Julia;
   each order is the one its side uses at the band's edge, so the band
   meets the unchanged rule outside it continuously at both ends. */
const R1_BLEND_LO = 9.0, R1_BLEND_HI = 10.0;
function rank1BlendRule(sharp) {
  const t = (sharp - R1_BLEND_LO) / (R1_BLEND_HI - R1_BLEND_LO);
  const w = t * t * (3.0 - 2.0 * t);
  const g = hermiteNodes(1, 8 * R1_BLEND_LO);
  const m = midpointRule(8 * R1_BLEND_HI + 1);
  return {
    F: g.F.concat(m.F),
    W: g.W.map(x => (1.0 - w) * x).concat(m.W.map(() => w / m.W.length)),
    hermite: true,   // kept in V's own axes, as python builds it
  };
}

function ruleFamily(V, D) {
  const n = V.length, r = V[0].length;
  let sharp = 0;
  for (let i = 0; i < n; i++) {
    const nv = Math.sqrt(V[i].reduce((a, b) => a + b * b, 0));
    sharp = Math.max(sharp, nv / Math.sqrt(Math.max(D[i], 1e-300)));
  }
  sharp *= Math.SQRT2;
  const cap = r === 1 ? 201 : r === 2 ? 41 : r === 3 ? 31 : 15;
  const sharpMax = r === 2 ? 3.75 : r === 3 ? 4.75 : 3.0;
  if (r >= 2 && sharp > sharpMax) return "halton";
  if (r === 1 && sharp > R1_BLEND_LO && sharp <= R1_BLEND_HI) return "hermite";
  if (r === 1 && Math.ceil(8 * sharp) > 80) return "midpoint";
  if (Math.pow(cap, r) > 100000) return "halton";
  return "hermite";
}

function canonicalRule(V, D, maxNodes) {
  const n = V.length, r = V[0].length;
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
  let rule;
  if (r >= 2 && sharp > sharpMax) {
    // escalate the FAMILY, not the order (matching python/R)
    rule = haltonRule(r, 8192);
  } else if (r === 1 && sharp > R1_BLEND_LO && sharp <= R1_BLEND_HI) {
    rule = rank1BlendRule(sharp);                           // (#783)
  } else if (r === 1 && Math.ceil(8 * sharp) > 80) {
    // rank-1 extreme sharpness (matching python/R): equal-weight
    // midpoint-quantile grid scaled with sharpness replaces GH
    rule = midpointRule(Math.min(Math.ceil(8 * sharp), 4001));
  } else if (Math.pow(cap, r) > 100000) {
    // high-rank tensor footgun (matching python/R): Halton fallback
    rule = haltonRule(r, 8192);
  } else {
    rule = { ...hermiteNodes(r, Math.min(Math.max(Math.ceil(8 * sharp), 15), cap)), hermite: true };
  }
  if (rule.W.length > maxNodes) {
    // a node budget (top-k: every node is a full count-program pass) is
    // met with a WHOLE rule of that size, as python's 2^10 scrambled
    // Sobol: the first 1024 nodes of a 3393-node midpoint grid are its
    // lower quarter, and top-1 missed the race by 0.50 (#340)
    rule = r === 1 ? midpointRule(maxNodes) : haltonRule(r, maxNodes);
  }
  return rule;
}

/* Runner-sized reductions as loops: Math.max(...D) spreads its array into
   call arguments, and past an engine's argument limit (~125k on node 24)
   a valid 130,000-runner field threw "Maximum call stack size exceeded"
   before any integration (#729). Same values as the spread, NaN included. */
function maxOf(a) {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) m = Math.max(m, a[i]);
  return m;
}
function minOf(a) {
  let m = Infinity;
  for (let i = 0; i < a.length; i++) m = Math.min(m, a[i]);
  return m;
}

/* mu moved next to the origin when a common offset dwarfs the field
   (python's races._translated_home). The origin is a gauge, but the
   lattice is built in the caller's units: at an offset of 1e15 its
   points collapsed to the float spacing there and every share came back
   NaN (#477). Shifting by a middle entry is exact for such offsets;
   below a million field scales nothing moves. */
export function translatedHome(mu, D) {
  const s = mu.slice().sort((a, b) => a - b);
  const off = s[Math.floor(s.length / 2)];
  const scale = (s[s.length - 1] - s[0]) + Math.sqrt(maxOf(D));
  return Math.abs(off) > 1e6 * scale ? mu.map(v => v - off) : mu;
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
  mu = translatedHome(mu, D);                      // #477
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
  checkTails(fn);
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
  const smax = maxOf(sd), smin = minOf(sd);
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
    const lo0 = bracket(minOf(muLo) - 9 * smax, step0, x => G(x, muLo) <= d, -1);
    const hi0 = bracket(maxOf(muHi) + 9 * smax, step0, x => G(x, muHi) >= 1 - d, +1);
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
  let d = asDelta(delta, "bulkWindow");
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
  if (!fn || fn === BASES.normal)
    [lo, hi, points] = rareWinnerEdges(muLo, muHi, sd, lo, hi, points);
  const out = new Array(points);
  const step = (hi - lo) / (points - 1);
  for (let t = 0; t < points; t++) out[t] = lo + t * step;
  return out;
}

/* Post-check of the bulk window's edges against each runner's OWN win
   integrand -- port of python's races._rare_winner_edges (#97). The bulk
   quantiles bound every share's error by delta absolutely, not
   relatively: at mu = [0, 20, 40] the middle runner wins near x = 10,
   past the right edge, and came back at 0.099 of its 1.04e-45 at any
   point count (#723). For a log-concave (normal) integrand the mass
   beyond an edge is at most g(edge) / |slope| (tangent bound) and the
   mass inside at least a chord integral next to the edge (best of a
   geometric ladder); a runner whose ratio reaches EDGE_REL_TOL, or whose
   integrand still rises at the edge, is truncated. The edge then moves
   out until every runner passes, the point count growing to keep the
   spacing, up to EDGE_POINTS_CAP times the request; past the cap the
   window stops there with a warning. One O(7 n) probe per edge:
   ordinary fields pass it and their lattice is unchanged. */
const EDGE_REL_TOL = 1e-6, EDGE_POINTS_CAP = 4;
const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);
let EDGE_WARNED = false;
function rareWinnerEdges(muLo, muHi, sd, lo, hi, points) {
  const n = sd.length;
  const width0 = hi - lo;
  if (!(width0 > 0) || n < 2) return [lo, hi, points];
  const spacing = width0 / Math.max(points - 1, 1);
  const maxWidth = spacing * (EDGE_POINTS_CAP * points - 1);
  const ws = [], logWs = [];
  for (let k = 0; k < 6; k++) { ws.push(0.5 * width0 * 0.25 ** k); logWs.push(Math.log(ws[k])); }
  const logSd = sd.map(Math.log);
  // log g_i = log f_i + sum_{j != i} log S_j and its slope, at x
  const z = new Float64Array(n), logS = new Float64Array(n), haz = new Float64Array(n);
  const terms = (x, m, logg, slope) => {
    let sumLogS = 0, sumHaz = 0;
    for (let i = 0; i < n; i++) {
      const zi = (x - m[i]) / sd[i], ls = logndtr(-zi);
      const h = Math.exp(-0.5 * zi * zi - LOG_SQRT_2PI - ls) / sd[i];
      z[i] = zi; logS[i] = ls; haz[i] = h;
      sumLogS += ls; sumHaz += h;
    }
    for (let i = 0; i < n; i++) {
      logg[i] = -0.5 * z[i] * z[i] - LOG_SQRT_2PI - logSd[i] + sumLogS - logS[i];
      if (slope) slope[i] = -z[i] / sd[i] - sumHaz + haz[i];
    }
  };
  const gAt = new Float64Array(n), sAt = new Float64Array(n);
  const inner = ws.map(() => new Float64Array(n));
  const bad = (edge, m, sign) => {
    // sign +1: right edge (the integrand must FALL outward), -1: left
    terms(edge, m, gAt, sAt);
    ws.forEach((w, k) => terms(edge - sign * w, m, inner[k], null));
    for (let i = 0; i < n; i++) {
      const g = gAt[i], out = sign * sAt[i];
      if (out >= 0) return true;
      const tail = g - Math.log(Math.max(-out, 1e-300));
      let cmax = -Infinity;
      for (let k = 0; k < ws.length; k++) {
        const gi = inner[k][i], d = Math.abs(gi - g);
        const c = d < 1e-12
          ? g + logWs[k]
          : Math.max(gi, g) + logWs[k] + Math.log(-Math.expm1(-d)) - Math.log(Math.max(d, 1e-300));
        cmax = Math.max(cmax, c);
      }
      // NaN (an unrepresentable field) is not evidence of truncation
      if (tail - cmax >= Math.log(EDGE_REL_TOL)) return true;
    }
    return false;
  };
  let newLo = lo, newHi = hi;
  for (const [sign, m] of [[+1, muHi], [-1, muLo]]) {
    let edge = sign > 0 ? newHi : newLo;
    if (!bad(edge, m, sign)) continue;
    let step = maxOf(sd);
    for (let it = 0; it < 60; it++) {
      const cand = edge + sign * step;
      if (Math.max(newHi, cand) - Math.min(newLo, cand) > maxWidth) {
        edge = sign > 0 ? newLo + maxWidth : newHi - maxWidth;
        if (!EDGE_WARNED && typeof console !== "undefined") {
          EDGE_WARNED = true;
          console.warn(
            "bulk window: a rare winner's integrand extends past the window " +
            `even at ${EDGE_POINTS_CAP}x the point budget, so its share is ` +
            "truncated (relatively; the absolute error stays below delta). " +
            'Raise points= or use window: "span" for relative accuracy on ' +
            "such shares.");
        }
        break;
      }
      edge = cand;
      if (!bad(edge, m, sign)) break;
      step *= 2;
    }
    if (sign > 0) newHi = edge; else newLo = edge;
  }
  if (newLo === lo && newHi === hi) return [lo, hi, points];
  const pts = Math.ceil((newHi - newLo) / spacing) + 1;
  return [newLo, newHi, Math.max(points, Math.min(pts, EDGE_POINTS_CAP * points))];
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
/* The lattice window: "bulk" or "span", nothing else. Every other value
   used to mean span, so window: "bulkk" moved a share by 39 points
   against the bulk default it misspelled (#582). */
export function asWindow(win, where = "raceProbabilities") {
  if (win !== "bulk" && win !== "span")
    throw new Error(`${where}: window must be "bulk" or "span"; got ${JSON.stringify(win)}`);
  return win;
}

/* The bulk window's tail mass: a number in (0, 0.5). delta: -1 hung the
   relaxation loop (a negative times 100 never reaches 1e-4), 1 and 2
   priced a different race, a string was coerced and NaN returned NaN
   (#445, matching python's races._as_delta). */
export function asDelta(delta, where = "raceProbabilities") {
  if (typeof delta !== "number" || !Number.isFinite(delta) ||
      !(delta > 0 && delta < 0.5))
    throw new Error(`${where}: delta must be a number in (0, 0.5); got ${typeof delta === "string" ? JSON.stringify(delta) : String(delta)}`);
  return delta;
}

export function forwardGrid(Mall, sd, st, points, win = "bulk",
                            delta = 1e-12) {
  points = asPoints(points);           // #444
  asWindow(win, "forwardGrid");
  asDelta(delta, "forwardGrid");
  let x;
  if (win === "bulk") {
    x = bulkWindow(Mall, sd, points, delta, st.fn || null);
  } else {
    let mn = Infinity, mx = -Infinity;
    for (const row of Mall) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const smax = maxOf(sd);
    x = new Array(points);
    const lo = mn - st.left * smax, hi = mx + st.right * smax;
    for (let t = 0; t < points; t++) x[t] = lo + t * (hi - lo) / (points - 1);
  }
  let dx = x[1] - x[0];
  // extreme-sharpness lattice refinement (matching python/R)
  const smin = minOf(sd);
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
export function _factorNodeRule(V, D, maxNodes = Infinity) {
  const n = asLoadings(V, V.length).length;
  const st = setup(new Array(n).fill(0), V, D, null, null, "normal");
  if (st.W.length <= maxNodes) return { V: st.V, F: st.F, W: st.W };
  return { V: st.V, ...factorRule(st.V, st.D, null, null, maxNodes) };
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

/* A two-runner normal race is ONE Gaussian contrast, priced in closed
   form as python's race_probabilities (and R) do: the lattice put a
   20-sd pair tail at 0.107x its exact 1.04e-45 (#86). Under a caller's
   nodes F/W the contrast variance is dV' Cov_W(F) dV + D0 + D1, with the
   node mean shifting the contrast; the default rule stands for the
   standard normal, so dV' dV. Each tail directly (ndtr(u), ndtr(-u)):
   full relative precision, and exactly permutation-equivariant. */
/* The pair under a caller's node law, exactly (#742), as python's
   _pair_mixture: given node q runner 0 wins with probability
   Phi((gap - proj_q) / sd), proj_q = (V0 - V1) . F_q, sd = sqrt(D0 + D1);
   the race is the weighted mixture. Each tail summed directly. */
function pairSd(d0, d1) {
  const lo = Math.min(d0, d1), hi = Math.max(d0, d1);
  if (!(hi > 0)) return 1e-150;  // an exactly deterministic contrast
  return Math.sqrt(hi) * Math.sqrt(lo / hi + 1);
}
function pairMixture(gap, proj, w, d0, d1, returnSlopes) {
  const sd = pairSd(d0, d1);
  let p0 = 0, p1 = 0, dens = 0;
  for (let q = 0; q < proj.length; q++) {
    const u = (gap - proj[q]) / sd;
    p0 += w[q] * ndtr(u);
    p1 += w[q] * ndtr(-u);
    if (returnSlopes) dens += w[q] * npdf(u);
  }
  const t = p0 + p1;
  const p = [p0 / t, p1 / t];
  if (!returnSlopes) return p;
  dens /= sd;
  return { p, slopes: [-dens, -dens] };
}
/* The gap at which pairMixture prices `target`, as python's
   _pair_mixture_gap: p0 increases in the gap, so Newton on
   log p_small - log target_small from the closed form's gap g0,
   safeguarded by the bracket the iterates establish (a step leaving it
   bisects; while one side is open the step is h, doubling). Direct sums
   unless p_small underflows, then the log domain. null if no root. */
function pairMixtureGap(target, proj, w, d0, d1, g0, h) {
  const sd = pairSd(d0, d1);
  const small = target[0] <= target[1] ? 0 : 1;
  const sign = small === 0 ? 1 : -1;
  const logt = Math.log(target[small]);
  const lse = terms => {
    let m = -Infinity;
    for (const v of terms) if (v > m) m = v;
    if (!Number.isFinite(m)) return m;
    let acc = 0;
    for (const v of terms) acc += Math.exp(v - m);
    return m + Math.log(acc);
  };
  if (!(Number.isFinite(g0) && Number.isFinite(h) && h > 0)) return null;
  let g = g0, lo = -Infinity, hi = Infinity;
  for (let it = 0; it < 200; it++) {
    const u = proj.map(s => sign * (g - s) / sd);
    let p = 0;
    for (let q = 0; q < u.length; q++) p += w[q] * ndtr(u[q]);
    let logp, ratio;
    if (p > 1e-250) {
      logp = Math.log(p);
      let d = 0;
      for (let q = 0; q < u.length; q++) d += w[q] * Math.exp(-0.5 * u[q] * u[q]);
      ratio = d / Math.sqrt(2 * Math.PI) / p;
    } else {
      const lw = w.map(v => (v > 0 ? Math.log(v) : -Infinity));
      logp = lse(u.map((x, q) => lw[q] + logndtr(x)));
      ratio = Math.exp(lse(u.map((x, q) => lw[q] - 0.5 * x * x))
                       - 0.5 * Math.log(2 * Math.PI) - logp);
    }
    const f = sign * (logp - logt);           // increasing in g
    if (f === 0) return g;
    if (!Number.isFinite(f)) return null;
    if (f > 0) hi = g; else lo = g;
    let gn = ratio > 0 ? g - f * sd / ratio : NaN;
    // quadratic convergence: from 1e-8 the step lands at rounding
    if (Math.abs(f) < 1e-8 && lo < gn && gn < hi) return gn;
    if (!(lo < gn && gn < hi)) {
      if (Number.isFinite(lo) && Number.isFinite(hi)) gn = 0.5 * (lo + hi);
      else { gn = f > 0 ? g - h : g + h; h *= 2; }
    }
    if (Math.abs(gn - g) <= 2 * Number.EPSILON * Math.max(Math.abs(g), sd)) return gn;
    g = gn;
  }
  return null;
}

function pairClosedForm(st, nodesGiven, returnSlopes) {
  const { mu, V, D, F, W } = st;
  const dV = V[0].map((v, j) => v - V[1][j]);
  let sdD, shift = 0;
  if (nodesGiven) {
    // relative weights, through the max (#415)
    let wmax = 0;
    for (const w of W) wmax = Math.max(wmax, w);
    const wr = W.map(w => w / wmax), ws = wr.reduce((a, b) => a + b, 0);
    // V -> aV, F -> F/a is the same race: at a = 1e308 the opposite rows
    // overflowed dV before the tiny nodes met it (NaN race, #86). Then
    // each row is projected first and the projections subtracted.
    const proj = dV.every(Number.isFinite)
      ? F.map(f => f.reduce((acc, x, j) => acc + x * dV[j], 0))
      : F.map(f => f.reduce((acc, x, j) => acc + x * V[0][j], 0) -
                   f.reduce((acc, x, j) => acc + x * V[1][j], 0));
    // the caller's F, W ARE the factor law, which need not be normal:
    // the pair is the exact mixture over the nodes, as the lattice
    // prices it (#742: a centred two-point law priced 0.5 for an exact
    // 0.75 when only its mean and covariance were matched)
    return pairMixture(mu[1] - mu[0], proj, wr.map(w => w / ws), D[0], D[1],
                       returnSlopes);
  } else {
    // in units of the largest term: no overflow, no absolute floor (#100)
    let big = Math.max(Math.sqrt(D[0]), Math.sqrt(D[1]));
    for (const v of dV) big = Math.max(big, Math.abs(v));
    let v = 0;
    if (big > 0) {
      for (const x of dV) v += (x / big) ** 2;
      v += D[0] / big / big + D[1] / big / big;
    }
    sdD = big * Math.sqrt(v);
    // the sd itself may exceed the float range (a rank-two 7e307
    // contrast priced [0.5, 0.5] for Phi(1/sqrt 2)): u in big's unit, as
    // python's big_unit (#86)
    if (!Number.isFinite(sdD)) {
      const u = (mu[1] / big - mu[0] / big) / Math.sqrt(v);
      const p = [ndtr(u), ndtr(-u)];
      if (!returnSlopes) return p;
      const dens = npdf(u) / big / Math.sqrt(v);
      return { p, slopes: [-dens, -dens] };
    }
  }
  if (!(sdD > 0)) sdD = 1e-150;     // an exactly deterministic contrast
  // the node mean raises each runner's location by V_i . Fm, so the
  // contrast mu1 - mu0 loses dV . Fm: adding it priced the shifted
  // runner as the favourite (#86)
  const u = (mu[1] - mu[0] - shift) / sdD;
  const p = [ndtr(u), ndtr(-u)];
  if (!returnSlopes) return p;
  const dens = npdf(u) / sdD;       // dp_i/dmu_i, equal and negative
  return { p, slopes: [-dens, -dens] };
}

export function raceProbabilities(mu, opts = {}) {
  const { V = null, D = null, F = null, W = null, base = "normal",
          points = 257, returnSlopes = false, window: win = "bulk",
          delta = 1e-12, structure = null, qa = 9, qf = 15 } = opts;
  checkOpts(opts, FORWARD_OPTS, "raceProbabilities", OPT_HINTS);
  // one of two names: every value but "bulk" -- a typo, null, a number --
  // silently meant span and moved a share by 39 points (#582). undefined
  // is omission, as for every option.
  if (win !== "bulk" && win !== "span")
    throw new Error(`raceProbabilities: window must be "bulk" or "span"; got ${typeof win === "string" ? JSON.stringify(win) : String(win)}`);
  asDelta(delta);                                         // (#445)
  const c = collapseStructure(structure, V, D, F, W, "raceProbabilities");
  if (c.structure) {
    return dispatchProbabilities(mu, c.structure, { base, points, qa, qf, returnSlopes, window: win, delta });
  }
  const st = setup(mu, c.V, c.D, F, W, base);
  if (base === "normal" && st.mu.length === 2) {
    asPoints(points);
    return pairClosedForm(st, F != null, returnSlopes);
  }
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
        // bounded below: an overflowing log (mu = 1e308) gave -Infinity
        // and -Inf - -Inf a NaN race; f is exactly 0 wherever it binds
        li[t] = S <= 1e-300 && st.fn.logSf
          ? Math.max(st.fn.logSf(z), -1e300) || -1e300 : Math.log(S);
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

/* Own-slopes of the race on the kink-split lattice (core.kinkLattice),
   for a base that declares kinks: the uniform rectangle sum of the slope
   integrand -f'(z), which jumps at each runner's kink, is first order --
   a laplace own-slope came out 6x too large and abilitiesFromRace missed
   an exact self-generated target by 0.79 (#267). Normalised by the split
   total, as python's race_probabilities(..., kink_split=True). */
function splitRaceSlopes(mu, { V, D, F, W, base, points }) {
  const st = setup(mu, V, D, F, W, base);
  const n = st.mu.length;
  const sd = st.D.map(Math.sqrt);
  const Mall = condMeans(st.mu, st.V, st.F);
  const { x } = forwardGrid(Mall, sd, st, points, "bulk", 1e-12);
  const p = new Array(n).fill(0), slope = new Array(n).fill(0);
  for (let q = 0; q < st.F.length; q++) {
    const Mq = Mall[q], wq = st.W[q];
    const { xs, ws } = kinkLattice(x, Mq, sd, st.fn.kinks);
    const T = xs.length;
    const L = new Array(T).fill(0);
    const logS = new Array(n), fArr = new Array(n), fpArr = new Array(n);
    for (let i = 0; i < n; i++) {
      const li = new Array(T), fi = new Array(T), fpi = new Array(T);
      for (let t = 0; t < T; t++) {
        const [S, f, fp] = st.fn((xs[t] - Mq[i]) / sd[i]);
        li[t] = Math.log(S); fi[t] = f / sd[i]; fpi[t] = fp;
        L[t] += li[t];
      }
      logS[i] = li; fArr[i] = fi; fpArr[i] = fpi;
    }
    for (let i = 0; i < n; i++) {
      const sd2 = sd[i] * sd[i];
      let si = 0, sl = 0;
      for (let t = 0; t < T; t++) {
        const rest = Math.exp(Math.min(Math.max(L[t] - logS[i][t], -745), 0));
        si += fArr[i][t] * rest * ws[t];
        sl += -fpArr[i][t] / sd2 * rest * ws[t];
      }
      p[i] += wq * si;
      slope[i] += wq * sl;
    }
  }
  const total = p.reduce((a, b) => a + b, 0);
  return slope.map(v => v / total);
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
    // a probability floor, in the final normalised scale (#592)
    ({ target, floored } = applyFloor(target, 1, targetFloor));
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
    const mx = maxOf(target);
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
    // the contrast sd in units of the largest term: D = [9e307, 9e307]
    // summed to Infinity and returned [-Infinity, Infinity], and the
    // absolute 1e-300 variance floor changed the model below contrast sd
    // 1e-150 (#100)
    const shared = sigV(0, 0) + sigV(1, 1) - 2 * sigV(0, 1);
    const big = Math.max(Math.abs(shared), Dn[0], Dn[1]);
    const sdD = big > 0
      ? Math.sqrt(big) * Math.sqrt(Math.max(shared / big + (Math.min(Dn[0], Dn[1]) / big + Math.max(Dn[0], Dn[1]) / big), 0))
      : 0;
    const shift = Vc[1].reduce((acc, v, c) => acc + (v - Vc[0][c]) * Fm[c], 0);
    // invert the SMALLER share: for [1, 1e-16] the normalized first
    // share rounds to exactly 1 and invNormalRational(1) is NaN, which
    // was certified as converged while [1e-16, 1] was finite (#412)
    const gap = (target[0] <= target[1]
      ? sdD * invNormalRational(target[0])
      : -sdD * invNormalRational(target[1])) - shift;
    // the forward prices the exact mixture over the caller's nodes
    // (#742): invert THAT map by a monotone 1-D solve from this start
    if (V != null && F != null) {
      const Fv = asFactorNodes(F, r, "F");
      const Wq = asWeights(W, Fv.length, "W");
      const proj = Fv.map(fq => fq.reduce((acc, x, c) => acc + x * (Vc[0][c] - Vc[1][c]), 0));
      const gs = pairMixtureGap(target, proj, Wq, Dn[0], Dn[1], gap, sdD);
      if (gs !== null && Number.isFinite(gs)) {
        // certified by the very map the forward evaluates
        const sol = [-0.5 * gs, 0.5 * gs];
        const ph = pairMixture(gs, proj, Wq, Dn[0], Dn[1], false);
        // no 1e-300 floor: a subnormal target is a real share the solve
        // reprices exactly (#91); an exact 0 gives Infinity
        const r1 = Math.max(...ph.map((v, i) => Math.abs(Math.log(v) - logt[i])));
        if (r1 < tol) return inverseReturn(sol, true, r1, 0, floored, tol, returnInfo);
      }
    }
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
  let cached = null;  // forward of an accepted Aitken trial (#583)
  const fnBase = typeof base === "function" ? base : BASES[base];
  const kinked = !!(fnBase && Array.isArray(fnBase.kinks) && fnBase.kinks.length);
  const sweepForward = m => {
    let { p: praw, slopes: sl } = raceProbabilities(m, { V, D, F, W, base, points, returnSlopes: true });
    // a kinked base: the slope from the split lattice, the value from the
    // map being inverted (#267)
    if (kinked) sl = splitRaceSlopes(m, { V, D, F, W, base, points });
    const phat = praw.map(v => Math.max(v, 1e-300));
    return { resid: phat.map((v, i) => Math.log(v) - logt[i]),
             // in the field's units: an absolute -1e-6 took over past
             // scale 1e6, and D = 1e20 stopped 0.062 short (#100)
             dlogp: sl.map((v, i) => Math.min(v / phat[i], -1e-6 / scale)) };
  };
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    let { resid, dlogp } = cached || sweepForward(mu);
    cached = null;
    let rmax = maxOf(resid.map(Math.abs));
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
          // collinear steps decaying geometrically: sum the tail (Aitken),
          // as a repriced, capped trial
          const trial = aitkenTrial(mu, step, ratio, scale, sweepForward,
                                    rmax, ["resid", "dlogp"]);
          if (trial) {
            mu = trial.mu;
            cached = trial.fwd;
            prevStep = null;
            extrapolated = true;
          }
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
  let maxLogResidual = maxOf(resid.map(Math.abs));
  // exhausted after a step no sweep priced: keep the better of it and the
  // preceding priced iterate, as python's _jacobi_sweeps and core.mjs
  // jacobiSweeps do -- one more sweep returned a residual of 1.66 where
  // the budget before it had 0.71 (#646)
  if (!(maxLogResidual < tol) && prev && !(maxLogResidual < prev.rmax)) {
    mu = prev.mu;
    maxLogResidual = prev.rmax;
  }
  return inverseReturn(mu, maxLogResidual < tol, maxLogResidual, iters,
                       floored, tol, returnInfo);
}

// filled in by structures.mjs to avoid a cycle
export let dispatchProbabilities = () => { throw new Error("import structures.mjs first"); };
export let dispatchAbilities = () => { throw new Error("import structures.mjs first"); };
export function _setDispatch(dp, da) { dispatchProbabilities = dp; dispatchAbilities = da; }
