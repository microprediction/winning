// The classic state-price lattice calibration -- port of
// winning/lattice.py + lattice_calibration.py, dead heats included.
import { ndtr, npdf, interpClamped, mean, checkOpts, OPT_HINTS, asFiniteVector, rescaledTarget,
         asIterations, isVector } from "./core.mjs";

/* Each exported call declares its own option keys; see checkOpts in
   core.mjs for why an options object needs this at all. */
const SOLVE_FOR_IMPLIED_OFFSETS_OPTS = new Set(["offsetSamples", "guess", "nIter"]);
// inline-destructured signatures bypassed checkOpts, so {sacle: 0.3} and
// {unti: 0.1} priced the defaults without a word (#561)
const SKEW_NORMAL_DENSITY_OPTS = new Set(["loc", "scale", "a"]);
const DIVIDEND_IMPLIED_ABILITY_OPTS = new Set(["nanValue", "unit"]);

/* The interpolation table is built BY MAPPING over offsetSamples, and
   read back with interpClamped, whose xp must ascend. Descending
   offsets are what make the prices ascend, so the direction of this
   argument is a precondition of the algorithm and not a presentation
   choice. Hand it an ascending array and the binary search runs on a
   descending table: every lookup falls off an end and clamps, so three
   distinct prices came back as the lattice boundary [24, -25, -25] and
   repriced 0.53 away from the target -- with no error at all (#274).

   python raises ValueError("Not descending") and R stops with
   "offset_samples must be descending" for the same input; only the
   browser guessed. Ties are legal in all three: a repeated offset is a
   flat step in the table, not an ascent.

   An empty array and a non-finite entry are the same class of thing --
   an offsetSamples the algorithm cannot use -- and the browser was
   silently wrong on both, returning [undefined, ...] and [-5, -5, -5]
   respectively. They are refused here rather than downstream, so the
   message names the argument the caller actually passed. */
export function asDescendingOffsets(offsets, where = "offsetSamples") {
  if (!Array.isArray(offsets) && !ArrayBuffer.isView(offsets))
    throw new Error(`${where} must be an array of offsets; got ${typeof offsets}`);
  const n = offsets.length;
  if (n === 0)
    throw new Error(`${where} is empty; there is nothing to interpolate against`);
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(offsets[i]))
      throw new Error(`${where}[${i}] = ${offsets[i]} is not a finite offset`);
  }
  for (let i = 0; i + 1 < n; i++) {
    if (offsets[i + 1] > offsets[i])
      throw new Error(
        `${where} must be descending: ${where}[${i + 1}] = ${offsets[i + 1]} ` +
        `is above ${where}[${i}] = ${offsets[i]}. The interpolation table is ` +
        `built in this order and read as an ascending price curve, so an ` +
        `ascending array silently returns lattice-boundary abilities.`);
  }
  return Array.from(offsets, Number);
}


/* The one boundary for a classic atom vector (#339), as python's
   as_classic_density: a finite, nonnegative, odd-length (2L+1, L >= 3)
   array with positive total, returned normalised so raw histogram
   counts are the same law as their frequencies. Entries down to -1e-12
   of the total are round-off and are clipped. Mass 10 used to price at
   -61, a negative atom as a finite tie, an even length as a silently
   truncated lattice, and L = 1 gave the inverse an empty table and NaN
   abilities. L <= 2 is refused because lowHigh pins every offset to
   [-L+2, L-2], a single point. */
export const MIN_CLASSIC_L = 3;
export function asClassicDensity(density, where = "density") {
  if (!Array.isArray(density) && !ArrayBuffer.isView(density))
    throw new Error(`${where} must be an array of lattice atoms; got ${typeof density}`);
  const n = density.length;
  if (n === 0) throw new Error(`${where} must be a nonempty vector of lattice atoms`);
  if (n % 2 !== 1)
    throw new Error(`${where} must have odd length 2L+1 on the symmetric lattice; got length ${n}`);
  if ((n - 1) / 2 < MIN_CLASSIC_L)
    throw new Error(`${where} has L = ${(n - 1) / 2}; the classic lattice needs L >= ` +
      `${MIN_CLASSIC_L} (length >= ${2 * MIN_CLASSIC_L + 1}) to represent distinct offsets`);
  let total = 0, lo = Infinity;
  for (let i = 0; i < n; i++) {
    const v = Number(density[i]);
    if (!Number.isFinite(v)) throw new Error(`${where}[${i}] = ${density[i]} is a non-finite atom`);
    total += v; lo = Math.min(lo, v);
  }
  // finite atoms can overflow only in the SUM: 1e308 counts per atom
  // normalised to all zeros (#484). Rescale by the max first, only then.
  let scale = 1;
  if (!Number.isFinite(total)) {
    scale = 0;
    for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(Number(density[i])));
    total = 0; lo /= scale;
    for (let i = 0; i < n; i++) total += Number(density[i]) / scale;
  }
  if (!(total > 0)) throw new Error(`${where} has no positive mass`);
  if (lo < -1e-12 * total) throw new Error(`${where} has a negative atom (${lo * scale})`);
  return Array.from(density, v => Math.max(Number(v) / scale, 0) / total);
}

/* Target state prices carry only relative mass (#377): the inverse read
   p and c*p off its table as different ordinates, so a 10% overround
   moved relative abilities by 0.9 lattice units and a 10x book came back
   as an all-tie race. Finite, nonnegative, positive total; normalised. */
export function asClassicPrices(prices, where = "prices") {
  if (!Array.isArray(prices) && !ArrayBuffer.isView(prices))
    throw new Error(`${where} must be an array of prices; got ${typeof prices}`);
  if (prices.length === 0) throw new Error(`${where} must be a nonempty vector`);
  let total = 0;
  for (let i = 0; i < prices.length; i++) {
    const v = Number(prices[i]);
    if (!Number.isFinite(v)) throw new Error(`${where}[${i}] = ${prices[i]} is a non-finite entry`);
    if (v < 0) throw new Error(`${where}[${i}] = ${v} is a negative entry`);
    total += v;
  }
  if (!(total > 0)) throw new Error(`${where} has no positive mass`);
  if (!Number.isFinite(total)) {
    // finite entries, overflowing sum: through the max (#484)
    let mx = 0;
    for (let i = 0; i < prices.length; i++) mx = Math.max(mx, Number(prices[i]));
    return rescaledTarget(Array.from(prices, v => Number(v) / mx));
  }
  return Array.from(prices, v => Number(v) / total);
}

export function pdfToCdf(f) {
  const c = new Array(f.length);
  let s = 0;
  for (let i = 0; i < f.length; i++) { s += f[i]; c[i] = s; }
  return c;
}
export function cdfToPdf(c) {
  const f = new Array(c.length);
  let prev = 0;
  for (let i = 0; i < c.length; i++) { f[i] = c[i] - prev; prev = c[i]; }
  return f;
}
export function impliedL(density) { return (density.length - 1) >> 1; }

function integerShift(cdf, k) {
  const m = cdf.length;
  k = Math.max(-(m - 1), Math.min(m - 1, k));
  if (k < 0) {
    const a = -k;
    const out = cdf.slice(a);
    const last = cdf[m - 1];
    for (let i = 0; i < a; i++) out.push(last);
    return out;
  }
  if (k === 0) return cdf.slice();
  const out = new Array(k).fill(0);
  for (let i = 0; i < m - k; i++) out.push(cdf[i]);
  // mass shifted past the top atom lumps on it, mirroring the negative
  // branch; truncating left a sub-probability CDF (#373)
  out[m - 1] = cdf[m - 1];
  return out;
}
function lowHigh(offset, L) {
  if (offset > -L + 2 && offset < L - 2) {
    const lo = Math.floor(offset), up = Math.ceil(offset);
    const r = offset - lo;
    return [[lo, 1 - r], [up, r]];
  }
  if (offset >= L - 2) return [[L - 2, 1], [L - 1, 0]];
  return [[-L + 1, 0], [-L + 2, 1]];
}
function shiftedCdf(cdf, offset, L) {
  const [[a, ac], [b, bc]] = lowHigh(offset, L);
  const sa = integerShift(cdf, a), sb = integerShift(cdf, b);
  return sa.map((v, i) => ac * v + bc * sb[i]);
}
/* Exact dead-heat pricing (#418, #362, #348, #373); mirrors
   winning/classic/lattice.py. A runner's equal-split winner claim is

     P_i = sum_t f_i(t) int_0^1 prod_{j != i} (S_j(t) + u f_j(t)) du,

   since 1/(1+M) = int_0^1 u^M du. The integrand has degree n-1 in u, so
   n//2 + 1 Gauss-Legendre nodes are exact. The field is
   G_q(t) = prod_j (S_j + u_q f_j); a runner's opponents are
   G_q / (S_i + u_q f_i), capped at one and nonincreasing in t. The old
   fold kept only the minimum's CDF and MEAN multiplicity, so a
   compact-support pair priced [0.917, 0.050] (#418), twenty iid
   three-atom runners summed to 0.887 (#362), and a fractional offset was
   paid as two integer runners neither of which was in the field (#348).
   The lattice is padded by L-1 atoms per side so no shift loses mass. */
function gaussLegendre01(n) {
  const x = new Array(n), w = new Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));
    let dp = 1;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) {
        const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1; p1 = p2;
      }
      dp = n * (z * p1 - p0) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x[n - 1 - i] = 0.5 * (z + 1);
    w[n - 1 - i] = 1 / ((1 - z * z) * dp * dp);
  }
  return { nodes: x, weights: w };
}
const exactNodes = n => Math.trunc(n / 2) + 1;
function paddedBaseCdf(density) {
  const L = impliedL(density);
  const pad = Math.max(L - 1, 0);
  const d = new Array(pad).fill(0).concat(Array.from(density, Number), new Array(pad).fill(0));
  return pdfToCdf(d);
}
function exactField(cdfs, nodes) {
  const m = cdfs[0].length;
  const G = nodes.map(() => new Array(m).fill(1));
  for (const c of cdfs) {
    const f = cdfToPdf(c);
    for (let q = 0; q < nodes.length; q++) {
      const row = G[q], u = nodes[q];
      for (let t = 0; t < m; t++) row[t] *= Math.max(1 - c[t], 0) + u * f[t];
    }
  }
  return G;
}
function exactPayoff(cdf, G, gl) {
  const f = cdfToPdf(cdf);
  let total = 0;
  for (let q = 0; q < gl.nodes.length; q++) {
    const row = G[q], u = gl.nodes[q];
    let run = 1, acc = 0;
    for (let t = 0; t < cdf.length; t++) {
      const den = Math.max(1 - cdf[t], 0) + u * f[t];
      if (den > 0) run = Math.min(run, Math.min(row[t] / den, 1));
      acc += run * f[t];
    }
    total += gl.weights[q] * acc;
  }
  return total;
}
/* Node doubling for big fields, as python's Q_START: exactness needs
   n//2 + 1 nodes, O(n^2 L) for a big field, but a smooth lattice is
   numerically low-degree. Start at 8 nodes, double (capped at exact)
   until two rules agree to 1e-14. The inverse table, a preconditioner,
   uses at most 16. */
const EXACT_Q_START = 8, EXACT_TABLE_NODES = 16, EXACT_TOL = 1e-14;
function exactStatePrices(cdfs) {
  const exact = exactNodes(cdfs.length);
  let q = Math.min(EXACT_Q_START, exact), prev = null;
  for (;;) {
    const gl = gaussLegendre01(q);
    const G = exactField(cdfs, gl.nodes);
    const p = cdfs.map(c => exactPayoff(c, G, gl));
    if (q >= exact || (prev && p.every((x, i) => Math.abs(x - prev[i]) <= EXACT_TOL))) return p;
    prev = p;
    q = Math.min(2 * q, exact);
  }
}
function exactImplicitPrices(base, fieldCdfs, offsets, L) {
  const gl = gaussLegendre01(Math.min(exactNodes(fieldCdfs.length), EXACT_TABLE_NODES));
  const G = exactField(fieldCdfs, gl.nodes);
  return offsets.map(k => exactPayoff(shiftedCdf(base, k, L), G, gl));
}

/* Contestant offsets (and an inverse's warm start) are finite numbers.
   lowHigh has two tested branches and an unconditional fallback to the
   most FAVOURABLE boundary, so NaN, undefined and "bad" all fell
   through to it: one invalid entry in a tied pair priced 0.9999978 to
   win. null and true were coerced to 0 and 1 (#438). A Float64Array is
   accepted and copied to a plain array, since mapping a typed array to
   CDF arrays coerced each CDF to NaN (#329). */
function asOffsets(x, where) {
  return asFiniteVector(x, where, "offset");
}

export function statePricesFromOffsets(density, offsets) {
  density = asClassicDensity(density);
  offsets = asOffsets(offsets, "offsets");
  const L = impliedL(density);
  const base = paddedBaseCdf(density);
  return exactStatePrices(Array.from(offsets, o => shiftedCdf(base, o, L)));
}

export function solveForImpliedOffsets(prices, density, opts = {}) {
  checkOpts(opts, SOLVE_FOR_IMPLIED_OFFSETS_OPTS, "solveForImpliedOffsets", OPT_HINTS);
  density = asClassicDensity(density);
  prices = asClassicPrices(prices);
  const L = impliedL(density);
  let { offsetSamples = null, guess = null, nIter = 3 } = opts;
  // a loop bound: 0, negative and NaN returned the target PROBABILITIES
  // as if they were offsets (27 points off on reprice), a fraction ran
  // Math.ceil of itself, and Infinity never returned (#401)
  asIterations(nIter, "solveForImpliedOffsets");
  prices = asFiniteVector(prices, "prices", "price");
  if (offsetSamples === null || offsetSamples === undefined) {
    offsetSamples = [];
    for (let k = Math.trunc(L / 2) - 1; k >= -Math.trunc(L / 2); k--) offsetSamples.push(k);
  } else {
    offsetSamples = asDescendingOffsets(offsetSamples, "offsetSamples");
  }
  // One starting offset per target price. The default was trunc(L/3)
  // offsets -- lattice width as contestant count, a five-runner first
  // field for a two-runner target at L=15 -- and any length was taken
  // (#369); dividendImpliedAbility always passed zeros.
  if (guess === null || guess === undefined) {
    guess = new Array(prices.length).fill(0);
  } else {
    guess = asOffsets(guess, "guess");        // finite numbers (#438)
    if (guess.length !== prices.length)
      throw new Error(`guess must have one starting offset per price: got ` +
        `${guess.length} for ${prices.length} prices`);
  }
  // Defect correction a <- a + T^{-1}(p) - T^{-1}(P(a)) against the
  // exact engine, so the fixed point is the exact forward map; reading
  // p straight off the table converges to the table's own bias.
  const base = paddedBaseCdf(density);
  let implied = Array.from(guess, Number);
  let cdfs = implied.map(o => shiftedCdf(base, o, L));
  for (let it = 0; it < nIter; it++) {
    const table = exactImplicitPrices(base, cdfs, offsetSamples, L);
    const current = exactStatePrices(cdfs);
    implied = implied.map((a, i) => a + interpClamped(prices[i], table, offsetSamples)
      - interpClamped(current[i], table, offsetSamples));
    cdfs = implied.map(o => shiftedCdf(base, o, L));
  }
  return implied;
}

export function skewNormalDensity(L, unit, opts = {}) {
  checkOpts(opts, SKEW_NORMAL_DENSITY_OPTS, "skewNormalDensity", OPT_HINTS);
  const { loc = 0, scale = 1.0, a = 2.0 } = opts;
  const n = 2 * L + 1;
  const density = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = unit * (i - L);
    const t = (x - loc) / scale;
    density[i] = 2 / scale * npdf(t) * ndtr(a * t);
  }
  let s = density.reduce((x, y) => x + y, 0);
  let d = density.map(v => v / s);
  // center, then apply the reference's density-vector fractional shift
  let m = 0;
  for (let i = 0; i < n; i++) m += d[i] * (i - L);
  d = cdfToPdf(shiftedCdf(pdfToCdf(d), -m, L));
  return shiftedCdf(d, loc / unit, L);       // reference quirk: cdf-machinery on the density
}

/* Port of StatePricer.prices_from_dividends. Only a MISSING quote --
   null, undefined or NaN -- becomes nanValue. A non-positive dividend,
   and -Infinity with it, is worth nothing and prices at 0; +Infinity
   prices at 1/Infinity = 0 on its own.

   The browser used `Number.isFinite(x) ? x : nanValue`, which conflates
   every non-finite value with a missing quote, and it divided by the
   dividend unconditionally. So an infinite-dividend entrant got
   1/2000 of the book instead of nothing, a dividend of 0 produced
   Infinity and then NaN after normalising, and a negative dividend came
   back as a NEGATIVE probability (#242). Normalising only when the
   total is positive is python's rule too: an all-infinite book is all
   zeros, not 0/0. */
export function pricesFromDividends(dividends, nanValue = 2000) {
  // a plain array whatever came in: a typed input used to come back
  // typed, and the next call in dividendImpliedAbility could not map it
  // (#329)
  if (!isVector(dividends))
    throw new Error(`dividends must be an array; got ${typeof dividends}`);
  const p = Array.from(dividends).map(x => {
    const v = (x === null || x === undefined || Number.isNaN(x))
      ? nanValue : Number(x);
    return v <= 0 ? 0 : 1 / v;
  });
  const s = p.reduce((a, b) => a + b, 0);
  return s > 0 ? p.map(v => v / s) : p;
}

export function dividendImpliedAbility(dividends, density, opts = {}) {
  checkOpts(opts, DIVIDEND_IMPLIED_ABILITY_OPTS, "dividendImpliedAbility", OPT_HINTS);
  const { nanValue = 2000, unit = 1.0 } = opts;
  const p = pricesFromDividends(dividends, nanValue);
  const guess = new Array(p.length).fill(0);
  return solveForImpliedOffsets(p, density, { guess }).map(v => v * unit);
}
