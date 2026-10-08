// Top-k memberships q_i = P(X_i among the k smallest), their mu- and
// sigma-Jacobians, the rank marginals, and the inversions: locations
// from one membership curve, (loc, scale) jointly from two. Port of
// winning/factor/topk.py -- the cavity count distribution with
// stable-direction deconvolution; see the python module docstring for
// the derivations and the two-branch refusal of exact-rank targets.
import { TINY, hermite1, solve, checkOpts, OPT_HINTS, asLoadings, asIdio,
         asAbilities, asFiniteVector, asIterations, asTolerance,
         jacobiSweeps, floorableTarget } from "./core.mjs";
import { BASES, _factorNodeRule } from "./races.mjs";

/* Each exported call declares its own option keys; see checkOpts in
   core.mjs for why an options object needs this at all. */

/* python refuses V here rather than not implementing it, and the reason is
   worth carrying: the browser used to solve the independent model and
   report converged: true (#200). */
const LOC_SCALE_HINTS = {
  V: "loc/scale calibration with fixed factor loadings is " +
     "under-identified: without the joint rescaling gauge, two curves " +
     "carry 2n - 2 numbers against 2n - 1 unknowns and a flat direction " +
     "survives. Use abilitiesFromTopk({V}) at fixed scales. Python raises " +
     "NotImplementedError here for the same reason.",
};
const TOP_K_PROBABILITIES_OPTS = new Set(["V", "D", "base", "points", "qa"]);
const BOTTOM_K_PROBABILITIES_OPTS = new Set(["V", "D", "base", "points", "qa"]);
const TOP_K_JACOBIANS_OPTS = new Set(["V", "D", "base", "points", "qa"]);
const RANK_PROBABILITIES_OPTS = new Set(["V", "D", "base", "points", "qa"]);
const ABILITIES_FROM_TOPK_OPTS = new Set(["V", "D", "base", "points", "qa", "nIter", "tol", "targetFloor", "returnInfo"]);
const LOC_SCALE_FROM_TOPK_PAIR_OPTS = new Set(["D0", "base", "points", "nIter", "tol", "ridge", "mu0", "returnInfo"]);
const LOC_SCALE_FROM_WIN_AND_SECOND_OPTS = new Set(["D0", "base", "points", "nIter", "tol", "ridge", "mu0", "returnInfo"]);
const ABILITIES_FROM_RANK_MARGINAL_OPTS = new Set(["D", "base", "points", "nIter", "tol", "mu0", "returnInfo"]);


function baseFn(base) {
  const fn = typeof base === "function" ? base : BASES[base];
  if (typeof fn !== "function")
    throw new Error(`unknown base ${JSON.stringify(base)}; known: ${Object.keys(BASES).join(", ")}, or a function`);
  return fn;
}

const clip01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);

/* searched in STANDARDISED units (centred, over the widest sd) and mapped
   back: memberships are invariant under mu -> a + c mu, sd -> c sd, but an
   absolute 1e-12 floor on the widest sd gave a field written in units of
   1e-18 a window 2e5 times its width (#370, as python/R). */
function countWindow(mu, sd, k, fn, delta = 1e-12, padSds = 2.0) {
  const n = mu.length;
  const m0 = mu.reduce((a, b) => a + b, 0) / n;
  const c = Math.max(...sd);
  if (!(Number.isFinite(c) && c > 0)) throw new Error("top-k window needs positive scales");
  const [lo, hi] = countWindowStd(mu.map(v => (v - m0) / c), sd.map(v => v / c),
                                  k, fn, delta, padSds);
  return [m0 + c * lo, m0 + c * hi];
}

function countWindowStd(mu, sd, k, fn, delta, padSds) {
  const n = mu.length;
  const smax = Math.max(...sd);
  const meanCount = x => {
    let t = 0;
    for (let i = 0; i < n; i++) t += 1 - fn((x - mu[i]) / sd[i])[0];
    return t;
  };
  let lo = Math.min(...mu) - 9 * smax;
  let step = 9 * smax;
  for (let it = 0; it < 60; it++) {
    if (meanCount(lo) <= delta) break;
    lo -= step; step *= 2;
  }
  const targetHi = Math.min(
    k + 2 * Math.log(1 / delta) + Math.sqrt(2 * (k + 1) * Math.log(1 / delta)),
    n - 1e-4);
  let hi = Math.max(...mu) + 9 * smax;
  step = 9 * smax;
  for (let it = 0; it < 60; it++) {
    if (meanCount(hi) >= targetHi) break;
    hi += step; step *= 2;
  }
  let a = lo, b = hi;
  for (let it = 0; it < 70; it++) {
    const m = 0.5 * (a + b);
    if (meanCount(m) < delta) a = m; else b = m;
  }
  const xlo = a;
  a = xlo; b = hi;
  for (let it = 0; it < 70; it++) {
    const m = 0.5 * (a + b);
    if (meanCount(m) < targetHi) a = m; else b = m;
  }
  return [xlo - padSds * smax, b + padSds * smax];
}

function baseGrid(fn, x, mu, sd) {
  const L = x.length, n = mu.length;
  const S = [], f = [], fp = [], z = [];
  for (let t = 0; t < L; t++) {
    const Sr = new Array(n), fr = new Array(n), fpr = new Array(n),
          zr = new Array(n);
    for (let i = 0; i < n; i++) {
      const zz = (x[t] - mu[i]) / sd[i];
      const [s, ff, ffp] = fn(zz);
      Sr[i] = s; fr[i] = ff; fpr[i] = ffp; zr[i] = zz;
    }
    S.push(Sr); f.push(fr); fp.push(fpr); z.push(zr);
  }
  return { S, f, fp, z };
}

function countDistribution(F) {
  const L = F.length, n = F[0].length;
  const C = new Array(L);
  for (let t = 0; t < L; t++) {
    const row = new Array(n + 1).fill(0);
    row[0] = 1;
    for (let j = 0; j < n; j++) {
      const q = F[t][j];
      for (let m = Math.min(j + 1, n); m >= 1; m--)
        row[m] = row[m] * (1 - q) + row[m - 1] * q;
      row[0] *= 1 - q;
    }
    C[t] = row;
  }
  return C;
}

function leaveOneOutCdf(C, F, k) {
  const L = F.length, n = F[0].length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(L);
    for (let t = 0; t < L; t++) {
      const Fi = F[t][i], Si = 1 - Fi;
      if (Si >= Fi) {
        const s = Math.max(Si, TINY);
        let Q = clip01(C[t][0] / s), acc = Q;
        for (let m = 1; m < k; m++) {
          Q = clip01((C[t][m] - Fi * Q) / s);
          acc += Q;
        }
        row[t] = clip01(acc);
      } else {
        const fi = Math.max(Fi, TINY);
        let Qb = clip01(C[t][n] / fi), acc = Qb;
        for (let m = n - 2; m >= k; m--) {
          Qb = clip01((C[t][m + 1] - Si * Qb) / fi);
          acc += Qb;
        }
        row[t] = clip01(1 - acc);
      }
    }
    out[i] = row;
  }
  return out;
}

function looPmf(C, F, i) {
  // full leave-one-out pmf Q[t][m], m = 0..n-1, stable direction per t
  const L = F.length, n = F[0].length;
  const Q = new Array(L);
  for (let t = 0; t < L; t++) {
    const Fi = F[t][i], Si = 1 - Fi;
    const row = new Array(n);
    if (Si >= Fi) {
      const s = Math.max(Si, TINY);
      row[0] = clip01(C[t][0] / s);
      for (let m = 1; m < n; m++)
        row[m] = clip01((C[t][m] - Fi * row[m - 1]) / s);
    } else {
      const fi = Math.max(Fi, TINY);
      row[n - 1] = clip01(C[t][n] / fi);
      for (let m = n - 2; m >= 0; m--)
        row[m] = clip01((C[t][m + 1] - Si * row[m + 1]) / fi);
    }
    Q[t] = row;
  }
  return Q;
}

function pairPmfAt(Qi, F, i, k) {
  // P(N_{-ij} = k-1) for every j != i at every lattice point
  const L = F.length, n = F[0].length;
  const out = new Array(n);
  for (let j = 0; j < n; j++) {
    const row = new Array(L).fill(0);
    if (j !== i) {
      for (let t = 0; t < L; t++) {
        const Fj = F[t][j], Sj = 1 - Fj;
        if (Sj >= Fj) {
          const s = Math.max(Sj, TINY);
          let Q = clip01(Qi[t][0] / s);
          for (let m = 1; m < k; m++) Q = clip01((Qi[t][m] - Fj * Q) / s);
          row[t] = Q;
        } else {
          const fj = Math.max(Fj, TINY);
          let Qb = clip01(Qi[t][n - 1] / fj);
          for (let m = n - 3; m >= k - 1; m--)
            Qb = clip01((Qi[t][m + 1] - Sj * Qb) / fj);
          row[t] = Qb;
        }
      }
    }
    out[j] = row;
  }
  return out;
}

function resolvedPoints(lo, hi, sd, points) {
  // about two points per NARROWEST sd, capped at 8193. The window is set
  // by the widest runner and the grid by `points`, so a heterogeneous
  // field leaves the narrowest density between samples, and the mass
  // check cannot see it: that check is one scalar (memberships sum to k)
  // and runner-level errors of opposite sign cancel in it (#224).
  const smin = Math.max(Math.min(...sd), 1e-300);
  // a Number is already a double, so this cannot overflow the way R and
  // Julia did (#228); !isFinite still guards a zero-width window
  const need = Math.ceil((hi - lo) / (0.5 * smin)) + 1;
  if ((!Number.isFinite(need) || need > 8193) && typeof console !== "undefined")
    console.warn(
      `top-k lattice cannot resolve the narrowest runner even at 8193 ` +
      `points (min sd ${smin.toExponential(1)} over a window of ` +
      `${(hi - lo).toPrecision(3)}); memberships may carry percent-level ` +
      "error the mass check cannot see.");
  return Math.max(points, dyadicPoints(need));
}

/* The adaptive count rounded UP to a dyadic lattice 2^m + 1 (capped at
   8193), so it is piecewise constant over a factor-of-two band of the
   narrowest scale instead of stepping by one point at every ceil. A
   count that moved 611 -> 610 under a 1e-5 relative change of one sd
   jumped q by 1.9e-6, so a central difference of the public map read
   -0.307 against the continuum Jsigma -0.0029, and the loc/scale
   inverse, consuming that Jacobian while its trials crossed the
   boundary, stopped at logit residual 3e-4 on an exact target (#515).
   Costs at most twice the points, only when the adaptive count binds. */
export function dyadicPoints(need) {
  if (!(need > 2)) return 2;
  if (!(need < 8193)) return 8193;
  return Math.min(2 ** Math.ceil(Math.log2(need - 1)) + 1, 8193);
}

function topkGrid(mu, sd, k, fn, points, delta = 1e-12) {
  const [lo, hi] = countWindow(mu, sd, k, fn, delta);
  points = resolvedPoints(lo, hi, sd, points);
  const x = new Array(points);
  const dx = (hi - lo) / (points - 1);
  for (let t = 0; t < points; t++) x[t] = lo + t * dx;
  const g = baseGrid(fn, x, mu, sd);
  const F = g.S.map(row => row.map(s => clip01(1 - s)));
  return { x, dx, F, ...g };
}

/* P(i among the k LARGEST) directly: the top-k integral with the count of
   runners ABOVE x. 1 - topK(n - k) cancelled -- 0 for a 7.7e-24 last
   place at 513 points and residue 3.6e-15 at 8193 (#365). */
function bottomkIndependent(mu, sd, k, fn, points) {
  const n = mu.length;
  const refl = z => { const b = fn(-z); return [1 - b[0], b[1], -b[2]]; };
  const [lr, hr] = countWindow(mu.map(v => -v), sd, k, refl);
  const lo = -hr, hi = -lr;
  points = resolvedPoints(lo, hi, sd, points);
  const x = new Array(points);
  const dx = (hi - lo) / (points - 1);
  for (let t = 0; t < points; t++) x[t] = lo + t * dx;
  const g = baseGrid(fn, x, mu, sd);
  const G = g.S.map(row => row.map(clip01));
  const C = countDistribution(G);
  const cdf = leaveOneOutCdf(C, G, k);
  const q = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s0 = 0;
    for (let t = 0; t < points; t++) s0 += (g.f[t][i] / sd[i]) * cdf[i][t];
    q[i] = s0 * dx;
  }
  return q;
}

function topkWithSlopes(mu, sd, k, fn, points) {
  const n = mu.length;
  const { dx, F, f, fp } = topkGrid(mu, sd, k, fn, points);
  const C = countDistribution(F);
  const cdf = leaveOneOutCdf(C, F, k);
  const q = new Array(n).fill(0), slopes = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s0 = 0, s1 = 0;
    for (let t = 0; t < F.length; t++) {
      s0 += (f[t][i] / sd[i]) * cdf[i][t];
      s1 -= (fp[t][i] / (sd[i] * sd[i])) * cdf[i][t];
    }
    q[i] = s0 * dx;
    slopes[i] = s1 * dx;
  }
  return { q, slopes };
}

function checkedTopk(raw, k, kind, massTol = 5e-3) {
  const t = raw.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(t) || Math.abs(t - k) > massTol * k)
    throw new Error(
      `${kind} captured total membership ${t.toFixed(4)} where exactly ` +
      `${k} slots exist: the window or the deconvolution missed part ` +
      `of the field. Raise points=, or report this field.`);
  // the clip is not mass-neutral: re-check what is returned (#99)
  const q = raw.map(v => v * (k / t));
  if (Math.max(...q) - 1 > massTol)
    throw new Error(`${kind} produced a membership of ${Math.max(...q).toFixed(4)} > 1: ` +
                    "the lattice did not resolve this field. Raise points=.");
  const out = q.map(clip01);
  const t2 = out.reduce((a, b) => a + b, 0);
  if (Math.abs(t2 - k) > massTol * k)
    throw new Error(`${kind} memberships total ${t2.toFixed(4)} after clipping, ` +
                    `where exactly ${k} slots exist`);
  return out;
}

function factorNodes(V, n, qa, D = null) {
  const Vm = asLoadings(V, n).map(row => row.slice());
  const r = Vm[0].length;
  if (r > 2)
    throw new Error("topKProbabilities is implemented for factor rank <= 2");
  if (qa === undefined && r > 0) {
    // qa omitted (the default): the general race's sharpness rule, as
    // python and R. A fixed qa = 15 Gauss-Hermite rule missed the race by
    // 10.7 points on a sharp rank-one field and moved a top-2 membership
    // 1.3 points under a covariance-preserving rotation (#340). Past the
    // race's 8192-node escalation the first 1024 Halton points are used:
    // every node here is a full count-program pass.
    const rule = _factorNodeRule(Vm, D == null ? new Array(n).fill(1) : D);
    let F = rule.F, W = rule.W;
    if (W.length > 1024) { F = F.slice(0, 1024); W = new Array(1024).fill(1 / 1024); }
    return { Vm: rule.V, nodes: F, w: W };
  }
  if (qa === undefined) qa = 15;   // an explicit value, null included, goes to hermite1's check
  for (let c = 0; c < r; c++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += Vm[i][c];
    m /= n;
    for (let i = 0; i < n; i++) Vm[i][c] -= m;
  }
  const h = hermite1(qa);
  let nodes, w;
  if (r === 0) {
    // the EMPTY PRODUCT: rank 1 was special-cased and every other rank
    // fell into the rank-2 tensor, so an (n, 0) matrix -- the documented
    // spelling of "no factors" -- became RangeError: Invalid array
    // length from a negative-length tensor (#309)
    nodes = [[]]; w = [1];
  } else if (r === 1) {
    nodes = h.nodes.map(v => [v]);
    w = h.weights.slice();
  } else {
    nodes = []; w = [];
    for (const a of h.nodes) for (const b of h.nodes) nodes.push([a, b]);
    for (const u of h.weights) for (const v of h.weights) w.push(u * v);
    const s = w.reduce((a, b) => a + b, 0);
    w = w.map(v => v / s);
  }
  return { Vm, nodes, w };
}

/* The depth of a top-k curve is a COUNT, so it is an integer.

   Every guard truncated first -- `Math.trunc(k)` here, `int(k)` in
   python, `as.integer(k)` in R -- and then range-checked the truncated
   value, so a fractional depth passed and was silently floored:
   topKProbabilities(mu, 1.5) returned the top-1 curve and 2.5 the
   top-2 one, with no warning and a mass of 1 or 2 rather than the 1.5
   or 2.5 asked for. k=0, k=n and k>n were all refused; only the
   non-integer slipped through, which is the one case the message
   "k must be in [1, n-1]" reads as permitting. */
function asDepth(k, n, where = "k") {
  if (!Number.isFinite(k))
    throw new Error(`${where} must be a whole number of places; got ${k}`);
  if (k !== Math.trunc(k))
    throw new Error(
      `${where} must be a whole number of places; got ${k}. A top-k ` +
      `curve counts finishers, so there is no top-${k}.`);
  if (!(k >= 1 && k <= n - 1))
    throw new Error(`${where} must be in [1, n-1]; got k=${k}, n=${n}`);
  return k;
}

export function topKProbabilities(mu, k, opts = {}) {
  checkOpts(opts, TOP_K_PROBABILITIES_OPTS, "topKProbabilities", OPT_HINTS);
  const { V = null, D = null, base = "normal", points = 513, qa } = opts;
  mu = asAbilities(mu);        // finite and nonempty, before any lattice (#440)
  const n = mu.length;
  k = asDepth(k, n);
  const sd = asIdio(D, n).map(Math.sqrt);
  const fn = baseFn(base);
  if (!V) return checkedTopk(topkWithSlopes(mu, sd, k, fn, points).q,
                             k, "top-k race");
  const { Vm, nodes, w } = factorNodes(V, n, qa, D);
  const raw = new Array(n).fill(0);
  for (let q = 0; q < nodes.length; q++) {
    const shifted = mu.map((m, i) => {
      let s = m;
      for (let c = 0; c < nodes[q].length; c++) s += Vm[i][c] * nodes[q][c];
      return s;
    });
    const node = topkWithSlopes(shifted, sd, k, fn, points).q;
    for (let i = 0; i < n; i++) raw[i] += w[q] * node[i];
  }
  return checkedTopk(raw, k, "top-k race");
}

export function bottomKProbabilities(mu, k, opts = {}) {
  checkOpts(opts, BOTTOM_K_PROBABILITIES_OPTS, "bottomKProbabilities", OPT_HINTS);
  const { V = null, D = null, base = "normal", points = 513, qa } = opts;
  mu = asAbilities(mu);
  const n = mu.length;
  k = asDepth(k, n);
  const sd = asIdio(D, n).map(Math.sqrt);
  const fn = typeof base === "function" ? base : BASES[base];
  if (!V) return checkedTopk(bottomkIndependent(mu, sd, k, fn, points), k, "bottom-k race");
  const { Vm, nodes, w } = factorNodes(V, n, qa, D);
  const raw = new Array(n).fill(0);
  for (let q = 0; q < nodes.length; q++) {
    const shifted = mu.map((m, i) => {
      let s = m;
      for (let c = 0; c < nodes[q].length; c++) s += Vm[i][c] * nodes[q][c];
      return s;
    });
    const node = bottomkIndependent(shifted, sd, k, fn, points);
    for (let i = 0; i < n; i++) raw[i] += w[q] * node[i];
  }
  return checkedTopk(raw, k, "bottom-k race");
}

export function topKJacobians(mu, k, opts = {}) {
  checkOpts(opts, TOP_K_JACOBIANS_OPTS, "topKJacobians", OPT_HINTS);
  const { D = null, base = "normal", points = 513, V = null,
          qa } = opts;
  mu = asAbilities(mu);        // an Infinity used to return all-NaN Jacobians (#440)
  const n = mu.length;
  k = asDepth(k, n);
  if (V) {
    // exact node mixture: the factor shift commutes with d/dmu, d/dsigma
    const { Vm, nodes, w } = factorNodes(V, n, qa, D);
    const Jmu = [], Jsigma = [];
    for (let i = 0; i < n; i++) {
      Jmu.push(new Array(n).fill(0));
      Jsigma.push(new Array(n).fill(0));
    }
    for (let q = 0; q < nodes.length; q++) {
      const shifted = mu.map((m, i) => {
        let s = m;
        for (let c = 0; c < nodes[q].length; c++) s += Vm[i][c] * nodes[q][c];
        return s;
      });
      const node = topKJacobians(shifted, k, { D, base, points });
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          Jmu[i][j] += w[q] * node.Jmu[i][j];
          Jsigma[i][j] += w[q] * node.Jsigma[i][j];
        }
    }
    return { Jmu, Jsigma };
  }
  const Dv = asIdio(D, n);
  const sd = Dv.map(Math.sqrt);
  const fn = baseFn(base);
  const { dx, F, f, fp, z } = topkGrid(mu, sd, k, fn, points);
  const L = F.length;
  const dens = [];
  for (let t = 0; t < L; t++) dens.push(f[t].map((v, i) => v / sd[i]));
  const C = countDistribution(F);
  const Jm = [], Js = [];
  for (let i = 0; i < n; i++) {
    const Qi = looPmf(C, F, i);
    const pair = pairPmfAt(Qi, F, i, k);
    const rowMu = new Array(n).fill(0), rowSd = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let sm = 0, ss = 0;
      for (let t = 0; t < L; t++) {
        const kern = pair[j][t] * dens[t][i];
        sm += kern * dens[t][j];
        ss += kern * z[t][j] * dens[t][j];
      }
      rowMu[j] = sm * dx;
      rowSd[j] = ss * dx;
    }
    rowMu[i] = -rowMu.reduce((a, b) => a + b, 0);
    let own = 0;
    for (let t = 0; t < L; t++) {
      let cdfI = 0;
      for (let m = 0; m < k; m++) cdfI += Qi[t][m];
      own += (-(z[t][i] * fp[t][i] + f[t][i]) / Dv[i]) * cdfI;
    }
    rowSd[i] = own * dx;
    Jm.push(rowMu); Js.push(rowSd);
  }
  return { Jmu: Jm, Jsigma: Js };
}

export function rankProbabilities(mu, opts = {}) {
  checkOpts(opts, RANK_PROBABILITIES_OPTS, "rankProbabilities", OPT_HINTS);
  const { V = null, D = null, base = "normal", points = 513, qa } = opts;
  mu = asAbilities(mu);        // an empty field threw RangeError (#440)
  const n = mu.length;
  const sd = asIdio(D, n).map(Math.sqrt);
  const fn = baseFn(base);

  // one factor node: the cavity count pmf against each runner's own
  // density, as in python's rank_probabilities.one_node
  const oneNode = (m) => {
    const { dx, F, f } = topkGrid(m, sd, n - 1, fn, points);
    const C = countDistribution(F);
    const out = [];
    for (let i = 0; i < n; i++) {
      const Qi = looPmf(C, F, i);
      const row = new Array(n).fill(0);
      for (let t = 0; t < F.length; t++) {
        const d = f[t][i] / sd[i];
        for (let mm = 0; mm < n; mm++) row[mm] += Qi[t][mm] * d;
      }
      out.push(row.map(v => v * dx));
    }
    return out;
  };

  // Loadings used to be destructured away and the independent rank matrix
  // returned -- plausible, doubly stochastic, and for the wrong model
  // (#199). The mixture is the same one topKProbabilities takes.
  let P;
  if (!V) {
    P = oneNode(mu);
  } else {
    const { Vm, nodes, w } = factorNodes(V, n, qa, D);
    P = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let q = 0; q < nodes.length; q++) {
      const shifted = mu.map((m, i) => {
        let acc = m;
        for (let c = 0; c < nodes[q].length; c++) acc += Vm[i][c] * nodes[q][c];
        return acc;
      });
      const node = oneNode(shifted);
      for (let i = 0; i < n; i++)
        for (let mm = 0; mm < n; mm++) P[i][mm] += w[q] * node[i][mm];
    }
  }
  // Check what is RETURNED. Row normalisation makes every row exact and
  // MOVES the columns, so a matrix that passed the raw check could fail
  // the stated identity afterwards with nothing looking (#203). The
  // columns are not forced: alternating scaling would make both exact by
  // hiding the under-resolution that caused it, and the same field at
  // 2001 points has a column error of 3.5e-9.
  // BOTH checks: independent, and mistaken for alternatives once. Row
  // normalisation can ERASE a gross raw defect and leave the result just
  // inside tolerance (#221). The raw check sees the quadrature, the post
  // check sees what the caller gets.
  const defects = (M) => {
    const rows = M.map(r => r.reduce((a, b) => a + b, 0));
    const cols = new Array(n).fill(0);
    for (const r of M) for (let m = 0; m < n; m++) cols[m] += r[m];
    return [Math.max(...rows.map(v => Math.abs(v - 1))),
            Math.max(...cols.map(v => Math.abs(v - 1)))];
  };
  const reject = (where, re, ce) => {
    throw new Error(
      `rank marginals defective ${where}: row-sum error ` +
      `${re.toExponential(2)}, column-sum error ${ce.toExponential(2)}. ` +
      "Raise points=.");
  };
  const finite = (M) => M.every(r => r.every(Number.isFinite));
  let [reRaw, ceRaw] = defects(P);
  if (!finite(P) || reRaw > 5e-3 || ceRaw > 5e-3)
    reject("before normalisation", reRaw, ceRaw);
  const raw = P.map(r => r.reduce((a, b) => a + b, 0));
  P = P.map((r, i) => r.map(v => clip01(v / Math.max(raw[i], 1e-300))));
  const [reOut, ceOut] = defects(P);
  if (!finite(P) || reOut > 5e-3 || ceOut > 5e-3)
    reject("in the RETURNED matrix after row normalisation", reOut, ceOut);
  return P;
}

function validatedTarget(q, k, n, targetFloor) {
  let target = asFiniteVector(q, "target", "membership");
  if (target.length !== n)
    throw new Error(`target has ${target.length} entries for ${n} runners`);
  // finite BEFORE flooring: NaN passed `v <= 0` and surfaced later as a
  // RangeError from the lattice sizing (#110)
  const bad = target.findIndex(v => !Number.isFinite(v));
  if (bad >= 0) throw new Error(`target[${bad}] = ${target[bad]} is not finite`);
  let floored = new Array(n).fill(false);
  if (targetFloor != null) {
    if (!(typeof targetFloor === "number" && targetFloor > 0 && Number.isFinite(targetFloor)))
      throw new Error("targetFloor must be positive");
    target = floorableTarget(target, k);   // a membership floor (#592)
    floored = target.map(v => v < targetFloor);
    target = target.map(v => Math.max(v, targetFloor));
  } else if (target.some(v => v <= 0)) {
    throw new Error(
      "all target memberships must be positive: a zero top-k probability " +
      "has no finite inverse. Pass targetFloor to floor small entries " +
      "deliberately.");
  }
  const s = target.reduce((a, b) => a + b, 0);
  target = target.map(v => v * (k / s));
  if (target.some(v => v >= 1))
    throw new Error(
      "after renormalizing to k slots, a target membership is >= 1: " +
      "certain membership has no finite inverse.");
  return { target, floored };
}

export function abilitiesFromTopk(q, k, opts = {}) {
  checkOpts(opts, ABILITIES_FROM_TOPK_OPTS, "abilitiesFromTopk", OPT_HINTS);
  const { V = null, D = null, base = "normal", points = 513, qa,
          nIter = 80, tol = 1e-8, targetFloor = null,
          returnInfo = false } = opts;
  // a finite positive budget: Infinity on a two-cycling field never
  // returned, and 0/negative/NaN returned the warm start (#413)
  asIterations(nIter, "abilitiesFromTopk");
  asTolerance(tol, "abilitiesFromTopk");
  if (!(Array.isArray(q) || ArrayBuffer.isView(q)))
    throw new Error("abilitiesFromTopk: target must be an array of memberships");
  const n = q.length;
  k = asDepth(k, n);
  const { target, floored } = validatedTarget(q, k, n, targetFloor);
  const sd = asIdio(D, n).map(Math.sqrt);
  const fn = baseFn(base);
  const fac = V ? factorNodes(V, n, qa, D) : null;

  const logitT = target.map(v => Math.log(v) - Math.log1p(-v));
  const logT = target.map(Math.log);
  const mLog = logT.reduce((a, b) => a + b, 0) / n;
  // in the field's own units, as python/R (#100)
  const dsrt = sd.map(v => v * v).sort((a, b) => a - b);
  const medD = n % 2 ? dsrt[(n - 1) / 2] : 0.5 * (dsrt[n / 2 - 1] + dsrt[n / 2]);
  let sv = 0;
  if (fac) { for (const row of fac.Vm) sv += row.reduce((a, b) => a + b * b, 0); sv /= n; }
  const scale = Math.sqrt(medD + sv);
  const mu0 = logT.map(v => -(v - mLog) / 2 * scale);
  // Damping, as python's abilities_from_topk after #151: a pair is
  // bipartite and two-cycles undamped, and so does ANY field at k = 1
  // whose top two runners hold nearly all the mass. The sweeps then
  // adapt the damping to the contraction they observe (#408).
  const top2 = (k === 1 && n > 2)
    ? target.slice().sort((a, b) => b - a).slice(0, 2).reduce((a, b) => a + b, 0)
    : 1;
  const alpha = (n === 2 || (k === 1 && top2 > 0.8)) ? 0.7 : 1.0;
  const forward = mu => {
    let qraw, sl;
    if (!fac) {
      ({ q: qraw, slopes: sl } = topkWithSlopes(mu, sd, k, fn, points));
      sl = sl.slice();
    } else {
      qraw = new Array(n).fill(0);
      sl = new Array(n).fill(0);
      for (let j = 0; j < fac.nodes.length; j++) {
        const shifted = mu.map((m, i) => {
          let s = m;
          for (let c = 0; c < fac.nodes[j].length; c++)
            s += fac.Vm[i][c] * fac.nodes[j][c];
          return s;
        });
        const node = topkWithSlopes(shifted, sd, k, fn, points);
        for (let i = 0; i < n; i++) {
          qraw[i] += fac.w[j] * node.q[i];
          sl[i] += fac.w[j] * node.slopes[i];
        }
      }
    }
    const qhat = checkedTopk(qraw, k, "top-k inversion");
    const resid = qhat.map((v, i) =>
      Math.log(Math.max(v, 1e-300)) - Math.log(Math.max(1 - v, 1e-300))
      - logitT[i]);
    const dres = sl.map((v, i) =>
      Math.min(v / Math.max(qhat[i] * (1 - qhat[i]), 1e-300), -1e-6 / scale));
    return { resid, dres };
  };
  const out = jacobiSweeps(mu0, forward, scale, alpha, nIter, tol);
  const mu = out.mu, residMax = out.residMax, iters = out.iterations;
  const converged = out.converged;
  if (!converged && !returnInfo)
    console.warn(`abilitiesFromTopk did not converge: max |logit residual| ` +
                 `${residMax.toExponential(2)} after ${iters} iterations`);
  if (returnInfo)
    return { mu, info: { converged, maxLogitResidual: residMax,
                         iterations: iters, floored } };
  return mu;
}

export function locScaleFromTopkPair(q1, k1, q2, k2, opts = {}) {
  checkOpts(opts, LOC_SCALE_FROM_TOPK_PAIR_OPTS, "locScaleFromTopkPair", { ...OPT_HINTS, ...LOC_SCALE_HINTS });
  const { D0 = null, base = "normal", points = 513, nIter = 60,
          tol = 1e-8, ridge = 0.0, mu0 = null,
          returnInfo = false } = opts;
  asIterations(nIter, "locScaleFromTopkPair");
  asTolerance(tol, "locScaleFromTopkPair");
  if (!(Array.isArray(q1) || ArrayBuffer.isView(q1)))
    throw new Error("locScaleFromTopkPair: q1 must be an array of memberships");
  const n = q1.length;
  k1 = asDepth(k1, n, "k1");
  k2 = asDepth(k2, n, "k2");
  if (k1 === k2)
    throw new Error("k1 == k2 gives one curve twice: scale is unidentified");
  const t1 = validatedTarget(q1, k1, n, null).target;
  const t2 = validatedTarget(q2, k2, n, null).target;
  const lt1 = t1.map(v => Math.log(v) - Math.log1p(-v));
  const lt2 = t2.map(v => Math.log(v) - Math.log1p(-v));

  // D0 through asIdio: a scalar threw `D0.map is not a function`, and the
  // FALSY invalid values 0 and NaN were read as "omitted" and the solve
  // certified converged after ignoring them (#254)
  let sd = D0 != null ? asIdio(D0, n, "D0").map(Math.sqrt) : new Array(n).fill(1);
  if (typeof ridge !== "number" || !Number.isFinite(ridge) || ridge < 0)
    throw new Error(`locScaleFromTopkPair: ridge must be a finite non-negative number; got ${ridge}`);
  let mu;
  if (mu0 != null) {
    const m00 = asAbilities(mu0, "mu0");
    if (m00.length !== n) throw new Error(`mu0 has ${m00.length} entries for ${n} runners`);
    const m0 = m00.reduce((a, b) => a + b, 0) / n;
    // the return gauge applied to the start too: an exact warm start came
    // back in physical units (#360)
    const c0 = Math.exp(sd.reduce((a, b) => a + Math.log(b), 0) / n);
    mu = m00.map(v => (v - m0) / c0);
    sd = sd.map(v => v / c0);
  } else {
    const [ka, ta] = k1 < k2 ? [k1, t1] : [k2, t2];
    // warm start only: the LM loop refines, loose tolerance by design
    mu = abilitiesFromTopk(ta, ka,
      { D: sd.map(v => v * v), base, points, nIter: 20, tol: 1e-3,
        returnInfo: true }).mu;
  }
  const sqr = Math.sqrt(Math.max(ridge, 0));

  const logits = (m, s) => {
    const qh1 = topKProbabilities(m, k1, { D: s.map(v => v * v), base, points })
      .map(v => Math.min(Math.max(v, 1e-300), 1 - 1e-15));
    const qh2 = topKProbabilities(m, k2, { D: s.map(v => v * v), base, points })
      .map(v => Math.min(Math.max(v, 1e-300), 1 - 1e-15));
    const r = qh1.map((v, i) => Math.log(v) - Math.log1p(-v) - lt1[i]).concat(
      qh2.map((v, i) => Math.log(v) - Math.log1p(-v) - lt2[i]),
      s.map(v => sqr * Math.log(v)));
    return { r, qh1, qh2 };
  };
  const fitMax = r => {
    let d = 0;
    for (let i = 0; i < 2 * n; i++) d = Math.max(d, Math.abs(r[i]));
    return d;
  };

  let { r, qh1, qh2 } = logits(mu, sd);
  let cost = r.reduce((a, b) => a + b * b, 0);
  let residMax = fitMax(r);
  let lam = 1e-6, iters = 0, lastAccepted = true, lastGrad = Infinity;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    if (residMax < tol) break;
    const J = [];
    for (const [kk, qh] of [[k1, qh1], [k2, qh2]]) {
      const { Jmu, Jsigma } = topKJacobians(mu, kk,
        { D: sd.map(v => v * v), base, points });
      for (let i = 0; i < n; i++) {
        const g = 1 / Math.max(qh[i] * (1 - qh[i]), 1e-300);
        const row = new Array(2 * n);
        for (let j = 0; j < n; j++) {
          row[j] = Jmu[i][j] * g;
          row[n + j] = Jsigma[i][j] * sd[j] * g;
        }
        J.push(row);
      }
    }
    for (let i = 0; i < n; i++) {
      const row = new Array(2 * n).fill(0);
      row[n + i] = sqr;
      J.push(row);
    }
    const rows = J.length;
    const JtJ = [], Jtr = new Array(2 * n).fill(0);
    for (let a = 0; a < 2 * n; a++) {
      const row = new Array(2 * n).fill(0);
      for (let b = 0; b < 2 * n; b++)
        for (let t = 0; t < rows; t++) row[b] += J[t][a] * J[t][b];
      JtJ.push(row);
      for (let t = 0; t < rows; t++) Jtr[a] += J[t][a] * r[t];
    }
    lastGrad = Math.max(...Jtr.map(Math.abs));
    let accepted = false;
    for (let attempt = 0; attempt < 8; attempt++) {
      const A = JtJ.map((row, a) =>
        row.map((v, b) => v + (a === b ? lam : 0)));
      let step;
      try {
        step = solve(A, Jtr.map(v => -v));
      } catch (e) {
        lam *= 8; continue;
      }
      let muN = mu.map((v, i) => v + step[i]);
      let lsN = sd.map((v, i) =>
        Math.min(Math.max(Math.log(v) + step[n + i], -3), 3));
      const lsMean = lsN.reduce((a, b) => a + b, 0) / n;
      const c = Math.exp(lsMean);
      const sdN = lsN.map(v => Math.exp(v - lsMean));
      const muMean = muN.reduce((a, b) => a + b, 0) / n;
      muN = muN.map(v => (v - muMean) / c);
      let out;
      try {
        out = logits(muN, sdN);
      } catch (e) {
        lam *= 8; continue;
      }
      const costN = out.r.reduce((a, b) => a + b * b, 0);
      if (costN < cost) {
        mu = muN; sd = sdN; r = out.r; qh1 = out.qh1; qh2 = out.qh2;
        cost = costN;
        residMax = fitMax(r);
        lam = Math.max(lam / 3, 1e-10);
        accepted = true;
        break;
      }
      lam *= 8;
    }
    lastAccepted = accepted;
    if (!accepted) break;
  }
  // with a ridge the penalized optimum keeps a nonzero fit residual by
  // design: an LM stall there is the answer, not a failure
  // A ridge stall is convergence only at a stationary point of the
  // penalized objective AND on a board passing the exact nesting
  // necessity P(top k1) <= P(top k2) for k1 < k2, as in python (#353,
  // #105): every rejected step used to be certified.
  const [loT, hiT] = k1 < k2 ? [t1, t2] : [t2, t1];
  const nested = loT.every((v, i) => v <= hiT[i] + 1e-12);
  const fitConverged = residMax < tol;
  const stationary = fitConverged ||
    (!lastAccepted && iters > 0 && lastGrad <= 1e-6 * Math.max(1, cost));
  const converged = fitConverged || (sqr > 0 && stationary && nested);
  if (!converged && !returnInfo)
    console.warn(`locScaleFromTopkPair did not converge: max |logit ` +
                 `residual| ${residMax.toExponential(2)} after ${iters} iterations`);
  if (returnInfo)
    return { mu, sd, info: { converged, maxLogitResidual: residMax,
                             iterations: iters, fitConverged, stationary,
                             nested } };
  return { mu, sd };
}

export function locScaleFromWinAndSecond(pWin, pSecond, opts = {}) {
  checkOpts(opts, LOC_SCALE_FROM_WIN_AND_SECOND_OPTS, "locScaleFromWinAndSecond", { ...OPT_HINTS, ...LOC_SCALE_HINTS });
  // win plus EXACTLY-second marginals: P(2nd) + P(win) = P(top-2),
  // the well-posed pair. Each marginal renormalized to unit mass.
  pWin = asFiniteVector(pWin, "pWin", "probability");
  pSecond = asFiniteVector(pSecond, "pSecond", "probability");
  const n = pWin.length;
  if (pSecond.length !== n)
    throw new Error("pWin and pSecond must have equal length");
  if (pWin.some(v => v <= 0) || pSecond.some(v => v <= 0))
    throw new Error("all win and second probabilities must be positive");
  const s1 = pWin.reduce((a, b) => a + b, 0);
  const s2 = pSecond.reduce((a, b) => a + b, 0);
  const p1 = pWin.map(v => v / s1);
  const top2 = p1.map((v, i) => v + pSecond[i] / s2);
  return locScaleFromTopkPair(p1, 1, top2, 2, opts);
}

function rankMarginalWithJacobian(mu, sd, r, fn, points) {
  const n = mu.length;
  const { dx, F, f } = topkGrid(mu, sd, n - 1, fn, points);
  const L = F.length;
  const dens = [];
  for (let t = 0; t < L; t++) dens.push(f[t].map((v, i) => v / sd[i]));
  const C = countDistribution(F);
  const p = new Array(n), J = [];
  for (let i = 0; i < n; i++) {
    const Qi = looPmf(C, F, i);
    let pi = 0;
    for (let t = 0; t < L; t++) pi += Qi[t][r - 1] * dens[t][i];
    p[i] = pi * dx;
    // P(N_{-ij} = n-1) is identically zero (only n-2 others exist);
    // deconvolving it injects window-edge-sensitive junk
    const hiPair = r <= n - 1 ? pairPmfAt(Qi, F, i, r) : null;
    const loPair = r >= 2 ? pairPmfAt(Qi, F, i, r - 1) : null;
    const row = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let s = 0;
      for (let t = 0; t < L; t++) {
        let c = hiPair ? hiPair[j][t] : 0;
        if (loPair) c -= loPair[j][t];
        s += c * dens[t][j] * dens[t][i];
      }
      row[j] = s * dx;
    }
    row[i] = -row.reduce((a, b) => a + b, 0);
    J.push(row);
  }
  return { p, J };
}

export function abilitiesFromRankMarginal(p, r, opts = {}) {
  checkOpts(opts, ABILITIES_FROM_RANK_MARGINAL_OPTS, "abilitiesFromRankMarginal", OPT_HINTS);
  // invert one EXACT-rank marginal at frozen scales: two-branched for
  // r >= 2, mu0 selects the branch. See the python docstring.
  const { mu0 = null, D = null, base = "normal", points = 513,
          nIter = 60, tol = 1e-8, returnInfo = false } = opts;
  asIterations(nIter, "abilitiesFromRankMarginal");
  asTolerance(tol, "abilitiesFromRankMarginal");
  p = asFiniteVector(p, "target", "probability");
  const n = p.length;
  // a whole-number rank, refused before coercion: Math.trunc(1.5) solved
  // first place silently (#317)
  if (typeof r !== "number" || !Number.isFinite(r) || r !== Math.trunc(r))
    throw new Error(`r must be a whole-number rank; got ${r}`);
  if (!(r >= 1 && r <= n))
    throw new Error(`rank must be in [1, n]; got r=${r}, n=${n}`);
  if (p.some(v => !Number.isFinite(v)))
    throw new Error("rank probabilities must be finite");
  if (p.some(v => v <= 0))
    throw new Error("all rank probabilities must be positive");
  const s = p.reduce((a, b) => a + b, 0);
  const logt = p.map(v => Math.log(v / s));
  const sd = asIdio(D, n).map(Math.sqrt);
  const fn = baseFn(base);
  const zz = [-2.3, -1.1, -0.35, 0.6, 1.7];
  const symmetric = zz.every(z => Math.abs(fn(z)[0] + fn(-z)[0] - 1) < 1e-12);
  if (mu0 == null && n % 2 === 1 && r === (n + 1) / 2 && symmetric)
    // the exact middle rank of an odd field under a symmetric base is even
    // in mu: zero Jacobian at the zero start, no way off it (#378)
    throw new Error(
      `the exact middle rank r=${r} of an odd field (n=${n}) under a ` +
      "symmetric base is unchanged by mu -> -mu, so the zero start is " +
      "stationary and the inverse two-branched. Pass mu0 to choose the branch.");
  let mu;
  if (mu0 != null) {
    const m00 = asAbilities(mu0, "mu0");
    if (m00.length !== n) throw new Error(`mu0 has ${m00.length} entries for ${n} runners`);
    const m0 = m00.reduce((a, b) => a + b, 0) / n;
    mu = m00.map(v => v - m0);
  } else {
    mu = new Array(n).fill(0);
  }

  let { p: phat, J } = rankMarginalWithJacobian(mu, sd, r, fn, points);
  let resid = phat.map((v, i) => Math.log(Math.max(v, 1e-300)) - logt[i]);
  let cost = resid.reduce((a, b) => a + b * b, 0);
  let residMax = Math.max(...resid.map(Math.abs));
  let lam = 1e-6, iters = 0;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    if (residMax < tol) break;
    const Jlog = J.map((row, i) =>
      row.map(v => v / Math.max(phat[i], 1e-300)));
    const A = [], g = new Array(n).fill(0);
    for (let a = 0; a < n; a++) {
      const row = new Array(n).fill(0);
      for (let b = 0; b < n; b++)
        for (let t = 0; t < n; t++) row[b] += Jlog[t][a] * Jlog[t][b];
      A.push(row);
      for (let t = 0; t < n; t++) g[a] += Jlog[t][a] * resid[t];
    }
    let accepted = false;
    for (let attempt = 0; attempt < 8; attempt++) {
      const Ad = A.map((row, a) => row.map((v, b) => v + (a === b ? lam : 0)));
      let step;
      try {
        step = solve(Ad, g.map(v => -v));
      } catch (e) {
        lam *= 8; continue;
      }
      let muN = mu.map((v, i) => v + step[i]);
      const mm = muN.reduce((a, b) => a + b, 0) / n;
      muN = muN.map(v => v - mm);
      const nx = rankMarginalWithJacobian(muN, sd, r, fn, points);
      const rN = nx.p.map((v, i) =>
        Math.log(Math.max(v, 1e-300)) - logt[i]);
      const costN = rN.reduce((a, b) => a + b * b, 0);
      if (costN < cost) {
        mu = muN; phat = nx.p; J = nx.J; resid = rN; cost = costN;
        residMax = Math.max(...resid.map(Math.abs));
        lam = Math.max(lam / 3, 1e-10);
        accepted = true;
        break;
      }
      lam *= 8;
    }
    if (!accepted) break;
  }
  const converged = residMax < tol;
  if (!converged && !returnInfo)
    console.warn(`abilitiesFromRankMarginal did not converge: max |log ` +
                 `residual| ${residMax.toExponential(2)} after ${iters} iterations`);
  if (returnInfo)
    return { mu, info: { converged, maxLogResidual: residMax,
                         iterations: iters } };
  return mu;
}
