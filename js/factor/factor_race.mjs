/* Factor-correlated race transforms: JavaScript port of
 * winning.factor.core (Python canonical). Min-wins convention.
 * Parity against committed test vectors: run `node test_parity.mjs`.
 */

const SQRT2 = Math.SQRT2;
const SQRT_PI = Math.sqrt(Math.PI);
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const PFLOOR = 1e-15;

/* erf by alternating series, accurate for |x| <= ~1.5 */
function erfSeries(x) {
  let term = x, sum = x;
  for (let n = 1; n < 60; n++) {
    term *= -x * x / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) < 1e-18 * Math.abs(sum)) break;
  }
  return (2 / SQRT_PI) * sum;
}

/* scaled complementary error function erfcx(x) = e^{x^2} erfc(x), x >= 1,
 * by the classical continued fraction (modified Lentz). */
function erfcx(x) {
  const tiny = 1e-30;
  let f = tiny, C = f, D = 0;
  // CF: erfcx(x) = (1/sqrt(pi)) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
  for (let n = 0; n < 400; n++) {
    const a = n === 0 ? 1.0 : n / 2.0;
    const b = n === 0 ? 0.0 : x;
    // first step: b0 = x handled by starting the wrap
    const bb = n === 0 ? x : b;
    D = bb + a * D;
    if (Math.abs(D) < tiny) D = tiny;
    C = bb + a / C;
    if (Math.abs(C) < tiny) C = tiny;
    D = 1 / D;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-17) break;
  }
  return f / SQRT_PI;
}

/* log of the standard normal CDF, tail-stable (port of scipy.log_ndtr use) */
export function logndtr(z) {
  if (z >= 1.0) {
    // log(1 - Phi(-z)); Phi(-z) computed via the negative branch
    return Math.log1p(-Math.exp(logndtr(-z)));
  }
  if (z > -1.0) {
    return Math.log(0.5 * (1 + erfSeries(z / SQRT2)));
  }
  // z <= -1: Phi(z) = 0.5 * erfcx(-z/sqrt2) * exp(-z^2/2)
  const x = -z / SQRT2;
  return Math.log(0.5 * erfcx(x)) - 0.5 * z * z;
}

/* One fixed grid over every conditional mean, padded by the WIDEST
   runner -- and then refined until it resolves the NARROWEST one.

   The span is set by the widest runner and by the farthest conditional
   mean, the spacing by `points`, so a distant or diffuse runner could
   leave the close ones between samples. Shares are normalised and hid
   it; the unnormalised outputs could not: a runner 100 units behind a
   close pair cut their tie density by 10.3% (#349), a removed favourite
   left a deletion row 15% short of mass that normalisation then spread
   over the survivors (#330), and the home page's 301-point t4 field
   showed 0.2-point errors (#391). The spacing is now held to half the
   smallest sd, capped at MAX_POINTS (with a warning past it), which is
   the resolution target python's forward_grid warns at. A well-resolved
   field is untouched: the refinement only ever adds points. */
const MAX_POINTS = 16385;
function lattice(Mall, sd, points, spans, resolution = 1) {
  let mMin = Infinity, mMax = -Infinity, sMax = 0, sMin = Infinity;
  for (const row of Mall) for (const v of row) { if (v < mMin) mMin = v; if (v > mMax) mMax = v; }
  for (const s of sd) { if (s > sMax) sMax = s; if (s < sMin) sMin = s; }
  const lo = mMin - spans[0] * sMax, hi = mMax + spans[1] * sMax;
  // `resolution` is the base's own feature width in standard units (1
  // for a smooth base; a skew-normal's Phi(alpha u) step is ~1/alpha
  // wide, and a lattice coarser than that aliases the density itself)
  const need = Math.ceil((hi - lo) / (0.5 * sMin * resolution)) + 1;
  if (need > points) {
    if (need > MAX_POINTS && typeof console !== "undefined")
      console.warn(`factor race lattice cannot resolve the narrowest runner ` +
                   `(sd ${sMin.toExponential(1)} over ${(hi - lo).toPrecision(3)} ` +
                   `units) even at ${MAX_POINTS} points`);
    points = Math.min(need, MAX_POINTS);
  }
  const x = new Float64Array(points);
  for (let l = 0; l < points; l++) x[l] = lo + (hi - lo) * l / (points - 1);
  return { x, dx: (hi - lo) / (points - 1) };
}

/* ---- the boundary: the same contracts as docs/js/winning/core.mjs ----
   This module is standalone (it is deployed as docs/assets/js and three
   pages import it directly), so it carries its own copies. */
const isVec = (x) => Array.isArray(x) || (ArrayBuffer.isView(x) && !(x instanceof DataView));
function finiteVec(x, where, what = "entry") {
  if (!isVec(x)) throw new Error(`${where} must be an array of numbers`);
  if (x.length === 0) throw new Error(`${where} is empty`);
  for (let i = 0; i < x.length; i++)
    if (typeof x[i] !== "number" || !Number.isFinite(x[i]))
      throw new Error(`${where}[${i}] = ${x[i]} is not a finite ${what}`);
  return Array.from(x);
}
/* D: a finite positive scalar or exactly one variance per contestant.
   An extra entry used to widen the lattice by its own sd and misprice
   the field it did not belong to (#254). */
function asIdio(D, n, where = "D") {
  let v;
  if (typeof D === "number") v = new Array(n).fill(D);
  else if (isVec(D)) {
    if (D.length !== n)
      throw new Error(`${where} must be one idiosyncratic variance per contestant; got ${D.length} for ${n}`);
    v = Array.from(D);
  } else throw new Error(`${where}: expected a number or array`);
  for (let i = 0; i < n; i++) {
    if (typeof v[i] !== "number" || !Number.isFinite(v[i]))
      throw new Error(`${where} has a non-finite entry at ${i}`);
    if (!(v[i] > 0))
      throw new Error(`${where}[${i}] = ${v[i]} must be a strictly positive variance`);
  }
  return v;
}
/* V: (n, rank) rows (rank 0 allowed), plain or typed, rectangular. */
function asLoadings(V, n, where = "V") {
  if (!isVec(V) || V.length !== n)
    throw new Error(`${where} must have one loading row per contestant; got ${isVec(V) ? V.length : typeof V} for ${n}`);
  const rows = Array.from(V, (r, i) => {
    if (!isVec(r)) throw new Error(`${where}[${i}] must be a row of loadings`);
    return Array.from(r);
  });
  const r = rows[0].length;
  rows.forEach((row, i) => {
    if (row.length !== r)
      throw new Error(`${where} is ragged: row ${i} has ${row.length} loadings, row 0 has ${r}`);
    row.forEach((v, d) => {
      if (typeof v !== "number" || !Number.isFinite(v))
        throw new Error(`${where}[${i}][${d}] = ${v} is not finite`);
    });
  });
  return rows;
}
/* F: a rectangular Q x rank(V) node matrix. condMeans read the rank off
   F[0], so a uniformly short F priced a LOWER-rank model (0.083 share
   move) and a ragged one silently ignored later coordinates (#290). */
function asNodes(F, r, where = "F") {
  if (!isVec(F) || F.length === 0)
    throw new Error(`${where} must be a nonempty array of factor nodes`);
  return Array.from(F, (row, q) => {
    if (!isVec(row) || row.length !== r)
      throw new Error(`${where}[${q}] has ${isVec(row) ? row.length : "no"} coordinate(s) but the loadings have rank ${r}`);
    const out = Array.from(row);
    out.forEach((v, d) => {
      if (typeof v !== "number" || !Number.isFinite(v))
        throw new Error(`${where}[${q}][${d}] = ${v} is not finite`);
    });
    return out;
  });
}
function asBudget(x, dflt, where, name) {
  if (x === undefined || x === null) return dflt;
  if (!Number.isInteger(x) || x < 1)
    throw new Error(`${where}: ${name} must be a finite positive integer; got ${x}. ` +
                    "0, negatives and NaN used to mean the default or skip the " +
                    "solve, and Infinity could hang the page (#428).");
  return x;
}
function asPositive(x, dflt, where, name) {
  if (x === undefined || x === null) return dflt;
  if (typeof x !== "number" || !Number.isFinite(x) || !(x > 0))
    throw new Error(`${where}: ${name} must be a finite positive number; got ${x}`);
  return x;
}
function baseOf(b) {
  const base = (b && typeof b === "object") ? b : BASES[b || "normal"];
  if (!base) throw new Error("unknown base: " + b);
  return base;
}

/* Standardized bases (zero mean, unit variance), matching Python
 * winning.factor.races.BASES. Each returns, for standardized z and
 * runner scale sd: ls = log survival, fx = density in x units, and
 * ds = the slope integrand d/dmu of the density (-f'(z)/sd^2). */
const EULER = 0.5772156649015329;
const GUMBEL_C = Math.PI / Math.sqrt(6);
const BASES = {
  normal: {
    spans: [8, 8],
    eval(z, sd) {
      const fx = Math.exp(-0.5 * z * z) / (sd * SQRT_2PI);
      return { ls: logndtr(-z), fx, ds: z * fx / sd };
    },
  },
  gumbel: {
    spans: [22, 8],
    eval(z, sd) {
      const u = Math.min(z * GUMBEL_C - EULER, 30.0);
      const eu = Math.exp(u);
      const S = Math.max(Math.exp(-eu), 1e-300);
      const fz = GUMBEL_C * eu * S;
      return { ls: -eu, fx: fz / sd,
        ds: -GUMBEL_C * GUMBEL_C * eu * S * (1 - eu) / (sd * sd) };
    },
  },
};

/* Bases without elementary survival functions are tabulated once on a
 * fine grid (dx = 0.005): survival by cumulative trapezoid, slope by
 * central differences, log-survival interpolated linearly (log-linear
 * tails). Python parity vs scipy is ~1e-6, not machine precision. */
function tabulatedBase(pdfStd, spans, { dpdf = null, dx = 0.005 } = {}) {
  // Two tiers. The inner one is fine enough for the bulk; the outer one
  // exists because clamping the lookup at the table edge made the density
  // FLAT beyond it, so a runner 100 sd behind kept the endpoint density
  // and the race reported a false longshot floor -- 3.5e-7 at gaps 80 and
  // 100 where the truth keeps falling (#182). Heavy tails are exactly
  // where that matters: a Student-t4 survival decays polynomially, so
  // there is no span past which the omitted mass is negligible, and the
  // old normalisation divided by the INNER mass alone, which is the same
  // error again.
  const HALF = 40;
  const NPTS = Math.round(2 * HALF / dx) + 1, DX = 2 * HALF / (NPTS - 1);
  const OUTER = 4000, NOUT = 8001, DXO = (OUTER - HALF) / (NOUT - 1);
  let g = null;
  const build = () => {
    const f = new Float64Array(NPTS);
    for (let k = 0; k < NPTS; k++) f[k] = pdfStd(-HALF + k * DX);
    const fR = new Float64Array(NOUT), fL = new Float64Array(NOUT);
    for (let k = 0; k < NOUT; k++) {
      fR[k] = pdfStd(HALF + k * DXO);
      fL[k] = pdfStd(-OUTER + k * DXO);
    }
    // one cumulative sweep across all three pieces, left to right, so the
    // survival is normalised by the WHOLE computed mass
    let C = 0;
    const cumL = new Float64Array(NOUT);
    for (let k = 1; k < NOUT; k++) { C += 0.5 * (fL[k - 1] + fL[k]) * DXO; cumL[k] = C; }
    const cum = new Float64Array(NPTS); cum[0] = C;
    for (let k = 1; k < NPTS; k++) { C += 0.5 * (f[k - 1] + f[k]) * DX; cum[k] = C; }
    const cumR = new Float64Array(NOUT); cumR[0] = C;
    for (let k = 1; k < NOUT; k++) { C += 0.5 * (fR[k - 1] + fR[k]) * DXO; cumR[k] = C; }
    const T = C;
    // No end correction on the cumulative. An Euler-Maclaurin term
    // (h^2/12)(f'(x) - f'(a)) was tried and measured WORSE than the plain
    // trapezoid, which is why #211's description and the changelog say it
    // was removed; it was left in the code by mistake (#216). It could not
    // have delivered its O(h^4): the cumulative is chained across three
    // pieces with two different step sizes, corrected per piece with that
    // piece's h and no correction at the joins, and f'(a) was taken as
    // fpL[0], which the centred-difference helper never writes, so it was
    // identically zero rather than the derivative at the left end. What
    // actually fixed the 1e-5 survival gap was the span widening in the
    // same PR, BASES.t4 [12,12] -> [24,24].
    const lsOf = (c) => {
      const a = new Float64Array(c.length);
      for (let k = 0; k < c.length; k++)
        a[k] = Math.log(Math.max((T - c[k]) / T, 1e-300));
      return a;
    };
    const dOf = (arr, h) => {
      const a = new Float64Array(arr.length);
      for (let k = 1; k < arr.length - 1; k++) a[k] = (arr[k + 1] - arr[k - 1]) / (2 * h);
      return a;
    };
    const fp = dOf(f, DX), fpR = dOf(fR, DXO), fpL = dOf(fL, DXO);
    g = { f, ls: lsOf(cum), fp,
          fR, lsR: lsOf(cumR), fpR,
          fL, lsL: lsOf(cumL), fpL };
  };
  return {
    spans,
    eval(z, sd) {
      if (!g) build();
      let arrF, arrLs, arrFp, t;
      // Both wings clamp at BOTH ends: the left wing's top index is
      // NOUT - 1 at z = -HALF exactly, and reading arr[k + 1] there gave
      // NaN, which the race then carried into the whole field.
      if (z >= HALF) {
        t = (z - HALF) / DXO; arrF = g.fR; arrLs = g.lsR; arrFp = g.fpR;
        t = Math.min(Math.max(t, 0), NOUT - 2);   // past 4000 sd a floor
      } else if (z <= -HALF) {                    // remains, now ~1e-13
        t = (z + OUTER) / DXO; arrF = g.fL; arrLs = g.lsL; arrFp = g.fpL;
        t = Math.min(Math.max(t, 0), NOUT - 2);   // rather than 3.5e-7
      } else {
        t = (z + HALF) / DX; arrF = g.f; arrLs = g.ls; arrFp = g.fp;
      }
      const k = Math.floor(t), a = t - k;
      const lerp = (arr) => arr[k] + a * (arr[k + 1] - arr[k]);
      if (dpdf) {
        // analytic density and slope where the base knows them (#331):
        // a centred difference of a table cannot follow a transition
        // narrower than its spacing, and the interpolated slope then
        // came out with the wrong SIGN
        return { ls: lerp(arrLs), fx: pdfStd(z) / sd, ds: -dpdf(z) / (sd * sd) };
      }
      return { ls: lerp(arrLs), fx: lerp(arrF) / sd, ds: -lerp(arrFp) / (sd * sd) };
    },
  };
}

const PHI = (z) => Math.exp(logndtr(z));

/* skew-normal with shape alpha, standardized to mean 0 variance 1;
 * returns a base object usable directly as opts.base */
export function skewNormalBase(alpha) {
  if (typeof alpha !== "number" || !Number.isFinite(alpha))
    throw new Error(`skewNormalBase: alpha must be a finite number; got ${alpha}`);
  const delta = alpha / Math.sqrt(1 + alpha * alpha);
  const m = delta * Math.sqrt(2 / Math.PI);
  const s = Math.sqrt(1 - m * m);
  const phi = (u) => Math.exp(-0.5 * u * u) / SQRT_2PI;
  const pdf = (z) => { const u = m + s * z;
    return s * 2 * phi(u) * PHI(alpha * u); };
  // d pdf / dz, analytic: the Phi(alpha u) transition is O(1/|alpha|)
  // wide, and a fixed 0.005 table differenced across it gave a POSITIVE
  // own slope at alpha = 400, so the inverse walked uphill (#331)
  const dpdf = (z) => { const u = m + s * z;
    return s * s * 2 * phi(u) * (alpha * phi(alpha * u) - u * PHI(alpha * u)); };
  // the survival is still tabulated; resolve the transition with a few
  // points across it, never coarser than the default
  const dx = Math.max(Math.min(0.005, 0.25 / Math.max(Math.abs(alpha), 1)), 1e-4);
  const b = tabulatedBase(pdf, [12, 12], { dpdf, dx });
  b.pdf = pdf;
  b.resolution = Math.min(1, 2 / Math.max(Math.abs(alpha), 1e-300));
  return b;
}

BASES.skew = skewNormalBase(3);
{
  // Student-t, nu = 4, standardized (sd = sqrt(2))
  const SQ2 = Math.SQRT2;
  // spans: how far past the extreme conditional mean the lattice runs, in
  // units of the largest sd. Student-t4 keeps real mass a long way out, so
  // 12 truncated it and the forward sat 1.01e-5 from scipy -- a hundred
  // times the python reference's own lattice error, and the standing
  // failure in test_parity.mjs. Measured against that fixture, at the
  // default 501 points:
  //
  //   span   forward error   inverse over 501..4001 points
  //     12       1.01e-5     2.9e-7 .. 2.3e-6
  //     20       1.75e-6     2.9e-7 .. 5.0e-7
  //     24       8.42e-7     2.9e-7 .. 3.5e-7
  //     32       2.37e-7     2.9e-7 .. 5.3e-7
  //     40       1.39e-7     NaN at 501, 1001 and 4001
  //
  // Both ends cost: too narrow drops tail mass, too wide spreads a fixed
  // point budget until the bulk is under-resolved, and past ~40 the
  // inverse returns NaN outright (its own fragility, not this constant --
  // reported separately). 24 sits where both are good with margin.
  BASES.t4 = tabulatedBase(
    (z) => { const u = SQ2 * z;
      return SQ2 * (3 / 8) * Math.pow(1 + u * u / 4, -2.5); },
    [24, 24]);
}

function condMeans(mu, V, F) {
  const Q = F.length, N = mu.length, K = F[0].length;
  const M = [];
  for (let q = 0; q < Q; q++) {
    const row = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = mu[i];
      for (let d = 0; d < K; d++) s += F[q][d] * V[i][d];
      row[i] = s;
    }
    M.push(row);
  }
  return M;
}

/* The standardized density of a base (zero mean, unit variance), as the
   engine integrates it -- for pages that DRAW the fitted performance
   law. fit.html drew a Gaussian for every base but Gumbel, so a t4 or
   skew-normal fit was shown as the wrong law (#351). */
export function basePdf(b) {
  const base = baseOf(b);
  return (z) => base.eval(z, 1).fx;
}

/* forward pass: shares (and optionally slopes, pairwise densities, deletions) */
/* The factor weights describe a LAW, so W and c*W are the same law and
   must give the same answer. The forward already did: it normalises its
   accumulated shares, so `p` was invariant to 1e-16 under any positive
   rescaling. But it returns the own-slopes UNNORMALISED, and the
   inverse divides those by the normalised probabilities -- so the
   Newton derivative carried a factor of c, and only the inverse moved.
   A self-generated target repriced 0.16 away after fifty iterations at
   c = 0.1, which is a calibration failure with no error (#281).

   python has the same helper-level mismatch and is saved by its front
   door: `races._setup` puts W through `winning.shapes.as_weights`. This
   standalone module has no such boundary, so it grows one, with the
   same contract -- finite, non-negative, positive total, normalised --
   which also settles zero and signed weights rather than letting them
   through to produce a normalised, plausible, wrong answer.

   Deliberately NOT also scaling the slope by the forward total: python
   does not, and a silent divergence between the ports is worse than
   either behaviour. This makes the two agree. */
export function asWeights(W, nNodes, where = "W") {
  if (!Array.isArray(W) && !ArrayBuffer.isView(W))
    throw new Error(`${where} must be an array of factor-node weights; got ${typeof W}`);
  if (W.length !== nNodes)
    throw new Error(`${where} must have one weight per factor node; got ${W.length} for ${nNodes}`);
  let top = 0;
  for (let i = 0; i < W.length; i++) {
    const v = Number(W[i]);
    if (!Number.isFinite(v))
      throw new Error(`${where}[${i}] = ${W[i]} is not a finite weight`);
    if (v < 0)
      throw new Error(`${where}[${i}] = ${v} is a negative weight; a factor law has no negative mass`);
    if (v > top) top = v;
  }
  if (!(top > 0))
    throw new Error(`${where} must have a positive total; got 0`);
  // through the LARGEST weight, never the raw total: [1e308, 1e308]
  // overflows only in the sum, and W / Infinity made every weight zero
  // and every probability NaN (#415)
  const u = Array.from(W, (v) => Number(v) / top);
  const total = u.reduce((a, b) => a + b, 0);
  return u.map((v) => v / total);
}

export function winProbabilitiesFactor(mu, V, D, F, W, opts = {}) {
  const points = asBudget(opts.points, 501, "winProbabilitiesFactor", "points");
  if (points < 3) throw new Error("winProbabilitiesFactor: points must be at least 3");
  const base = baseOf(opts.base);
  mu = finiteVec(mu, "mu", "ability");
  const N = mu.length;
  V = asLoadings(V, N);
  D = asIdio(D, N);
  F = asNodes(F, V[0].length);
  const Q = F.length;
  W = asWeights(W, Q, "W");
  const sd = D.map(Math.sqrt);
  // gauge-fix matching the python reference: center each factor's
  // loadings across contestants (a common column cannot move an argmin)
  const r0 = V[0].length;
  const colMean = new Array(r0).fill(0);
  for (const row of V) for (let j = 0; j < r0; j++) colMean[j] += row[j] / N;
  V = V.map((row) => row.map((v, j) => v - colMean[j]));
  const M = condMeans(mu, V, F);
  const { x, dx } = lattice(M, sd, points, base.spans, base.resolution || 1);
  const L = x.length;
  const p = new Float64Array(N);
  const slope = new Float64Array(N);
  // the derivative of the RETURNED (normalised) share needs the raw
  // total's derivative too: d a_j / d mu_i = w_ij for j != i, so this is
  // the row sum of the tie densities, accumulated on the same pass (#371)
  const wantCross = !!opts.ownLogSlope;
  const cross = new Float64Array(N);
  const w = opts.pairwise ? Array.from({ length: N }, () => new Float64Array(N)) : null;
  const q = opts.deletions ? Array.from({ length: N }, () => new Float64Array(N)) : null;

  const logS = Array.from({ length: N }, () => new Float64Array(L));
  const f = Array.from({ length: N }, () => new Float64Array(L));
  const dsl = Array.from({ length: N }, () => new Float64Array(L));
  const logSfield = new Float64Array(L);

  for (let c = 0; c < Q; c++) {
    logSfield.fill(0);
    for (let i = 0; i < N; i++) {
      for (let l = 0; l < L; l++) {
        const z = (x[l] - M[c][i]) / sd[i];
        const { ls, fx, ds } = base.eval(z, sd[i]);
        logS[i][l] = ls;
        logSfield[l] += ls;
        f[i][l] = fx;
        dsl[i][l] = ds;
      }
    }
    const Wc = W[c];
    for (let i = 0; i < N; i++) {
      let acc = 0, accS = 0;
      for (let l = 0; l < L; l++) {
        let e = logSfield[l] - logS[i][l];
        if (e > 0) e = 0;
        const rest = e < -745 ? 0 : Math.exp(e);
        acc += f[i][l] * rest;
        accS += dsl[i][l] * rest;
      }
      p[i] += Wc * acc * dx;
      slope[i] += Wc * accS * dx;
    }
    if (w || wantCross) {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        if (j === i) continue;
        if (!w && j < i) continue;           // symmetric: one triangle for cross
        let acc = 0;
        for (let l = 0; l < L; l++) {
          let e = logSfield[l] - logS[i][l] - logS[j][l];
          if (e > 0) e = 0;
          acc += f[i][l] * f[j][l] * (e < -745 ? 0 : Math.exp(e));
        }
        if (w) w[i][j] += Wc * acc * dx;
        if (wantCross) {
          if (w) cross[i] += Wc * acc * dx;
          else { cross[i] += Wc * acc * dx; cross[j] += Wc * acc * dx; }
        }
      }
    }
    if (q) {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (j === i) continue;
          let acc = 0;
          for (let l = 0; l < L; l++) {
            let e = logSfield[l] - logS[i][l] - logS[j][l];
            if (e > 0) e = 0;
            acc += f[j][l] * (e < -745 ? 0 : Math.exp(e));
          }
          q[i][j] += Wc * acc * dx;
        }
      }
    }
  }
  let total = 0;
  for (const v of p) total += v;
  const out = Array.from(p, (v) => v / total);
  const res = { p: out, total, slope: Array.from(slope), points: L };
  if (wantCross) {
    // d log p_i / d mu_i of the normalised share: own term minus the
    // total's, a_i' / a_i - (a_i' + sum_j w_ij) / T
    res.ownLogSlope = Array.from(p, (a, i) =>
      slope[i] / Math.max(a, 1e-300) - (slope[i] + cross[i]) / total);
  }
  if (w) res.w = w.map((r) => Array.from(r));
  if (q) {
    // each removal row is its own race and carries its own mass check:
    // the full race's `total` says nothing about it -- 0.99984 there sat
    // beside a deletion row 13.4 points wrong (#330)
    const mass = q.map((r) => r.reduce((a, b) => a + b, 0));
    const worst = Math.max(...mass.map((m) => Math.abs(m - 1)));
    if (!(worst <= 5e-3))
      throw new Error(
        `deletion rows captured mass ${Math.min(...mass).toFixed(4)}..` +
        `${Math.max(...mass).toFixed(4)} where each is a whole race: the ` +
        "lattice cannot resolve a removal counterfactual; raise points=");
    res.deletionMass = mass;
    res.deletions = q.map((r, i) => Array.from(r, (v) => v / mass[i]));
  }
  return res;
}

/* inverse transform: damped coordinatewise Newton (port of
 * abilities_from_probabilities_factor).
 *
 * Returns mu; with opts.returnInfo, {mu, converged, residual,
 * iterations}. Without it a non-converged solve WARNS rather than
 * returning an uncertified calibration silently (#358, #428). */
export function abilitiesFromProbabilitiesFactor(pTarget, V, D, F, W, opts = {}) {
  const where = "abilitiesFromProbabilitiesFactor";
  const nIter = asBudget(opts.nIter, 50, where, "nIter");
  const tol = asPositive(opts.tol, 1e-6, where, "tol");
  const points = asBudget(opts.points, 501, where, "points");
  const base = opts.base || "normal";
  baseOf(base);
  const target = finiteVec(pTarget, "target", "probability");
  const N = target.length;
  let psum = 0;
  for (const v of target) { if (v <= 0) throw new Error("targets must be positive"); psum += v; }
  const p = target.map((v) => v / psum);
  const logp = p.map(Math.log);
  V = asLoadings(V, N);
  D = asIdio(D, N);
  const r = V[0].length;
  F = asNodes(F, r);
  W = asWeights(W, F.length, "W");
  const sd = D.map(Math.sqrt);
  const fwdOpts = { points, base };
  const finish = (mu, converged, residual, iterations) => {
    if (!converged && !opts.returnInfo && typeof console !== "undefined")
      console.warn(`${where} did not converge: max |log residual| ` +
                   `${residual.toExponential(2)} after ${iterations} iterations ` +
                   `(tol ${tol}). Pass returnInfo: true for the diagnostics.`);
    return opts.returnInfo ? { mu, converged, residual, iterations } : mu;
  };
  if (N === 1) return finish([0], true, 0, 0);

  // The factor law actually represented, as python's abilities_from_race
  // measures it: V -> aV, F -> F/a is the SAME forward, but reading the
  // factor variance off raw sum(V_i^2) inflated the warm start and the
  // step cap by a^2 and missed by 89 points at a = 100 (#443).
  const Vc = (() => {
    const cm = new Array(r).fill(0);
    for (const row of V) for (let j = 0; j < r; j++) cm[j] += row[j] / N;
    return V.map((row) => row.map((v, j) => v - cm[j]));
  })();
  const Fm = new Array(r).fill(0);
  F.forEach((fq, k) => fq.forEach((v, d) => { Fm[d] += W[k] * v; }));
  const CovF = Array.from({ length: r }, (_, a) => Array.from({ length: r }, (_, b) =>
    F.reduce((acc, fq, k) => acc + W[k] * (fq[a] - Fm[a]) * (fq[b] - Fm[b]), 0)));
  const quad = (u, v) => u.reduce((acc, ua, a) =>
    acc + ua * CovF[a].reduce((acc2, c, b) => acc2 + c * v[b], 0), 0);
  const sigV = Vc.map((row) => quad(row, row));
  const shift = Vc.map((row) => row.reduce((acc, v, d) => acc + v * Fm[d], 0));
  const stepCap = D.map((d, i) => Math.sqrt(d + sigV[i]));

  const floor = Math.max(1e-9, 1e-4 / N);
  const ident = p.map((v) => v > floor);
  const residualOf = (phat) => {
    let res = 0, any = false;
    for (let i = 0; i < N; i++) if (ident[i]) { any = true; res = Math.max(res, Math.abs(Math.log(phat[i]) - logp[i])); }
    if (!any) for (let i = 0; i < N; i++) res = Math.max(res, Math.abs(Math.log(phat[i]) - logp[i]));
    return res;
  };

  if (N === 2) {
    // A pair is ONE number, the gap, and p_0 is monotone in it: solve
    // that scalar equation against the actual forward. The simultaneous
    // update is a two-cycle on K_2 (python's closed-form comment), and
    // this module had no pair branch at all, returning 0.770 for a
    // target of 0.8 (#198). The Gaussian closed form -- exact for
    // Gaussian nodes -- is the starting bracket, and the forward has
    // the last word, so a non-Gaussian rule is handled too.
    const s0 = Math.sqrt(D[0] + D[1] + quad(
      Vc[0].map((v, d) => v - Vc[1][d]), Vc[0].map((v, d) => v - Vc[1][d])));
    const g0 = s0 * ndtri(p[0]) - (shift[1] - shift[0]);
    const p0At = (g) => winProbabilitiesFactor([-0.5 * g, 0.5 * g], V, D, F, W, fwdOpts).p;
    let g = g0, ph = p0At(g), res = residualOf(ph.map((v) => Math.max(v, PFLOOR)));
    let it = 0;
    if (res >= tol) {
      // bracket the root of h(g) = log p0(g) - log p0* (increasing in g)
      const h = (gg) => Math.log(Math.max(p0At(gg)[0], PFLOOR)) - logp[0];
      let lo = g0, hi = g0, step = Math.max(0.25 * s0, 1e-6);
      let hlo = h(lo), hhi = hlo;
      while (hlo > 0 && it < nIter) { lo -= step; step *= 2; hlo = h(lo); it++; }
      step = Math.max(0.25 * s0, 1e-6);
      while (hhi < 0 && it < nIter) { hi += step; step *= 2; hhi = h(hi); it++; }
      for (; it < nIter; it++) {
        // secant inside the bracket, bisection when it leaves it
        let m = hhi !== hlo ? hi - hhi * (hi - lo) / (hhi - hlo) : 0.5 * (lo + hi);
        if (!(m > lo && m < hi)) m = 0.5 * (lo + hi);
        const hm = h(m);
        g = m;
        ph = p0At(g);
        res = residualOf(ph.map((v) => Math.max(v, PFLOOR)));
        if (res < tol) { it++; break; }
        if (hm < 0) { lo = m; hlo = hm; } else { hi = m; hhi = hm; }
      }
    }
    return finish([-0.5 * g, 0.5 * g], res < tol, res, it);
  }

  let mu;
  const anyV = Vc.some((row) => row.some((v) => v !== 0));
  if (opts.mu0) {
    mu = finiteVec(opts.mu0, "mu0", "ability");
    if (mu.length !== N) throw new Error(`mu0 has ${mu.length} entries for ${N}`);
  } else if (r >= 1 && anyV) {
    // warm start: the independent race at each runner's represented
    // total variance, shifted by the rule's mean
    const sdTot2 = D.map((d, i) => d + sigV[i]);
    const ind = abilitiesFromProbabilitiesFactor(
      p, V.map(() => [0]), sdTot2, [[0]], [1], { nIter, tol, points, base, returnInfo: true });
    mu = ind.mu.map((m, i) => m - shift[i]);
  } else {
    const m = logp.reduce((a, b) => a + b, 0) / N;
    mu = logp.map((v) => (v - m) / 2.0);
  }
  // two runners holding nearly all the mass two-cycle undamped, at any N
  // (python's abilities_from_race; #358): start damped there
  const top2 = p.slice().sort((a, b) => b - a).slice(0, 2).reduce((a, b) => a + b, 0);
  let damp = top2 > 0.8 ? 0.7 : 1.0;
  let prevRes = Infinity, res = Infinity, iters = 0;
  let prevDelta = null;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    const fwd = winProbabilitiesFactor(mu, V, D, F, W, { ...fwdOpts, ownLogSlope: true });
    const phat = fwd.p.map((v) => Math.max(v, PFLOOR));
    const resid = phat.map((v, i) => Math.log(v) - logp[i]);
    res = residualOf(phat);
    if (!Number.isFinite(res)) break;
    if (res < tol) break;
    if (res > prevRes * 1.2) damp = Math.max(0.25, damp * 0.5);
    prevRes = res;
    const delta = new Array(N);
    for (let i = 0; i < N; i++) {
      // the derivative of the share the forward RETURNS (normalised), not
      // the raw own-density slope divided by it: on a coarse lattice the
      // two parted by 85% and the solve missed by 22 points (#371)
      let dlogp = fwd.ownLogSlope[i];
      const ceil = -1e-3 / (sd[i] + 1e-9);
      if (!(dlogp <= ceil)) dlogp = ceil;           // also catches NaN (#210)
      let d = damp * resid[i] / dlogp;
      if (d > stepCap[i]) d = stepCap[i];
      if (d < -stepCap[i]) d = -stepCap[i];
      delta[i] = d;
    }
    // an observed two-cycle (consecutive steps pointing back) damps, as
    // python's _jacobi_sweeps estimates it from the step pair
    if (prevDelta) {
      let dot = 0, nn = 0;
      for (let i = 0; i < N; i++) { dot += delta[i] * prevDelta[i]; nn += prevDelta[i] * prevDelta[i]; }
      if (nn > 0 && dot / nn < -0.5) damp = Math.max(0.25, damp * 0.7);
    }
    prevDelta = delta;
    let mean = 0;
    for (let i = 0; i < N; i++) { mu[i] -= delta[i]; mean += mu[i]; }
    mean /= N;
    for (let i = 0; i < N; i++) mu[i] -= mean;
  }
  if (!Number.isFinite(res) || mu.some((v) => !Number.isFinite(v)))
    throw new Error(`${where}: the iteration produced a non-finite residual; ` +
                    "the lattice cannot resolve this field (raise points= or narrow the base's spans)");
  return finish(mu, res < tol, res, iters);
}

/* inverse normal cdf (Acklam, refined by one Halley step) */
function ndtri(p) {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969,
             138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887,
             66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
             -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143,
             3.75440866190742];
  const pl = 0.02425;
  let x;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  } else if (p > 1 - pl) {
    return -ndtri(1 - p);
  } else {
    const q = p - 0.5, r2 = q * q;
    x = (((((a[0]*r2+a[1])*r2+a[2])*r2+a[3])*r2+a[4])*r2+a[5])*q /
        (((((b[0]*r2+b[1])*r2+b[2])*r2+b[3])*r2+b[4])*r2+1);
  }
  const e = Math.exp(logndtr(x)) - p;
  const u = e * SQRT_2PI * Math.exp(0.5 * x * x);
  return x - u / (1 + 0.5 * x * u);
}
