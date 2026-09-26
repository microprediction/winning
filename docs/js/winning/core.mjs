// Core numerics for the winning JS port. The python numpy implementation
// is the spec; parity/check.mjs pins this port to its vectors.

export const TINY = 1e-300;

/* ---- normal cdf/log-cdf: series + scaled complementary erf ---------- */
function erfSeries(x) {
  let s = x, t = x;
  for (let n = 1; n < 120; n++) {
    t *= -x * x / n;
    s += t / (2 * n + 1);
    if (Math.abs(t) < 1e-19 * Math.abs(s)) break;
  }
  return (2 / Math.sqrt(Math.PI)) * s;
}
function erfcx(x) {
  // Laplace continued fraction, accurate for x >= 2.5
  let cf = 0;
  for (let k = 60; k >= 1; k--) cf = (k / 2) / (x + cf);
  return 1 / (Math.sqrt(Math.PI) * (x + cf));
}
export function ndtr(z) {
  const x = z / Math.SQRT2;
  if (x >= 2.5) return 1 - 0.5 * erfcx(x) * Math.exp(-x * x);
  if (x <= -2.5) return 0.5 * erfcx(-x) * Math.exp(-x * x);
  return 0.5 * (1 + erfSeries(x));
}
export function logndtr(z) {
  if (z > -3.5355) return Math.log(ndtr(z));
  const x = -z / Math.SQRT2;
  return Math.log(0.5 * erfcx(x)) - x * x;
}
export function npdf(z) {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

/* ---- symmetric tridiagonal eigen (QL, implicit shifts): nodes and
   first-row eigenvector components, for Golub-Welsch ----------------- */
export function tridiagEigen(d, e) {
  const n = d.length;
  const diag = d.slice();
  const off = e.slice(); off.push(0);
  const z = new Array(n).fill(0); z[0] = 1;
  // full first-row tracking needs the rotation applied to a row vector
  const row = new Array(n).fill(0); row[0] = 1;
  for (let l = 0; l < n; l++) {
    let iter = 0;
    let m;
    do {
      for (m = l; m < n - 1; m++) {
        const dd = Math.abs(diag[m]) + Math.abs(diag[m + 1]);
        if (Math.abs(off[m]) <= 1e-16 * dd) break;
      }
      if (m !== l) {
        if (iter++ === 50) throw new Error("tridiagEigen: no convergence");
        let g = (diag[l + 1] - diag[l]) / (2 * off[l]);
        let r = Math.hypot(g, 1);
        g = diag[m] - diag[l] + off[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1, c = 1, p = 0;
        for (let i = m - 1; i >= l; i--) {
          let f = s * off[i], b = c * off[i];
          r = Math.hypot(f, g);
          off[i + 1] = r;
          if (r === 0) { diag[i + 1] -= p; off[m] = 0; break; }
          s = f / r; c = g / r;
          g = diag[i + 1] - p;
          r = (diag[i] - g) * s + 2 * c * b;
          p = s * r;
          diag[i + 1] = g + p;
          g = c * r - b;
          const ri1 = row[i + 1], ri = row[i];
          row[i + 1] = s * ri + c * ri1;
          row[i] = c * ri - s * ri1;
        }
        if (off[l] !== 0 || m - 1 >= l) { diag[l] -= p; off[l] = g; off[m] = 0; }
      }
    } while (m !== l);
  }
  const idx = diag.map((v, i) => i).sort((a, b) => diag[a] - diag[b]);
  return { values: idx.map(i => diag[i]), first: idx.map(i => row[i]) };
}

/* probabilists' Hermite nodes/weights (weights normalized to sum 1),
   matching R's .hermite1 (parity with python hermegauss to fp) */
export function hermite1(order) {
  const d = new Array(order).fill(0);
  const e = [];
  for (let i = 1; i < order; i++) e.push(Math.sqrt(i));
  const { values, first } = tridiagEigen(d, e);
  let w = first.map(v => v * v);
  const s = w.reduce((a, b) => a + b, 0);
  w = w.map(v => v / s);
  return { nodes: values, weights: w };
}

/* pruned product rule: first coordinate slowest, prune WITHOUT
   renormalizing (matching the reference exactly) */
export function hermiteNodes(k, order = 15, prune = 1e-7) {
  const h = hermite1(order);
  if (k === 1) return { F: h.nodes.map(x => [x]), W: h.weights.slice() };
  const F = [], W = [];
  const idx = new Array(k).fill(0);
  const total = Math.pow(order, k);
  for (let t = 0; t < total; t++) {
    let rem = t;
    const node = new Array(k), digits = new Array(k);
    for (let dPos = k - 1; dPos >= 0; dPos--) {   // last coordinate fastest
      digits[dPos] = rem % order;
      rem = Math.floor(rem / order);
    }
    let w = 1;
    for (let dPos = 0; dPos < k; dPos++) {
      node[dPos] = h.nodes[digits[dPos]];
      w *= h.weights[digits[dPos]];
    }
    F.push(node); W.push(w);
  }
  const wmax = Math.max(...W);
  const keepF = [], keepW = [];
  for (let i = 0; i < W.length; i++) {
    if (W[i] > prune * wmax) { keepF.push(F[i]); keepW.push(W[i]); }
  }
  // renormalize after pruning, as the reference does: it drops ~1e-7 of
  // the mass and a direct weighted mixture consumes W as-is. The port
  // omitted this and its weights summed to 1 - 2e-9 (surface audit).
  const wsum = keepW.reduce((a, b) => a + b, 0);
  return { F: keepF, W: keepW.map(w => w / wsum) };
}

/* ---- small dense linear algebra ------------------------------------ */
export function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const p = M[c][c];
    if (Math.abs(p) < 1e-300) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / p;
      for (let cc = c; cc <= n; cc++) M[r][cc] -= f * M[c][cc];
    }
  }
  return M.map((row, i) => (Math.abs(row[i]) > 1e-300 ? row[n] / row[i] : 0));
}

export function mean(v) { return v.reduce((a, b) => a + b, 0) / v.length; }
export function interpClamped(x, xp, fp) {
  // np.interp semantics: ascending xp, end-clamped
  if (x <= xp[0]) return fp[0];
  const last = xp.length - 1;
  if (x >= xp[last]) return fp[last];
  let lo = 0, hi = last;                 // largest j with xp[j] <= x
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  const d = xp[lo + 1] - xp[lo];
  if (d <= 0) return fp[lo];
  return fp[lo] + (x - xp[lo]) / d * (fp[lo + 1] - fp[lo]);
}

/* Caller-supplied factor nodes and their weights.

   `condMeans` dotted each node row against the loadings over the ROW's
   own length, so the rank was whatever each row happened to be. A
   uniformly short F silently priced a LOWER-RANK model (0.46444 where
   the rank-2 answer is 0.38173); a ragged F applied a different rank at
   different quadrature nodes and still returned finite, normalised
   probabilities; extra weights were ignored and too few produced NaN
   (#290). python refuses all of these -- `np.asarray(F, float)` will
   not build an array from ragged rows, and a wrong rank fails the
   matmul against V -- so this is the browser guessing alone.

   Distinct from #232 (the shape of V) and #281 (weight SCALE in the
   standalone js/factor module). */
export function asFactorNodes(F, rank, where = "F") {
  if (!Array.isArray(F))
    throw new Error(`${where} must be an array of factor nodes; got ${typeof F}`);
  if (F.length === 0)
    throw new Error(`${where} is empty; there are no quadrature nodes`);
  const out = new Array(F.length);
  for (let q = 0; q < F.length; q++) {
    const row = Array.isArray(F[q]) ? F[q]
      : (typeof F[q] === "number" ? [F[q]] : null);
    if (row === null)
      throw new Error(`${where}[${q}] must be a node of ${rank} coordinate(s)`);
    if (row.length !== rank)
      throw new Error(
        `${where}[${q}] has ${row.length} coordinate(s) but the loadings ` +
        `have rank ${rank}; a short node prices a lower-rank model and a ` +
        `ragged one prices a different rank at each node`);
    for (let c = 0; c < rank; c++) {
      if (!Number.isFinite(row[c]))
        throw new Error(`${where}[${q}][${c}] = ${row[c]} is not finite`);
    }
    out[q] = Array.from(row, Number);
  }
  return out;
}

/* One weight per node, normalised -- python's `as_weights` at the same
   door. The forward normalises its shares so a rescaling cancels there,
   but the spelling should not matter anywhere, and a mismatched length
   must not reach the kernel. */
export function asWeights(W, nNodes, where = "W") {
  if (!Array.isArray(W) && !ArrayBuffer.isView(W))
    throw new Error(`${where} must be an array of node weights; got ${typeof W}`);
  if (W.length !== nNodes)
    throw new Error(
      `${where} must have one weight per factor node; got ${W.length} ` +
      `for ${nNodes} nodes`);
  let total = 0;
  for (let q = 0; q < W.length; q++) {
    const v = Number(W[q]);
    if (!Number.isFinite(v))
      throw new Error(`${where}[${q}] = ${W[q]} is not a finite weight`);
    if (v < 0)
      throw new Error(`${where}[${q}] = ${v} is a negative weight`);
    total += v;
  }
  if (!(total > 0))
    throw new Error(`${where} must have a positive total; got ${total}`);
  return Array.from(W, v => Number(v) / total);
}

/* ---- options-object guards ---------------------------------------- *
 * A javascript object silently swallows a key nobody reads, so an API
 * whose options are an object cannot rely on the language to reject a
 * misspelling or an option belonging to a different call. That is not
 * hypothetical here: raceProbabilities({cov}) returned the INDEPENDENT
 * race, identical even for an all-zero covariance matrix; rankProbabilities
 * ({V}) returned the independent rank matrix, plausible and doubly
 * stochastic and for the wrong model; locScaleFromTopkPair({V}) reported
 * converged: true where python raises NotImplementedError because the
 * problem is underidentified.
 *
 * Every exported function taking an options object declares its OWN keys.
 * A shared union is not enough: one let each race API accept the other's
 * options and ignore them.
 */
export function checkOpts(opts, known, where, hints = {}) {
  for (const k of Object.keys(opts)) {
    if (known.has(k)) continue;
    if (hints[k]) throw new Error(`${where}: ${hints[k]}`);
    throw new Error(
      `${where}: unknown option '${k}'. Known: ${[...known].join(", ")}.`);
  }
}

/* Reasons that are worth more than "unknown option", shared by the modules
 * that can receive these keys. */
export const OPT_HINTS = {
  cov: "cov= is not supported in the browser port. Fit the covariance " +
       "first -- fitGrammar(C, k, m) in demo.mjs returns {V, D} for this " +
       "call -- or use the python package, whose race_probabilities(cov=) " +
       "also routes a degraded fit to GHK.",
};

/* ---- the loading-shape contract ------------------------------------ *
 * `V` means factor loadings in every public verb and is an (n, rank)
 * matrix, one ROW per contestant. This is the one place that rule is
 * decided, mirroring winning/shapes.py::as_loadings: a scalar, a
 * length-n vector, (n, rank) and (rank, n) are the same race, and
 * anything else raises rather than reaching the kernel.
 *
 * The browser used to index V directly and take the rank from
 * V[0].length, so a length-n vector threw an obscure `V[i].reduce is not
 * a function` and a RAGGED V was silently truncated to the first row's
 * width, producing an all-NaN answer with no complaint (#232).
 */
/* Idiosyncratic VARIANCES as the length-n vector the kernels contract
 * for, mirroring winning/shapes.py::as_idio. A scalar is the same
 * variance for every contestant; a wrong length, a non-finite entry or a
 * negative variance raises here rather than reaching the lattice.
 *
 * The browser normalised V and left D alone, so a scalar threw
 * `D.slice is not a function`, and -- the dangerous one -- a D with ONE
 * EXTRA entry was accepted and returned a normalised, plausible,
 * materially wrong answer: [0.888, 0.102, 0.010] where the race is
 * [0.507, 0.312, 0.181]. Short, zero, negative and NaN entries all
 * propagated NaN through forward and inverse alike (#254).
 *
 * `positive` is for the lattice kernels, which cannot represent an
 * exact zero variance.
 */
export function asIdio(D, n, where = "D", positive = true) {
  if (D == null) return new Array(n).fill(1);
  let v;
  if (typeof D === "number") {
    v = new Array(n).fill(D);
  } else if (Array.isArray(D)) {
    if (D.length !== n)
      throw new Error(
        `${where} must be one idiosyncratic variance per contestant; ` +
        `got ${D.length} for ${n}`);
    v = D.slice();
  } else {
    throw new Error(`${where}: expected a number or array`);
  }
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(v[i]))
      throw new Error(`${where} has a non-finite entry at ${i}`);
    if (v[i] < 0)
      throw new Error(
        `${where}[${i}] = ${v[i]} is a negative variance`);
    if (positive && v[i] === 0)
      throw new Error(
        `${where} must be strictly positive here: entry ${i} is zero, ` +
        "which the lattice cannot represent");
  }
  return v;
}

export function asLoadings(V, n, where = "V") {
  if (V == null) return null;
  if (typeof V === "number") {
    if (!Number.isFinite(V)) throw new Error(`${where}: not a finite number`);
    return Array.from({ length: n }, () => [V]);        // scalar: rank 1
  }
  if (!Array.isArray(V)) throw new Error(`${where}: expected a number or array`);
  if (V.length === 0) throw new Error(`${where}: empty`);
  if (!Array.isArray(V[0])) {                            // a flat vector
    if (V.length !== n)
      throw new Error(
        `${where}: a flat vector must have one entry per contestant; ` +
        `got ${V.length} for ${n}`);
    if (!V.every(Number.isFinite)) throw new Error(`${where}: non-finite entry`);
    return V.map(v => [v]);
  }
  const widths = new Set(V.map(r => (Array.isArray(r) ? r.length : -1)));
  if (widths.has(-1))
    throw new Error(`${where}: mixed rows and scalars`);
  if (widths.size !== 1)
    throw new Error(
      `${where}: ragged -- rows have widths ` +
      `${[...widths].sort((a, b) => a - b).join(", ")}; every contestant ` +
      "needs the same number of factors");
  const w = [...widths][0];
  if (w === 0) throw new Error(`${where}: rows are empty`);
  let M = V;
  if (V.length !== n) {
    if (w !== n)
      throw new Error(
        `${where}: shape ${V.length}x${w} matches neither (n, rank) nor ` +
        `(rank, n) for n = ${n}`);
    M = Array.from({ length: n }, (_, i) => V.map(row => row[i]));  // (rank,n)
  }
  if (!M.every(r => r.every(Number.isFinite)))
    throw new Error(`${where}: non-finite entry`);
  return M;
}

/* First d primes, generated. Literal tables put silent cliffs in the
 * Halton constructions: 16 in demo.mjs and 24 here, past which
 * `primes[dim]` is undefined, the radical inverse becomes NaN, and the
 * whole probability vector comes back NaN (#233). The same tabulated
 * cliff was in the R GHK at 30 (#190). */
export function firstPrimes(d) {
  const out = [];
  for (let c = 2; out.length < d; c++) {
    let isP = true;
    for (const q of out) {
      if (q * q > c) break;
      if (c % q === 0) { isP = false; break; }
    }
    if (isP) out.push(c);
  }
  return out;
}
