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
  // `new Array("15")` is a ONE-element array, so an order read from a
  // form control (a string) silently became the one-node rule, node 0
  // weight 1: the factor integral vanished and finite, normalised
  // prices came back for a different model. null and booleans did the
  // same (#442). The order is a count; it is checked as one here, at
  // the one place every quadrature caller reaches.
  order = asCount(order, "order", 1, "hermite1");
  const d = new Array(order).fill(0);
  const e = [];
  for (let i = 1; i < order; i++) e.push(Math.sqrt(i));
  const { values, first } = tridiagEigen(d, e);
  let w = first.map(v => v * v);
  const s = w.reduce((a, b) => a + b, 0);
  w = w.map(v => v / s);
  return { nodes: values, weights: w };
}

/* Pruned product Gauss-Hermite rule, built ONE DIMENSION AT A TIME.

   This used to materialise the whole `order ** k` tensor and prune it
   afterwards, which is the defect python closed in #155 and the browser
   kept. Two ways it fails. `Math.max(...W)` spreads every weight as an
   argument list, so rank 5 at order 15 -- 759,375 nodes -- died with
   `RangeError: Maximum call stack size exceeded` before pruning ever
   ran. And rewriting that line as a loop only moves the wall: rank 5 at
   order 41 is 115,856,201 nodes to build and then throw away.

   A partial product whose weight cannot reach the threshold whatever
   the remaining factors contribute -- each at most `wmax` -- cannot be
   in the answer, so it is dropped as soon as it appears. The kept set
   and its ORDER are exactly those of the full tensor pruned once: the
   test for a partial product is implied by the test for every full
   product extending it, and the loops below extend in the same order
   the tensor enumerated (first coordinate slowest, last fastest).

   Weights are renormalised at the end, as the reference does: pruning
   drops ~1e-7 of the mass and a direct weighted mixture consumes W
   as-is. (#268)

   k = 0 is the EMPTY PRODUCT -- one node of weight 1 with no columns --
   so a zero-rank loading matrix integrates over the zero-dimensional
   factor space and prices the independent race through this same path.
   The guard here used to demand k >= 1, which refused it; python, R and
   julia all answer it, so refusing made this port the odd one out
   (#68). */
export function asCount(x, name, minimum, who = "hermiteNodes") {
  // Number.isInteger is false for "15", true and null, so a string from
  // a form control is refused rather than coerced (#442)
  if (!Number.isInteger(x))
    throw new Error(`${who} needs an integer ${name}; got ${typeof x === "string" ? JSON.stringify(x) : x}`);
  if (x < minimum)
    throw new Error(`${who} needs ${name} >= ${minimum}; got ${x}`);
  return x;
}

/* An iteration budget is a finite positive whole number of sweeps.

   Every browser solver used its budget as a bare loop bound, so 0, a
   negative number and NaN skipped the solve and returned the warm start
   (or, in the classic solver, the target PROBABILITIES as if they were
   offsets) as an ordinary answer; a fraction ran Math.ceil of itself;
   and Infinity removed the only hard bound, so a stalled iteration
   froze the page (#401, #413, #421, #428). Refused here, before any
   lattice work. */
export function asIterations(n, where, name = "nIter") {
  if (!Number.isInteger(n) || n < 1)
    throw new Error(
      `${where}: ${name} must be a finite positive integer number of ` +
      `iterations; got ${typeof n === "string" ? JSON.stringify(n) : n}. ` +
      "A zero, negative or NaN budget skips the solve and returns the " +
      "warm start as if it were an answer, and Infinity can hang the page.");
  return n;
}

/* A convergence tolerance is a finite positive number. */
export function asTolerance(tol, where, name = "tol") {
  if (typeof tol !== "number" || !Number.isFinite(tol) || !(tol > 0))
    throw new Error(`${where}: ${name} must be a finite positive number; got ${tol}`);
  return tol;
}

/* A numeric vector, plain or typed, as a plain Array.

   A Float64Array's .map keeps the typed container and coerces whatever
   the callback returns to a number, so `mu.map(() => [0])` -- the
   default loadings -- became a flat vector of NaN and the race failed
   with "F[0] has 1 coordinate(s) but the loadings have rank undefined"
   (#334); the classic lattice crashed the same way mapping offsets to
   CDF arrays (#329). Normalising once at each door makes a typed vector
   and a plain one the same input, as numpy arrays and lists are in
   python. */
export const isVector = x => Array.isArray(x) || (ArrayBuffer.isView(x) && !(x instanceof DataView));

export function asFiniteVector(x, where, what = "entry", { allowEmpty = false } = {}) {
  if (!isVector(x))
    throw new Error(`${where} must be an array of numbers; got ${x === null ? "null" : typeof x}`);
  if (x.length === 0 && !allowEmpty)
    throw new Error(`${where} is empty`);
  for (let i = 0; i < x.length; i++) {
    // typeof first: Number.isFinite already refuses strings, null and
    // booleans, but the message should say what arrived
    if (typeof x[i] !== "number" || !Number.isFinite(x[i]))
      throw new Error(
        `${where}[${i}] = ${typeof x[i] === "string" ? JSON.stringify(x[i]) : x[i]} ` +
        `is not a finite ${what}`);
  }
  return Array.from(x, Number);
}

/* Abilities: a nonempty vector of finite locations. The winner race
   checked this at its door and the top-k/rank family did not, so a NaN
   threw RangeError: Invalid array length, an Infinity reported a NaN
   "mass defect" with advice to raise points=, and topKJacobians
   returned two all-NaN matrices without complaint (#440). */
export function asAbilities(mu, where = "mu") {
  if (!isVector(mu))
    throw new Error(`${where} must be an array of abilities; got ${typeof mu}`);
  if (mu.length === 0)
    throw new Error(`${where} is empty; a race needs at least one contestant`);
  for (let i = 0; i < mu.length; i++) {
    if (typeof mu[i] !== "number" || !Number.isFinite(mu[i]))
      throw new Error(
        `${where}[${i}] = ${mu[i]} is not finite; an ability is a finite ` +
        `location on the performance scale`);
  }
  return Array.from(mu, Number);
}

export function hermiteNodes(k, order = 15, prune = 1e-7) {
  k = asCount(k, "k", 0);
  order = asCount(order, "order", 1);
  if (k === 0) return { F: [[]], W: [1] };
  const h = hermite1(order);
  if (k === 1) return { F: h.nodes.map(x => [x]), W: h.weights.slice() };

  // one-dimensional max, so this is `order` values and never the tensor
  let wmax = h.weights[0];
  for (let i = 1; i < order; i++) if (h.weights[i] > wmax) wmax = h.weights[i];
  const floor = prune * Math.pow(wmax, k);

  let F = h.nodes.map(x => [x]);
  let W = h.weights.slice();
  for (let d = 1; d < k; d++) {
    // what the remaining k - 1 - d coordinates can still contribute
    const reach = Math.pow(wmax, k - 1 - d);
    const nF = [], nW = [];
    for (let i = 0; i < F.length; i++) {
      const row = F[i], wi = W[i];
      for (let j = 0; j < order; j++) {
        const w = wi * h.weights[j];
        if (w * reach > floor) { nF.push([...row, h.nodes[j]]); nW.push(w); }
      }
    }
    F = nF; W = nW;
  }
  const wsum = W.reduce((a, b) => a + b, 0);
  return { F, W: W.map(w => w / wsum) };
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
  if (!isVector(F))
    throw new Error(`${where} must be an array of factor nodes; got ${typeof F}`);
  if (F.length === 0)
    throw new Error(`${where} is empty; there are no quadrature nodes`);
  const out = new Array(F.length);
  for (let q = 0; q < F.length; q++) {
    const row = isVector(F[q]) ? F[q]
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
  if (!isVector(W))
    throw new Error(`${where} must be an array of node weights; got ${typeof W}`);
  if (W.length !== nNodes)
    throw new Error(
      `${where} must have one weight per factor node; got ${W.length} ` +
      `for ${nNodes} nodes`);
  let top = 0;
  for (let q = 0; q < W.length; q++) {
    const v = W[q];
    if (typeof v !== "number" || !Number.isFinite(v))
      throw new Error(`${where}[${q}] = ${W[q]} is not a finite weight`);
    if (v < 0)
      throw new Error(`${where}[${q}] = ${v} is a negative weight`);
    if (v > top) top = v;
  }
  if (!(top > 0))
    throw new Error(`${where} must have a positive total; got 0`);
  // through the LARGEST weight, never the raw total: [1e308, 1e308]
  // overflows only in the sum, and W / Infinity made every weight zero
  // and every probability NaN (#415)
  const u = Array.from(W, v => Number(v) / top);
  const total = u.reduce((a, b) => a + b, 0);
  return u.map(v => v / total);
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
  } else if (isVector(D)) {          // a typed D is the same input (#334)
    if (D.length !== n)
      throw new Error(
        `${where} must be one idiosyncratic variance per contestant; ` +
        `got ${D.length} for ${n}`);
    v = Array.from(D);
  } else {
    throw new Error(`${where}: expected a number or array`);
  }
  for (let i = 0; i < n; i++) {
    if (typeof v[i] !== "number" || !Number.isFinite(v[i]))
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
  if (!isVector(V)) throw new Error(`${where}: expected a number or array`);
  if (V.length === 0) throw new Error(`${where}: empty`);
  // typed vectors and typed rows are the same input as plain ones (#334)
  if (!Array.isArray(V) || !isVector(V[0])) {            // a flat vector
    if (V.length !== n)
      throw new Error(
        `${where}: a flat vector must have one entry per contestant; ` +
        `got ${V.length} for ${n}`);
    const flat = Array.from(V);
    if (flat.some(isVector)) throw new Error(`${where}: mixed rows and scalars`);
    if (!flat.every(v => typeof v === "number" && Number.isFinite(v)))
      throw new Error(`${where}: non-finite entry`);
    return flat.map(v => [v]);
  }
  V = V.map(r => (isVector(r) ? Array.from(r) : r));
  const widths = new Set(V.map(r => (Array.isArray(r) ? r.length : -1)));
  if (widths.has(-1))
    throw new Error(`${where}: mixed rows and scalars`);
  if (widths.size !== 1)
    throw new Error(
      `${where}: ragged -- rows have widths ` +
      `${[...widths].sort((a, b) => a - b).join(", ")}; every contestant ` +
      "needs the same number of factors");
  const w = [...widths][0];
  // w = 0 is an (n, 0) loading matrix: no factors, which the reference
  // accepts and prices as the independent race. Refusing it here made
  // this port the only one of four that could not express "no factors"
  // programmatically -- and the guard arrived in the commit whose whole
  // purpose was to match the reference (#68). A zero-width V with the
  // WRONG row count still fails the shape check below.
  let M = V;
  if (V.length !== n) {
    if (w !== n)
      throw new Error(
        `${where}: shape ${V.length}x${w} matches neither (n, rank) nor ` +
        `(rank, n) for n = ${n}`);
    M = Array.from({ length: n }, (_, i) => V.map(row => row[i]));  // (rank,n)
  }
  if (!M.every(r => r.every(v => typeof v === "number" && Number.isFinite(v))))
    throw new Error(`${where}: non-finite entry`);
  return M.map(r => r.slice());
}

/* Gauge-fix a loading matrix: subtract each factor's mean loading
 * across contestants. A common loading column c adds the same c'f to
 * every performance and cannot move an argmin, so PV prices the
 * IDENTICAL race -- and unlike V it makes every downstream decision
 * (node family, node order, lattice window) invariant under
 * V -> V + 1c'. One place, because two modules needed it and the one
 * that had it privately (the standalone copy, #139) did not stop this
 * one shipping without it (#303).
 */
export function gaugeCenter(V) {
  if (!V || !V.length) return V;
  const n = V.length, r = V[0].length;
  const colMean = new Array(r).fill(0);
  for (const row of V) for (let j = 0; j < r; j++) colMean[j] += row[j] / n;
  return V.map(row => row.map((x, j) => x - colMean[j]));
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

/* Own-slope-preconditioned Jacobi sweeps on the mean-zero quotient,
   port of python's races._jacobi_sweeps (see its docstring for the
   measurements behind each rule): Richardson damping estimated from
   consecutive steps and held persistently, a transient penalty after a
   sweep that contracts neither the max nor the rms residual, and Aitken
   summation of a geometrically decaying monotone mode.

   `forward(mu)` returns {resid, dres}: residual (model minus target) and
   own-slopes, negative and bounded away from zero. Returns {mu,
   converged, residMax, iterations}. The top-k inverse kept the
   original fixed `n > 2 ? 1 : 0.7` damping after python moved to this,
   so a field with two live contenders and a tail two-cycled to the
   iteration limit and repriced 3.8 points wrong (#408). */
export function jacobiSweeps(mu, forward, scale, alpha, nIter, tol) {
  let residMax = Infinity, residRms = Infinity, iters = 0;
  let prev = null, prevStep = null;
  let alphaBase = alpha, penalty = 1;
  const n = mu.length;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    let { resid, dres } = forward(mu);
    residMax = Math.max(...resid.map(Math.abs));
    residRms = Math.sqrt(resid.reduce((a, b) => a + b * b, 0) / n);
    if (residMax < tol) break;
    if (prev && residMax >= prev.residMax && residRms >= prev.residRms) {
      if (penalty > 0.1) {
        penalty = Math.max(0.5 * penalty, 0.1);
        ({ mu, resid, dres, residMax, residRms } = prev);
        prevStep = null;
      }
    } else if (prev && penalty < 1) {
      penalty = Math.min(1, penalty / 0.75);
    }
    prev = { mu, resid, dres, residMax, residRms };
    const a = alphaBase * penalty;
    let step = resid.map((v, i) => {
      const lim = Math.min(2, 10 * Math.abs(v)) * scale;
      return Math.min(Math.max(a * v / dres[i], -lim), lim);
    });
    const sm = step.reduce((x, y) => x + y, 0) / n;
    step = step.map(v => v - sm);
    if (prevStep) {
      const na = Math.sqrt(prevStep.reduce((x, y) => x + y * y, 0));
      const nb = Math.sqrt(step.reduce((x, y) => x + y * y, 0));
      if (na > 0) {
        const dot = step.reduce((x, y, i) => x + y * prevStep[i], 0);
        const rho = dot / (na * na);
        const cosn = nb > 0 ? dot / (na * nb) : 0;
        const ratio = nb / na;
        if (rho < 0) {
          const lam = 1 - (1 - rho) / a;
          alphaBase = Math.min(Math.max(2 / (2 - lam), 0.1), 1);
        } else if (cosn > 0.999 && ratio > 0.5 && ratio < 0.999 && residMax > 1e3 * tol) {
          mu = mu.map((m, i) => m - step[i] / (1 - ratio));
          prevStep = null;
          continue;
        }
      }
    }
    prevStep = step;
    mu = mu.map((m, i) => m - step[i]);
  }
  return { mu, converged: residMax < tol, residMax, iterations: iters };
}
