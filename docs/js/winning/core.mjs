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
  if (z === Infinity) return 0;
  if (z === -Infinity) return -Infinity;
  // the reflected lower tail: log(ndtr(z)) formed 1 - Phi(-z) and was
  // exactly 0 from z ~ 8.3, though log Phi(z) is representable to z ~ 38
  // (#696); this matches js/factor's logndtr and scipy's log_ndtr
  if (z >= 1) return Math.log1p(-Math.exp(logndtr(-z)));
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

/* A lattice size: a whole number of at least two points (python's
   shapes.as_points). Every lattice divides by points - 1, so 0, 1, null
   and true reached the race grid and came back all NaN, -1 and 2.5 as
   "Invalid array length", and the top-k grid silently replaced them with
   its own adaptive size (#444). */
export function asPoints(points, name = "points", minimum = 2) {
  if (typeof points !== "number" || !Number.isFinite(points) || !Number.isInteger(points))
    throw new Error(`${name} must be a whole number of lattice points; got ` +
                    `${typeof points === "string" ? JSON.stringify(points) : String(points)}`);
  if (points < minimum)
    throw new Error(`${name} must be at least ${minimum}: a lattice of ${points} point` +
                    `${points === 1 ? "" : "s"} has no spacing to integrate on`);
  return points;
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

/* Math.max / Math.min of an array without spreading it: a spread passes
   every entry as an argument and overflows the call stack around 1.3e5
   runners (#747, #461). Same result as the spread, including NaN
   propagation and max(-0, 0) = 0. */
export function maxOf(a) {
  let m = -Infinity;
  for (const v of a) {
    if (v !== v) return NaN;
    if (v > m || (v === 0 && m === 0 && Object.is(m, -0))) m = v;
  }
  return m;
}
export function minOf(a) {
  let m = Infinity;
  for (const v of a) {
    if (v !== v) return NaN;
    if (v < m || (v === 0 && m === 0 && Object.is(v, -0))) m = v;
  }
  return m;
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

/* Callable-base metadata vectors, through the same door as every other
   numeric input: a plain Array OR a typed array of finite numbers.
   Array.isArray alone read a Float64Array([25, 8]) span as "no span" and
   fell back to [12, 12], truncating a rare mode's integrand so a 0.005
   win price came back 5e-324; a Float64Array([0]) kink list was read as
   a smooth base and the #267 split-lattice repair never ran (#907).
   Absent metadata is null; present-but-malformed metadata throws here,
   before any grid is built, rather than becoming a silent fallback. */
export function baseSpan(fn, where = "base.span") {
  const s = typeof fn === "function" ? fn.span : undefined;
  if (s === undefined || s === null) return null;
  if (!isVector(s) || s.length !== 2)
    throw new Error(`${where} must be [left, right], two positive finite numbers of standard ` +
      `units; got ${isVector(s) ? `length ${s.length}` : typeof s}`);
  const out = [Number(s[0]), Number(s[1])];
  for (const v of out)
    if (!Number.isFinite(v) || !(v > 0))
      throw new Error(`${where} must be [left, right], two positive finite numbers; got ${v}`);
  return out;
}
export function baseKinks(fn, where = "base.kinks") {
  const k = typeof fn === "function" ? fn.kinks : undefined;
  if (k === undefined || k === null) return null;
  if (!isVector(k))
    throw new Error(`${where} must be an array of finite kink locations in standard units; got ${typeof k}`);
  const out = Array.from(k, Number);
  for (let i = 0; i < out.length; i++)
    if (!Number.isFinite(out[i]))
      throw new Error(`${where}[${i}] must be a finite kink location; got ${k[i]}`);
  return out.length ? out : null;
}

const AS_FINITE_VECTOR_OPTS = new Set(["allowEmpty"]);
export function asFiniteVector(x, where, what = "entry", opts = {}) {
  checkOpts(opts, AS_FINITE_VECTOR_OPTS, "asFiniteVector");
  const { allowEmpty = false } = opts;
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
  // np.interp semantics: ascending xp, end-clamped. Only x BELOW xp[0]
  // clamps: at a duplicated lower knot the search takes the last of the
  // plateau as numpy does, where this returned the first and sent a
  // zero-priced classic runner across the field (#842)
  if (x < xp[0]) return fp[0];
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

/* A quadrature rule with its null nodes removed -- python's
   shapes.as_factor_law. Nodes of weight zero, and the lightest nodes
   whose COMBINED relative mass is at most 1e-15, change no share at
   double precision, but every lattice sizes its window from all the
   nodes it is handed: a zero (or 1e-300) atom at f = 1000 widened the
   window until the real density spikes fell between lattice points
   (0.69 against 0.54 on one share), and padded block nodes threw
   "captured total mass 0" (#416). */
export function asFactorLaw(F, W, rank, whereF = "F", whereW = "W") {
  const Fv = asFactorNodes(F, rank, whereF);
  const Wn = asWeights(W, Fv.length, whereW);
  const order = Wn.map((_, q) => q).sort((a, b) => Wn[a] - Wn[b] || a - b);
  const drop = new Array(Wn.length).fill(false);
  let cum = 0;
  for (const q of order) {
    cum += Wn[q];
    if (cum > 1e-15) break;
    drop[q] = true;
  }
  if (!drop.some(Boolean)) return { F: Fv, W: Wn };
  const keepF = [], keepW = [];
  let tot = 0;
  for (let q = 0; q < Wn.length; q++)
    if (!drop[q]) { keepF.push(Fv[q]); keepW.push(Wn[q]); tot += Wn[q]; }
  return { F: keepF, W: keepW.map(v => v / tot) };
}

/* A finite nonnegative vector normalised to total `mass`, scale-safely.
   A relative target is a law up to a positive factor, but summing first
   overflows when every ENTRY is finite: [0.8, 0.6, 0.4, 0.2] * 1e308 has
   an infinite total, and v / Infinity zeroed the target before any
   solver ran (#461, #463, #483, #484). Rescale by the max only when the
   sum overflows, so ordinary inputs stay bit-identical (as #326) -- or
   when mass / sum does: a finite sum below ~1/MAX_VALUE (4e-309 * [0.2,
   0.3, 0.5]) made every entry Infinity (#461 residual). */
export function rescaledTarget(v, mass = 1) {
  let u = Array.from(v);
  let s = u.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(s) || !Number.isFinite(mass / s)) {
    // a loop, not Math.max(...u): a runner-sized spread passes the
    // engine's argument limit and threw RangeError at 150k (#461)
    let mx = -Infinity;
    for (let i = 0; i < u.length; i++) mx = Math.max(mx, Math.abs(u[i]));
    u = u.map(x => x / mx);
    s = u.reduce((a, b) => a + b, 0);
  }
  return u.map(x => x * (mass / s));
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
  // centred on the first row, so an exactly common column is exactly
  // zero: summing row / n left -2e292 of a [1e308] * 9 column (and a
  // residue at 1e20) as a huge loading, and the race went uniform (#471)
  const v0 = V[0];
  const colMean = v0.slice();
  for (const row of V) for (let j = 0; j < r; j++) colMean[j] += (row[j] - v0[j]) / n;
  const out = V.map(row => row.map((x, j) => x - colMean[j]));
  // a column whose differences overflow ([1, 1, -1] * 1e308: row - v0 is
  // -inf) is centred in units of its largest entry, (V/s - mean) * s, as
  // python's _gauge_center does (#471)
  for (let j = 0; j < r; j++) {
    if (out.every(row => Number.isFinite(row[j]))) continue;
    let s = 0;
    for (const row of V) s = Math.max(s, Math.abs(row[j]));
    let m = 0;
    for (const row of V) m += row[j] / s / n;
    for (let i = 0; i < n; i++) out[i][j] = (V[i][j] / s - m) * s;
  }
  return out;
}

/* A representation of a Gaussian factor law that depends on V V' alone
 * (port of python's ratings _canonical_loadings). Gaussian factors
 * enter only through V V': a sign flip, a column permutation, an
 * orthogonal rotation or a zero column is the SAME race. A fixed node
 * cloud is not invariant under those maps, so moving two loading
 * columns from positions 0, 1 to 68, 69 of an 80-column matrix moved a
 * share by 0.078 through the plain-Halton projections, and a column
 * swap in a block moved one by 0.234 (#130).
 *
 * One-sided Jacobi (Hestenes) on the columns: V Q = C with Q orthogonal,
 * C's columns orthogonal, sorted by norm, null directions (norm <=
 * rtol x the largest) dropped, each sign set by the column's sum of
 * cubes (a permutation of contestants does not change it). Returns
 * { C (n x r'), Q (r x r') } with V Q = C, so a node f' for C is the
 * node Q f' for V. Rotations within a repeated singular value cannot be
 * resolved, as in python. */
export function canonicalBasis(V, rtol = 1e-12) {
  const n = V.length, r = n ? V[0].length : 0;
  // a * b and the cubes overflowed for finite loadings near 1e80, so no
  // rotation ran and V, VQ got different node rules (#130 residual):
  // work on an exact power-of-two rescale outside 2^+-100, undo on C
  let amax = 0;
  for (const row of V) for (const v of row) amax = Math.max(amax, Math.abs(v));
  if (amax > 0 && Number.isFinite(amax) && (amax > 2 ** 100 || amax < 2 ** -100)) {
    const e = Math.max(-1022, Math.min(1022, -Math.round(Math.log2(amax))));
    const out = canonicalBasis(V.map(row => row.map(v => v * 2 ** e)), rtol);
    const h = 2 ** (-e / 2);   // two steps: 2^-e alone can overflow
    return { C: out.C.map(row => row.map(v => v * h * h)), Q: out.Q };
  }
  const A = V.map(row => row.slice());
  const Q = Array.from({ length: r }, (_, i) => Array.from({ length: r }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let p = 0; p < r - 1; p++) for (let q = p + 1; q < r; q++) {
      let a = 0, b = 0, g = 0;
      for (let i = 0; i < n; i++) { const x = A[i][p], y = A[i][q]; a += x * x; b += y * y; g += x * y; }
      if (!(Math.abs(g) > 1e-15 * Math.sqrt(a * b)) || g === 0) continue;
      rotated = true;
      const zeta = (b - a) / (2 * g);
      const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.hypot(1, zeta));
      const c = 1 / Math.hypot(1, t), s = c * t;
      for (let i = 0; i < n; i++) {
        const x = A[i][p], y = A[i][q];
        A[i][p] = c * x - s * y; A[i][q] = s * x + c * y;
      }
      for (let i = 0; i < r; i++) {
        const x = Q[i][p], y = Q[i][q];
        Q[i][p] = c * x - s * y; Q[i][q] = s * x + c * y;
      }
    }
    if (!rotated) break;
  }
  const norm = Array.from({ length: r }, (_, j) => Math.hypot(...A.map(row => row[j])));
  const top = Math.max(0, ...norm);
  const keep = [...Array(r).keys()]
    .filter(j => norm[j] > rtol * top && norm[j] > 0)
    .sort((x, y) => norm[y] - norm[x] || x - y);
  const C = A.map(() => []), Qk = Q.map(() => []);
  for (const j of keep) {
    let cube = 0, abs3 = 0, big = 0;
    for (let i = 0; i < n; i++) {
      const v = A[i][j];
      cube += v * v * v; abs3 += Math.abs(v) ** 3;
      if (Math.abs(v) > Math.abs(big)) big = v;
    }
    // cube-symmetric column: the largest entry decides, as python
    if (Math.abs(cube) <= 1e-12 * abs3) cube = big;
    const sg = cube < 0 ? -1 : 1;
    for (let i = 0; i < n; i++) C[i].push(sg * A[i][j]);
    for (let i = 0; i < r; i++) Qk[i].push(sg * Q[i][j]);
  }
  return { C, Q: Qk };
}

/* First d primes, generated. Literal tables put silent cliffs in the
 * Halton constructions: 16 in demo.mjs and 24 here, past which
 * `primes[dim]` is undefined, the radical inverse becomes NaN, and the
 * whole probability vector comes back NaN (#233). The same tabulated
 * cliff was in the R GHK at 30 (#190). */
export function firstPrimes(d) {
  // a count: NaN and negatives returned [] (a rank-zero rule), 2.5 three
  // primes, and Infinity never returned (#557)
  asCount(d, "count of primes", 0, "firstPrimes");
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
/* The Aitken-extrapolated iterate as a bounded, checked trial (#583),
   port of python's races._aitken_trial: the summed tail step / (1 -
   ratio), capped in the max norm at twice the field scale (never below
   the ordinary step), and repriced. Returns {mu, fwd}, or null -- take
   the ordinary step -- when the forward throws, is non-finite, or its
   max residual grows tenfold; milder rises are left to the sweeps'
   backtracking (requiring contraction stalls the dense n = 30
   correlation, whose residual rises while its near-duplicate mode is
   summed). Unchecked, an exact normal top-1 target at residual 20 with
   ratio 0.998 jumped 539x to abilities in [-894, 249] and ended at
   residual 715 after 120 sweeps. `fwdKeys` names the residual and
   slope fields of the forward's result. */
export function aitkenTrial(mu, step, ratio, scale, forward, residMax,
                            fwdKeys = ["resid", "dres"]) {
  let jump = step.map(v => v / (1 - ratio));
  const big = maxOf(jump.map(Math.abs));
  const cap = Math.max(2 * scale, maxOf(step.map(Math.abs)));
  if (big > cap) jump = jump.map(v => v * (cap / big));
  const cand = mu.map((m, i) => m - jump[i]);
  let fwd;
  try { fwd = forward(cand); } catch (e) { return null; }
  const rc = fwd[fwdKeys[0]], dc = fwd[fwdKeys[1]];
  if (!rc.every(Number.isFinite) || !dc.every(Number.isFinite)) return null;
  if (maxOf(rc.map(Math.abs)) > 10 * residMax) return null;
  return { mu: cand, fwd };
}

/* Gauss-Legendre nodes and weights on [-1, 1] (Newton on P_m), cached.
   Python's numpy leggauss; agreement to rounding. */
const LEGENDRE = new Map();
export function legendre(m) {
  if (LEGENDRE.has(m)) return LEGENDRE.get(m);
  const t = new Array(m), w = new Array(m);
  for (let i = 0; i < m; i++) {
    let x = Math.cos(Math.PI * (i + 0.75) / (m + 0.5));
    let dp = 1;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = x;
      for (let j = 2; j <= m; j++) {
        const p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
        p0 = p1; p1 = p2;
      }
      if (m === 1) { p1 = x; p0 = 1; }
      dp = m * (x * p1 - p0) / (x * x - 1);
      const dx = p1 / dp;
      x -= dx;
      if (Math.abs(dx) < 1e-15) break;
    }
    t[m - 1 - i] = x;
    w[m - 1 - i] = 2 / ((1 - x * x) * dp * dp);
  }
  const out = { t, w };
  LEGENDRE.set(m, out);
  return out;
}

/* Nodes and weights over the window of the uniform lattice x, split at
   every runner's kink and Gauss-Legendre on each piece: python's
   races.kink_lattice, same budget rule (the two outer pieces at a
   quarter weight, at least four nodes a piece, at most len(x) / 4 kinks
   kept, merged onto equal-width bins beyond that). A kinked base
   (laplace) makes the slope integrand -f'(z) JUMP at every runner's
   location, so a uniform rectangle sum of a slope or Jacobian is only
   first order there; on these pieces it is spectral (#267). Used for the
   inverses' derivatives only: the forward maps keep the uniform lattice. */
export function kinkLattice(x, M, sd, kinks) {
  const a = x[0], b = x[x.length - 1], L = x.length;
  let K = [];
  for (let i = 0; i < M.length; i++)
    for (const c of kinks) {
      const v = M[i] + c * sd[i];
      if (v > a && v < b) K.push(v);
    }
  K = Array.from(new Set(K)).sort((u, v) => u - v);
  const cap = Math.max(1, Math.floor(L / 4));
  if (K.length > cap) {
    const width = (b - a) / cap;
    const seen = new Set(), kept = [];
    for (const v of K) {
      const bin = Math.floor((v - a) / width);
      if (!seen.has(bin)) { seen.add(bin); kept.push(v); }
    }
    K = kept;
  }
  const cuts = [a, ...K, b];
  const P = cuts.length - 1;
  const eff = new Array(P);
  let tot = 0;
  for (let j = 0; j < P; j++) {
    eff[j] = (cuts[j + 1] - cuts[j]) * (j === 0 || j === P - 1 ? 0.25 : 1);
    tot += eff[j];
  }
  const xs = [], ws = [];
  for (let j = 0; j < P; j++) {
    const m = Math.max(4, Math.round(L * eff[j] / tot));
    const { t, w } = legendre(m);
    const mid = 0.5 * (cuts[j + 1] + cuts[j]), half = 0.5 * (cuts[j + 1] - cuts[j]);
    for (let r = 0; r < m; r++) { xs.push(mid + half * t[r]); ws.push(half * w[r]); }
  }
  return { xs, ws };
}

/* A target about to be floored, as a law of total `mass`: the floor is a
   probability, so it applies to normalized entries, never the caller's
   arbitrary units -- [c, 0] floored at 1e-6 returned ability gaps of 0,
   4.75 and 7.03 for c = 1e-6, 1, 1e6 (#592). Port of python's
   races._floorable; dividing by the max first keeps the sum finite. */
export function floorableTarget(target, mass) {
  if (target.some(v => v < 0))
    throw new Error("target entries must be non-negative: targetFloor " +
                    "floors small and zero shares, not negative ones");
  const mx = maxOf(target);
  const t = target.map(v => v / mx);
  const s = t.reduce((a, b) => a + b, 0);
  return t.map(v => v * (mass / s));
}

/* Floor a target at `floor` IN ITS FINAL NORMALISED SCALE (python
   races._apply_floor, #592): normalise to `mass`, raise the entries below
   the floor to exactly the floor and take that mass proportionally from
   the others, repeating if that pushes another entry under. Clamping and
   then renormalising scaled the floor itself back down ([.7, .3, 0] at
   floor .2 repriced the floored runner at 0.167). An infeasible floor
   (no mass left for the unfloored entries) throws. */
export function applyFloor(target, mass, floor) {
  const t = floorableTarget(target, mass);
  let floored = t.map(v => v < floor);
  let scaled = t;
  for (let it = 0; it < t.length; it++) {
    const m = floored.filter(Boolean).length;
    const free = mass - floor * m;
    const rest = t.reduce((a, v, i) => a + (floored[i] ? 0 : v), 0);
    if (!(free > 0 && rest > 0))
      throw new Error(`targetFloor=${floor} is infeasible: flooring ${m} of ${t.length} ` +
                      `entries at it leaves no mass for the rest of a target of total ${mass}`);
    scaled = t.map((v, i) => (floored[i] ? floor : v * (free / rest)));
    const newly = scaled.map((v, i) => !floored[i] && v < floor);
    if (!newly.some(Boolean)) break;
    floored = floored.map((f, i) => f || newly[i]);
  }
  return { target: scaled, floored };
}

export function jacobiSweeps(mu, forward, scale, alpha, nIter, tol) {
  let residMax = Infinity, residRms = Infinity, iters = 0;
  let prev = null, prevStep = null;
  let alphaBase = alpha, penalty = 1;
  let cached = null;  // forward of an accepted Aitken trial
  const n = mu.length;
  let backoffs = 0;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    let fw = cached;
    if (!fw) {
      try {
        fw = forward(mu);
      } catch (e) {
        // the forward refused this iterate (a lattice defect, or a
        // membership above one): back off halfway toward the last priced
        // iterate, as python's _jacobi_sweeps (#267). The first forward
        // still throws; a persistent refusal returns the last priced
        // iterate as not converged.
        if (!prev) throw e;
        if (backoffs >= 20)
          return { mu: prev.mu, converged: prev.residMax < tol,
                   residMax: prev.residMax, iterations: iters };
        backoffs += 1;
        mu = prev.mu.map((m, i) => m + 0.5 * (mu[i] - m));
        penalty = Math.max(0.5 * penalty, 0.1);
        prevStep = null;
        continue;
      }
    }
    let { resid, dres } = fw;
    cached = null;
    residMax = maxOf(resid.map(Math.abs));
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
          const trial = aitkenTrial(mu, step, ratio, scale, forward, residMax);
          if (trial) {  // sum the geometric tail
            mu = trial.mu;
            cached = trial.fwd;
            prevStep = null;
            continue;
          }
        }
      }
    }
    prevStep = step;
    mu = mu.map((m, i) => m - step[i]);
  }
  if (!(residMax < tol)) {
    // exhausted after a step no sweep priced: price the returned iterate
    // (one forward, only here) and keep the better of the two, so the
    // diagnostics describe the mu returned (#497, port of python)
    // A refusal here (the last step crossed the lattice/membership guard)
    // returns the last priced iterate as not converged, as the in-loop
    // backoff does; with nothing priced it still throws (#823).
    let fwd = cached;
    if (!fwd) {
      try {
        fwd = forward(mu);
      } catch (e) {
        if (!prev) throw e;
        return { mu: prev.mu, converged: false,
                 residMax: prev.residMax, iterations: iters };
      }
    }
    const rEnd = maxOf(fwd.resid.map(Math.abs));
    if (prev && !(rEnd < prev.residMax)) mu = prev.mu;
    else residMax = rEnd;
  }
  return { mu, converged: residMax < tol, residMax, iterations: iters };
}
