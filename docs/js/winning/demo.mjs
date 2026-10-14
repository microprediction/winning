// Demo support: seeded correlation generators (randomcov's ensembles in
// miniature), dense linear algebra for the in-browser grammar fit, and
// a Monte Carlo sampler to race against.
import { hermite1, interpClamped, solve, firstPrimes, asLoadings,
         isVector, asCount, asIdio } from "./core.mjs";
import { treeCluster } from "./blocks.mjs";

/* Halton sequence through the normal quantile: equal-weight nodes for
   E over N(0, I_r). The fitted grammar has rank k+m > 2, where tensor
   Gauss-Hermite grids explode; low-discrepancy nodes are the right
   family there (same escalation as the python and R engines). */
function invNormal(p) {
  // Acklam-style rational approximation, adequate for node placement
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
  if (p > 1 - pl) return -invNormal(1 - p);
  const q = p - 0.5, r2 = q * q;
  return (((((a[0]*r2+a[1])*r2+a[2])*r2+a[3])*r2+a[4])*r2+a[5])*q /
         (((((b[0]*r2+b[1])*r2+b[2])*r2+b[3])*r2+b[4])*r2+1);
}
export function haltonNormalNodes(r, count) {
  // a factor dimension >= 0 and a node count >= 1, both whole: NaN gave a
  // rank-zero rule, 2.5 three-dimensional nodes, Infinity hung (#557)
  asCount(r, "rank", 0, "haltonNormalNodes");
  asCount(count, "count", 1, "haltonNormalNodes");
  const F = [], W = new Array(count).fill(1 / count);
  const primes = firstPrimes(r);   // generated: a literal table had a
  // silent cliff at rank 17, past which every node was NaN (#233)
  for (let idx = 0; idx < count; idx++) {
    const node = [];
    for (let dim = 0; dim < r; dim++) {
      const base = primes[dim];
      let i = idx + 21, f = 1 / base, h = 0;
      while (i > 0) { h += f * (i % base); i = Math.floor(i / base); f /= base; }
      node.push(invNormal(Math.min(Math.max(h, 1e-12), 1 - 1e-12)));
    }
    F.push(node);
  }
  return { F, W };
}

/* ---- seeded rng (mulberry32) + normals ------------------------------ */
export function rng(seed) {
  let s = seed >>> 0;
  const u = () => {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const n = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let a, b, r2;
    do { a = 2 * u() - 1; b = 2 * u() - 1; r2 = a * a + b * b; }
    while (r2 >= 1 || r2 === 0);
    const f = Math.sqrt(-2 * Math.log(r2) / r2);
    spare = b * f;
    return a * f;
  };
  return { u, n };
}

/* ---- generators: return { C } (dense) or { structure } (grammar) ---- */
export const GENERATORS = {
  "factor (rank 2)": (n, r) => {
    const V = [], D = [];
    for (let i = 0; i < n; i++) {
      V.push([0.65 * r.n(), 0.35 * r.n()]);
      D.push(Math.max(1 - V[i][0] ** 2 - V[i][1] ** 2, 0.08));
    }
    return { structure: { kind: "Factor", V, D },
             note: "exactly in the grammar: no fit needed" };
  },
  "sectors (blocks)": (n, r) => {
    const nc = Math.max(2, Math.round(n / 8));
    const cluster = [], loading = [], D = [];
    for (let i = 0; i < n; i++) {
      cluster.push(Math.floor(r.u() * nc));
      const rho = 0.35 + 0.4 * r.u();
      loading.push(Math.sqrt(rho));
      D.push(1 - rho);
    }
    return { structure: { kind: "Blocks", cluster, loading, D },
             note: "exactly in the grammar: no fit needed" };
  },
  "hierarchy (tree)": (n, r) => {
    const nc = Math.max(2, Math.round(n / 6));
    const cluster = [], loading = [], D = [];
    for (let i = 0; i < n; i++) cluster.push(i % nc);
    // random binary merges over clusters
    let nodes = [...Array(nc).keys()];
    const parent = new Array(nc).fill(-1);
    const strength = new Array(nc).fill(0);
    while (nodes.length > 1) {
      const i = Math.floor(r.u() * nodes.length);
      let j = Math.floor(r.u() * (nodes.length - 1));
      if (j >= i) j++;
      const [a, b] = [nodes[Math.max(i, j)], nodes[Math.min(i, j)]];
      parent.push(-1); strength.push(0.25 + 0.35 * r.u());
      const t = parent.length - 1;
      parent[a] = t; parent[b] = t;
      nodes = nodes.filter(x => x !== a && x !== b).concat([t]);
    }
    for (let i = 0; i < n; i++) {
      loading.push(0.3 + 0.25 * r.u());
      let pv = 0, u = cluster[i];
      while (parent[u] >= 0) { pv += strength[parent[u]] ** 2; u = parent[u]; }
      D.push(Math.max(1 - loading[i] ** 2 - pv, 0.06));
    }
    return { structure: { kind: "Tree", cluster, loading, D, parent, strength },
             note: "exactly in the grammar: no fit needed" };
  },
  "AR(1) chain (dense)": (n, r) => {
    const rho = 0.55 + 0.4 * r.u();
    const C = [];
    for (let i = 0; i < n; i++) {
      C.push([]);
      for (let j = 0; j < n; j++) C[i].push(Math.pow(rho, Math.abs(i - j)));
    }
    return { C, note: `rho = ${rho.toFixed(2)}; fitted to the grammar in-browser` };
  },
  "spiked spectrum (dense)": (n, r) => {
    // three spikes over a noise floor: C = corr(B B' + 0.5 I), PSD by
    // construction (the random-matrix caricature of an equity market)
    const B = [];
    for (let i = 0; i < n; i++) {
      B.push([0.7 * r.n(), 0.45 * r.n(), 0.3 * r.n()]);
    }
    const S = [];
    for (let i = 0; i < n; i++) {
      S.push([]);
      for (let j = 0; j < n; j++) {
        let v = i === j ? 0.5 : 0;
        for (let q = 0; q < 3; q++) v += B[i][q] * B[j][q];
        S[i].push(v);
      }
    }
    const C = S.map((row, i) => row.map((v, j) =>
      v / Math.sqrt(S[i][i] * S[j][j])));
    return { C, note: "dense but spectrally concentrated; fitted in-browser" };
  },
};

/* ---- dense support: jacobi eigh, cholesky, grammar fit -------------- */
export function jacobiEigh(Ain, maxSweeps = 12) {
  const n = Ain.length;
  const A = Ain.map(row => row.slice());
  const V = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  // thresholds RELATIVE to the matrix's own size: absolute ones made
  // the decomposition (and so every fit built on it) depend on the units
  // the covariance happened to be written in (#341)
  // squared norms overflowed past 1e154 (and underflowed below 1e-162),
  // so no rotation ran and the diagonal came back as the spectrum (#536):
  // rotate an exactly rescaled copy, and unscale the eigenvalues
  let amax = 0;
  for (const row of A) for (const v of row) amax = Math.max(amax, Math.abs(v));
  const e = pow4Scale(amax);
  if (e !== 0) {
    const r = jacobiEigh(A.map(row => row.map(v => v * 2 ** e)), maxSweeps);
    const down = 2 ** (-e / 2);
    return { values: r.values.map(v => v * down * down), vectors: r.vectors };
  }
  let tot = 0;
  for (let p = 0; p < n; p++) for (let q = 0; q < n; q++) tot += A[p][q] * A[p][q];
  const tiny = 1e-15 * Math.sqrt(tot);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (!(off > 1e-30 * tot)) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) <= tiny) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const vals = A.map((row, i) => row[i]);
  return { values: vals, vectors: V };   // columns of V are eigenvectors
}

/* A finite, square, symmetric, positive-semidefinite matrix, or a
   refusal -- python's _validate_covariance. fitGrammar projected and
   eigendecomposed whatever it was given: an asymmetric matrix and its
   transpose fitted races 4.6 points apart (the projection used row
   means for both sides), and an indefinite one was clipped into some
   PSD model and priced as if valid (#357). Returns the symmetrised
   copy. */
export function validateCovariance(C, name = "C") {
  if (!isVector(C) || C.length === 0)
    throw new Error(`${name} must be a nonempty square matrix`);
  const n = C.length;
  const M = C.map((row, i) => {
    if (!isVector(row) || row.length !== n)
      throw new Error(
        `${name} must be square: row ${i} has ` +
        `${isVector(row) ? row.length : "no"} entries for ${n} rows`);
    return Array.from(row, v => {
      if (typeof v !== "number" || !Number.isFinite(v))
        throw new Error(`${name} contains a non-finite entry (${v}) in row ${i}`);
      return v;
    });
  });
  let amax = 0, asym = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      amax = Math.max(amax, Math.abs(M[i][j]));
      asym = Math.max(asym, Math.abs(M[i][j] - M[j][i]));
    }
  if (asym > 1e-8 * Math.max(amax, 1e-300))
    throw new Error(
      `${name} is not symmetric (max asymmetry ${asym.toExponential(2)}); ` +
      "pass (C + C')/2 if the asymmetry is numerical noise -- a fit cannot " +
      "choose a triangle for you");
  const S = M.map((row, i) => row.map((v, j) => 0.5 * v + 0.5 * M[j][i]));
  // PSD in CORRELATION units (#666; python winning.factor.core): a
  // tolerance on the mean raw variance let a 1e12 variance hide a
  // correlation of 2, and its 1e-300 floor let [[0, c], [c, 0]] through
  // at c = 1e-309. No negative variance; a zero variance has a zero row;
  // the standardised positive-variance block has min eigenvalue >= -1e-8.
  const notPsd = why => new Error(
    `${name} is not positive semidefinite (${why}); a covariance has no ` +
    "negative variance in any direction, and clipping it silently invents " +
    "a different model");
  const pos = [];
  for (let i = 0; i < n; i++) {
    const d = S[i][i];
    if (d < 0) throw notPsd(`variance [${i}] = ${d.toExponential(2)} < 0`);
    if (d > 0) pos.push(i);
    else for (let j = 0; j < n; j++)
      if (j !== i && S[i][j] !== 0)
        throw notPsd(`variance [${i}] is zero but its row is not`);
  }
  if (pos.length) {
    const sd = pos.map(i => Math.sqrt(S[i][i]));
    const R = pos.map((i, a) => pos.map((j, b) => S[i][j] / sd[a] / sd[b]));
    const lamMin = Math.min(...jacobiEigh(R, 50).values);
    if (lamMin < -1e-8)
      throw notPsd("min eigenvalue of the correlation matrix " +
                   lamMin.toExponential(2));
  }
  return S;
}

/* argmin_{d >= 0} d'Gd/2 - c'd for G = P o P = aI + b11' -- python's
   _nnls_centered_gram (water-filling, O(n)). */
function nnlsCenteredGram(c, n) {
  const a = 1 - 2 / n, b = 1 / (n * n);
  let s = Math.max(c.reduce((x, y) => x + y, 0), 0) / (a + b * n);
  if (n <= 2) return new Array(n).fill(Math.max(s, 0) / n);
  for (let pass = 0; pass < 100; pass++) {
    let sum = 0, cnt = 0;
    for (const v of c) if (v > b * s) { sum += v; cnt++; }
    const sN = sum / (a + b * cnt);
    const done = Math.abs(sN - s) <= 1e-15 * Math.max(1, Math.abs(s));
    s = sN;
    if (done) break;
  }
  return c.map(v => Math.max((v - b * s) / a, 0));
}

/* How many of the leading `want` directions (of `values`, sorted
   descending) can be taken without cutting through a tied group --
   python's _warn_if_rank_splits_a_tie, applied rather than just warned
   about. A cut through a tie is not an approximation but an arbitrary
   choice of basis, so relabelling the contestants of an exchangeable
   covariance changed the fitted race by 0.6 points (#383). The tied
   group is DROPPED (its variance falls to the closing diagonal, which
   is the exchangeable representation) and the caller is told. */
function untiedRank(values, want, stage, tol = 1e-6) {
  const lam = values;
  if (want <= 0 || want >= lam.length || !(lam[0] > 0)) return want;
  const scale = lam[0];
  if (lam[want - 1] - lam[want] > tol * scale) return want;
  let lo = want - 1;
  while (lo > 0 && lam[lo - 1] - lam[lo] <= tol * scale) lo--;
  let hi = want;
  while (hi + 1 < lam.length && lam[hi] - lam[hi + 1] <= tol * scale) hi++;
  // A tie in a low-variance subspace is NOT harmless: once the leading
  // direction puts one contestant out of contention, the rest can be
  // decided entirely inside it. A 5% exemption here let the tie at 0.04
  // of [1, 0.04, 0.04] be sliced, and relabelling moved a contrast
  // variance 0.064 -> 0.094 (4.2 points of an exact pair price; #383).
  // So the tied group is always dropped -- that is the exchangeable
  // representation, and costs nothing -- but only a material tie (python's
  // 5% judgment) is worth a console message: every centred spectrum ends
  // in a tied bulk.
  if (lam[want - 1] >= 0.05 * scale && typeof console !== "undefined")
    console.warn(
      `fitGrammar: rank ${want} (${stage}) splits a tied eigenvalue: ` +
      `eigenvalues ${lo + 1} to ${hi + 1} are equal, so any ${want - lo} ` +
      `of those ${hi - lo + 1} directions are equally good and imply ` +
      `DIFFERENT races. Using rank ${lo} for this stage instead; rank ` +
      `${hi + 1} would include the whole group.`);
  return lo;
}

/* An exact power-of-four rescale bringing amax near 1, or 1 when amax is
   already within 2^+-200 (so ordinary inputs run bit-identically). The
   fitter squares entries and sums contrasts: at 1e308 they overflowed and
   it returned D = NaN (#536); a power of four keeps sqrt(scale) exact. */
function pow4Scale(amax) {
  if (!(amax > 0) || !Number.isFinite(amax) || (amax < 2 ** 200 && amax > 2 ** -200)) return 0;
  const e = -2 * Math.round(Math.log2(amax) / 2);
  return Math.max(-1022, Math.min(1022, e));
}

/* fitGrammar(C, k = 3, m = 4[, opts]) or fitGrammar(C, opts), opts =
   { k, m, method, maxIter, tol }. Returns { V, D } columns for
   raceProbabilities.

   method "eig" (the default): one eigendecomposition of P C P for the
   factor stage. When the diagonal is heterogeneous its centred part has
   structured eigenvalues of its own, so the top directions need not span
   the factor: an exact v v' + diag(D) was fitted with an 11.7x wrong
   contrast (#505).

   method "ipaf" (opt-in): iterated principal-axis factoring on the
   projected matrix -- alternate V = top-k eigencolumns of P(C - diag D)P
   and D = the closing diagonal of P(C - V V')P -- each step the exact
   minimiser of ||P(C - V V' - diag D)P||_F in its block, so the residual
   never rises. At most maxIter (default 500) iterations, each one
   O(n^3) Jacobi; stops when the relative residual changes by <= tol
   (default 1e-12). Two deterministic starts (SMC uniquenesses, then
   the eigen fit's diagonal), the smaller residual kept, so at most
   2 maxIter iterations. The result carries
   ipaf: { iterations, converged, residual } (iterations summed over the
   starts run; residual relative to ||P C P||_F; n >= 3 and k >= 1).
   The report also carries contrastResidualMax, and converged requires
   it to be <= 1 (a contrast off by its own size).

   Every fit carries fit.contrastResidualMax (non-enumerable, so the
   result still spreads as { V, D }), the worst pairwise contrast misfit
   |Var_fit(X_i - X_j) / Var_C(X_i - X_j) - 1| (python's
   contrast_residual_max); V and D do not depend on it. The residual stage then promotes the top-m eigencolumns
   of the remaining projected residual, not of P(C - V V')P. */
export function fitGrammar(C, k = 3, m = 4, opts = undefined) {
  if (k !== null && typeof k === "object") { opts = k; k = opts.k ?? 3; m = opts.m ?? 4; }
  opts = opts ?? {};
  const method = opts.method ?? "eig";
  if (method !== "eig" && method !== "ipaf")
    throw new Error(`fitGrammar: method must be "eig" or "ipaf"; got ${method}`);
  const maxIter = opts.maxIter ?? 500, tol = opts.tol ?? 1e-12;
  if (method === "ipaf" && !(Number.isInteger(maxIter) && maxIter >= 1 && tol >= 0))
    throw new Error("fitGrammar: maxIter must be a positive integer and tol >= 0");
  const ip = method === "ipaf" ? { maxIter, tol } : null;
  C = validateCovariance(C, "fitGrammar: C");
  let amax = 0;
  for (const row of C) for (const v of row) amax = Math.max(amax, Math.abs(v));
  const e = pow4Scale(amax);
  const up = 2 ** e;
  const Cs = e === 0 ? C : C.map(row => row.map(v => v * up));
  const fit = fitGrammarScaled(Cs, k, m, ip);
  // the worst pairwise contrast misfit, as python's contrast_residual_max:
  // the global residual cannot see a small choice-critical contrast (an
  // exact v v' + diag(D) fitted its 0-3 contrast 78x too large while ipaf
  // reported converged at residual 8e-8, #505). Scale free, so measured
  // in the scaled units; above CONTRAST_MISFIT = 1 ipaf does not claim
  // convergence.
  // Non-enumerable on the result, so { ...fit } still spreads as the
  // { V, D } options raceProbabilities takes (and refuses unknown keys
  // of); enumerable inside the opt-in ipaf report.
  const crm = contrastResidualMax(Cs, fit.V, fit.D);
  const withCrm = o => Object.defineProperty(o, "contrastResidualMax",
    { value: crm, enumerable: false });
  if (fit.ipaf) {
    fit.ipaf.contrastResidualMax = crm;
    if (fit.ipaf.converged && crm > CONTRAST_MISFIT) {
      fit.ipaf.converged = false;
      if (typeof console !== "undefined")
        console.warn(`fitGrammar: ipaf reached a stationary point (relative residual ` +
                     `${fit.ipaf.residual}) whose worst pairwise contrast is off by ` +
                     `${crm.toPrecision(3)} of its own size; not converged`);
    }
  }
  if (e === 0) return withCrm(fit);
  // undo in two exact steps: 2^-e alone overflows to Infinity past e = 1023
  const down = 2 ** (-e / 2);
  const out = { V: fit.V.map(row => row.map(v => v * down)),
                D: fit.D.map(d => d * down * down) };
  if (fit.ipaf) out.ipaf = fit.ipaf;
  withCrm(out);
  if (!out.D.every(Number.isFinite) || !out.V.flat().every(Number.isFinite))
    throw new Error("fitGrammar: the fitted grammar is not representable in " +
                    "double precision at this covariance scale");
  return out;
}

// A contrast off by its own size. python's cov= path warns at 0.05, but a
// fit to an ordinary SAMPLE covariance misses its contrasts by 5-15%
// from sampling noise alone: 0.05 flagged 315 of 480 ipaf fits on a
// fixed-seed battery of sample covariances of true k-factor models
// (n = 6-16, T = 500). The largest misfit over 1460 ordinary fits
// (n up to 30, T down to 60, the demo's dense generators) was 0.70;
// the #505 fixture is 76.9.
const CONTRAST_MISFIT = 1.0;
/* max over pairs of |fitted - target| contrast variance over the target
   contrast variance (python's _worst_contrast_ratio): the target formed
   as anchored differences (C_ii - C_ij) + (C_jj - C_ij), the fit from
   the loading differences, so neither sees a common shift. */
function contrastResidualMax(C, V, D) {
  const n = C.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += C[i][i] / n;
  const tiny = 1e-12 * Math.max(mean, 1e-300);
  let worst = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const target = (C[i][i] - C[i][j]) + (C[j][j] - C[i][j]);
      let fitted = D[i] + D[j];
      for (let r = 0; r < V[i].length; r++) fitted += (V[i][r] - V[j][r]) ** 2;
      worst = Math.max(worst, Math.abs(fitted - target) / Math.max(target, tiny));
    }
  }
  return worst;
}

function fitGrammarScaled(C, k, m, ip = null) {
  // rank-k + promoted residual on the PROJECTED residual (the package's
  // fit_covariance pipeline; blocks omitted for browser latency, and the
  // contrast heuristic stands in for the certified quotient ALS):
  // returns { V, D } columns for raceProbabilities. Only P C P is
  // choice-relevant, so every stage fits the projected matrix and the
  // closing diagonal solves (P.P) d = diag(P R P).
  // C arrives validated and symmetrized by fitGrammar (#357)
  if (!Number.isInteger(k) || k < 0 || !Number.isInteger(m) || m < 0)
    throw new Error(`fitGrammar: k and m must be non-negative integers; got k=${k}, m=${m}`);
  const n = C.length;
  if (n === 1) return { V: [[0]], D: [Math.max(C[0][0], 1e-6 * Math.max(C[0][0], 1e-300))] };
  // Everything below reads C only through its pairwise CONTRASTS,
  // Delta_ij = (C_ii - C_ij) + (C_jj - C_ij), formed as anchored
  // differences before any averaging. A common shift a 11' cancels from
  // every contrast, but summing raw rows near a = 2**51 rounded the +1 of
  // each variance away, P C P came back as I instead of I - 11'/4, and a
  // field whose six contrasts were all exactly 2 fitted contrasts of 2.06
  // and 2.86 (a 3.6-point price move). P C P = -P Delta P / 2 (#465).
  const Delta = C.map((row, i) => row.map((v, j) =>
    (i === j ? 0 : (row[i] - v) + (C[j][j] - v))));
  // the floor is RELATIVE to each runner's own contrast scale -- half its
  // smallest contrast, the most variance any shared part can leave it --
  // not to its raw variance, which a common shift inflates at will (#465).
  // Relative, not absolute: an absolute 0.03 floor added 0.06 to a
  // pair's only identifiable contrast and broke scale equivariance (#341).
  const own = Delta.map((row, i) => 0.5 * Math.min(...row.filter((_, j) => j !== i)));
  const meanOwn = own.reduce((a, b) => a + b, 0) / n;
  const floor = own.map(v => 1e-6 * Math.max(v, 1e-6 * Math.max(meanOwn, 1e-300)));
  if (n === 2) {
    // a pair has ONE choice-relevant number, Var(X0 - X1); python returns
    // it as two equal idiosyncratic halves with no factor, which prices
    // the pair exactly (no near-step factor integral to resolve)
    const half = 0.5 * Math.max(Delta[0][1], floor[0] + floor[1]);
    return { V: [[0], [0]], D: [half, half] };
  }
  const proj = M => {
    // P M P with P = I - 11'/n (for matrices with no large common part)
    const rm = M.map(row => row.reduce((a, b) => a + b, 0) / n);
    const tot = rm.reduce((a, b) => a + b, 0) / n;
    return M.map((row, i) => row.map((v, j) => v - rm[i] - rm[j] + tot));
  };
  const CP = proj(Delta).map(row => row.map(v => -0.5 * v));
  let meanCP = 0;
  for (let i = 0; i < n; i++) meanCP += CP[i][i] / n;
  // a direction is dead relative to the matrix's own (choice-relevant)
  // scale, not 1e-8 absolute, which changed the fitted rank under
  // rescaling (#341), nor the raw variance a common shift inflates (#465)
  const dead = 1e-8 * Math.max(meanCP, 1e-300);
  // P (C - sum col col') P, from the stable CP: P is linear
  const projMinus = colsNow => {
    const VV = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => {
      let s = 0;
      for (const col of colsNow) s += col[i] * col[j];
      return s;
    }));
    const VP = proj(VV);
    return CP.map((row, i) => row.map((v, j) => v - VP[i][j]));
  };
  const { values, vectors } = jacobiEigh(CP, 50);
  const order = values.map((v, i) => i).sort((a, b) => values[b] - values[a]);
  const kk = untiedRank(order.map(i => values[i]), Math.min(k, n), "factor stage");
  let cols = [];
  for (const idx of order.slice(0, kk)) {
    const lam = Math.max(values[idx], 0);
    if (lam > dead) cols.push(vectors.map(row => row[idx] * Math.sqrt(lam)));
  }
  // closing diagonal: min_d ||P(R - diag d)P||, R = C - VV', d >= floor,
  // as the LOWER-BOUNDED least squares (python's water-filling), not a
  // generic solve with a floor applied afterwards: P o P is singular at
  // n = 2, where the generic solve silently zeroed a coordinate (#341)
  const closeD = colsNow => {
    const RP = projMinus(colsNow);
    const a = 1 - 2 / n, b = 1 / (n * n);
    const rhs = RP.map((row, i) => row[i]);
    const fsum = floor.reduce((x, y) => x + y, 0);
    if (n <= 2) {
      const s = Math.max(rhs.reduce((x, y) => x + y, 0), 0) / (a + n * b) / n;
      return floor.map(f => Math.max(s, f));
    }
    const x = nnlsCenteredGram(rhs.map((v, i) => v - (a * floor[i] + b * fsum)), n);
    return floor.map((f, i) => f + x[i]);
  };
  // P (C - VV' - diag d) P
  const projResid = (colsNow, d) => {
    const R = projMinus(colsNow);
    const pd = proj(d.map((di, i) => d.map((_, j) => (i === j ? di : 0))));
    return R.map((row, i) => row.map((v, j) => v - pd[i][j]));
  };
  const fro = M => Math.sqrt(M.reduce((acc, row) => acc + row.reduce((t, v) => t + v * v, 0), 0));
  let report = null;
  if (ip && kk > 0) {
    // iterated principal-axis factoring on the projected matrix (#505)
    // Two deterministic starts, the smaller final residual kept: the
    // squared-multiple-correlation uniquenesses 1/(C^-1)_ii (when C is
    // well conditioned), then the eigen stage's own closing diagonal.
    // IPAF's linear rate can be near 1 from a start whose factor sits on
    // the wrong runners: from the eigen start alone the #505 fixture
    // needed ~1e5 iterations, from the SMC start it converges in tens.
    const scale = Math.max(fro(CP), 1e-300);
    const starts = [];
    const eC = jacobiEigh(C, 50);
    const lmax = Math.max(...eC.values);
    if (Math.min(...eC.values) > 1e-10 * lmax)
      starts.push(C.map((_, i) => {
        let s = 0;
        for (let q = 0; q < n; q++) s += eC.vectors[i][q] ** 2 / eC.values[q];
        return Math.max(1 / s, floor[i]);
      }));
    starts.push(closeD(cols));
    let best = null;
    for (const d0 of starts) {
      let d = d0, prev = Infinity, it = 0, converged = false, cs = [];
      for (it = 1; it <= ip.maxIter; it++) {
        const eA = jacobiEigh(projResid([], d), 50);
        const oA = eA.values.map((v, i) => i).sort((x, y) => eA.values[y] - eA.values[x]);
        cs = [];
        for (const idx of oA.slice(0, kk)) {
          const lam = Math.max(eA.values[idx], 0);
          if (lam > dead) cs.push(eA.vectors.map(row => row[idx] * Math.sqrt(lam)));
        }
        d = closeD(cs);
        const res = fro(projResid(cs, d)) / scale;
        // relative change, with an absolute 1e-15 floor: at a rounding-level
        // residual (an exact grammar member) the change is pure noise
        const stop = Math.abs(prev - res) <= ip.tol * res + 1e-15;
        prev = res;
        if (stop) { converged = true; break; }
      }
      const iters = Math.min(it, ip.maxIter);
      if (!best || prev < best.residual)
        best = { cols: cs, iterations: (best ? best.iterations : 0) + iters, converged, residual: prev };
      else best.iterations += iters;
      if (converged && prev <= 1e-12) break;
    }
    cols = best.cols;
    report = { iterations: best.iterations, converged: best.converged, residual: best.residual };
    if (!report.converged && typeof console !== "undefined")
      console.warn(`fitGrammar: ipaf stopped at maxIter=${ip.maxIter} ` +
                   `without converging (relative residual ${report.residual})`);
  }
  // projected residual, diagonal zeroed, top-m eigencolumns promoted
  // (ipaf: the full remaining residual P(C - VV' - diag D)P instead)
  const E = report ? projResid(cols, closeD(cols)) : projMinus(cols);
  for (let i = 0; i < n; i++) E[i][i] = 0;
  const eE = jacobiEigh(E, 50);
  const orderE = eE.values.map((v, i) => i).sort((a, b) => eE.values[b] - eE.values[a]);
  const mm = untiedRank(orderE.map(i => eE.values[i]), Math.min(m, n), "residual stage");
  for (const idx of orderE.slice(0, mm)) {
    const lam = Math.max(eE.values[idx], 0);
    if (lam > dead) cols.push(eE.vectors.map(row => row[idx] * Math.sqrt(lam)));
  }
  const Dc = closeD(cols);
  if (!cols.length) cols.push(new Array(n).fill(0));
  const V = [], D = [];
  for (let i = 0; i < n; i++) {
    V.push(cols.map(col => col[i]));
    D.push(Dc[i]);
  }
  return report ? { V, D, ipaf: report } : { V, D };
}

export function structureCov(s) {
  // dense covariance implied by a grammar structure (for the MC sampler).
  // Every public grammar, and loadings through the same shape door as
  // the pricing kernels: Independent and Nested fell through to an
  // all-zero matrix (Monte Carlo then simulated a deterministic race),
  // rank-r Blocks rows multiplied to NaN, and a flat rank-one Factor V
  // silently lost its factor (#337). An unknown kind is refused.
  if (!s || typeof s !== "object") throw new Error("structureCov: expected a structure");
  // the field size from contestant-bearing data, never from D alone: a
  // scalar D (the package's own spelling) gave n = undefined -- an empty
  // matrix for Independent(1) and loading errors "for undefined" (#563)
  let n;
  if (isVector(s.D)) n = s.D.length;
  else if (s.kind === "Factor" && isVector(s.V)) n = s.V.length;
  else if (isVector(s.cluster)) n = s.cluster.length;
  else
    throw new Error(
      `structureCov: a ${s.kind} structure with scalar D carries no field ` +
      "size; pass D as one variance per contestant");
  const D = asIdio(s.D, n, "D");
  const C = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? D[i] : 0)));
  const dot = (a, b) => a.reduce((acc, v, k) => acc + v * b[k], 0);
  const addBlocks = () => {
    const L = asLoadings(s.loading, n, "loading");
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (s.cluster[i] === s.cluster[j]) C[i][j] += dot(L[i], L[j]);
  };
  if (s.kind === "Independent") {
    // diag(D), already there
  } else if (s.kind === "Factor") {
    const V = asLoadings(s.V, n, "V");
    if (V) for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) C[i][j] += dot(V[i], V[j]);
  } else if (s.kind === "Blocks") {
    addBlocks();
  } else if (s.kind === "Nested") {
    addBlocks();
    if (s.coupling != null && s.gamma !== 0) {
      const g = asLoadings(s.coupling, n, "coupling");
      const g2 = (s.gamma ?? 1) ** 2;
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) C[i][j] += g2 * dot(g[i], g[j]);
    }
  } else if (s.kind === "Tree") {
    // Labels are arbitrary comparable values: the pricing kernels remap
    // them densely and this read them as node INDICES, so 10/20/30 sent
    // every lookup past the end of `parent` -- an empty ancestor set,
    // every shared factor silently gone -- and negative or string
    // labels threw `anc[...] is not iterable` (#306). Same remapper as
    // the kernels, imported rather than repeated.
    const cluster = treeCluster(s);                       // (#372)
    const anc = [];
    const nc = Math.max(...cluster) + 1;
    for (let c = 0; c < nc; c++) {
      const a = new Set();
      let u = c, hops = 0;
      while (s.parent[u] >= 0) {
        a.add(s.parent[u]); u = s.parent[u];
        if (++hops > s.parent.length)
          throw new Error("structureCov: Tree parent contains a cycle");
      }
      anc.push(a);
    }
    const L = asLoadings(s.loading, n, "loading");
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (const t of anc[cluster[i]])
          if (anc[cluster[j]].has(t)) v += s.strength[t] ** 2;
        if (s.cluster[i] === s.cluster[j]) v += dot(L[i], L[j]);
        C[i][j] += v;
      }
  } else {
    throw new Error(`structureCov: unknown structure kind ${JSON.stringify(s.kind)}`);
  }
  return C;
}

/* Cholesky with a RELATIVE pivot tolerance. The old absolute floor,
   sqrt(max(s, 1e-10)), replaced every valid pivot below 1e-10 -- so the
   same race written in smaller units simulated a different race (a
   valid 3-runner field at d = 1e-14 came out [0.26, 0.37, 0.37] for
   [0.73, 0.22, 0.05]) (#375). A pivot that is roundoff relative to its
   own diagonal is a zero pivot, and its column is zero, which is what a
   PSD factor of a singular matrix has there. */
export function cholesky(Cin) {
  const n = Cin.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = Cin[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        L[i][i] = s > 1e-13 * Math.abs(Cin[i][i]) ? Math.sqrt(s) : 0;
      } else {
        L[i][j] = L[j][j] > 0 ? s / L[j][j] : 0;
      }
    }
  }
  return L;
}

/* The factor to SIMULATE a race with, built in the race quotient.

   Adding a 11' to a covariance is a shared shift and cannot change an
   ordering, but a Cholesky of the full matrix loses the small
   choice-relevant eigenvalue to cancellation: C = [[a+1, a], [a, a+1]]
   at a = 2e15 kept a contrast variance of 1.5 instead of 2, and two
   million draws converged to [0.793, 0.207] instead of [0.760, 0.240]
   (#394). This anchors on runner 0: the differences x_j - x_0 have
   covariance S_jk = C_jk - C_j0 - C_0k + C_00, formed from C BEFORE any
   factorisation, and the returned lower-triangular L has a zero first
   row and chol(S) below it. mcBatch(mu, L, ...) then draws
   x_0 = mu_0 and x_j = mu_j + (x_j - x_0 noise), whose argmin is the
   race's. Scale-free by construction (#375). */
export function raceFactor(C) {
  // through the covariance boundary: cholesky zeroes non-positive pivots,
  // so an asymmetric or indefinite C was silently simulated as some other
  // PSD law (#564)
  C = validateCovariance(C, "raceFactor: C");
  const n = C.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  if (n < 2) return L;
  const S = [];
  for (let j = 1; j < n; j++) {
    const row = [];
    for (let k = 1; k < n; k++) row.push(C[j][k] - C[j][0] - C[0][k] + C[0][0]);
    S.push(row);
  }
  const Ls = cholesky(S);
  for (let j = 1; j < n; j++)
    for (let k = 1; k <= j; k++) L[j][k] = Ls[j - 1][k - 1];
  return L;
}

export function mcBatch(mu, L, r, batch, counts) {
  // frequency simulation: argmin of mu + L z, `batch` draws into counts
  const n = mu.length;
  const x = new Array(n);
  for (let b = 0; b < batch; b++) {
    const z = new Array(n);
    for (let i = 0; i < n; i++) z[i] = r.n();
    for (let i = 0; i < n; i++) {
      let s = mu[i];
      const Li = L[i];
      for (let k = 0; k <= i; k++) s += Li[k] * z[k];
      x[i] = s;
    }
    let best = 0;
    for (let i = 1; i < n; i++) if (x[i] < x[best]) best = i;
    counts[best]++;
  }
}

/* ---- the competition: GHK and Mendell-Elston --------------------------
   Both price alternative i through the difference vector u_j = x_j - x_i,
   j != i (min-wins: p_i = P(u > 0)), each with its own (n-1)-dimensional
   covariance and its own O(n^3/6) Cholesky or sweep. That per-alternative
   structure is the point the demo makes: pricing the whole field costs n
   times a single-alternative price, before a single draw is taken. */
import { ndtr, npdf, logndtr } from "./core.mjs";


function diffScale(S, d) {
  let t = 0;
  for (let a = 0; a < d; a++) t += S[a * d + a] / d;
  return Math.max(t, 1e-300);
}

function badCovariance(C, where) {
  for (let a = 0; a < C.length; a++)
    for (let b = 0; b < C.length; b++)
      if (typeof C[a][b] !== "number" || !Number.isFinite(C[a][b]))
        throw new Error(`${where}: C contains a non-finite entry (${C[a][b]}) in row ${a}`);
  for (let a = 0; a < C.length; a++)
    if (C[a][a] < 0)
      throw new Error(`${where}: C is not positive semidefinite (variance [${a}] = ${C[a][a]} < 0)`);
}

function diffProblem(mu, C, i, where) {
  // mean and covariance of (x_j - x_i)_{j != i}
  const n = mu.length, m = new Float64Array(n - 1);
  // C at the door, in O(n^2): square, finite, symmetric (the tolerance of
  // validateCovariance) and no negative variance. Both arms read one
  // triangle through S, so C and C' priced differently, and an
  // indefinite C came back as a confident book (#726). Definiteness of
  // what the arms actually use -- the difference covariance S -- is
  // checked by its own Cholesky pivots, at no extra order.
  if (!Array.isArray(C) || C.length !== n)
    throw new Error(`${where}: C must be a ${n} x ${n} covariance matrix`);
  for (let a = 0; a < n; a++)
    if (!isVector(C[a]) || C[a].length !== n)
      throw new Error(`${where}: C must be square; row ${a} is not ${n} entries`);
  let amax = 0, asym = 0, bad = 0;
  for (let a = 0; a < n; a++) {
    const Ca = C[a];
    for (let b = 0; b <= a; b++) {
      const v = Ca[b], w = C[b][a];
      bad += 0 * v + 0 * w;             // NaN iff an entry is not finite
      amax = Math.max(amax, Math.abs(v));
      asym = Math.max(asym, Math.abs(v - w));
    }
    if (Ca[a] < 0) bad = -1;
  }
  if (bad !== 0) badCovariance(C, where);
  if (asym > 1e-8 * Math.max(amax, 1e-300))
    throw new Error(`${where}: C is not symmetric (max asymmetry ${asym.toExponential(2)}); ` +
                    "pass (C + C')/2 if the asymmetry is numerical noise");
  const S = new Float64Array((n - 1) * (n - 1));
  const idx = [];
  for (let j = 0; j < n; j++) if (j !== i) idx.push(j);
  for (let a = 0; a < n - 1; a++) {
    m[a] = mu[idx[a]] - mu[i];
    for (let b = 0; b < n - 1; b++)
      S[a * (n - 1) + b] = C[idx[a]][idx[b]] - C[idx[a]][i] - C[i][idx[b]] + C[i][i];
  }
  return { m, S };
}

export function ghkPrepareOne(mu, C, i) {
  // per-alternative Cholesky of the difference covariance (the GHK setup
  // cost the wall-time axis charges honestly)
  const { m, S } = diffProblem(mu, C, i, "ghkPrepareOne");
  const d = mu.length - 1;
  const L = new Float64Array(d * d);
  // the pivot floor is relative to the difference problem's own scale:
  // an absolute 1e-12 replaced valid small contrast variances, so the
  // same race in smaller units simulated a different one (#375). And it
  // floors only a pivot that is not positive: raising every pivot below
  // it inflated a valid small one -- near-twins with variance 2^-43
  // priced [0.5, 0.158, 0.342] for the exact [0.5, 0.079, 0.421] (#409).
  // A pivot materially below zero (relative to its own variance, as
  // validateCovariance's -1e-8 in correlation units) is an indefinite
  // contrast covariance, not roundoff: eps priced it as some other law
  // (#726).
  const eps = 1e-12 * diffScale(S, d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b <= a; b++) {
      let s = S[a * d + b];
      for (let k = 0; k < b; k++) s -= L[a * d + k] * L[b * d + k];
      if (a === b) {
        if (s < -1e-8 * Math.abs(S[a * d + a])) negativePivot("ghkPrepareOne", s);
        L[a * d + a] = Math.sqrt(s > 0 ? s : eps);
      } else L[a * d + b] = s / L[b * d + b];
    }
  }
  return { m, L, d };
}

function negativePivot(where, s) {
  throw new Error(`${where}: C is not positive semidefinite: the ` +
    `contrast covariance has a negative pivot (${s.toExponential(2)}); ` +
    "clipping it silently invents a different model");
}

/* Definiteness of a difference covariance, by the same pivots
   ghkPrepareOne factors with (#726). */
function checkDiffDefinite(S, d, where) {
  const L = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b <= a; b++) {
      let s = S[a * d + b];
      for (let k = 0; k < b; k++) s -= L[a * d + k] * L[b * d + k];
      if (a === b) {
        if (s < -1e-8 * Math.abs(S[a * d + a])) negativePivot(where, s);
        L[a * d + a] = Math.sqrt(s > 0 ? s : 0);
      } else L[a * d + b] = L[b * d + b] > 0 ? s / L[b * d + b] : 0;
    }
  }
}

export function ghkSampleOne(prob, reps, r) {
  // GHK sequential-conditioning importance sampler: mean weight over reps
  const { m, L, d } = prob;
  const e = new Float64Array(d);
  let sum = 0;
  for (let rep = 0; rep < reps; rep++) {
    let w = 1;
    for (let k = 0; k < d; k++) {
      let partial = m[k];
      const Lk = k * d;
      for (let l = 0; l < k; l++) partial += L[Lk + l] * e[l];
      const a = -partial / L[Lk + k];
      // the upper tail as the reflected lower one: 1 - ndtr(a) cancelled
      // to 0 from a ~ 8.3 and a 1e-14 cutoff zeroed it before that, so a
      // two-runner gap of 11 priced 0 for 3.7e-15 (#555). Draw e > a as
      // -Phi^{-1}((1 - u) q), which is Phi^{-1}(Fa + u q) without the
      // rounded-to-one sum.
      const q = ndtr(-a);
      w *= q;
      if (!(q > 0)) { w = 0; break; }
      const u1 = 1 - r.u(), v = u1 * q;
      // below 1e-300 the draw is inverted from its LOG: flooring v there
      // put e[k] inside the truncation (e <= a), so a correlated field
      // whose first step had q = 2.9e-316 priced 8.6e-322 for >= 2.9e-316
      // (#555 residual)
      e[k] = v >= 1e-300 ? -invNormal(Math.min(v, 1 - 1e-15))
        : -invNormalFromLog(Math.log(u1) + logndtr(-a));
    }
    sum += w;
  }
  return sum;
}

/* Phi^{-1}(exp(lp)) for a lower-tail log probability lp < log 0.02425:
   Acklam's tail branch reads p only through sqrt(-2 log p), so it takes
   the log directly; one Newton step on log Phi refines it. */
function invNormalFromLog(lp) {
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
             -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143,
             3.75440866190742];
  const q = Math.sqrt(-2 * lp);
  const z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  const lz = logndtr(z);
  return z - (lz - lp) * Math.exp(lz - (-0.5 * z * z - 0.5 * Math.log(2 * Math.PI)));
}

export function mendellElstonOne(mu, C, i) {
  // Mendell-Elston analytic sequential moment approximation: condition on
  // u_k > 0 one coordinate at a time, propagating truncated-normal moments
  // and pretending normality is preserved (it is not; the bias is the
  // flat line this arm draws).
  const { m, S } = diffProblem(mu, C, i, "mendellElstonOne");
  const d = mu.length - 1;
  checkDiffDefinite(S, d, "mendellElstonOne");     // #726
  let logp = 0;
  // Hardest constraint FIRST, as python's _order_variables does.
  // Sequential moment matching is order dependent -- each step pretends
  // the conditioned remainder is still normal -- so processing in raw
  // contestant order made the answer depend on the LABELS: permuting a
  // four-runner race, evaluating, and undoing the permutation moved a
  // share by 3.9 percentage points (#287).
  //
  // python orders by a_t/sqrt(C_tt) descending, and its z is the
  // negative of this port's, so here it is m[k]/sqrt(S[k,k])
  // ASCENDING: the smallest survival probability goes first, while the
  // normal approximation is still exact. The order is fixed once from
  // the initial moments, as python fixes it, not re-sorted as the
  // conditioning proceeds.
  //
  // Exact ties used to fall back to INPUT order (a stable sort), so equal
  // abilities -- a routine input -- were still label dependent: 0.0087 on
  // a four-runner field and 0.0147 on a five-runner one (#287). Ties are
  // now broken by quantities that move WITH the labels: the difference
  // variance, then the sorted covariances to the other differences. Two
  // constraints still tied on all of those are interchangeable by an
  // automorphism of the problem and give the same answer either way.
  const eps = 1e-12 * diffScale(S, d);
  const alive = [];
  for (let a = 0; a < d; a++) alive.push(a);
  const hard = a => m[a] / Math.sqrt(Math.max(S[a * d + a], eps));
  const rowKey = a => {
    const r = [];
    for (let b = 0; b < d; b++) if (b !== a) r.push(S[a * d + b]);
    return r.sort((x, y) => y - x);
  };
  const keys = alive.map(a => ({ h: hard(a), v: S[a * d + a], r: rowKey(a) }));
  alive.sort((a, b) => {
    const A = keys[a], B = keys[b];
    if (A.h !== B.h) return A.h - B.h;
    if (A.v !== B.v) return B.v - A.v;
    for (let t = 0; t < A.r.length; t++) if (A.r[t] !== B.r[t]) return B.r[t] - A.r[t];
    return 0;
  });
  while (alive.length) {
    const k = alive.shift();
    const skk = Math.max(S[k * d + k], eps), sk = Math.sqrt(skk);
    const z = m[k] / sk;
    const Pz = Math.max(ndtr(z), 1e-300);
    logp += Math.log(Pz);
    const lam = npdf(z) / Pz;
    const del = lam * (lam + z);
    for (const j of alive) m[j] += (S[k * d + j] / sk) * lam;
    for (let aj = 0; aj < alive.length; aj++)
      for (let ak = aj; ak < alive.length; ak++) {
        const j = alive[aj], l = alive[ak];
        const upd = (S[k * d + j] * S[k * d + l] / skk) * del;
        S[j * d + l] -= upd;
        if (l !== j) S[l * d + j] -= upd;
      }
  }
  return Math.exp(logp);
}
