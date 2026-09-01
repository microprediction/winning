// Demo support: seeded correlation generators (randomcov's ensembles in
// miniature), dense linear algebra for the in-browser grammar fit, and
// a Monte Carlo sampler to race against.
import { hermite1, interpClamped, solve } from "./core.mjs";

/* Halton sequence through the normal quantile: equal-weight nodes for
   E over N(0, I_r). The fitted grammar has rank k+m > 2, where tensor
   Gauss-Hermite grids explode; low-discrepancy nodes are the right
   family there (same escalation as the python and R engines). */
const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
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
  const F = [], W = new Array(count).fill(1 / count);
  for (let idx = 0; idx < count; idx++) {
    const node = [];
    for (let dim = 0; dim < r; dim++) {
      const base = PRIMES[dim];
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
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-18 * n * n) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-14) continue;
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

export function fitGrammar(C, k = 3, m = 4) {
  // rank-k + promoted residual on the PROJECTED residual (the package's
  // fit_covariance pipeline; blocks omitted for browser latency, and the
  // contrast heuristic stands in for the certified quotient ALS):
  // returns { V, D } columns for raceProbabilities. Only P C P is
  // choice-relevant, so every stage fits the projected matrix and the
  // closing diagonal solves (P.P) d = diag(P R P).
  const n = C.length;
  const proj = M => {
    // P M P with P = I - 11'/n
    const rm = M.map(row => row.reduce((a, b) => a + b, 0) / n);
    const tot = rm.reduce((a, b) => a + b, 0) / n;
    return M.map((row, i) => row.map((v, j) => v - rm[i] - rm[j] + tot));
  };
  const CP = proj(C);
  const { values, vectors } = jacobiEigh(CP);
  const order = values.map((v, i) => i).sort((a, b) => values[b] - values[a]);
  const cols = [];
  for (const idx of order.slice(0, k)) {
    const lam = Math.max(values[idx], 0);
    cols.push(vectors.map(row => row[idx] * Math.sqrt(lam)));
  }
  // projected residual, diagonal zeroed, top-m eigencolumns promoted
  const E = proj(C.map((row, i) => row.map((v, j) => {
    let s = v;
    for (const col of cols) s -= col[i] * col[j];
    return s;
  })));
  for (let i = 0; i < n; i++) E[i][i] = 0;
  const eE = jacobiEigh(E);
  const orderE = eE.values.map((v, i) => i).sort((a, b) => eE.values[b] - eE.values[a]);
  for (const idx of orderE.slice(0, m)) {
    const lam = Math.max(eE.values[idx], 0);
    if (lam > 1e-8) cols.push(eE.vectors.map(row => row[idx] * Math.sqrt(lam)));
  }
  // closing diagonal: (P.P) d = diag(P R P), R = C - VV'
  const R = C.map((row, i) => row.map((v, j) => {
    let s = v;
    for (const col of cols) s -= col[i] * col[j];
    return s;
  }));
  const RP = proj(R);
  const G = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const p = (i === j ? 1 - 1 / n : -1 / n);
      return p * p;
    }));
  const d = solve(G, RP.map((row, i) => row[i]));
  const V = [], D = [];
  for (let i = 0; i < n; i++) {
    V.push(cols.map(col => col[i]));
    D.push(Math.max(d[i], 0.03));
  }
  return { V, D };
}

export function structureCov(s) {
  // dense covariance implied by a grammar structure (for the MC sampler)
  const n = s.D.length;
  const C = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, () => 0));
  if (s.kind === "Factor") {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (let r = 0; r < s.V[i].length; r++) v += s.V[i][r] * s.V[j][r];
        C[i][j] = v + (i === j ? s.D[i] : 0);
      }
  } else if (s.kind === "Blocks") {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let v = (s.cluster[i] === s.cluster[j]) ? s.loading[i] * s.loading[j] : 0;
        C[i][j] = v + (i === j ? s.D[i] : 0);
      }
  } else if (s.kind === "Tree") {
    const anc = [];
    const nc = Math.max(...s.cluster) + 1;
    for (let c = 0; c < nc; c++) {
      const a = new Set();
      let u = c;
      while (s.parent[u] >= 0) { a.add(s.parent[u]); u = s.parent[u]; }
      anc.push(a);
    }
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let v = 0;
        for (const t of anc[s.cluster[i]])
          if (anc[s.cluster[j]].has(t)) v += s.strength[t] ** 2;
        if (s.cluster[i] === s.cluster[j]) v += s.loading[i] * s.loading[j];
        C[i][j] = v + (i === j ? s.D[i] : 0);
      }
  }
  return C;
}

export function cholesky(Cin) {
  const n = Cin.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = Cin[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][i] = Math.sqrt(Math.max(s, 1e-10));
      else L[i][j] = s / L[j][j];
    }
  }
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
import { ndtr, npdf } from "./core.mjs";

function invNormalCdf(p) { return invNormal(Math.min(Math.max(p, 1e-15), 1 - 1e-15)); }

function diffProblem(mu, C, i) {
  // mean and covariance of (x_j - x_i)_{j != i}
  const n = mu.length, m = new Float64Array(n - 1);
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
  const { m, S } = diffProblem(mu, C, i);
  const d = mu.length - 1;
  const L = new Float64Array(d * d);
  for (let a = 0; a < d; a++) {
    for (let b = 0; b <= a; b++) {
      let s = S[a * d + b];
      for (let k = 0; k < b; k++) s -= L[a * d + k] * L[b * d + k];
      if (a === b) L[a * d + a] = Math.sqrt(Math.max(s, 1e-12));
      else L[a * d + b] = s / L[b * d + b];
    }
  }
  return { m, L, d };
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
      const Fa = ndtr(a), q = 1 - Fa;
      w *= q;
      if (q < 1e-14) { w = 0; break; }
      e[k] = invNormalCdf(Fa + r.u() * q);
    }
    sum += w;
  }
  return sum;
}

export function mendellElstonOne(mu, C, i) {
  // Mendell-Elston analytic sequential moment approximation: condition on
  // u_k > 0 one coordinate at a time, propagating truncated-normal moments
  // and pretending normality is preserved (it is not; the bias is the
  // flat line this arm draws).
  const { m, S } = diffProblem(mu, C, i);
  const d = mu.length - 1;
  let logp = 0;
  const alive = [];
  for (let a = 0; a < d; a++) alive.push(a);
  while (alive.length) {
    const k = alive.shift();
    const skk = Math.max(S[k * d + k], 1e-12), sk = Math.sqrt(skk);
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
