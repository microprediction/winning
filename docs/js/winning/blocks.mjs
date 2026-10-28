// Block, nested and tree races -- port of winning/factor/blocks.py.
import { TINY, ndtr, npdf, hermite1, mean, solve, interpClamped, checkOpts,
         OPT_HINTS, asLoadings, firstPrimes, asAbilities, asIdio, isVector,
         asFactorNodes, asWeights, asFactorLaw, asIterations, asTolerance,
         asFiniteVector, rescaledTarget, canonicalBasis, asPoints, gaugeCenter } from "./core.mjs";
import { invNormalRational, translatedHome } from "./races.mjs";

/* Each exported call declares its own option keys; see checkOpts in
   core.mjs for why an options object needs this at all. */
const BLOCK_RACE_PROBABILITIES_OPTS = new Set(["points", "qa", "nodes"]);
const NESTED_RACE_PROBABILITIES_OPTS = new Set(["points", "qa", "qf", "coupling", "gamma"]);
const TREE_RACE_PROBABILITIES_OPTS = new Set(["points", "qa", "leaves"]);
const BLOCK_RACE_JACOBIAN_OPTS = new Set(["points", "qa"]);
const NESTED_RACE_JACOBIAN_OPTS = new Set(["points", "qa", "qf", "coupling", "gamma"]);
const TREE_RACE_JACOBIAN_OPTS = new Set(["points", "qa", "leaves"]);
const ABILITIES_FROM_BLOCK_RACE_OPTS = new Set(["points", "qa", "maxIter", "tol"]);


// Exported because a SECOND reader of cluster labels existed and had a
// private rule: demo.mjs's structureCov indexed by the raw label, so a
// tree relabelled 0,1,2 -> 10,20,30 silently lost every ancestor factor
// and negative or string labels threw (#306). One place decides what a
// label means, as the python side decided in #146.
export function clusterIndex(cluster) {
  const lv = [...new Set(cluster)].sort((a, b) => (a > b ? 1 : a < b ? -1 : 0));
  const map = new Map(lv.map((v, i) => [v, i]));
  return cluster.map(c => map.get(c));
}

/* Each runner's LEAF NODE index in a tree (python blocks.tree_leaf_index).
   leaves == null keeps the sorted-label order, so labels 0..nC-1 are the
   leaf nodes; any other spelling lands on leaves by its sort order, and
   renaming [0, 0, 1, 2] to [30, 30, 10, 20] priced a different tree when
   the leaves sit at unequal depths (#372). leaves[j] = the label of leaf
   node j states the map explicitly. */
export function treeLeafIndex(cluster, leaves = null) {
  if (leaves == null) return clusterIndex(cluster);
  if (!isVector(leaves))
    throw new Error("leaves must be an array: leaves[j] is the label of leaf node j (#372)");
  const pos = new Map();
  leaves.forEach((lab, j) => {
    if (pos.has(lab)) throw new Error(`leaves names label ${JSON.stringify(lab)} twice (#372)`);
    pos.set(lab, j);
  });
  const seen = new Array(leaves.length).fill(false);
  const out = cluster.map((lab, i) => {
    if (!pos.has(lab))
      throw new Error(`runner ${i} has cluster label ${JSON.stringify(lab)}, which leaves does not map to a leaf node (#372)`);
    seen[pos.get(lab)] = true;
    return pos.get(lab);
  });
  const miss = seen.indexOf(false);
  if (miss >= 0)
    throw new Error(`leaf ${miss} (label ${JSON.stringify(leaves[miss])}) carries no runner: every leaf node must hold a cluster (#372)`);
  return out;
}

/* A Tree structure's runner-to-leaf indices (0..nC-1): the explicit map
   when given, the sorted-label order otherwise. */
export const treeCluster = s => treeLeafIndex(s.cluster, s.leaves);

/* One door for the block/nested/tree arguments.

   The leaf-cluster `loading` was read raw: the rank came from
   loading[0].length and each row was reduced over its OWN length, so a
   transposed (1, n) or (r, n) spelling -- the same race in python --
   threw or went NaN, and a RAGGED row was silently zero-padded, pricing a
   different covariance (a 2.4-point share move) with no complaint (#335).
   The loading now goes through asLoadings like V does everywhere else;
   mu and D through their own doors (#440, #254). */
function blockArgs(mu, cluster, loading, D, where) {
  mu = asAbilities(mu, "mu");
  const n = mu.length;
  if (!isVector(cluster) || cluster.length !== n)
    throw new Error(
      `${where}: cluster must have one label per contestant; got ` +
      `${isVector(cluster) ? cluster.length : typeof cluster} for ${n}`);
  if (loading == null)
    throw new Error(`${where}: loading is required (use 0 for no cluster effect)`);
  let L = asLoadings(loading, n, "loading");
  // rank zero is the independent race: as a zero rank-one loading, not
  // the rank >= 3 route's 8192 identical empty nodes (2.2 s vs 4 ms, #508)
  if (L.length && L[0].length === 0) L = L.map(() => [0]);
  L = oneClusterGauge(L, cluster);
  D = asIdio(D, n, "D");
  // the origin is a gauge: at a common offset of 1e13 the absolute grid
  // collapsed below the float spacing and every grammar threw (or its
  // Jacobian came back 0) though the contrasts were exact (#477)
  return { mu: translatedHome(mu, D), cluster: Array.from(cluster), L, D };
}

/* With ONE cluster its loading shock is shared by every contestant and
   cancels from every comparison, so only the centred loadings are
   identified: a common c in every row widened the lattice until c = 40
   threw a 2.5% mass defect, and the Jacobian moved by 0.074 at c = 50
   (#471). Scale-safe centring (core's gaugeCenter); never per cluster
   when there are several, where a cluster's common part is a real
   between-cluster effect. */
function oneClusterGauge(L, cluster) {
  for (let i = 1; i < cluster.length; i++) if (cluster[i] !== cluster[0]) return L;
  return gaugeCenter(L);
}

/* A tree is a tree: one root, every parent an existing node, no cycles,
   and the cluster nodes are leaves. The kernels follow parent pointers
   with `while (par[u] >= 0)`, so a cycle -- [1, 0], or a self-parent --
   spun synchronously forever and froze the page (#328). Checked once,
   in linear time, before any walk. */
/* Strengths with every COMMON ancestor's set to zero (port of python's
   blocks._without_common_shock). A node above every cluster adds the same
   shock to everyone and cancels from every difference; kept, it widened
   the lattice window, so root strength 50 on a race identical to the
   root-zero one raised a 3.7% mass defect at 257 points (#489). `parent`
   uses -1 for the root. */
export function withoutCommonShock(parent, strength, nC) {
  const lam = Array.from(strength, Number);
  const hits = new Array(parent.length).fill(0);
  for (let c = 0; c < nC; c++) {
    let u = c;
    while (parent[u] >= 0) { u = parent[u]; hits[u] += 1; }
  }
  for (let t = 0; t < lam.length; t++) if (hits[t] === nC) lam[t] = 0;
  return lam;
}

export function validateTree(parent, strength, nC, where = "tree") {
  if (!isVector(parent))
    throw new Error(`${where}: parent must be an array of node indices`);
  const nT = parent.length;
  if (!isVector(strength) || strength.length !== nT)
    throw new Error(
      `${where}: strength must have one entry per tree node; got ` +
      `${isVector(strength) ? strength.length : typeof strength} for ${nT}`);
  const par = Array.from(parent);
  for (let t = 0; t < nT; t++) {
    const u = par[t];
    if (!Number.isInteger(u) || u < -1 || u >= nT)
      throw new Error(`${where}: parent[${t}] = ${u} is not -1 or a node index in [0, ${nT})`);
    if (u === t)
      throw new Error(`${where}: parent contains a cycle: ${t} -> ${t}`);
    if (typeof strength[t] !== "number" || !Number.isFinite(strength[t]))
      throw new Error(`${where}: strength[${t}] = ${strength[t]} is not finite`);
  }
  const roots = [];
  for (let t = 0; t < nT; t++) if (par[t] < 0) roots.push(t);
  if (roots.length !== 1)
    throw new Error(
      `${where}: a tree has exactly one root (parent -1); got ${roots.length}` +
      (roots.length ? ` (${roots.join(", ")})` : ""));
  // cycle check: every node must reach the root within nT hops
  const state = new Uint8Array(nT);         // 0 unseen, 1 on path, 2 done
  for (let t = 0; t < nT; t++) {
    const path = [];
    let u = t;
    while (u >= 0 && state[u] === 0) { state[u] = 1; path.push(u); u = par[u]; }
    if (u >= 0 && state[u] === 1) {
      const k = path.indexOf(u);
      throw new Error(
        `${where}: parent contains a cycle: ${path.slice(k).concat([u]).join(" -> ")}`);
    }
    for (const v of path) state[v] = 2;
  }
  if (nC > nT)
    throw new Error(`${where}: ${nC} clusters but only ${nT} tree nodes`);
  for (let t = 0; t < nT; t++)
    if (par[t] >= 0 && par[t] < nC)
      throw new Error(
        `${where}: node ${t} has parent ${par[t]}, a cluster node; the first ` +
        `${nC} nodes are the clusters and must be leaves`);
  return { parent: par, strength: Array.from(strength, Number) };
}

function stableOrder(inv) {
  return inv.map((v, i) => i).sort((a, b) => inv[a] - inv[b] || a - b);
}
// Winner-bulk window covering every RETAINED conditional race (max-wins):
// per-runner conditional locations range over [mu - amp, mu + amp] with amp
// the largest shared-effect shift the quadrature nodes can produce, and the
// IDIOSYNCRATIC sd sets the local scale. Replaces an independent-marginal
// proxy whose lower crossing drifted upward with n under near-common shocks
// while the winner sat O(1) lower (28% of mass lost on a 400-runner cluster
// at correlation 0.99; fourteenth review).
function windowNodes(mu, sd, amp, delta = 1e-12, padSds = 2) {
  const n = mu.length;
  const mLo = mu.map((m, i) => m - amp[i]);
  const mHi = mu.map((m, i) => m + amp[i]);
  // the field's own scale (sds are positive): an absolute 1e-12 floor
  // stopped the window shrinking with the model, and the same block race
  // in units of 1e-14 lost 0.7% of its mass and raised (#100)
  const smax = Math.max(...sd);
  const F = (x, m) => {
    let ls = 0;
    for (let i = 0; i < n; i++) ls += Math.log(Math.max(ndtr((x - m[i]) / sd[i]), TINY));
    return Math.exp(ls);
  };
  let lo = Math.min(...mLo) - 9 * smax;
  let step = 9 * smax;
  for (let it = 0; it < 60; it++) {
    if (F(lo, mLo) <= delta) break;
    lo -= step; step *= 2;
  }
  let hi = Math.max(...mHi) + 9 * smax;
  step = 9 * smax;
  for (let it = 0; it < 60; it++) {
    if (F(hi, mHi) >= 1 - delta) break;
    hi += step; step *= 2;
  }
  let a = lo, b = hi;
  for (let it = 0; it < 70; it++) {
    const m = 0.5 * (a + b);
    if (F(m, mLo) < delta) a = m; else b = m;
  }
  const xlo = a;
  a = xlo; b = hi;
  for (let it = 0; it < 70; it++) {
    const m = 0.5 * (a + b);
    if (F(m, mHi) < 1 - delta) a = m; else b = m;
  }
  return [xlo - padSds * smax, b + padSds * smax];
}

// Raw lattice mass is a diagnostic, not a nuisance: a material defect means
// the window missed the winner's region, and normalizing it away returns
// confident wrong shares. Stop instead.
/* The mass tolerance in force (python blocks._MASS_TOL): an inverse's
   internal trial states are priced under a looser one, and the state it
   returns is re-priced strictly (#656). */
let MASS_TOL = 5e-3;
export function withTrialMassTolerance(fn) {
  const prev = MASS_TOL;
  MASS_TOL = Math.max(prev, 0.05);
  try { return fn(); } finally { MASS_TOL = prev; }
}

function checkedMass(raw, kind, massTol = MASS_TOL) {
  const t = raw.reduce((a, b) => a + b, 0);
  // NaN compares false against any tolerance, so test finiteness
  // explicitly or a NaN mass sails through the check
  if (!Number.isFinite(t) || Math.abs(t - 1) > massTol) {
    throw new Error(
      `${kind} lattice captured total mass ${t.toFixed(4)} (defect ` +
      `${Math.abs(t - 1).toExponential(2)} > ${massTol}): the window missed ` +
      `part of the winner distribution and the shares would be silently ` +
      `wrong if normalized. Raise points, or report this field: the ` +
      `node-aware window should have covered it.`);
  }
  return raw.map(v => v / t);
}

function clusterNodes(r, qa) {
  const h = hermite1(qa);
  if (r === 1) return { nodes: h.nodes.map(x => [x]), w: h.weights.slice() };
  if (r === 2) {
    const nodes = [], w = [];
    for (let a = 0; a < qa; a++) for (let b = 0; b < qa; b++) {
      nodes.push([h.nodes[a], h.nodes[b]]);
      w.push(h.weights[a] * h.weights[b]);
    }
    const s = w.reduce((x, y) => x + y, 0);
    return { nodes, w: w.map(v => v / s) };
  }
  // rank >= 3: the tensor grid explodes, so escalate the FAMILY, as
  // races.mjs does for high-rank loadings and as python does with
  // scrambled Sobol. The old message told the caller to "supply nodes
  // explicitly", which no nested verb accepts -- checkOpts rejects a
  // `nodes` option before pricing -- so it was a remediation that could
  // not be followed (#264).
  const Q = 8192;
  const nodes = [], w = new Array(Q).fill(1 / Q);
  const primes = firstPrimes(r);
  for (let idx = 0; idx < Q; idx++) {
    const node = [];
    for (let dim = 0; dim < r; dim++) {
      const b = primes[dim];
      let i = idx + 21, f = 1 / b, h = 0;
      while (i > 0) { h += f * (i % b); i = Math.floor(i / b); f /= b; }
      node.push(invNormalRational(Math.min(Math.max(h, 1e-12), 1 - 1e-12)));
    }
    nodes.push(node);
  }
  return { nodes, w };
}

/* Each cluster's rank-r effect is its own N(0, I) draw, so a separate
   orthogonal map per cluster -- V_c -> V_c Q_c -- is the SAME race. The
   node cloud is not invariant under those maps: moving one block's two
   active columns from positions 0, 1 to 68, 69 of an 80-column loading
   moved a share by 0.195, and swapping them by 0.234, through the
   high-dimensional Halton projections (#130). Each cluster's rows are
   put in canonical form (canonicalBasis: principal axes, null
   directions dropped, cube-sum signs) and zero-padded to the largest
   cluster rank, so the rule depends on each V_c V_c' alone. Rank two
   too (python's blocks._canonical_block_loadings): the product
   Gauss-Hermite rule is symmetric under column signs and swaps but not
   rotations, and a 45-degree rotation of one block moved a share by
   0.0052 (#787). At rank two the product grid is laid at 45 degrees to
   the principal axes, where its default error was ~10x smaller than on
   them. A rank-one loading is Gauss-Hermite, symmetric, and kept. */
function canonicalBlockLoadings(L, cluster) {
  if (L[0].length < 2) return L;
  const inv = clusterIndex(cluster);
  const nC = Math.max(...inv) + 1;
  const members = Array.from({ length: nC }, () => []);
  inv.forEach((c, i) => members[c].push(i));
  const parts = members.map(ix => canonicalBasis(ix.map(i => L[i])).C);
  const R = Math.max(1, ...parts.map(C => (C.length ? C[0].length : 0)));
  const out = L.map(() => new Array(R).fill(0));
  members.forEach((ix, c) => ix.forEach((i, k) => {
    parts[c][k].forEach((v, j) => { out[i][j] = v; });
  }));
  if (R !== 2) return out;
  const h = Math.SQRT1_2;          // python's _ROT45: [[h, -h], [h, h]]
  return out.map(([a, b]) => [h * a + h * b, -h * a + h * b]);
}

/* python's blocks._warn_if_sharp: the fixed Gauss-Hermite cluster rule
   does not escalate, and past sharpness 3 (loading norm over
   idiosyncratic sd) its error lands on group shares -- 5e-2 TV at
   sharpness 18. Once per (qa, rounded sharpness), not once per call. */
const SHARP_WARNED = new Set();
function warnIfSharp(V, sd, qa, threshold = 3.0) {
  let s = 0;
  V.forEach((row, i) => {
    s = Math.max(s, Math.hypot(...row) / Math.max(sd[i], 1e-300));
  });
  const key = `${qa}:${s.toPrecision(2)}`;
  if (!(s > threshold) || SHARP_WARNED.has(key) || typeof console === "undefined") return;
  SHARP_WARNED.add(key);
  console.warn(
    `block/nested quadrature is a fixed ${qa}-node Gauss-Hermite rule and ` +
    `this field's cluster sharpness is ${s.toPrecision(2)} (loading over ` +
    `idiosyncratic sd); past ${threshold} the conditional integrand is a ` +
    "near-step and Gauss-Hermite fails at any order, on group shares. " +
    "Restructure sharp shared effects into the factor grammar, which " +
    "escalates its node family; raising qa does not fix this.");
}

/* The nested coupling is one global factor: V -> V Q is the same law,
   so its node rule is built for the canonical coupling (#130). */
function canonicalCoupling(g) {
  if (g[0].length < 3) return g;     // tensor GH in the caller's axes, as python
  // g -> g + 1c' adds one common shock c'f to every contestant: the same
  // race. Canonicalizing the raw Gram matrix let that shift rotate the
  // Halton cloud and moved a share by 0.0015 (#698); centre first.
  const { C } = canonicalBasis(gaugeCenter(g));
  return C[0].length ? C : g.map(() => [0]);
}

/* The caller-node override goes through the same doors as F/W in the
   factor race: a node matrix of the loading's rank and exactly one
   finite non-negative weight per node, normalised. Signed weights moved
   a share by 0.17 and surplus weights were silently ignored (#290). */
function checkedNodes(o, r) {
  if (!o || typeof o !== "object" || !("nodes" in o) || !("w" in o))
    throw new Error("nodes must be {nodes, w}: the factor nodes and their weights");
  // paired, with null nodes dropped: zero-weight padding sized the
  // block window and threw "captured total mass 0" (#416)
  const law = asFactorLaw(o.nodes, o.w, r, "nodes.nodes", "nodes.w");
  return { nodes: law.F, w: law.W };
}

function fieldPass(muO, sdO, shifts, cO, nC, x) {
  // returns S[c][q][t], logF[i][q][t], pdf[i][q][t]
  const n = muO.length, Q = shifts[0].length, P = x.length;
  const S = Array.from({ length: nC }, () => Array.from({ length: Q }, () => new Array(P).fill(0)));
  const logF = [], pdf = [];
  for (let i = 0; i < n; i++) {
    const li = [], pi = [];
    for (let q = 0; q < Q; q++) {
      const lrow = new Array(P), prow = new Array(P);
      const mi = muO[i] + shifts[i][q], si = sdO[i];
      for (let t = 0; t < P; t++) {
        const z = (x[t] - mi) / si;
        lrow[t] = Math.log(Math.max(ndtr(z), TINY));
        prow[t] = npdf(z) / si;
        S[cO[i]][q][t] += lrow[t];
      }
      li.push(lrow); pi.push(prow);
    }
    logF.push(li); pdf.push(pi);
  }
  return { S, logF, pdf };
}

function blockMax(mu, sd, cluster, loading, points, qa, nodesOverride) {
  // `loading` is the canonical (n, r) matrix from blockArgs
  const n = mu.length;
  const isMat = true;
  const r = loading[0].length;
  const inv = clusterIndex(cluster);
  const ord = stableOrder(inv);
  const muO = ord.map(i => mu[i]), sdO = ord.map(i => sd[i]);
  const VO = ord.map(i => (isMat ? loading[i] : [loading[i]]));
  const cO = ord.map(i => inv[i]);
  const nC = Math.max(...cO) + 1;
  const { nodes, w } = nodesOverride
    ? checkedNodes(nodesOverride, r) : clusterNodes(r, qa);
  if (!nodesOverride && r <= 2) warnIfSharp(VO, sdO, qa);
  const Q = nodes.length;
  const shifts = VO.map(vi => nodes.map(nq => vi.reduce((a, b, k) => a + b * nq[k], 0)));
  // a custom rule is sized by the shifts it actually makes: the bound
  // ||v|| max ||f|| counted node coordinates in the loadings' null space,
  // and an irrelevant +-1000 there failed a 257-point call on mass (#807).
  // The automatic rule keeps its bound.
  let amp;
  if (nodesOverride) {
    amp = shifts.map(row => Math.max(...row.map(Math.abs)));
  } else {
    const maxNodeNorm = Math.max(...nodes.map(nq => Math.sqrt(nq.reduce((a, b) => a + b * b, 0))));
    amp = VO.map(vi => Math.sqrt(vi.reduce((a, b) => a + b * b, 0)) * maxNodeNorm);
  }
  const [lo, hi] = windowNodes(muO, sdO, amp);
  const P = asPoints(points);              // #444
  const x = new Array(P);
  for (let t = 0; t < P; t++) x[t] = lo + t * (hi - lo) / (P - 1);
  const dx = x[1] - x[0];
  const { S, logF, pdf } = fieldPass(muO, sdO, shifts, cO, nC, x);
  const G = Array.from({ length: nC }, () => new Array(P).fill(0));
  for (let c = 0; c < nC; c++)
    for (let q = 0; q < Q; q++)
      for (let t = 0; t < P; t++) G[c][t] += w[q] * Math.exp(Math.min(S[c][q][t], 0));
  const logGAll = new Array(P).fill(0);
  const logG = G.map(gc => gc.map(v => Math.log(Math.max(v, TINY))));
  for (let c = 0; c < nC; c++) for (let t = 0; t < P; t++) logGAll[t] += logG[c][t];
  const pO = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const c = cO[i];
    let acc = 0;
    for (let t = 0; t < P; t++) {
      let h = 0;
      for (let q = 0; q < Q; q++)
        h += w[q] * pdf[i][q][t] * Math.exp(Math.min(S[c][q][t] - logF[i][q][t], 0));
      const rest = Math.exp(Math.min(logGAll[t] - logG[c][t], 0));
      acc += h * rest;
    }
    pO[i] = Math.max(acc * dx, 0);
  }
  const p = new Array(n);
  ord.forEach((orig, pos) => { p[orig] = pO[pos]; });
  return p;
}

export function blockRaceProbabilities(mu, cluster, loading, D, opts = {}) {
  checkOpts(opts, BLOCK_RACE_PROBABILITIES_OPTS, "blockRaceProbabilities", OPT_HINTS);
  const { points = 257, qa = 9, nodes = null } = opts;
  const a = blockArgs(mu, cluster, loading, D, "blockRaceProbabilities");
  const sd = a.D.map(Math.sqrt);
  // a caller's nodes are in the caller's coordinates, so only the
  // automatic rule gets the canonical loadings
  const L = nodes ? a.L : canonicalBlockLoadings(a.L, a.cluster);
  const p = blockMax(a.mu.map(v => -v), sd, a.cluster, L, points, qa, nodes);
  return checkedMass(p, "block race");
}

/* (V, D) of a two-runner Blocks, Nested or Tree race as a factor model
   with the same covariance, after the grammar's own doors (port of
   python's structures._pair_factor, #86): one column block per cluster
   holding the member loadings, Nested's coupling scaled by gamma, and one
   column per tree node holding its strength for every runner beneath. */
export function pairFactor(mu, s) {
  const where = `${s.kind} race`;
  if (s.kind === "Tree") {
    const a = blockArgs(mu, treeCluster(s), s.loading, s.D, where);
    if (a.L[0].length > 1)
      throw new Error("tree races take scalar (rank-one) leaf-cluster loadings; rank-r " +
                      "leaf effects are supported by the block grammar only.");
    const inv = clusterIndex(a.cluster);
    const nC = Math.max(...inv) + 1;
    const tv = validateTree(s.parent, s.strength, nC, where);
    const nT = tv.parent.length;
    const V = a.mu.map((_, i) => {
      const row = new Array(nC + nT).fill(0);
      row[inv[i]] = a.L[i][0];
      for (let u = inv[i]; tv.parent[u] >= 0; u = tv.parent[u])
        row[nC + tv.parent[u]] = tv.strength[tv.parent[u]];
      return row;
    });
    return { V, D: a.D };
  }
  if (s.kind !== "Blocks" && s.kind !== "Nested") throw new Error("unknown structure " + s.kind);
  const a = blockArgs(mu, s.cluster, s.loading, s.D, where);
  const inv = clusterIndex(a.cluster), r = a.L[0].length;
  const nC = Math.max(...inv) + 1;
  const V = a.L.map((li, i) => {
    const row = new Array(nC * r).fill(0);
    for (let j = 0; j < r; j++) row[inv[i] * r + j] = li[j];
    return row;
  });
  if (s.kind === "Nested") {
    asGamma(s.gamma, "nestedRaceProbabilities");
    if (s.coupling != null && s.gamma !== 0) {
      const g = canonicalCoupling(asLoadings(s.coupling, 2, "coupling"));
      g.forEach((gi, i) => { for (const v of gi) V[i].push(s.gamma * v); });
    }
  }
  return { V, D: a.D };
}

/* gamma (g @ node), the global factor's shift (port of python's
   blocks._coupled_shift). Only gamma g is identified: coupling c g with
   gamma / c is the same race, but at c = 1e308 the raw g @ node
   overflowed before gamma was applied (NaN lattice mass, #722). That
   spelling is kept wherever it is finite; otherwise gamma is absorbed
   into g first. The sign of gamma is not identified either (gamma f and
   -gamma f are the same Gaussian law), but the rank-3+ Halton rule is
   not symmetric under f -> -f, so gamma = -1 priced differently from
   gamma = 1 (#871): |gamma| is used. */
function coupledShift(g, gamma, node) {
  gamma = Math.abs(gamma);
  const s = g.map(gi => gi.reduce((a, b, r) => a + b * node[r], 0));
  if (s.every(Number.isFinite)) return s.map(v => gamma * v);
  return g.map(gi => gi.reduce((a, b, r) => a + (gamma * b) * node[r], 0));
}

export function asGamma(gamma, where) {
  if (typeof gamma !== "number" || !Number.isFinite(gamma))
    throw new Error(`${where}: gamma must be a finite number; got ${typeof gamma === "string" ? JSON.stringify(gamma) : String(gamma)}`);
  return gamma;
}

export function nestedRaceProbabilities(mu, cluster, loading, D, opts = {}) {
  checkOpts(opts, NESTED_RACE_PROBABILITIES_OPTS, "nestedRaceProbabilities", OPT_HINTS);
  const { coupling = null, gamma = 1.0, points = 257, qa = 9, qf = 15 } = opts;
  // gamma and coupling validated BEFORE the no-global-factor shortcut:
  // truthiness sent NaN/false/"" couplings to the block race (#586), and
  // the Jacobian let "2", true, false and null select a coupling the
  // forward refuses (#540). Only an omitted coupling (null/undefined) or
  // an exact numeric gamma of 0 means no global factor.
  asGamma(gamma, "nestedRaceProbabilities");
  if (coupling == null || gamma === 0)
    return blockRaceProbabilities(mu, cluster, loading, D, { points, qa });
  const a = blockArgs(mu, cluster, loading, D, "nestedRaceProbabilities");
  mu = a.mu;
  // The shared contract, as python's atleast_2d-plus-transpose does
  // it: a (rank, n) coupling is the SAME race as (n, rank). This
  // normalised by hand and never transposed, so the transposed
  // spelling was read as rank n and refused (#264).
  const g = canonicalCoupling(asLoadings(coupling, mu.length, "coupling"));
  const k = g[0].length;
  let fn, fw;
  if (k === 1) {
    const h = hermite1(qf);
    fn = h.nodes.map(x => [x]); fw = h.weights;
  } else {
    const cn = clusterNodes(k, qf);
    fn = cn.nodes; fw = cn.w;
  }
  const n = mu.length;
  const sd = a.D.map(Math.sqrt);
  const La = canonicalBlockLoadings(a.L, a.cluster);     // #130
  const p = new Array(n).fill(0);
  for (let q = 0; q < fn.length; q++) {
    // average the RAW conditional masses (each near one) and normalize
    // once: normalizing each conditional separately hides a window defect
    const shift = coupledShift(g, gamma, fn[q]);
    const shifted = mu.map((m, i) => -(m + shift[i]));
    const pq = blockMax(shifted, sd, a.cluster, La, points, qa, null);
    for (let i = 0; i < n; i++) p[i] += fw[q] * pq[i];
  }
  return checkedMass(p, "nested race");
}

/* tree machinery shared by forward and jacobian */
function treeInternals(mu, cluster, loading, D, parent, strength, points, qa) {
  const a = blockArgs(mu, cluster, loading, D, "tree race");
  if (a.L[0].length > 1) {
    throw new Error(
      "tree races take scalar (rank-one) leaf-cluster loadings; rank-r " +
      "leaf effects are supported by the block grammar only.");
  }
  loading = a.L.map(v => v[0]);
  cluster = a.cluster;
  const m = a.mu.map(v => -v);
  const sd = a.D.map(Math.sqrt);
  const inv = clusterIndex(cluster);
  const nC = Math.max(...inv) + 1;
  const tv = validateTree(parent, strength, nC, "tree race");   // #328
  const lam = withoutCommonShock(tv.parent, tv.strength, nC);   // #489
  const par = tv.parent;                             // -1 = root (python style)
  const n = m.length, nT = par.length;
  const ord = stableOrder(inv);
  const muO = ord.map(i => m[i]), sdO = ord.map(i => sd[i]);
  const vO = ord.map(i => loading[i]), cO = ord.map(i => inv[i]);
  const h = hermite1(qa);
  const an = h.nodes, aw = h.weights;
  const depth = new Array(nT).fill(0);        // |lam| path sums, for the window
  // traversal order must be TREE depth in hops, not the |lam| path sum:
  // zero strengths (from_linkage's floored merges) tie the path sums and a
  // tied sort visits children before their parents, reading cavities still
  // at their initial value (raw mass 3.0 on a 6-leaf zero-strength linkage).
  const depthHops = new Array(nT).fill(0);
  for (let t = 0; t < nT; t++) {
    let s = 0, d = 0, u = t;
    while (par[u] >= 0) { s += Math.abs(lam[par[u]]); d += 1; u = par[u]; }
    depth[t] = s;
    depthHops[t] = d;
  }
  const pathVar = new Array(nC).fill(0);
  for (let c = 0; c < nC; c++) {
    let s = 0, u = c;
    while (par[u] >= 0) { s += lam[par[u]] * lam[par[u]]; u = par[u]; }
    pathVar[c] = s;
  }
  const maxAn = Math.max(...an.map(Math.abs));
  const amp = muO.map((mm, i) => (Math.abs(vO[i]) + depth[cO[i]]) * maxAn);
  const [lo, hi] = windowNodes(muO, sdO, amp);
  const P = asPoints(points);              // #444
  const x = new Array(P);
  for (let t = 0; t < P; t++) x[t] = lo + t * (hi - lo) / (P - 1);
  const dx = x[1] - x[0];
  const shifts = vO.map(vi => an.map(a => vi * a));
  const { S, logF, pdf } = fieldPass(muO, sdO, shifts, cO, nC, x);
  const G = Array.from({ length: nT }, () => new Array(P).fill(0));
  for (let c = 0; c < nC; c++)
    for (let q = 0; q < aw.length; q++)
      for (let t = 0; t < P; t++) G[c][t] += aw[q] * Math.exp(Math.min(S[c][q][t], 0));
  const children = Array.from({ length: nT }, () => []);
  let root = -1;
  for (let t = 0; t < nT; t++) {
    if (par[t] >= 0) children[par[t]].push(t); else root = t;
  }
  const shiftEval = (g, delta) => x.map(xt => interpClamped(xt + delta, x, g));
  const up = [];
  for (let t = nC; t < nT; t++) up.push(t);
  up.sort((a, b) => depthHops[b] - depthHops[a] || a - b);
  for (const t of up) {
    const acc = new Array(P).fill(0);
    for (let q = 0; q < aw.length; q++) {
      const prod = new Array(P).fill(1);
      for (const c of children[t]) {
        const sh = shiftEval(G[c], lam[t] * an[q]);
        for (let tt = 0; tt < P; tt++) prod[tt] *= sh[tt];
      }
      for (let tt = 0; tt < P; tt++) acc[tt] += aw[q] * prod[tt];
    }
    G[t] = acc.map(v => Math.max(v, 0));
  }
  const R = Array.from({ length: nT }, () => new Array(P).fill(1));
  const down = [];
  for (let t = 0; t < nT; t++) down.push(t);
  down.sort((a, b) => depthHops[a] - depthHops[b] || a - b);
  for (const t of down) {
    const pa = par[t];
    if (pa < 0) continue;
    const sm = new Array(P).fill(0);
    for (let q = 0; q < aw.length; q++) {
      const sh = shiftEval(R[pa], -lam[pa] * an[q]);
      for (let tt = 0; tt < P; tt++) sm[tt] += aw[q] * sh[tt];
    }
    const prod = new Array(P).fill(1);
    for (const s of children[pa]) if (s !== t)
      for (let tt = 0; tt < P; tt++) prod[tt] *= G[s][tt];
    R[t] = sm.map((v, tt) => Math.max(v * prod[tt], 0));
  }
  const hMat = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(P).fill(0);
    const c = cO[i];
    for (let q = 0; q < aw.length; q++)
      for (let t = 0; t < P; t++)
        row[t] += aw[q] * pdf[i][q][t] * Math.exp(Math.min(S[c][q][t] - logF[i][q][t], 0));
    hMat.push(row);
  }
  return { n, nC, nT, ord, cO, x, dx, S, logF, pdf, G, R, hMat, root, aw };
}

export function treeRaceProbabilities(mu, cluster, loading, D, parent, strength, opts = {}) {
  checkOpts(opts, TREE_RACE_PROBABILITIES_OPTS, "treeRaceProbabilities", OPT_HINTS);
  const { points = 257, qa = 9, leaves = null } = opts;
  if (leaves != null) cluster = treeLeafIndex(cluster, leaves);      // (#372)
  const I = treeInternals(mu, cluster, loading, D, parent, strength, points, qa);
  const pO = new Array(I.n).fill(0);
  for (let i = 0; i < I.n; i++) {
    let acc = 0;
    const rc = I.R[I.cO[i]];
    for (let t = 0; t < I.x.length; t++) acc += I.hMat[i][t] * rc[t];
    pO[i] = Math.max(acc * I.dx, 0);
  }
  const p = new Array(I.n);
  I.ord.forEach((orig, pos) => { p[orig] = pO[pos]; });
  return checkedMass(p, "tree race");
}

function withinBlockTerm(J, I, negate = true) {
  // exact same-cluster tie terms, overwriting J[idx][idx]
  const P = I.x.length;
  for (let c = 0; c < I.nC; c++) {
    const idx = [];
    for (let i = 0; i < I.n; i++) if (I.cO[i] === c) idx.push(i);
    if (idx.length === 1) continue;
    const Rc = I.R ? I.R[c] : I._rest[c];
    for (let a = 0; a < idx.length; a++) {
      for (let b = 0; b < idx.length; b++) {
        let term = 0;
        for (let q = 0; q < I.aw.length; q++) {
          let s = 0;
          for (let t = 0; t < P; t++) {
            const lo2 = Math.exp(Math.min(
              I.S[c][q][t] - I.logF[idx[a]][q][t] - I.logF[idx[b]][q][t], 0));
            s += I.pdf[idx[a]][q][t] * I.pdf[idx[b]][q][t] * lo2 * Rc[t];
          }
          term += I.aw[q] * s;
        }
        J[idx[a]][idx[b]] = -term * I.dx;
      }
    }
  }
}

/* The FORWARD block kernel prices rank-r per-cluster loadings; this
   Jacobian is written for rank one only. It takes each loading row as
   a number -- `Math.abs(vO[i])` for the quadrature amplitude, `vi * a`
   for the shift -- and javascript coerces a ONE-element array to its
   number, so an (n, 1) column keeps working, but a rank-2 row is NaN
   at both. The NaN then propagates through every cell, so the call
   returned an all-NaN matrix and said nothing (#271).

   python's `block_race_jacobian` detects this and raises; the browser
   matches it rather than inventing a second behaviour. Every other
   Jacobian door here -- nestedRaceJacobian, the structured raceJacobian
   front door, polishRace -- funnels through this function, so the one
   guard covers all of them.

   This is not #212 (a finite-grid derivative mismatch for loadings
   that ARE supported) and not #264 (the nested `coupling` rank). */
function assertRankOneLoading(loading, where) {
  if (!Array.isArray(loading)) return;
  const rows = loading.filter(Array.isArray);
  if (rows.length === 0) return;                 // plain numbers: rank one
  const rank = Math.max(...rows.map(r => r.length));
  if (rank > 1 && loading.length > 1)
    throw new Error(
      `${where} supports rank-one cluster loadings only; the rank-${rank} ` +
      `block Jacobian is not implemented (the FORWARD rank-${rank} kernel ` +
      `is). Use finite differences of blockRaceProbabilities, or the ` +
      `factor grammar if the loadings are global.`);
}

export function blockRaceJacobian(mu, cluster, loading, D, opts = {}) {
  checkOpts(opts, BLOCK_RACE_JACOBIAN_OPTS, "blockRaceJacobian", OPT_HINTS);
  const { points = 257, qa = 9 } = opts;
  const ba = blockArgs(mu, cluster, loading, D, "blockRaceJacobian");
  // after normalisation, so a (1, n) rank-one row is rank one here too
  // and the rank-r refusal of #271 sees the canonical rank (#335)
  assertRankOneLoading(ba.L, "blockRaceJacobian");
  loading = ba.L.map(v => v[0]);
  cluster = ba.cluster;
  const m = ba.mu.map(v => -v);
  const sd = ba.D.map(Math.sqrt);
  const inv = clusterIndex(cluster);
  const ord = stableOrder(inv);
  const muO = ord.map(i => m[i]), sdO = ord.map(i => sd[i]);
  const vO = ord.map(i => loading[i]), cO = ord.map(i => inv[i]);
  const n = m.length, nC = Math.max(...cO) + 1;
  const h = hermite1(qa);
  const an = h.nodes, aw = h.weights;
  const maxAnJ = Math.max(...an.map(Math.abs));
  const ampJ = muO.map((mm, i) => Math.abs(vO[i]) * maxAnJ);
  const [lo, hi] = windowNodes(muO, sdO, ampJ);
  const P = asPoints(points);              // #444
  const x = new Array(P);
  for (let t = 0; t < P; t++) x[t] = lo + t * (hi - lo) / (P - 1);
  const dx = x[1] - x[0];
  const shifts = vO.map(vi => an.map(a => vi * a));
  const { S, logF, pdf } = fieldPass(muO, sdO, shifts, cO, nC, x);
  const G = Array.from({ length: nC }, () => new Array(P).fill(0));
  for (let c = 0; c < nC; c++)
    for (let q = 0; q < aw.length; q++)
      for (let t = 0; t < P; t++) G[c][t] += aw[q] * Math.exp(Math.min(S[c][q][t], 0));
  const logG = G.map(gc => gc.map(v => Math.log(Math.max(v, TINY))));
  const logGAll = new Array(P).fill(0);
  for (let c = 0; c < nC; c++) for (let t = 0; t < P; t++) logGAll[t] += logG[c][t];
  const Rc = logG.map(lg => lg.map((v, t) => Math.exp(Math.min(logGAll[t] - v, 0))));
  const hMat = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(P).fill(0);
    for (let q = 0; q < aw.length; q++)
      for (let t = 0; t < P; t++)
        row[t] += aw[q] * pdf[i][q][t] * Math.exp(Math.min(S[cO[i]][q][t] - logF[i][q][t], 0));
    hMat.push(row);
  }
  const U = hMat.map((row, i) => row.map((v, t) =>
    v * Rc[cO[i]][t] / Math.sqrt(Math.max(Math.exp(Math.min(logGAll[t], 0)), TINY)) * Math.sqrt(dx)));
  const J = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let t = 0; t < P; t++) s += U[i][t] * U[j][t];
      J[i][j] = -s;
    }
  withinBlockTerm(J, { n, nC, cO, x, dx, S, logF, pdf, aw, R: Rc });
  for (let i = 0; i < n; i++) J[i][i] = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += J[i][j];
    J[i][i] = -s;
  }
  const Jf = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let a = 0; a < n; a++)
    for (let b = 0; b < n; b++) Jf[ord[a]][ord[b]] = J[a][b];
  return Jf.map(row => row.map(v => -v));         // min-wins chain rule
}

export function nestedRaceJacobian(mu, cluster, loading, D, opts = {}) {
  checkOpts(opts, NESTED_RACE_JACOBIAN_OPTS, "nestedRaceJacobian", OPT_HINTS);
  const { coupling = null, gamma = 1.0, points = 257, qa = 9, qf = 15 } = opts;
  asGamma(gamma, "nestedRaceJacobian");     // as the forward (#540, #586)
  if (coupling == null || gamma === 0)
    return blockRaceJacobian(mu, cluster, loading, D, { points, qa });
  mu = asAbilities(mu);
  mu = translatedHome(mu, asIdio(D, mu.length, "D"));     // before the node shifts (#477)
  // The shared contract, as python's atleast_2d-plus-transpose does
  // it: a (rank, n) coupling is the SAME race as (n, rank). This
  // normalised by hand and never transposed, so the transposed
  // spelling was read as rank n and refused (#264).
  const g = canonicalCoupling(asLoadings(coupling, mu.length, "coupling"));
  const k = g[0].length;
  let fn, fw;
  if (k === 1) {
    const h = hermite1(qf);
    fn = h.nodes.map(x => [x]); fw = h.weights;
  } else {
    const cn = clusterNodes(k, qf);
    fn = cn.nodes; fw = cn.w;
  }
  const n = mu.length;
  const J = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let q = 0; q < fn.length; q++) {
    const shift = coupledShift(g, gamma, fn[q]);
    const shifted = mu.map((m, i) => m + shift[i]);
    const Jq = blockRaceJacobian(shifted, cluster, loading, D, { points, qa });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) J[i][j] += fw[q] * Jq[i][j];
  }
  return J;
}

export function treeRaceJacobian(mu, cluster, loading, D, parent, strength, opts = {}) {
  checkOpts(opts, TREE_RACE_JACOBIAN_OPTS, "treeRaceJacobian", OPT_HINTS);
  const { points = 257, qa = 9, leaves = null } = opts;
  if (leaves != null) cluster = treeLeafIndex(cluster, leaves);      // (#372)
  // With any ancestor effect the cross-cluster Gram term is not the
  // derivative (it factorises the two-leaf cavity as R_i R_j / G_root):
  // two leaves under a common root priced invariantly in the root
  // strength while J[0][1] moved 0.2467 -> 0.1792 at strength 0.8 (#350).
  // There the matrix is the central difference of the exact forward, as
  // in python's tree_race_jacobian.
  const nC = new Set(cluster).size;
  // a common shock cancels: without it a bare root takes the exact path
  const tvJ = validateTree(parent, strength, nC, "treeRaceJacobian");
  if (withoutCommonShock(tvJ.parent, tvJ.strength, nC).slice(nC).some(v => v !== 0)) {
    // the step from the CANONICAL D and loadings: raw D.map threw for a
    // scalar D and a typed loading made the step NaN, though the forward
    // accepts both spellings (#596)
    const a = blockArgs(mu, cluster, loading, D, "treeRaceJacobian");
    mu = a.mu;
    const n = mu.length;
    const tot = a.D.map((d, i) => d + a.L[i].reduce((s2, v) => s2 + v * v, 0));
    const srt = tot.slice().sort((a, b) => a - b);
    const med = n % 2 ? srt[(n - 1) / 2] : 0.5 * (srt[n / 2 - 1] + srt[n / 2]);
    const h = 1e-5 * Math.sqrt(med);
    const J = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      const up = mu.slice(), dn = mu.slice();
      up[j] += h; dn[j] -= h;
      const pu = treeRaceProbabilities(up, cluster, loading, D, parent, strength, { points, qa });
      const pd = treeRaceProbabilities(dn, cluster, loading, D, parent, strength, { points, qa });
      for (let i = 0; i < n; i++) J[i][j] = (pu[i] - pd[i]) / (2 * h);
    }
    return J;
  }
  const I = treeInternals(mu, cluster, loading, D, parent, strength, points, qa);
  const P = I.x.length;
  const Gr = I.G[I.root].map(v => Math.max(v, TINY));
  const U = I.hMat.map((row, i) => row.map((v, t) =>
    v * I.R[I.cO[i]][t] / Math.sqrt(Gr[t]) * Math.sqrt(I.dx)));
  const J = Array.from({ length: I.n }, () => new Array(I.n).fill(0));
  for (let i = 0; i < I.n; i++)
    for (let j = 0; j < I.n; j++) {
      let s = 0;
      for (let t = 0; t < P; t++) s += U[i][t] * U[j][t];
      J[i][j] = -s;
    }
  withinBlockTerm(J, I);
  for (let i = 0; i < I.n; i++) J[i][i] = 0;
  for (let i = 0; i < I.n; i++) {
    let s = 0;
    for (let j = 0; j < I.n; j++) s += J[i][j];
    J[i][i] = -s;
  }
  const Jf = Array.from({ length: I.n }, () => new Array(I.n).fill(0));
  for (let a = 0; a < I.n; a++)
    for (let b = 0; b < I.n; b++) Jf[I.ord[a]][I.ord[b]] = J[a][b];
  return Jf.map(row => row.map(v => -v));
}

/* sqrt(median(D_i + |loading_i|^2)): the block race's contrast scale */
export function blockScale(loading, D, n, cluster = null) {
  // through the SAME shape contract the forwards use: indexing raw
  // arguments made a scalar D, a scalar or (1, n) loading, or a scalar
  // coupling NaN, and the inverse refused its own first iterate (#597)
  let L = asLoadings(loading, n, "loading");
  if (isVector(cluster) && cluster.length === n) L = oneClusterGauge(L, cluster);   // #471
  const Dv = asIdio(D, n, "D");
  const tot = [];
  for (let i = 0; i < n; i++) tot.push(Dv[i] + L[i].reduce((a, b) => a + b * b, 0));
  tot.sort((a, b) => a - b);
  const h = Math.floor(n / 2);
  return Math.sqrt(n % 2 ? tot[h] : 0.5 * (tot[h - 1] + tot[h]));
}

export function abilitiesFromBlockRace(pTarget, cluster, loading, D, opts = {}) {
  checkOpts(opts, ABILITIES_FROM_BLOCK_RACE_OPTS, "abilitiesFromBlockRace", OPT_HINTS);
  const { points = 257, qa = 9, tol = 1e-10, maxIter = 25 } = opts;
  // the budget is a loop bound: 0/negative/NaN skipped Newton and
  // returned the globaliser's state, a fraction ran Math.ceil of itself
  // while reporting the fraction, and Infinity with a sub-machine tol
  // spun forever on a step no line search could accept (#421)
  asIterations(maxIter, "abilitiesFromBlockRace", "maxIter");
  asTolerance(tol, "abilitiesFromBlockRace");
  let pT = asFiniteVector(pTarget, "target", "probability");
  if (pT.some(v => v < 0))
    throw new Error("abilitiesFromBlockRace: a target probability is negative");
  pT = rescaledTarget(pT);                 // overflow-safe (#463)
  const n = pT.length;
  const floor = Math.max(1e-14, Math.min(...pT.filter(v => v > 0)) * 1e-3);
  pT = pT.map(v => Math.max(v, floor));
  const s = pT.reduce((a, b) => a + b, 0);
  pT = pT.map(v => v / s);
  const lt = pT.map(Math.log);
  const forward = m => blockRaceProbabilities(m, cluster, loading, D, { points, qa });
  const lm = mean(lt);
  // the field's own scale: the start, the fixed-point step and the Newton
  // trust radius were absolute numbers, so loading -> c loading,
  // D -> c^2 D missed by 0.40 of a share at c = 1e-3 (#100, as python)
  const scale = blockScale(loading, D, n, cluster);
  const residualAt = m => {
    let pv = forward(m).map(v => Math.max(v, TINY));
    const sv = pv.reduce((a, b) => a + b, 0);
    pv = pv.map(v => v / sv);
    return { pv, r: pv.map((v, i) => Math.log(v) - lt[i]) };
  };
  // trial states (warm start, line search) under the looser trial mass
  // tolerance: they only steer, and the strict 0.5% check rejected a
  // self-generated target at its own warm start (#656). The returned
  // state is re-priced below under the strict check.
  const solved = withTrialMassTolerance(() => {
    let mu = lt.map(v => -(v - lm) * scale);
    let eta = 1.0;
    let lp = forward(mu).map(v => Math.log(Math.max(v, TINY)));
    let err = Math.max(...lp.map((v, i) => Math.abs(v - lt[i])));
    for (let it = 0; it < 200 && err >= 0.2; it++) {
      let muN = mu.map((m, i) => m - eta * (lt[i] - lp[i]) * scale);
      const mm = mean(muN);
      muN = muN.map(v => v - mm);
      const lpN = forward(muN).map(v => Math.log(Math.max(v, TINY)));
      const eN = Math.max(...lpN.map((v, i) => Math.abs(v - lt[i])));
      if (eN < err) { mu = muN; lp = lpN; err = eN; eta = Math.min(eta * 1.2, 1.5); }
      else { eta *= 0.5; if (eta < 1e-4) break; }
    }
    let iterations = 0;
    for (let it = 0; it < maxIter; it++) {
      const { pv, r } = residualAt(mu);
      const cur = Math.max(...r.map(Math.abs));
      if (cur < tol) return { mu, iterations };
      iterations = it + 1;
      const J = blockRaceJacobian(mu, cluster, loading, D, { points, qa });
      const A = J.map((row, i) => row.map(v => v / pv[i] * scale + 1 / n));
      let step = solve(A, r.map(v => -v));
      const nn = Math.sqrt(step.reduce((a, b) => a + b * b, 0));
      if (nn > 5) step = step.map(v => v * 5 / nn);
      step = step.map(v => v * scale);
      let accepted = false;
      for (let k = 0; k < 8; k++) {
        let muN = mu.map((m, i) => m + step[i]);
        const mm = mean(muN);
        muN = muN.map(v => v - mm);
        let pN = forward(muN).map(v => Math.max(v, TINY));
        const sN = pN.reduce((a, b) => a + b, 0);
        pN = pN.map(v => v / sN);
        if (Math.max(...pN.map((v, i) => Math.abs(Math.log(v) - lt[i]))) < cur) {
          mu = muN; accepted = true; break;
        }
        step = step.map(v => v * 0.5);
      }
      // no line-search step improved: the state cannot change, so further
      // sweeps would repeat this one exactly. Stop and report (#421).
      if (!accepted) break;
    }
    return { mu, iterations };
  });
  const mu = solved.mu, iterations = solved.iterations;
  const { pv } = residualAt(mu);
  return { mu: mu.map(v => v - mean(mu)),
           residual: Math.max(...pv.map((v, i) => Math.abs(Math.log(v) - lt[i]))),
           iterations };
}
