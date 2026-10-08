// One race, five covariance grammars -- port of winning/factor/structures.py.
import { raceProbabilities, abilitiesFromRace, _setDispatch } from "./races.mjs";
import { blockRaceProbabilities, nestedRaceProbabilities, treeRaceProbabilities,
         abilitiesFromBlockRace, validateTree } from "./blocks.mjs";
import { mean, asIterations, asTolerance, asFiniteVector, isVector, rescaledTarget,
         asLoadings, asIdio } from "./core.mjs";

export const Independent = D => ({ kind: "Independent", D });
export const Factor = (V, D) => ({ kind: "Factor", V, D });
export const Blocks = (cluster, loading, D) => ({ kind: "Blocks", cluster, loading, D });
export const Nested = (cluster, loading, D, coupling, gamma = 1.0) =>
  ({ kind: "Nested", cluster, loading, D, coupling, gamma });
export const Tree = (cluster, loading, D, parent, strength) =>
  ({ kind: "Tree", cluster, loading, D, parent, strength });

/* the tree race whose implied correlation IS the cophenetic matrix of a
   scipy-style linkage; negative cophenetic correlation floored at zero.
   A non-monotonic linkage (an inversion) is refused, as in python. */
export function treeFromLinkage(Z) {
  const n = Z.length + 1;
  const nT = 2 * n - 1;
  const parent = new Array(nT).fill(-1);
  // d[t] = 1 - rho_t = 2 h^2, kept directly: 1 - (1 - 2 h^2) cancels to a
  // few ulps for near-duplicate leaves (python Tree.from_linkage, #430)
  const d = new Array(nT).fill(1);
  for (let k = 0; k < Z.length; k++) {
    const a = Math.round(Z[k][0]), b = Math.round(Z[k][1]), h = Z[k][2];
    const t = n + k;
    parent[a] = t; parent[b] = t;
    d[t] = Math.min(2 * h * h, 1);
  }
  const strength = new Array(nT).fill(0);
  // the nonnegative increments are a PREMISE, checked as in python's
  // Tree.from_linkage: centroid and median linkage invert, and clipping
  // the negative increment returns a covariance that is not the
  // cophenetic one promised -- here D = [1, 1, 0.5] for a unit-variance
  // model (#133)
  const bad = [];
  for (let t = n; t < nT; t++) {
    const pa = parent[t];
    const lam2 = (pa >= 0 ? d[pa] : 1) - d[t];      // rho_t - rho_pa
    if (lam2 < -1e-9) bad.push([t, lam2]);
    strength[t] = Math.sqrt(Math.max(lam2, 0));
  }
  if (bad.length) {
    const [t0, d0] = bad.reduce((w, x) => (x[1] < w[1] ? x : w));
    throw new Error(
      `this linkage is not monotonic: node ${t0} merges ` +
      `${Number((-d0).toPrecision(3))} BELOW its parent, and ${bad.length} ` +
      "node(s) do. A tree race is a nested variance decomposition, so an " +
      "inversion has no representation in it -- clipping the negative " +
      "increment would return a different covariance from the cophenetic " +
      "one this promises. Use a monotonic method (average, complete, " +
      "ward), or build the Tree with explicit parent/strength.");
  }
  const D = [];
  for (let i = 0; i < n; i++) {
    // A leaf with no parent only happens for the one-leaf tree, which an
    // EMPTY linkage is the valid scipy-style spelling of. javascript has
    // no negative indexing, so rho[-1] was undefined and D came out
    // [NaN]; pricing that tree then failed instead of returning the
    // certain probability [1] (#241). Python gets the same answer by
    // accident -- numpy wraps rho[-1] to the single zero entry -- and the
    // guard on the line above this loop was already written correctly.
    const pa = parent[i];
    // EXACTLY 2 h^2: the old absolute floor Math.max(., 1e-10) changed
    // every near-duplicate branch (h = 1e-6 priced 0.556 where the
    // cophenetic model gives Phi(1) = 0.841), #430
    D.push(pa >= 0 ? d[pa] : 1);
  }
  const zero = D.findIndex((v) => !(v > 0));
  if (zero >= 0)
    throw new Error(
      `leaf ${zero} merges at height 0: coincident leaves have zero ` +
      "idiosyncratic variance, which a tree race cannot price. Merge the " +
      "duplicates, or perturb them deliberately.");
  return Tree([...Array(n).keys()], new Array(n).fill(0), D, parent, strength);
}

/* scale: the field's contrast scale. A dimensionless start and an
   absolute step made the same Nested/Tree race miss by 0.40 of a share
   at c = 1e-3 (#100). */
export function invertGeneric(p, forward, tol = 1e-9, maxIter = 400, scale = 1) {
  return invertGenericInfo(p, forward, tol, maxIter, scale).mu;
}

/* The same iteration, with what it achieved. A target here is already
   validated and normalised by the caller (abilitiesFromRace owns the
   contract, #387), so the old 1e-300 clamp no longer turns a zero share
   into a finite "inverse"; it stays only as protection for the logs of
   the FORWARD's own output. */
function invertGenericInfo(p, forward, tol = 1e-9, maxIter = 400, scale = 1) {
  asIterations(maxIter, "invertGeneric", "maxIter");
  asTolerance(tol, "invertGeneric");
  let pv = asFiniteVector(p, "target", "probability");
  if (pv.some(v => v <= 0))
    throw new Error(
      "all target probabilities must be positive: a zero share has no " +
      "finite inverse. Pass targetFloor to abilitiesFromRace to floor " +
      "small entries deliberately.");
  pv = rescaledTarget(pv);                 // overflow-safe (#463)
  const lt = pv.map(v => Math.log(v));
  const lm = mean(lt);
  let mu = lt.map(v => -(v - lm) * scale);
  let eta = 1.0;
  let lp = forward(mu).map(v => Math.log(Math.max(v, 1e-300)));
  let err = Math.max(...lp.map((v, i) => Math.abs(v - lt[i])));
  let iters = 0;
  for (let it = 0; it < maxIter && err >= tol; it++) {
    iters = it + 1;
    let muN = mu.map((m, i) => m - eta * (lt[i] - lp[i]) * scale);
    const mm = mean(muN);
    muN = muN.map(v => v - mm);
    const lpN = forward(muN).map(v => Math.log(Math.max(v, 1e-300)));
    const e = Math.max(...lpN.map((v, i) => Math.abs(v - lt[i])));
    if (e < err) { mu = muN; lp = lpN; err = e; eta = Math.min(eta * 1.2, 1.5); }
    else { eta *= 0.5; if (eta < 1e-4) break; }
  }
  return { mu, converged: err < tol, maxLogResidual: err, iterations: iters };
}

/* the hierarchical kernels are Gaussian hard races on their own lattice
   windows: base/window/delta were dropped silently (#89) */
function refuseHierarchical(kind, opts) {
  const { base = "normal", window: win = "bulk", delta = 1e-12 } = opts;
  const bad = [];
  if (base !== "normal") bad.push("base");
  if (win !== "bulk") bad.push("window");
  if (delta !== 1e-12) bad.push("delta");
  if (bad.length)
    throw new Error(`${kind} races do not support ${bad.join(", ")}: the ` +
      "block/nested/tree kernels are Gaussian hard races on their own " +
      "lattice windows. Use structure Factor (or V/D).");
}

function structureScale(s) {
  if (!isVector(s.cluster))
    throw new Error(`${s.kind}: cluster must have one label per contestant`);
  const n = s.cluster.length;
  // canonical shapes, as the forwards read them: raw indexing made a
  // scalar D, a scalar or (1, n) loading and a scalar coupling NaN (#597)
  const L = asLoadings(s.loading, n, "loading");
  const Dv = asIdio(s.D, n, "D");
  const G = s.kind === "Nested" && s.coupling != null ? asLoadings(s.coupling, n, "coupling") : null;
  const tot = [];
  for (let i = 0; i < n; i++) {
    let v = Dv[i] + L[i].reduce((a, b) => a + b * b, 0);
    if (G) v += (s.gamma ?? 1) ** 2 * G[i].reduce((a, b) => a + b * b, 0);
    tot.push(v);
  }
  if (s.kind === "Tree") {
    // the walk below follows parent pointers: validate first, as the
    // kernels do, or a cyclic tree spins here before they can refuse it
    validateTree(s.parent, s.strength, new Set(s.cluster).size, "Tree");
    // add each leaf cluster's ancestor variance (labels remapped as the
    // kernel does: sorted unique labels are leaf node ids)
    const labels = [...new Set(s.cluster)].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    const idx = new Map(labels.map((v, k) => [v, k]));
    for (let i = 0; i < n; i++) {
      let u = idx.get(s.cluster[i]), a = 0;
      while (s.parent[u] >= 0) { u = s.parent[u]; a += s.strength[u] ** 2; }
      tot[i] += a;
    }
  }
  tot.sort((a, b) => a - b);
  const h = Math.floor(n / 2);
  return Math.sqrt(n % 2 ? tot[h] : 0.5 * (tot[h - 1] + tot[h]));
}

function dp(mu, s, opts) {
  const { base = "normal", points = 257, qa = 9, qf = 15, returnSlopes = false,
          window: win = "bulk", delta = 1e-12 } = opts;
  // window and delta are forwarded: they used to be validated and then
  // dropped, so a structured window: "span" priced the bulk window (#89)
  if (s.kind === "Independent")
    return raceProbabilities(mu, { D: s.D, base, points, returnSlopes, window: win, delta });
  if (s.kind === "Factor")
    return raceProbabilities(mu, { V: s.V, D: s.D, base, points, returnSlopes, window: win, delta });
  refuseHierarchical(s.kind, opts);
  if (returnSlopes) throw new Error("returnSlopes: Independent/Factor only");
  if (s.kind === "Blocks")
    return blockRaceProbabilities(mu, s.cluster, s.loading, s.D, { points, qa });
  if (s.kind === "Nested")
    return nestedRaceProbabilities(mu, s.cluster, s.loading, s.D,
                                   { coupling: s.coupling, gamma: s.gamma, points, qa, qf });
  if (s.kind === "Tree")
    return treeRaceProbabilities(mu, s.cluster, s.loading, s.D, s.parent, s.strength,
                                 { points, qa });
  throw new Error("unknown structure " + s.kind);
}
/* Structured inversion. Returns {mu, converged, maxLogResidual,
   iterations} for every grammar, so abilitiesFromRace reports one info
   shape whichever grammar it was given (#387). The target arrives
   validated, floored and normalised. */
function da(p, s, opts) {
  // base reaches the V/D race, and the hierarchical kernels refuse the
  // controls they would otherwise silently ignore (#89)
  // nIter is the caller's when given: a hierarchical solve defaults to 400
  // sweeps, but an explicit budget is honoured and reported (#89)
  const { points = 257, qa = 9, qf = 15, nIter, tol = 1e-8, base = "normal" } = opts;
  if (s.kind === "Independent" || s.kind === "Factor") {
    const r = abilitiesFromRace(p, { V: s.kind === "Factor" ? s.V : null,
                                     D: s.D, points, base, nIter: nIter ?? 60, tol, returnInfo: true });
    return r;
  }
  refuseHierarchical(s.kind, opts);
  if (s.kind === "Blocks") {
    const r = abilitiesFromBlockRace(p, s.cluster, s.loading, s.D,
                                     { points, qa, tol, maxIter: nIter ?? 25 });
    return { mu: r.mu, converged: r.residual < tol, maxLogResidual: r.residual,
             iterations: r.iterations };
  }
  if (s.kind === "Nested")
    return invertGenericInfo(p, m => nestedRaceProbabilities(m, s.cluster, s.loading, s.D,
      { coupling: s.coupling, gamma: s.gamma, points, qa, qf }), tol, nIter ?? 400, structureScale(s));
  if (s.kind === "Tree")
    return invertGenericInfo(p, m => treeRaceProbabilities(m, s.cluster, s.loading, s.D,
      s.parent, s.strength, { points, qa }), tol, nIter ?? 400, structureScale(s));
  throw new Error("unknown structure " + s.kind);
}
_setDispatch(dp, da);

/* average-linkage agglomerative clustering on a distance matrix,
   returning a scipy-style linkage Z (for treeFromLinkage and demos) */
/* A distance matrix: nonempty, square, finite, symmetric to a relative
   1e-8, zero diagonal, non-negative. averageLinkage copied the UPPER
   triangle only, so a matrix and its transpose built different trees
   and moved a tree-race share by 9.7 points, where scipy's squareform
   refuses both (#361). Tolerance-level asymmetry is averaged away. */
export function asDistanceMatrix(dist, where = "averageLinkage") {
  if (!isVector(dist) || dist.length === 0)
    throw new Error(`${where}: dist must be a nonempty square matrix`);
  const n = dist.length;
  const M = Array.from(dist, (row, i) => {
    if (!isVector(row) || row.length !== n)
      throw new Error(`${where}: dist must be square; row ${i} has ` +
                      `${isVector(row) ? row.length : "no"} entries for ${n} rows`);
    return Array.from(row, v => {
      if (typeof v !== "number" || !Number.isFinite(v))
        throw new Error(`${where}: dist has a non-finite entry (${v}) in row ${i}`);
      return v;
    });
  });
  let amax = 0;
  for (const r of M) for (const v of r) amax = Math.max(amax, Math.abs(v));
  const tol = 1e-8 * Math.max(amax, 1e-300);
  for (let i = 0; i < n; i++) {
    if (Math.abs(M[i][i]) > tol)
      throw new Error(`${where}: dist[${i}][${i}] = ${M[i][i]}; a distance matrix has a zero diagonal`);
    for (let j = 0; j < n; j++) {
      if (Math.abs(M[i][j] - M[j][i]) > tol)
        throw new Error(
          `${where}: distance matrix must be symmetric: dist[${i}][${j}] = ` +
          `${M[i][j]} but dist[${j}][${i}] = ${M[j][i]}`);
      if (M[i][j] < -tol)
        throw new Error(`${where}: dist[${i}][${j}] = ${M[i][j]} is a negative distance`);
    }
  }
  return M.map((r, i) => r.map((v, j) => (i === j ? 0 : Math.max(0.5 * v + 0.5 * M[j][i], 0))));
}

export function averageLinkage(dist) {
  dist = asDistanceMatrix(dist);
  const n = dist.length;
  let active = [...Array(n).keys()].map(i => ({ id: i, members: [i] }));
  const D = new Map();
  const key = (a, b) => (a < b ? a + "_" + b : b + "_" + a);
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) D.set(key(i, j), dist[i][j]);
  const Z = [];
  let nextId = n;
  while (active.length > 1) {
    let best = Infinity, bi = 0, bj = 1;
    for (let i = 0; i < active.length; i++)
      for (let j = i + 1; j < active.length; j++) {
        const d = D.get(key(active[i].id, active[j].id));
        if (d < best) { best = d; bi = i; bj = j; }
      }
    const a = active[bi], b = active[bj];
    Z.push([a.id, b.id, best, a.members.length + b.members.length]);
    const merged = { id: nextId++, members: a.members.concat(b.members) };
    const rest = active.filter((_, k) => k !== bi && k !== bj);
    for (const c of rest) {
      const da = D.get(key(a.id, c.id)), db = D.get(key(b.id, c.id));
      D.set(key(merged.id, c.id),
        (da * a.members.length + db * b.members.length)
          / (a.members.length + b.members.length));
    }
    active = rest.concat([merged]);
  }
  return Z;
}

/* cut a linkage into at most k clusters: scipy fcluster(Z, k,
   criterion="maxclust").

   This removed the k-1 last linkage ROWS, which is not a distance cut
   when merge heights tie: two equal-height merges were split by row
   order, making three clusters where scipy -- for which a tied level is
   indivisible -- returns two, and a block race moved 3.3 points (#346).
   maxclust takes the smallest threshold t (over "below every merge" and
   the merge heights) at which no more than k clusters remain, merging
   every node whose subtree maximum height is <= t (scipy's monocrit
   maxdists, so an inverted linkage is handled the same way). */
export function cutLinkage(Z, n, k) {
  if (!Number.isInteger(n) || n < 1)
    throw new Error(`cutLinkage: n must be a positive integer; got ${n}`);
  if (!Number.isInteger(k) || k < 1)
    throw new Error(`cutLinkage: k must be a positive integer number of clusters; got ${k}`);
  if (!isVector(Z) || Z.length !== n - 1)
    throw new Error(`cutLinkage: a linkage of ${n} leaves has ${n - 1} rows; got ${isVector(Z) ? Z.length : typeof Z}`);
  const nT = 2 * n - 1;
  const parent = new Array(nT).fill(-1);
  const maxd = new Array(nT).fill(-Infinity);
  for (let m = 0; m < Z.length; m++) {
    const a = Math.round(Z[m][0]), b = Math.round(Z[m][1]), h = Number(Z[m][2]);
    if (!(a >= 0 && a < n + m && b >= 0 && b < n + m) || !Number.isFinite(h))
      throw new Error(`cutLinkage: row ${m} of the linkage is malformed`);
    // each cluster is merged exactly once, and never with itself: a
    // self-merge left the tree disconnected and k = 1 returned two
    // clusters (#603), as scipy's is_valid_linkage refuses
    if (a === b || parent[a] !== -1 || parent[b] !== -1)
      throw new Error(`cutLinkage: row ${m} uses the same cluster more than once`);
    parent[a] = n + m;
    parent[b] = n + m;
    maxd[n + m] = Math.max(h, maxd[a], maxd[b]);
  }
  const clustersAt = t => {
    // a leaf belongs to its highest ancestor whose subtree max <= t
    const root = new Array(n);
    for (let i = 0; i < n; i++) {
      let u = i;
      while (parent[u] >= 0 && maxd[parent[u]] <= t) u = parent[u];
      root[i] = u;
    }
    return root;
  };
  const thresholds = [-Infinity, ...[...new Set(maxd.slice(n))].sort((x, y) => x - y)];
  let root = null;
  for (const t of thresholds) {
    const r = clustersAt(t);
    if (new Set(r).size <= k) { root = r; break; }
  }
  if (root === null) root = clustersAt(Infinity);
  const labels = new Array(n);
  const map = new Map();
  for (let i = 0; i < n; i++) {
    if (!map.has(root[i])) map.set(root[i], map.size);
    labels[i] = map.get(root[i]);
  }
  return labels;
}
