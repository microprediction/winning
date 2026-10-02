// One race, five covariance grammars -- port of winning/factor/structures.py.
import { raceProbabilities, abilitiesFromRace, _setDispatch } from "./races.mjs";
import { blockRaceProbabilities, nestedRaceProbabilities, treeRaceProbabilities,
         abilitiesFromBlockRace, blockScale } from "./blocks.mjs";
import { mean } from "./core.mjs";

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
   at c = 1e-3 (#100). returnInfo reports convergence: the bare array
   used to come back silently after the hidden budget (#89). */
export function invertGeneric(p, forward, tol = 1e-9, maxIter = 400,
                              scale = 1, returnInfo = false) {
  let pv = p.slice();
  const s = pv.reduce((a, b) => a + b, 0);
  pv = pv.map(v => v / s);
  const lt = pv.map(v => Math.log(Math.max(v, 1e-300)));
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
  if (returnInfo)
    return { mu, converged: err < tol, maxLogResidual: err, iterations: iters };
  if (!(err < tol))
    console.warn(`structured inverse did not converge: max |log residual| ` +
                 `${err.toExponential(2)} (tol ${tol})`);
  return mu;
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
  const n = s.D.length;
  const tot = [];
  for (let i = 0; i < n; i++) {
    const L = Array.isArray(s.loading[i]) ? s.loading[i] : [Number(s.loading[i])];
    let v = Number(s.D[i]) + L.reduce((a, b) => a + b * b, 0);
    if (s.kind === "Nested" && s.coupling) {
      const g = Array.isArray(s.coupling[i]) ? s.coupling[i] : [Number(s.coupling[i])];
      v += (s.gamma ?? 1) ** 2 * g.reduce((a, b) => a + b * b, 0);
    }
    tot.push(v);
  }
  if (s.kind === "Tree") {
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
function da(p, s, opts) {
  // base, nIter, tol and returnInfo are the caller's: every branch used
  // to drop them and run hidden defaults, a logistic request calibrated a
  // Gaussian race, and a stalled Tree solve came back as a bare array (#89)
  const { points = 257, qa = 9, qf = 15, base = "normal", nIter, tol,
          returnInfo = false, targetFloor = null } = opts;
  const ctl = {};
  if (nIter !== undefined) ctl.nIter = nIter;
  if (tol !== undefined) ctl.tol = tol;
  if (s.kind === "Independent")
    return abilitiesFromRace(p, { D: s.D, points, base, returnInfo, targetFloor, ...ctl });
  if (s.kind === "Factor")
    return abilitiesFromRace(p, { V: s.V, D: s.D, points, base, returnInfo, targetFloor, ...ctl });
  refuseHierarchical(s.kind, opts);
  const tolH = tol ?? 1e-9;
  let fwd;
  if (s.kind === "Blocks")
    fwd = m => blockRaceProbabilities(m, s.cluster, s.loading, s.D, { points, qa });
  else if (s.kind === "Nested")
    fwd = m => nestedRaceProbabilities(m, s.cluster, s.loading, s.D,
      { coupling: s.coupling, gamma: s.gamma, points, qa, qf });
  else if (s.kind === "Tree")
    fwd = m => treeRaceProbabilities(m, s.cluster, s.loading, s.D,
      s.parent, s.strength, { points, qa });
  else throw new Error("unknown structure " + s.kind);
  if (s.kind === "Blocks") {
    const out = abilitiesFromBlockRace(p, s.cluster, s.loading, s.D,
      { points, qa, tol: tol ?? 1e-10, maxIter: nIter ?? 25 });
    if (!returnInfo) return out.mu;
    const lt = p.map(v => Math.log(v / p.reduce((a, b) => a + b, 0)));
    const lp = fwd(out.mu).map(v => Math.log(Math.max(v, 1e-300)));
    const err = Math.max(...lp.map((v, i) => Math.abs(v - lt[i])));
    return { mu: out.mu, converged: err < (tol ?? 1e-8), maxLogResidual: err,
             iterations: out.iterations };
  }
  return invertGeneric(p, fwd, tolH, nIter ?? 400, structureScale(s), returnInfo);
}
_setDispatch(dp, da);

/* average-linkage agglomerative clustering on a distance matrix,
   returning a scipy-style linkage Z (for treeFromLinkage and demos) */
export function averageLinkage(dist) {
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

/* cut a linkage into k clusters (fcluster maxclust equivalent) */
export function cutLinkage(Z, n, k) {
  const parent = new Array(2 * n - 1).fill(-1);
  for (let m = 0; m < Z.length; m++) {
    parent[Z[m][0]] = n + m;
    parent[Z[m][1]] = n + m;
  }
  // remove the k-1 highest merges
  const cutIds = new Set();
  for (let m = Z.length - 1; m >= Z.length - (k - 1) && m >= 0; m--) cutIds.add(n + m);
  const labels = new Array(n);
  const rootOf = i => {
    let u = i;
    while (parent[u] >= 0 && !cutIds.has(parent[u])) u = parent[u];
    return u;
  };
  const map = new Map();
  for (let i = 0; i < n; i++) {
    const r = rootOf(i);
    if (!map.has(r)) map.set(r, map.size);
    labels[i] = map.get(r);
  }
  return labels;
}
