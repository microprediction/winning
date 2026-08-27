// One race, five covariance grammars -- port of winning/factor/structures.py.
import { raceProbabilities, abilitiesFromRace, _setDispatch } from "./races.mjs";
import { blockRaceProbabilities, nestedRaceProbabilities, treeRaceProbabilities,
         abilitiesFromBlockRace } from "./blocks.mjs";
import { mean } from "./core.mjs";

export const Independent = D => ({ kind: "Independent", D });
export const Factor = (V, D) => ({ kind: "Factor", V, D });
export const Blocks = (cluster, loading, D) => ({ kind: "Blocks", cluster, loading, D });
export const Nested = (cluster, loading, D, coupling, gamma = 1.0) =>
  ({ kind: "Nested", cluster, loading, D, coupling, gamma });
export const Tree = (cluster, loading, D, parent, strength) =>
  ({ kind: "Tree", cluster, loading, D, parent, strength });

/* the tree race whose implied correlation IS the cophenetic matrix of a
   scipy-style linkage; negative cophenetic correlation floored at zero */
export function treeFromLinkage(Z) {
  const n = Z.length + 1;
  const nT = 2 * n - 1;
  const parent = new Array(nT).fill(-1);
  const rho = new Array(nT).fill(0);
  for (let k = 0; k < Z.length; k++) {
    const a = Math.round(Z[k][0]), b = Math.round(Z[k][1]), h = Z[k][2];
    const t = n + k;
    parent[a] = t; parent[b] = t;
    rho[t] = Math.max(1 - 2 * h * h, 0);
  }
  const strength = new Array(nT).fill(0);
  for (let t = n; t < nT; t++) {
    const pa = parent[t];
    strength[t] = Math.sqrt(Math.max(rho[t] - (pa >= 0 ? rho[pa] : 0), 0));
  }
  const D = [];
  for (let i = 0; i < n; i++) D.push(Math.max(1 - rho[parent[i]], 1e-10));
  return Tree([...Array(n).keys()], new Array(n).fill(0), D, parent, strength);
}

export function invertGeneric(p, forward, tol = 1e-9, maxIter = 400) {
  let pv = p.slice();
  const s = pv.reduce((a, b) => a + b, 0);
  pv = pv.map(v => v / s);
  const lt = pv.map(v => Math.log(Math.max(v, 1e-300)));
  const lm = mean(lt);
  let mu = lt.map(v => -(v - lm));
  let eta = 1.0;
  let lp = forward(mu).map(v => Math.log(Math.max(v, 1e-300)));
  let err = Math.max(...lp.map((v, i) => Math.abs(v - lt[i])));
  for (let it = 0; it < maxIter && err >= tol; it++) {
    let muN = mu.map((m, i) => m - eta * (lt[i] - lp[i]));
    const mm = mean(muN);
    muN = muN.map(v => v - mm);
    const lpN = forward(muN).map(v => Math.log(Math.max(v, 1e-300)));
    const e = Math.max(...lpN.map((v, i) => Math.abs(v - lt[i])));
    if (e < err) { mu = muN; lp = lpN; err = e; eta = Math.min(eta * 1.2, 1.5); }
    else { eta *= 0.5; if (eta < 1e-4) break; }
  }
  return mu;
}

function dp(mu, s, opts) {
  const { base = "normal", points = 257, qa = 9, qf = 15, returnSlopes = false } = opts;
  if (s.kind === "Independent")
    return raceProbabilities(mu, { D: s.D, base, points, returnSlopes });
  if (s.kind === "Factor")
    return raceProbabilities(mu, { V: s.V, D: s.D, base, points, returnSlopes });
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
  const { points = 257, qa = 9, qf = 15 } = opts;
  if (s.kind === "Independent") return abilitiesFromRace(p, { D: s.D, points });
  if (s.kind === "Factor") return abilitiesFromRace(p, { V: s.V, D: s.D, points });
  if (s.kind === "Blocks")
    return abilitiesFromBlockRace(p, s.cluster, s.loading, s.D, { points, qa }).mu;
  if (s.kind === "Nested")
    return invertGeneric(p, m => nestedRaceProbabilities(m, s.cluster, s.loading, s.D,
      { coupling: s.coupling, gamma: s.gamma, points, qa, qf }));
  if (s.kind === "Tree")
    return invertGeneric(p, m => treeRaceProbabilities(m, s.cluster, s.loading, s.D,
      s.parent, s.strength, { points, qa }));
  throw new Error("unknown structure " + s.kind);
}
_setDispatch(dp, da);
