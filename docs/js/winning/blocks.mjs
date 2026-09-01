// Block, nested and tree races -- port of winning/factor/blocks.py.
import { TINY, ndtr, npdf, hermite1, mean, solve, interpClamped } from "./core.mjs";

function clusterIndex(cluster) {
  const lv = [...new Set(cluster)].sort((a, b) => (a > b ? 1 : a < b ? -1 : 0));
  const map = new Map(lv.map((v, i) => [v, i]));
  return cluster.map(c => map.get(c));
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
  const smax = Math.max(Math.max(...sd), 1e-12);
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
function checkedMass(raw, kind, massTol = 5e-3) {
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
  throw new Error("rank >= 3 cluster effects: supply nodes explicitly");
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
  const n = mu.length;
  const isMat = Array.isArray(loading[0]);
  const r = isMat ? loading[0].length : 1;
  const inv = clusterIndex(cluster);
  const ord = stableOrder(inv);
  const muO = ord.map(i => mu[i]), sdO = ord.map(i => sd[i]);
  const VO = ord.map(i => (isMat ? loading[i] : [loading[i]]));
  const cO = ord.map(i => inv[i]);
  const nC = Math.max(...cO) + 1;
  const { nodes, w } = nodesOverride || clusterNodes(r, qa);
  const Q = nodes.length;
  const maxNodeNorm = Math.max(...nodes.map(nq => Math.sqrt(nq.reduce((a, b) => a + b * b, 0))));
  const amp = VO.map(vi => Math.sqrt(vi.reduce((a, b) => a + b * b, 0)) * maxNodeNorm);
  const [lo, hi] = windowNodes(muO, sdO, amp);
  const P = points;
  const x = new Array(P);
  for (let t = 0; t < P; t++) x[t] = lo + t * (hi - lo) / (P - 1);
  const dx = x[1] - x[0];
  const shifts = VO.map(vi => nodes.map(nq => vi.reduce((a, b, k) => a + b * nq[k], 0)));
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
  const { points = 257, qa = 9, nodes = null } = opts;
  const sd = D.map(Math.sqrt);
  const p = blockMax(mu.map(v => -v), sd, cluster, loading, points, qa, nodes);
  return checkedMass(p, "block race");
}

export function nestedRaceProbabilities(mu, cluster, loading, D, opts = {}) {
  const { coupling = null, gamma = 1.0, points = 257, qa = 9, qf = 15 } = opts;
  if (!coupling || gamma === 0)
    return blockRaceProbabilities(mu, cluster, loading, D, { points, qa });
  const g = Array.isArray(coupling[0]) ? coupling : coupling.map(v => [v]);
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
  const sd = D.map(Math.sqrt);
  const p = new Array(n).fill(0);
  for (let q = 0; q < fn.length; q++) {
    // average the RAW conditional masses (each near one) and normalize
    // once: normalizing each conditional separately hides a window defect
    const shifted = mu.map((m, i) => -(m + gamma * g[i].reduce((a, b, r) => a + b * fn[q][r], 0)));
    const pq = blockMax(shifted, sd, cluster, loading, points, qa, null);
    for (let i = 0; i < n; i++) p[i] += fw[q] * pq[i];
  }
  return checkedMass(p, "nested race");
}

/* tree machinery shared by forward and jacobian */
function treeInternals(mu, cluster, loading, D, parent, strength, points, qa) {
  if (Array.isArray(loading[0])) {
    if (loading[0].length > 1) {
      throw new Error(
        "tree races take scalar (rank-one) leaf-cluster loadings; rank-r " +
        "leaf effects are supported by the block grammar only.");
    }
    loading = loading.map(v => v[0]);
  }
  const m = mu.map(v => -v);
  const sd = D.map(Math.sqrt);
  const lam = strength.slice();
  const par = parent.slice();                        // -1 = root (python style)
  const n = m.length, nT = par.length;
  const inv = clusterIndex(cluster);
  const nC = Math.max(...inv) + 1;
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
  const P = points;
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
  const { points = 257, qa = 9 } = opts;
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

export function blockRaceJacobian(mu, cluster, loading, D, opts = {}) {
  const { points = 257, qa = 9 } = opts;
  const m = mu.map(v => -v);
  const sd = D.map(Math.sqrt);
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
  const P = points;
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
  const { coupling = null, gamma = 1.0, points = 257, qa = 9, qf = 15 } = opts;
  if (!coupling || gamma === 0)
    return blockRaceJacobian(mu, cluster, loading, D, { points, qa });
  const g = Array.isArray(coupling[0]) ? coupling : coupling.map(v => [v]);
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
    const shifted = mu.map((m, i) => m + gamma * g[i].reduce((a, b, r) => a + b * fn[q][r], 0));
    const Jq = blockRaceJacobian(shifted, cluster, loading, D, { points, qa });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) J[i][j] += fw[q] * Jq[i][j];
  }
  return J;
}

export function treeRaceJacobian(mu, cluster, loading, D, parent, strength, opts = {}) {
  const { points = 257, qa = 9 } = opts;
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

export function abilitiesFromBlockRace(pTarget, cluster, loading, D, opts = {}) {
  const { points = 257, qa = 9, tol = 1e-10, maxIter = 25 } = opts;
  let pT = pTarget.slice();
  let s = pT.reduce((a, b) => a + b, 0);
  pT = pT.map(v => v / s);
  const n = pT.length;
  const floor = Math.max(1e-14, Math.min(...pT.filter(v => v > 0)) * 1e-3);
  pT = pT.map(v => Math.max(v, floor));
  s = pT.reduce((a, b) => a + b, 0);
  pT = pT.map(v => v / s);
  const lt = pT.map(Math.log);
  const forward = m => blockRaceProbabilities(m, cluster, loading, D, { points, qa });
  const lm = mean(lt);
  let mu = lt.map(v => -(v - lm));
  let eta = 1.0;
  let lp = forward(mu).map(v => Math.log(Math.max(v, TINY)));
  let err = Math.max(...lp.map((v, i) => Math.abs(v - lt[i])));
  for (let it = 0; it < 200 && err >= 0.2; it++) {
    let muN = mu.map((m, i) => m - eta * (lt[i] - lp[i]));
    const mm = mean(muN);
    muN = muN.map(v => v - mm);
    const lpN = forward(muN).map(v => Math.log(Math.max(v, TINY)));
    const eN = Math.max(...lpN.map((v, i) => Math.abs(v - lt[i])));
    if (eN < err) { mu = muN; lp = lpN; err = eN; eta = Math.min(eta * 1.2, 1.5); }
    else { eta *= 0.5; if (eta < 1e-4) break; }
  }
  for (let it = 0; it < maxIter; it++) {
    let pv = forward(mu).map(v => Math.max(v, TINY));
    const sv = pv.reduce((a, b) => a + b, 0);
    pv = pv.map(v => v / sv);
    const r = pv.map((v, i) => Math.log(v) - lt[i]);
    const cur = Math.max(...r.map(Math.abs));
    if (cur < tol) return { mu: mu.map(v => v - mean(mu)), residual: cur, iterations: it };
    const J = blockRaceJacobian(mu, cluster, loading, D, { points, qa });
    const A = J.map((row, i) => row.map(v => v / pv[i] + 1 / n));
    let step = solve(A, r.map(v => -v));
    const nn = Math.sqrt(step.reduce((a, b) => a + b * b, 0));
    if (nn > 5) step = step.map(v => v * 5 / nn);
    for (let k = 0; k < 8; k++) {
      let muN = mu.map((m, i) => m + step[i]);
      const mm = mean(muN);
      muN = muN.map(v => v - mm);
      let pN = forward(muN).map(v => Math.max(v, TINY));
      const sN = pN.reduce((a, b) => a + b, 0);
      pN = pN.map(v => v / sN);
      if (Math.max(...pN.map((v, i) => Math.abs(Math.log(v) - lt[i]))) < cur) { mu = muN; break; }
      step = step.map(v => v * 0.5);
    }
  }
  let pv = forward(mu).map(v => Math.max(v, TINY));
  const sv = pv.reduce((a, b) => a + b, 0);
  pv = pv.map(v => v / sv);
  return { mu: mu.map(v => v - mean(mu)),
           residual: Math.max(...pv.map((v, i) => Math.abs(Math.log(v) - lt[i]))),
           iterations: maxIter };
}
