// Polish a race onto linear constraints -- port of winning/factor/polish.py
// (augmented Lagrangian with a compact BFGS inner solver standing in for
// SLSQP; agrees with the reference optimum to optimizer tolerance).
import { mean } from "./core.mjs";
import { raceProbabilities, abilitiesFromRace, BASES } from "./races.mjs";
import { blockRaceJacobian, nestedRaceJacobian, treeRaceJacobian } from "./blocks.mjs";

export function raceJacobian(mu, opts = {}) {
  const { V = null, D = null, base = "normal", points = 501, structure = null,
          qa = 9, qf = 15 } = opts;
  if (structure) {
    const s = structure;
    if (s.kind === "Independent") return raceJacobian(mu, { D: s.D, base, points });
    if (s.kind === "Factor") return raceJacobian(mu, { V: s.V, D: s.D, base, points });
    if (s.kind === "Blocks")
      return blockRaceJacobian(mu, s.cluster, s.loading, s.D, { points, qa });
    if (s.kind === "Nested")
      return nestedRaceJacobian(mu, s.cluster, s.loading, s.D,
        { coupling: s.coupling, gamma: s.gamma, points, qa, qf });
    if (s.kind === "Tree")
      return treeRaceJacobian(mu, s.cluster, s.loading, s.D, s.parent, s.strength,
        { points, qa });
    throw new Error("race_jacobian: unknown structure");
  }
  const n = mu.length;
  const Dv = D || new Array(n).fill(1);
  const Vv = V || mu.map(() => [0]);
  return raceJacobianExplicit(mu, Vv, Dv, base, points);
}

import { hermiteNodes } from "./core.mjs";
function raceJacobianExplicit(mu, V, D, base, points) {
  const n = mu.length;
  const sd = D.map(Math.sqrt);
  let F = [[0]], W = [1];
  const hasV = V.some(row => row.some(v => v !== 0));
  if (hasV) {
    let sharp = 0;
    for (let i = 0; i < n; i++) {
      const nv = Math.sqrt(V[i].reduce((a, b) => a + b * b, 0));
      sharp = Math.max(sharp, nv / Math.sqrt(Math.max(D[i], 1e-300)));
    }
    const r = V[0].length;
    const cap = r === 1 ? 201 : r === 2 ? 41 : 15;
    const Q = Math.min(Math.max(Math.ceil(8 * sharp), 15), cap);
    ({ F, W } = hermiteNodes(r, Q));
  }
  const fn = typeof base === "function" ? base : BASES[base];
  const spans = { normal: [8, 8], gumbel: [22, 8] };
  const [left, right] = spans[base] || [12, 12];
  const Mall = F.map(fq => mu.map((m, i) => {
    let s = m;
    for (let r = 0; r < fq.length; r++) s += V[i][r] * fq[r];
    return s;
  }));
  let mn = Infinity, mx = -Infinity;
  for (const row of Mall) for (const v of row) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const smax = Math.max(...sd);
  const P = points;
  const x = new Array(P);
  const lo = mn - left * smax, hi = mx + right * smax;
  for (let t = 0; t < P; t++) x[t] = lo + t * (hi - lo) / (P - 1);
  const dx = x[1] - x[0];
  const J = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let q = 0; q < F.length; q++) {
    const Mq = Mall[q], wq = W[q];
    const L = new Array(P).fill(0);
    const logS = [], logf = [];
    for (let i = 0; i < n; i++) {
      const li = new Array(P), fi = new Array(P);
      for (let t = 0; t < P; t++) {
        const z = (x[t] - Mq[i]) / sd[i];
        const [S, f] = fn(z);
        li[t] = Math.log(S);
        fi[t] = Math.log(Math.max(f / sd[i], 1e-300));
        L[t] += li[t];
      }
      logS.push(li); logf.push(fi);
    }
    const P1 = [], P2 = [];
    for (let i = 0; i < n; i++) {
      const p1 = new Array(P), p2 = new Array(P);
      for (let t = 0; t < P; t++) {
        p1[t] = Math.exp(Math.min(Math.max(logf[i][t] + L[t] - logS[i][t], -745), 40));
        p2[t] = Math.exp(Math.min(Math.max(logf[i][t] - logS[i][t], -745), 40));
      }
      P1.push(p1); P2.push(p2);
    }
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let t = 0; t < P; t++) s += P1[i][t] * P2[j][t];
        J[i][j] += wq * s * dx;
      }
  }
  for (let i = 0; i < n; i++) J[i][i] = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += J[i][j];
    J[i][i] = -s;
  }
  return J;
}

export function concentrationMatrix(n, { nameCaps = null, groups = null } = {}) {
  const A = [], b = [];
  if (nameCaps != null) {
    const caps = Array.isArray(nameCaps) ? nameCaps : new Array(n).fill(nameCaps);
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(caps[i])) {
        const r = new Array(n).fill(0); r[i] = 1;
        A.push(r); b.push(caps[i]);
      }
    }
  }
  if (groups) for (const [idx, cap] of groups) {
    const r = new Array(n).fill(0);
    for (const i of idx) r[i] = 1;
    A.push(r); b.push(cap);
  }
  return { A, b };
}

function bfgsMin(x0, obj, grad, maxit = 80) {
  const n = x0.length;
  let x = x0.slice();
  let H = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  let g = grad(x), f = obj(x);
  for (let it = 0; it < maxit; it++) {
    const d = H.map(row => -row.reduce((a, v, j) => a + v * g[j], 0));
    let step = 1, fN, xN;
    const slope = d.reduce((a, v, j) => a + v * g[j], 0);
    if (slope > -1e-16) break;
    for (let ls = 0; ls < 30; ls++) {
      xN = x.map((v, j) => v + step * d[j]);
      fN = obj(xN);
      if (fN <= f + 1e-4 * step * slope) break;
      step *= 0.5;
    }
    const gN = grad(xN);
    const s = xN.map((v, j) => v - x[j]);
    const y = gN.map((v, j) => v - g[j]);
    const sy = s.reduce((a, v, j) => a + v * y[j], 0);
    if (sy > 1e-12) {
      const Hy = H.map(row => row.reduce((a, v, j) => a + v * y[j], 0));
      const yHy = y.reduce((a, v, j) => a + v * Hy[j], 0);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++)
          H[i][j] += ((sy + yHy) * s[i] * s[j]) / (sy * sy)
            - (Hy[i] * s[j] + s[i] * Hy[j]) / sy;
    }
    const done = Math.max(...gN.map(Math.abs)) < 1e-9
      || Math.abs(fN - f) < 1e-13 * (1 + Math.abs(f));
    x = xN; g = gN; f = fN;
    if (done) break;
  }
  return x;
}

export function polishRace(opts = {}) {
  const { p0 = null, mu0: mu0In = null, V = null, D = null, base = "normal",
          points = 257, nameCaps = null, groups = null, A = null, b = null,
          structure = null } = opts;
  const forward = m => raceProbabilities(m, { V, D, base, points, structure });
  const jac = m => raceJacobian(m, { V, D, base, points, structure });
  let mu0 = mu0In;
  if (!mu0) {
    if (!p0) throw new Error("give p0 or mu0");
    mu0 = abilitiesFromRace(p0, { V, D, base, points, structure });
  }
  const m0m = mean(mu0);
  mu0 = mu0.map(v => v - m0m);
  const n = mu0.length;
  const cm = concentrationMatrix(n, { nameCaps, groups });
  let A0 = cm.A, b0 = cm.b;
  if (A) { A0 = A0.concat(A); b0 = b0.concat(b); }
  if (!b0.length) return { p: forward(mu0), mu: mu0, info: { active: [] } };
  const applyA = p => A0.map(row => row.reduce((a, v, j) => a + v * p[j], 0));

  const solveAL = useFD => {
    let lam = new Array(b0.length).fill(0);
    let rho = 10;
    let m = mu0.slice();
    for (let outer = 0; outer < 12; outer++) {
      const obj = mm => {
        const mc = mm.map(v => v - mean(mm));
        const c = applyA(forward(mc)).map((v, k) => b0[k] - v);
        const psi = c.map((v, k) => Math.max(0, lam[k] - rho * v));
        return 0.5 * mc.reduce((a, v, j) => a + (v - mu0[j]) ** 2, 0)
          + psi.reduce((a, v, k) => a + (v * v - lam[k] * lam[k]), 0) / (2 * rho);
      };
      const grad = mm => {
        const mc = mm.map(v => v - mean(mm));
        const c = applyA(forward(mc)).map((v, k) => b0[k] - v);
        const psi = c.map((v, k) => Math.max(0, lam[k] - rho * v));
        let Jm;
        if (useFD) {
          const h = 1e-6;
          Jm = [];
          for (let j = 0; j < n; j++) {
            const e = new Array(n).fill(0); e[j] = h;
            const pp = forward(mc.map((v, i) => v + e[i]));
            const pm = forward(mc.map((v, i) => v - e[i]));
            Jm.push(pp.map((v, i) => (v - pm[i]) / (2 * h)));
          }
          // Jm is [j][i]; transpose to [i][j]
          Jm = Jm[0].map((_, i) => Jm.map(col => col[i]));
        } else {
          Jm = jac(mc);
        }
        const g = mc.map((v, j) => v - mu0[j]);
        for (let k = 0; k < b0.length; k++) {
          if (psi[k] === 0) continue;
          for (let j = 0; j < n; j++) {
            let aj = 0;
            for (let i = 0; i < n; i++) aj += A0[k][i] * Jm[i][j];
            g[j] += psi[k] * aj;
          }
        }
        const gm = mean(g);
        return g.map(v => v - gm);
      };
      m = bfgsMin(m, obj, grad, 80);
      m = m.map(v => v - mean(m));
      const c = applyA(forward(m)).map((v, k) => b0[k] - v);
      lam = c.map((v, k) => Math.max(0, lam[k] - rho * v));
      if (Math.max(0, -Math.min(...c)) < 1e-8 && outer > 0) break;
      rho = Math.min(rho * 3, 1e6);
    }
    return m;
  };
  let m = solveAL(false);
  let p = forward(m);
  let slack = applyA(p).map((v, k) => b0[k] - v);
  if (-Math.min(...slack) > 1e-6) {
    m = solveAL(true);
    p = forward(m);
    slack = applyA(p).map((v, k) => b0[k] - v);
  }
  return { p, mu: m,
           info: { active: slack.map((v, k) => [v, k]).filter(([v]) => v < 1e-6).map(([, k]) => k),
                   maxViolation: Math.max(0, -Math.min(...slack)),
                   muDistance: Math.sqrt(m.reduce((a, v, j) => a + (v - mu0[j]) ** 2, 0)) } };
}
