// Polish a race onto linear constraints -- port of winning/factor/polish.py
// (augmented Lagrangian with a compact BFGS inner solver standing in for
// SLSQP; agrees with the reference optimum to optimizer tolerance).
import { mean, checkOpts, OPT_HINTS, asLoadings, asFactorNodes, asWeights } from "./core.mjs";
import { raceProbabilities, abilitiesFromRace, BASES,
         forwardGrid } from "./races.mjs";
import { blockRaceJacobian, nestedRaceJacobian, treeRaceJacobian } from "./blocks.mjs";

export function raceJacobian(mu, opts = {}) {
  checkOpts(opts, RACE_JACOBIAN_OPTS, "raceJacobian", OPT_HINTS);
  const { V = null, D = null, F = null, W = null, base = "normal",
          points = 501, structure = null, qa = 9, qf = 15 } = opts;
  if (structure) {
    const s = structure;
    if (s.kind === "Independent") return raceJacobian(mu, { D: s.D, base, points });
    if (s.kind === "Factor")
      return raceJacobian(mu, { V: s.V, D: s.D, F, W, base, points });
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
  return raceJacobianExplicit(mu, Vv, Dv, base, points, F, W);
}

import { hermiteNodes } from "./core.mjs";

/* Each exported call declares its own option keys; see checkOpts in
   core.mjs for why an options object needs this at all. */
const RACE_JACOBIAN_OPTS = new Set(["V", "D", "F", "W", "base", "points", "qa", "qf", "structure"]);
const POLISH_RACE_OPTS = new Set(["V", "D", "F", "W", "base", "points", "structure", "A", "b", "groups", "nameCaps", "p0", "mu0", "fdFallback"]);

/* F0/W0: a caller-supplied factor law. raceProbabilities has always
   accepted one, and this always built its own standard-normal Hermite
   rule instead, so the browser could PRICE a discrete or otherwise
   non-Gaussian factor and could not differentiate or polish it -- the
   returned Jacobian was the derivative of a different model (#209).
   Used verbatim, exactly as the forward's setup() uses them. */
function raceJacobianExplicit(mu, V, D, base, points, F0 = null, W0 = null) {
  const n = mu.length;
  const sd = D.map(Math.sqrt);
  let F = [[0]], W = [1];
  V = asLoadings(V, n) || Array.from({ length: n }, () => [0]);   // #232
  const hasV = V.some(row => row.some(v => v !== 0));
  if (hasV && F0 && W0) {
    // the caller's nodes get the same contract the forward applies, or
    // the Jacobian differentiates a different-rank model than the
    // forward prices (#290)
    F = asFactorNodes(F0, V[0].length, "F");
    W = asWeights(W0, F.length, "W");
  } else if (hasV) {
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
  // The SAME lattice raceProbabilities integrates on. This built its own
  // -- a plain span window, no adaptive placement, no refinement -- so
  // it differentiated a different grid than the forward computed on, and
  // the two parted company exactly where the lattice is coarse relative
  // to the field: on a four-runner race whose variances span 4005x, at
  // 257 points, the analytic jacobian was 1.0e-3 from finite differences
  // of its own forward. Sharing the grid brings that to 1.5e-5 (#212).
  const { x, dx } = forwardGrid(Mall, sd, { V, left, right },
                                points, "bulk", 1e-12);
  const P = x.length;
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

/* These inputs are portfolio limits: name caps, sector caps. A typo that
   DROPS a constraint returns an ordinary-looking result that is simply
   under-constrained, which is worse than an error. Every malformed shape
   below used to do exactly that (#247):

     nameCaps of length n-1   the last name was left uncapped, and an 80%
                              position came back with maxViolation 0
     nameCaps of length n+1   silently truncated to n
     a group index of n       wrote past the row; the optimiser returned
                              NaN for every runner and called it feasible
     a negative group index   javascript stores it as a non-element
                              property, so the member was silently
                              ignored -- python would select the last name
   Python documents name_caps as "scalar or length-n" and broadcasts,
   which rejects either wrong length. A non-finite ENTRY is a documented
   feature there -- "NaN/None entries skipped", meaning no cap for that
   name -- so it stays one here. Negative group indices are refused in
   both ports rather than silently meaning different things: python's
   numpy indexing made -1 the LAST name while javascript dropped the
   member entirely, and neither is documented. */
export function concentrationMatrix(n, { nameCaps = null, groups = null } = {}) {
  const A = [], b = [];
  if (nameCaps != null) {
    let caps;
    if (Array.isArray(nameCaps)) {
      if (nameCaps.length !== n)
        throw new Error(
          `nameCaps must be a scalar or have one entry per contestant; ` +
          `got ${nameCaps.length} for ${n}`);
      caps = nameCaps;
    } else {
      caps = new Array(n).fill(nameCaps);
    }
    for (let i = 0; i < n; i++) {
      // A non-finite entry means NO cap for that name. That is python's
      // documented behaviour -- "NaN/None entries skipped" -- so it stays
      // a feature here, not an error; only the LENGTH was ever wrong.
      if (!Number.isFinite(caps[i])) continue;
      const r = new Array(n).fill(0); r[i] = 1;
      A.push(r); b.push(caps[i]);
    }
  }
  if (groups) for (const [idx, cap] of groups) {
    if (!Array.isArray(idx))
      throw new Error("each group is [indices, cap]; indices must be an array");
    if (!Number.isFinite(cap))
      throw new Error(`group cap for [${idx}] is not a finite number`);
    const r = new Array(n).fill(0);
    for (const i of idx) {
      if (!Number.isInteger(i) || i < 0 || i >= n)
        throw new Error(
          `group index ${i} is not an integer in [0, ${n}); a negative ` +
          "index is silently dropped by javascript rather than counting " +
          "from the end");
      r[i] = 1;
    }
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
  checkOpts(opts, POLISH_RACE_OPTS, "polishRace", OPT_HINTS);
  const { p0 = null, mu0: mu0In = null, V = null, D = null, F = null,
          W = null, base = "normal",
          points = 257, nameCaps = null, groups = null, A = null, b = null,
          structure = null, fdFallback = true } = opts;
  // F/W must reach the forward, the jacobian AND the inverse that makes
  // mu0. Polishing under a different factor law than the one the caller
  // priced with silently optimises the wrong model (#209).
  const forward = m => raceProbabilities(m, { V, D, F, W, base, points, structure });
  const jac = m => raceJacobian(m, { V, D, F, W, base, points, structure });
  let mu0 = mu0In;
  if (!mu0) {
    if (!p0) throw new Error("give p0 or mu0");
    mu0 = abilitiesFromRace(p0, { V, D, F, W, base, points, structure });
  }
  const m0m = mean(mu0);
  mu0 = mu0.map(v => v - m0m);
  const n = mu0.length;
  const cm = concentrationMatrix(n, { nameCaps, groups });
  let A0 = cm.A, b0 = cm.b;
  if (A) {
    // An explicit A/b pair is consumed row by row against mu of length
    // n; a row of the wrong width contributes nothing and the constraint
    // disappears in silence (#247).
    if (!Array.isArray(A) || !Array.isArray(b))
      throw new Error("A and b must both be arrays");
    if (A.length !== b.length)
      throw new Error(
        `A has ${A.length} rows and b has ${b.length} entries`);
    A.forEach((row, k) => {
      if (!Array.isArray(row) || row.length !== n)
        throw new Error(
          `A[${k}] must have one coefficient per contestant; got ` +
          `${Array.isArray(row) ? row.length : typeof row} for ${n}`);
      if (!row.every(Number.isFinite))
        throw new Error(`A[${k}] has a non-finite coefficient`);
      if (!Number.isFinite(b[k]))
        throw new Error(`b[${k}] is not a finite number`);
    });
    A0 = A0.concat(A); b0 = b0.concat(b);
  }
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
  if (fdFallback && -Math.min(...slack) > 1e-6) {
    m = solveAL(true);
    p = forward(m);
    slack = applyA(p).map((v, k) => b0[k] - v);
  }
  return { p, mu: m,
           info: { active: slack.map((v, k) => [v, k]).filter(([v]) => v < 1e-6).map(([, k]) => k),
                   maxViolation: Math.max(0, -Math.min(...slack)),
                   muDistance: Math.sqrt(m.reduce((a, v, j) => a + (v - mu0[j]) ** 2, 0)) } };
}
