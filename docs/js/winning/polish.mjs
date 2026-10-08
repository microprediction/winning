// Polish a race onto linear constraints -- port of winning/factor/polish.py
// (augmented Lagrangian with a compact BFGS inner solver standing in for
// SLSQP; agrees with the reference optimum to optimizer tolerance).
import { mean, checkOpts, OPT_HINTS, asLoadings, gaugeCenter, asAbilities,
         asIdio, isVector } from "./core.mjs";
import { raceProbabilities, abilitiesFromRace, BASES, SPANS,
         forwardGrid, factorRule, requireWholeRule,
         collapseStructure, translatedHome } from "./races.mjs";
import { blockRaceJacobian, nestedRaceJacobian, treeRaceJacobian } from "./blocks.mjs";

export function raceJacobian(mu, opts = {}) {
  checkOpts(opts, RACE_JACOBIAN_OPTS, "raceJacobian", OPT_HINTS);
  const { V = null, D = null, F = null, W = null, base = "normal",
          points = 501, structure = null, qa = 9, qf = 15 } = opts;
  const c = collapseStructure(structure, V, D, F, W, "raceJacobian");
  if (c.structure) {
    const s = c.structure;
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
  // the same doors as the forward: a scalar D threw `D.map is not a
  // function` here while raceProbabilities broadcast it (#254), and a
  // typed mu broke the default loadings (#334)
  mu = asAbilities(mu);
  const n = mu.length;
  const Dv = asIdio(c.D, n);
  mu = translatedHome(mu, Dv);                     // as the forward (#477)
  return raceJacobianExplicit(mu, c.V, Dv, base, points, F, W);
}

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
  // The independent default is the EMPTY node (no coordinates), not the
  // rank-one node [[0]]: with an (n, 0) loading matrix the conditional
  // mean below read V[i][0] * F[q][0] = undefined * 0 = NaN, so the
  // rank-zero Jacobian was all NaN and polishRace died on it, while the
  // rank-zero forward was already the independent race (#68). A zero
  // rank-one column gives the same answer either way.
  let F = [[]], W = [1];
  requireWholeRule(F0, W0, "raceJacobian");                     // #290
  V = asLoadings(V, n) || Array.from({ length: n }, () => [0]);   // #232
  // Gauge-fix before anything reads the loadings, exactly as the
  // forward path does: a common column cannot move an argmin, so it
  // must not move the Jacobian either. Uncentered, adding 1 to every
  // loading moved an entry by 0.0199 (#303).
  V = gaugeCenter(V);
  // The SAME factor rule the forward integrates over, from the one
  // function that chooses it: this used to re-derive only the Hermite
  // branch, so a sharp rank-one field was priced on a midpoint-quantile
  // rule and differentiated on a Hermite one, 0.058 apart (#325). A
  // caller's F/W is validated and used verbatim, as setup() uses it
  // (#209, #290).
  ({ F, W } = factorRule(V, D, F0, W0));
  const fn = typeof base === "function" ? base : BASES[base];
  if (typeof fn !== "function")
    throw new Error(`unknown base ${JSON.stringify(base)}`);
  const [left, right] = typeof base === "function" ? [12, 12] : (SPANS[base] || [12, 12]);
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
  const { x, dx } = forwardGrid(Mall, sd, { V, fn, left, right },
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
const CONCENTRATION_MATRIX_OPTS = new Set(["nameCaps", "groups"]);
export function concentrationMatrix(n, opts = {}) {
  // an inline-destructured signature skipped checkOpts: {namecaps: ...}
  // returned an EMPTY constraint set (#561)
  checkOpts(opts, CONCENTRATION_MATRIX_OPTS, "concentrationMatrix", OPT_HINTS);
  const { nameCaps = null, groups = null } = opts;
  const A = [], b = [];
  if (nameCaps != null) {
    let caps;
    if (isVector(nameCaps)) {
      // a Float64Array of caps fell into the SCALAR branch, was copied
      // into every slot as an object, failed Number.isFinite, and so
      // every cap was "no cap": polishRace returned the unconstrained
      // race with an empty active set (#334)
      if (nameCaps.length !== n)
        throw new Error(
          `nameCaps must be a scalar or have one entry per contestant; ` +
          `got ${nameCaps.length} for ${n}`);
      caps = Array.from(nameCaps);
    } else if (typeof nameCaps === "number") {
      caps = new Array(n).fill(nameCaps);
    } else {
      throw new Error(`nameCaps must be a number or an array; got ${typeof nameCaps}`);
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
  if (groups) for (const [idx0, cap] of groups) {
    const idx = isVector(idx0) ? Array.from(idx0) : idx0;
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
          points = 257, nameCaps = null, groups = null,
          structure = null, fdFallback = true } = opts;
  let { A = null, b = null } = opts;
  // F/W must reach the forward, the jacobian AND the inverse that makes
  // mu0. Polishing under a different factor law than the one the caller
  // priced with silently optimises the wrong model (#209).
  const fwdM = m => raceProbabilities(m, { V, D, F, W, base, points, structure });
  const jacM = m => raceJacobian(m, { V, D, F, W, base, points, structure });
  let mu0 = mu0In != null ? asAbilities(mu0In, "mu0") : null;
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
    if (!isVector(A) || !isVector(b))
      throw new Error("A and b must both be arrays");
    A = Array.from(A, row => (isVector(row) ? Array.from(row) : row));
    b = Array.from(b);
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
  if (!b0.length) return { p: fwdM(mu0), mu: mu0, info: { active: [] } };
  // Solve in the FIELD'S units. The objective, the penalty schedule and
  // every stopping tolerance below are absolute numbers, so the same race
  // written with abilities x c and variances x c^2 polished differently:
  // c = 1e-8 returned a reversed race certified converged, c = 100 the
  // original violating race (#593). The unit is 1 / median_i |J_ii| / p_i
  // at mu0 -- degree -1 in the ability scale, so u = mu / unit is exactly
  // equivariant -- and close to 1 for a unit-variance field.
  const unit = (() => {
    const pM = fwdM(mu0), JM = jacM(mu0);
    const r = [];
    for (let i = 0; i < n; i++)
      if (pM[i] > 1e-12 && Number.isFinite(JM[i][i]) && JM[i][i] !== 0)
        r.push(Math.abs(JM[i][i]) / pM[i]);
    if (!r.length) return 1;
    r.sort((a, b) => a - b);
    const h = Math.floor(r.length / 2);
    const med = r.length % 2 ? r[h] : 0.5 * (r[h - 1] + r[h]);
    return Number.isFinite(1 / med) && med > 0 ? 1 / med : 1;
  })();
  const forward = u => fwdM(u.map(v => v * unit));
  const jac = u => jacM(u.map(v => v * unit)).map(row => row.map(v => v * unit));
  const mu0M = mu0;
  mu0 = mu0M.map(v => v / unit);
  const applyA = p => A0.map(row => row.reduce((a, v, j) => a + v * p[j], 0));

  const jacOf = (mc, useFD) => {
    if (!useFD) return jac(mc);
    const h = 1e-6;
    let Jm = [];
    for (let j = 0; j < n; j++) {
      const e = new Array(n).fill(0); e[j] = h;
      const pp = forward(mc.map((v, i) => v + e[i]));
      const pm = forward(mc.map((v, i) => v - e[i]));
      Jm.push(pp.map((v, i) => (v - pm[i]) / (2 * h)));
    }
    // Jm is [j][i]; transpose to [i][j]
    return Jm[0].map((_, i) => Jm.map(col => col[i]));
  };
  // gauge-projected gradient of 0.5|m - mu0|^2 + sum_k psi_k (A p(m) - b)_k:
  // the augmented-Lagrangian gradient while solving, and the KKT
  // stationarity residual when psi is the multiplier estimate
  const lagrangianGrad = (mc, psi, useFD) => {
    const g = mc.map((v, j) => v - mu0[j]);
    if (psi.some(v => v !== 0)) {
      const Jm = jacOf(mc, useFD);
      for (let k = 0; k < b0.length; k++) {
        if (psi[k] === 0) continue;
        for (let j = 0; j < n; j++) {
          let aj = 0;
          for (let i = 0; i < n; i++) aj += A0[k][i] * Jm[i][j];
          g[j] += psi[k] * aj;
        }
      }
    }
    const gm = mean(g);
    return g.map(v => v - gm);
  };

  let converged = false;
  const solveAL = useFD => {
    let lam = new Array(b0.length).fill(0);
    let rho = 10;
    let m = mu0.slice();
    converged = false;
    for (let outer = 0; outer < 40; outer++) {
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
        return lagrangianGrad(mc, psi, useFD);
      };
      m = bfgsMin(m, obj, grad, 80);
      m = m.map(v => v - mean(m));
      const c = applyA(forward(m)).map((v, k) => b0[k] - v);
      const lamN = c.map((v, k) => Math.max(0, lam[k] - rho * v));
      const viol = Math.max(0, -Math.min(...c));
      const dLam = Math.max(...lamN.map((v, k) => Math.abs(v - lam[k])));
      const lamMax = Math.max(1, ...lamN);
      lam = lamN;
      // Stop on a KKT point, not on feasibility alone. A multiplier
      // carried over from an infeasible iterate can push the subproblem
      // PAST the boundary; that overshot point is feasible, and breaking
      // there returned a race 8.9% farther from mu0 than the nearest
      // feasible one, with an empty active set (#379). The multiplier
      // update lam <- max(0, lam - rho c) is a fixed point exactly when
      // complementarity holds (lam = 0 on a slack row, c = 0 on a
      // binding one), so feasibility plus a settled multiplier is the
      // test. rho grows only while the iterate is still infeasible.
      if (outer > 0 && viol < 1e-8 && dLam <= 1e-7 * lamMax) { converged = true; break; }
      // ...or certify the point DIRECTLY. Once infeasible iterates have
      // driven rho to 1e6, a harmless 3e-9 slack on a binding row moves
      // the multiplier by 3e-3 per outer step, so the fixed-point test
      // could never pass at a solved point: a third to a half of
      // ordinary capped races reported converged: false and paid for a
      // redundant finite-difference re-solve (2-2.6x, #547). Feasible,
      // lamN >= 0 by construction, complementarity (lamN_k = 0 unless
      // row k is within 1e-8 of binding, as the update forces), and a
      // stationary Lagrangian at lamN is the KKT certificate. The #379
      // overshoot fails it: every row is slack there, so lamN = 0 and the
      // residual is the whole distance m - mu0.
      if (viol < 1e-8) {
        const comp = Math.max(0, ...lamN.map((v, k) => (v > 0 ? v * Math.max(c[k], 0) : 0)));
        const r = lagrangianGrad(m, lamN, useFD);
        const scale = Math.max(1, ...m.map((v, j) => Math.abs(v - mu0[j])), ...lamN);
        if (comp <= 1e-7 * scale && Math.max(...r.map(Math.abs)) <= 1e-6 * scale) {
          converged = true; break;
        }
      }
      if (viol >= 1e-8) rho = Math.min(rho * 3, 1e6);
    }
    return m;
  };
  let m = solveAL(false);
  let p = forward(m);
  let slack = applyA(p).map((v, k) => b0[k] - v);
  if (fdFallback && (-Math.min(...slack) > 1e-6 || !converged)) {
    m = solveAL(true);
    p = forward(m);
    slack = applyA(p).map((v, k) => b0[k] - v);
  }
  const mOut = m.map(v => v * unit);
  return { p, mu: mOut,
           info: { active: slack.map((v, k) => [v, k]).filter(([v]) => v < 1e-6).map(([, k]) => k),
                   maxViolation: Math.max(0, -Math.min(...slack)),
                   converged,
                   muDistance: Math.sqrt(mOut.reduce((a, v, j) => a + (v - mu0M[j]) ** 2, 0)) } };
}
