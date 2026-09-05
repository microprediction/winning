// Top-k memberships q_i = P(X_i among the k smallest), their mu- and
// sigma-Jacobians, the rank marginals, and the inversions: locations
// from one membership curve, (loc, scale) jointly from two. Port of
// winning/factor/topk.py -- the cavity count distribution with
// stable-direction deconvolution; see the python module docstring for
// the derivations and the two-branch refusal of exact-rank targets.
import { TINY, hermite1, solve } from "./core.mjs";
import { BASES } from "./races.mjs";

const clip01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);

function countWindow(mu, sd, k, fn, delta = 1e-12, padSds = 2.0) {
  const n = mu.length;
  const smax = Math.max(Math.max(...sd), 1e-12);
  const meanCount = x => {
    let t = 0;
    for (let i = 0; i < n; i++) t += 1 - fn((x - mu[i]) / sd[i])[0];
    return t;
  };
  let lo = Math.min(...mu) - 9 * smax;
  let step = 9 * smax;
  for (let it = 0; it < 60; it++) {
    if (meanCount(lo) <= delta) break;
    lo -= step; step *= 2;
  }
  const targetHi = Math.min(
    k + 2 * Math.log(1 / delta) + Math.sqrt(2 * (k + 1) * Math.log(1 / delta)),
    n - 1e-4);
  let hi = Math.max(...mu) + 9 * smax;
  step = 9 * smax;
  for (let it = 0; it < 60; it++) {
    if (meanCount(hi) >= targetHi) break;
    hi += step; step *= 2;
  }
  let a = lo, b = hi;
  for (let it = 0; it < 70; it++) {
    const m = 0.5 * (a + b);
    if (meanCount(m) < delta) a = m; else b = m;
  }
  const xlo = a;
  a = xlo; b = hi;
  for (let it = 0; it < 70; it++) {
    const m = 0.5 * (a + b);
    if (meanCount(m) < targetHi) a = m; else b = m;
  }
  return [xlo - padSds * smax, b + padSds * smax];
}

function baseGrid(fn, x, mu, sd) {
  const L = x.length, n = mu.length;
  const S = [], f = [], fp = [], z = [];
  for (let t = 0; t < L; t++) {
    const Sr = new Array(n), fr = new Array(n), fpr = new Array(n),
          zr = new Array(n);
    for (let i = 0; i < n; i++) {
      const zz = (x[t] - mu[i]) / sd[i];
      const [s, ff, ffp] = fn(zz);
      Sr[i] = s; fr[i] = ff; fpr[i] = ffp; zr[i] = zz;
    }
    S.push(Sr); f.push(fr); fp.push(fpr); z.push(zr);
  }
  return { S, f, fp, z };
}

function countDistribution(F) {
  const L = F.length, n = F[0].length;
  const C = new Array(L);
  for (let t = 0; t < L; t++) {
    const row = new Array(n + 1).fill(0);
    row[0] = 1;
    for (let j = 0; j < n; j++) {
      const q = F[t][j];
      for (let m = Math.min(j + 1, n); m >= 1; m--)
        row[m] = row[m] * (1 - q) + row[m - 1] * q;
      row[0] *= 1 - q;
    }
    C[t] = row;
  }
  return C;
}

function leaveOneOutCdf(C, F, k) {
  const L = F.length, n = F[0].length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(L);
    for (let t = 0; t < L; t++) {
      const Fi = F[t][i], Si = 1 - Fi;
      if (Si >= Fi) {
        const s = Math.max(Si, TINY);
        let Q = clip01(C[t][0] / s), acc = Q;
        for (let m = 1; m < k; m++) {
          Q = clip01((C[t][m] - Fi * Q) / s);
          acc += Q;
        }
        row[t] = clip01(acc);
      } else {
        const fi = Math.max(Fi, TINY);
        let Qb = clip01(C[t][n] / fi), acc = Qb;
        for (let m = n - 2; m >= k; m--) {
          Qb = clip01((C[t][m + 1] - Si * Qb) / fi);
          acc += Qb;
        }
        row[t] = clip01(1 - acc);
      }
    }
    out[i] = row;
  }
  return out;
}

function looPmf(C, F, i) {
  // full leave-one-out pmf Q[t][m], m = 0..n-1, stable direction per t
  const L = F.length, n = F[0].length;
  const Q = new Array(L);
  for (let t = 0; t < L; t++) {
    const Fi = F[t][i], Si = 1 - Fi;
    const row = new Array(n);
    if (Si >= Fi) {
      const s = Math.max(Si, TINY);
      row[0] = clip01(C[t][0] / s);
      for (let m = 1; m < n; m++)
        row[m] = clip01((C[t][m] - Fi * row[m - 1]) / s);
    } else {
      const fi = Math.max(Fi, TINY);
      row[n - 1] = clip01(C[t][n] / fi);
      for (let m = n - 2; m >= 0; m--)
        row[m] = clip01((C[t][m + 1] - Si * row[m + 1]) / fi);
    }
    Q[t] = row;
  }
  return Q;
}

function pairPmfAt(Qi, F, i, k) {
  // P(N_{-ij} = k-1) for every j != i at every lattice point
  const L = F.length, n = F[0].length;
  const out = new Array(n);
  for (let j = 0; j < n; j++) {
    const row = new Array(L).fill(0);
    if (j !== i) {
      for (let t = 0; t < L; t++) {
        const Fj = F[t][j], Sj = 1 - Fj;
        if (Sj >= Fj) {
          const s = Math.max(Sj, TINY);
          let Q = clip01(Qi[t][0] / s);
          for (let m = 1; m < k; m++) Q = clip01((Qi[t][m] - Fj * Q) / s);
          row[t] = Q;
        } else {
          const fj = Math.max(Fj, TINY);
          let Qb = clip01(Qi[t][n - 1] / fj);
          for (let m = n - 3; m >= k - 1; m--)
            Qb = clip01((Qi[t][m + 1] - Sj * Qb) / fj);
          row[t] = Qb;
        }
      }
    }
    out[j] = row;
  }
  return out;
}

function topkGrid(mu, sd, k, fn, points, delta = 1e-12) {
  const [lo, hi] = countWindow(mu, sd, k, fn, delta);
  const x = new Array(points);
  const dx = (hi - lo) / (points - 1);
  for (let t = 0; t < points; t++) x[t] = lo + t * dx;
  const g = baseGrid(fn, x, mu, sd);
  const F = g.S.map(row => row.map(s => clip01(1 - s)));
  return { x, dx, F, ...g };
}

function topkWithSlopes(mu, sd, k, fn, points) {
  const n = mu.length;
  const { dx, F, f, fp } = topkGrid(mu, sd, k, fn, points);
  const C = countDistribution(F);
  const cdf = leaveOneOutCdf(C, F, k);
  const q = new Array(n).fill(0), slopes = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s0 = 0, s1 = 0;
    for (let t = 0; t < F.length; t++) {
      s0 += (f[t][i] / sd[i]) * cdf[i][t];
      s1 -= (fp[t][i] / (sd[i] * sd[i])) * cdf[i][t];
    }
    q[i] = s0 * dx;
    slopes[i] = s1 * dx;
  }
  return { q, slopes };
}

function checkedTopk(raw, k, kind, massTol = 5e-3) {
  const t = raw.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(t) || Math.abs(t - k) > massTol * k)
    throw new Error(
      `${kind} captured total membership ${t.toFixed(4)} where exactly ` +
      `${k} slots exist: the window or the deconvolution missed part ` +
      `of the field. Raise points=, or report this field.`);
  return raw.map(v => clip01(v * (k / t)));
}

function factorNodes(V, n, qa) {
  let Vm = V.map(r => (Array.isArray(r) ? r.slice() : [r]));
  const r = Vm[0].length;
  if (r > 2)
    throw new Error("topKProbabilities is implemented for factor rank <= 2");
  for (let c = 0; c < r; c++) {
    let m = 0;
    for (let i = 0; i < n; i++) m += Vm[i][c];
    m /= n;
    for (let i = 0; i < n; i++) Vm[i][c] -= m;
  }
  const h = hermite1(qa);
  let nodes, w;
  if (r === 1) {
    nodes = h.nodes.map(v => [v]);
    w = h.weights.slice();
  } else {
    nodes = []; w = [];
    for (const a of h.nodes) for (const b of h.nodes) nodes.push([a, b]);
    for (const u of h.weights) for (const v of h.weights) w.push(u * v);
    const s = w.reduce((a, b) => a + b, 0);
    w = w.map(v => v / s);
  }
  return { Vm, nodes, w };
}

export function topKProbabilities(mu, k, opts = {}) {
  const { V = null, D = null, base = "normal", points = 513, qa = 15 } = opts;
  const n = mu.length;
  k = Math.trunc(k);
  if (!(k >= 1 && k <= n - 1))
    throw new Error(`k must be in [1, n-1]; got k=${k}, n=${n}`);
  const sd = (D || new Array(n).fill(1)).map(Math.sqrt);
  const fn = typeof base === "function" ? base : BASES[base];
  if (!V) return checkedTopk(topkWithSlopes(mu, sd, k, fn, points).q,
                             k, "top-k race");
  const { Vm, nodes, w } = factorNodes(V, n, qa);
  const raw = new Array(n).fill(0);
  for (let q = 0; q < nodes.length; q++) {
    const shifted = mu.map((m, i) => {
      let s = m;
      for (let c = 0; c < nodes[q].length; c++) s += Vm[i][c] * nodes[q][c];
      return s;
    });
    const node = topkWithSlopes(shifted, sd, k, fn, points).q;
    for (let i = 0; i < n; i++) raw[i] += w[q] * node[i];
  }
  return checkedTopk(raw, k, "top-k race");
}

export function bottomKProbabilities(mu, k, opts = {}) {
  const n = mu.length;
  k = Math.trunc(k);
  if (!(k >= 1 && k <= n - 1))
    throw new Error(`k must be in [1, n-1]; got k=${k}, n=${n}`);
  return topKProbabilities(mu, n - k, opts).map(v => 1 - v);
}

export function topKJacobians(mu, k, opts = {}) {
  const { D = null, base = "normal", points = 513 } = opts;
  const n = mu.length;
  k = Math.trunc(k);
  if (!(k >= 1 && k <= n - 1))
    throw new Error(`k must be in [1, n-1]; got k=${k}, n=${n}`);
  const Dv = D || new Array(n).fill(1);
  const sd = Dv.map(Math.sqrt);
  const fn = typeof base === "function" ? base : BASES[base];
  const { dx, F, f, fp, z } = topkGrid(mu, sd, k, fn, points);
  const L = F.length;
  const dens = [];
  for (let t = 0; t < L; t++) dens.push(f[t].map((v, i) => v / sd[i]));
  const C = countDistribution(F);
  const Jm = [], Js = [];
  for (let i = 0; i < n; i++) {
    const Qi = looPmf(C, F, i);
    const pair = pairPmfAt(Qi, F, i, k);
    const rowMu = new Array(n).fill(0), rowSd = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let sm = 0, ss = 0;
      for (let t = 0; t < L; t++) {
        const kern = pair[j][t] * dens[t][i];
        sm += kern * dens[t][j];
        ss += kern * z[t][j] * dens[t][j];
      }
      rowMu[j] = sm * dx;
      rowSd[j] = ss * dx;
    }
    rowMu[i] = -rowMu.reduce((a, b) => a + b, 0);
    let own = 0;
    for (let t = 0; t < L; t++) {
      let cdfI = 0;
      for (let m = 0; m < k; m++) cdfI += Qi[t][m];
      own += (-(z[t][i] * fp[t][i] + f[t][i]) / Dv[i]) * cdfI;
    }
    rowSd[i] = own * dx;
    Jm.push(rowMu); Js.push(rowSd);
  }
  return { Jmu: Jm, Jsigma: Js };
}

export function rankProbabilities(mu, opts = {}) {
  const { D = null, base = "normal", points = 513 } = opts;
  const n = mu.length;
  const sd = (D || new Array(n).fill(1)).map(Math.sqrt);
  const fn = typeof base === "function" ? base : BASES[base];
  const { dx, F, f } = topkGrid(mu, sd, n - 1, fn, points);
  const C = countDistribution(F);
  const P = [];
  for (let i = 0; i < n; i++) {
    const Qi = looPmf(C, F, i);
    const row = new Array(n).fill(0);
    for (let t = 0; t < F.length; t++) {
      const d = f[t][i] / sd[i];
      for (let m = 0; m < n; m++) row[m] += Qi[t][m] * d;
    }
    P.push(row.map(v => v * dx));
  }
  const rows = P.map(r => r.reduce((a, b) => a + b, 0));
  const cols = new Array(n).fill(0);
  for (const r of P) for (let m = 0; m < n; m++) cols[m] += r[m];
  const bad = rows.some(v => Math.abs(v - 1) > 5e-3)
    || cols.some(v => Math.abs(v - 1) > 5e-3);
  if (bad) throw new Error("rank marginals defective; raise points=");
  return P.map((r, i) => r.map(v => clip01(v / rows[i])));
}

function validatedTarget(q, k, n, targetFloor) {
  let target = q.slice();
  if (target.length !== n)
    throw new Error(`target has ${target.length} entries for ${n} runners`);
  let floored = new Array(n).fill(false);
  if (targetFloor != null) {
    if (!(targetFloor > 0)) throw new Error("targetFloor must be positive");
    floored = target.map(v => v < targetFloor);
    target = target.map(v => Math.max(v, targetFloor));
  } else if (target.some(v => v <= 0)) {
    throw new Error(
      "all target memberships must be positive: a zero top-k probability " +
      "has no finite inverse. Pass targetFloor to floor small entries " +
      "deliberately.");
  }
  const s = target.reduce((a, b) => a + b, 0);
  target = target.map(v => v * (k / s));
  if (target.some(v => v >= 1))
    throw new Error(
      "after renormalizing to k slots, a target membership is >= 1: " +
      "certain membership has no finite inverse.");
  return { target, floored };
}

export function abilitiesFromTopk(q, k, opts = {}) {
  const { V = null, D = null, base = "normal", points = 513, qa = 15,
          nIter = 80, tol = 1e-8, targetFloor = null,
          returnInfo = false } = opts;
  const n = q.length;
  k = Math.trunc(k);
  if (!(k >= 1 && k <= n - 1))
    throw new Error(`k must be in [1, n-1]; got k=${k}, n=${n}`);
  const { target, floored } = validatedTarget(q, k, n, targetFloor);
  const sd = (D || new Array(n).fill(1)).map(Math.sqrt);
  const fn = typeof base === "function" ? base : BASES[base];
  const fac = V ? factorNodes(V, n, qa) : null;

  const logitT = target.map(v => Math.log(v) - Math.log1p(-v));
  const logT = target.map(Math.log);
  const mLog = logT.reduce((a, b) => a + b, 0) / n;
  let mu = logT.map(v => -(v - mLog) / 2);
  const alpha = n > 2 ? 1.0 : 0.7;
  let residMax = Infinity, iters = 0;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    let qraw, sl;
    if (!fac) {
      ({ q: qraw, slopes: sl } = topkWithSlopes(mu, sd, k, fn, points));
      sl = sl.slice();
    } else {
      qraw = new Array(n).fill(0);
      sl = new Array(n).fill(0);
      for (let j = 0; j < fac.nodes.length; j++) {
        const shifted = mu.map((m, i) => {
          let s = m;
          for (let c = 0; c < fac.nodes[j].length; c++)
            s += fac.Vm[i][c] * fac.nodes[j][c];
          return s;
        });
        const node = topkWithSlopes(shifted, sd, k, fn, points);
        for (let i = 0; i < n; i++) {
          qraw[i] += fac.w[j] * node.q[i];
          sl[i] += fac.w[j] * node.slopes[i];
        }
      }
    }
    const qhat = checkedTopk(qraw, k, "top-k inversion");
    const resid = qhat.map((v, i) =>
      Math.log(Math.max(v, 1e-300)) - Math.log(Math.max(1 - v, 1e-300))
      - logitT[i]);
    residMax = Math.max(...resid.map(Math.abs));
    if (residMax < tol) break;
    for (let i = 0; i < n; i++) {
      const dlogit = Math.min(
        sl[i] / Math.max(qhat[i] * (1 - qhat[i]), 1e-300), -1e-6);
      const lim = Math.min(2, 10 * Math.abs(resid[i]));
      let step = alpha * resid[i] / dlogit;
      if (step > lim) step = lim;
      if (step < -lim) step = -lim;
      mu[i] -= step;
    }
    const mm = mu.reduce((a, b) => a + b, 0) / n;
    mu = mu.map(v => v - mm);
  }
  const converged = residMax < tol;
  if (!converged && !returnInfo)
    console.warn(`abilitiesFromTopk did not converge: max |logit residual| ` +
                 `${residMax.toExponential(2)} after ${iters} iterations`);
  if (returnInfo)
    return { mu, info: { converged, maxLogitResidual: residMax,
                         iterations: iters, floored } };
  return mu;
}

export function locScaleFromTopkPair(q1, k1, q2, k2, opts = {}) {
  const { D0 = null, base = "normal", points = 513, nIter = 60,
          tol = 1e-8, ridge = 0.0, mu0 = null,
          returnInfo = false } = opts;
  const n = q1.length;
  k1 = Math.trunc(k1); k2 = Math.trunc(k2);
  if (k1 === k2)
    throw new Error("k1 == k2 gives one curve twice: scale is unidentified");
  for (const kk of [k1, k2])
    if (!(kk >= 1 && kk <= n - 1))
      throw new Error(`k must be in [1, n-1]; got k=${kk}, n=${n}`);
  const t1 = validatedTarget(q1, k1, n, null).target;
  const t2 = validatedTarget(q2, k2, n, null).target;
  const lt1 = t1.map(v => Math.log(v) - Math.log1p(-v));
  const lt2 = t2.map(v => Math.log(v) - Math.log1p(-v));

  let sd = D0 ? D0.map(Math.sqrt) : new Array(n).fill(1);
  let mu;
  if (mu0 != null) {
    const m0 = mu0.reduce((a, b) => a + b, 0) / n;
    mu = mu0.map(v => v - m0);
  } else {
    const [ka, ta] = k1 < k2 ? [k1, t1] : [k2, t2];
    // warm start only: the LM loop refines, loose tolerance by design
    mu = abilitiesFromTopk(ta, ka,
      { D: sd.map(v => v * v), base, points, nIter: 20, tol: 1e-3,
        returnInfo: true }).mu;
  }
  const sqr = Math.sqrt(Math.max(ridge, 0));

  const logits = (m, s) => {
    const qh1 = topKProbabilities(m, k1, { D: s.map(v => v * v), base, points })
      .map(v => Math.min(Math.max(v, 1e-300), 1 - 1e-15));
    const qh2 = topKProbabilities(m, k2, { D: s.map(v => v * v), base, points })
      .map(v => Math.min(Math.max(v, 1e-300), 1 - 1e-15));
    const r = qh1.map((v, i) => Math.log(v) - Math.log1p(-v) - lt1[i]).concat(
      qh2.map((v, i) => Math.log(v) - Math.log1p(-v) - lt2[i]),
      s.map(v => sqr * Math.log(v)));
    return { r, qh1, qh2 };
  };
  const fitMax = r => {
    let d = 0;
    for (let i = 0; i < 2 * n; i++) d = Math.max(d, Math.abs(r[i]));
    return d;
  };

  let { r, qh1, qh2 } = logits(mu, sd);
  let cost = r.reduce((a, b) => a + b * b, 0);
  let residMax = fitMax(r);
  let lam = 1e-6, iters = 0, lastAccepted = true;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    if (residMax < tol) break;
    const J = [];
    for (const [kk, qh] of [[k1, qh1], [k2, qh2]]) {
      const { Jmu, Jsigma } = topKJacobians(mu, kk,
        { D: sd.map(v => v * v), base, points });
      for (let i = 0; i < n; i++) {
        const g = 1 / Math.max(qh[i] * (1 - qh[i]), 1e-300);
        const row = new Array(2 * n);
        for (let j = 0; j < n; j++) {
          row[j] = Jmu[i][j] * g;
          row[n + j] = Jsigma[i][j] * sd[j] * g;
        }
        J.push(row);
      }
    }
    for (let i = 0; i < n; i++) {
      const row = new Array(2 * n).fill(0);
      row[n + i] = sqr;
      J.push(row);
    }
    const rows = J.length;
    const JtJ = [], Jtr = new Array(2 * n).fill(0);
    for (let a = 0; a < 2 * n; a++) {
      const row = new Array(2 * n).fill(0);
      for (let b = 0; b < 2 * n; b++)
        for (let t = 0; t < rows; t++) row[b] += J[t][a] * J[t][b];
      JtJ.push(row);
      for (let t = 0; t < rows; t++) Jtr[a] += J[t][a] * r[t];
    }
    let accepted = false;
    for (let attempt = 0; attempt < 8; attempt++) {
      const A = JtJ.map((row, a) =>
        row.map((v, b) => v + (a === b ? lam : 0)));
      let step;
      try {
        step = solve(A, Jtr.map(v => -v));
      } catch (e) {
        lam *= 8; continue;
      }
      let muN = mu.map((v, i) => v + step[i]);
      let lsN = sd.map((v, i) =>
        Math.min(Math.max(Math.log(v) + step[n + i], -3), 3));
      const lsMean = lsN.reduce((a, b) => a + b, 0) / n;
      const c = Math.exp(lsMean);
      const sdN = lsN.map(v => Math.exp(v - lsMean));
      const muMean = muN.reduce((a, b) => a + b, 0) / n;
      muN = muN.map(v => (v - muMean) / c);
      let out;
      try {
        out = logits(muN, sdN);
      } catch (e) {
        lam *= 8; continue;
      }
      const costN = out.r.reduce((a, b) => a + b * b, 0);
      if (costN < cost) {
        mu = muN; sd = sdN; r = out.r; qh1 = out.qh1; qh2 = out.qh2;
        cost = costN;
        residMax = fitMax(r);
        lam = Math.max(lam / 3, 1e-10);
        accepted = true;
        break;
      }
      lam *= 8;
    }
    lastAccepted = accepted;
    if (!accepted) break;
  }
  // with a ridge the penalized optimum keeps a nonzero fit residual by
  // design: an LM stall there is the answer, not a failure
  const converged = residMax < tol || (sqr > 0 && !lastAccepted);
  if (!converged && !returnInfo)
    console.warn(`locScaleFromTopkPair did not converge: max |logit ` +
                 `residual| ${residMax.toExponential(2)} after ${iters} iterations`);
  if (returnInfo)
    return { mu, sd, info: { converged, maxLogitResidual: residMax,
                             iterations: iters } };
  return { mu, sd };
}

export function locScaleFromWinAndSecond(pWin, pSecond, opts = {}) {
  // win plus EXACTLY-second marginals: P(2nd) + P(win) = P(top-2),
  // the well-posed pair. Each marginal renormalized to unit mass.
  const n = pWin.length;
  if (pSecond.length !== n)
    throw new Error("pWin and pSecond must have equal length");
  if (pWin.some(v => v <= 0) || pSecond.some(v => v <= 0))
    throw new Error("all win and second probabilities must be positive");
  const s1 = pWin.reduce((a, b) => a + b, 0);
  const s2 = pSecond.reduce((a, b) => a + b, 0);
  const p1 = pWin.map(v => v / s1);
  const top2 = p1.map((v, i) => v + pSecond[i] / s2);
  return locScaleFromTopkPair(p1, 1, top2, 2, opts);
}

function rankMarginalWithJacobian(mu, sd, r, fn, points) {
  const n = mu.length;
  const { dx, F, f } = topkGrid(mu, sd, n - 1, fn, points);
  const L = F.length;
  const dens = [];
  for (let t = 0; t < L; t++) dens.push(f[t].map((v, i) => v / sd[i]));
  const C = countDistribution(F);
  const p = new Array(n), J = [];
  for (let i = 0; i < n; i++) {
    const Qi = looPmf(C, F, i);
    let pi = 0;
    for (let t = 0; t < L; t++) pi += Qi[t][r - 1] * dens[t][i];
    p[i] = pi * dx;
    const hiPair = pairPmfAt(Qi, F, i, r);
    const loPair = r >= 2 ? pairPmfAt(Qi, F, i, r - 1) : null;
    const row = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let s = 0;
      for (let t = 0; t < L; t++) {
        let c = hiPair[j][t];
        if (loPair) c -= loPair[j][t];
        s += c * dens[t][j] * dens[t][i];
      }
      row[j] = s * dx;
    }
    row[i] = -row.reduce((a, b) => a + b, 0);
    J.push(row);
  }
  return { p, J };
}

export function abilitiesFromRankMarginal(p, r, opts = {}) {
  // invert one EXACT-rank marginal at frozen scales: two-branched for
  // r >= 2, mu0 selects the branch. See the python docstring.
  const { mu0 = null, D = null, base = "normal", points = 513,
          nIter = 60, tol = 1e-8, returnInfo = false } = opts;
  const n = p.length;
  r = Math.trunc(r);
  if (!(r >= 1 && r <= n))
    throw new Error(`rank must be in [1, n]; got r=${r}, n=${n}`);
  if (p.some(v => v <= 0))
    throw new Error("all rank probabilities must be positive");
  const s = p.reduce((a, b) => a + b, 0);
  const logt = p.map(v => Math.log(v / s));
  const sd = (D || new Array(n).fill(1)).map(Math.sqrt);
  const fn = typeof base === "function" ? base : BASES[base];
  let mu;
  if (mu0 != null) {
    const m0 = mu0.reduce((a, b) => a + b, 0) / n;
    mu = mu0.map(v => v - m0);
  } else {
    mu = new Array(n).fill(0);
  }

  let { p: phat, J } = rankMarginalWithJacobian(mu, sd, r, fn, points);
  let resid = phat.map((v, i) => Math.log(Math.max(v, 1e-300)) - logt[i]);
  let cost = resid.reduce((a, b) => a + b * b, 0);
  let residMax = Math.max(...resid.map(Math.abs));
  let lam = 1e-6, iters = 0;
  for (let it = 0; it < nIter; it++) {
    iters = it + 1;
    if (residMax < tol) break;
    const Jlog = J.map((row, i) =>
      row.map(v => v / Math.max(phat[i], 1e-300)));
    const A = [], g = new Array(n).fill(0);
    for (let a = 0; a < n; a++) {
      const row = new Array(n).fill(0);
      for (let b = 0; b < n; b++)
        for (let t = 0; t < n; t++) row[b] += Jlog[t][a] * Jlog[t][b];
      A.push(row);
      for (let t = 0; t < n; t++) g[a] += Jlog[t][a] * resid[t];
    }
    let accepted = false;
    for (let attempt = 0; attempt < 8; attempt++) {
      const Ad = A.map((row, a) => row.map((v, b) => v + (a === b ? lam : 0)));
      let step;
      try {
        step = solve(Ad, g.map(v => -v));
      } catch (e) {
        lam *= 8; continue;
      }
      let muN = mu.map((v, i) => v + step[i]);
      const mm = muN.reduce((a, b) => a + b, 0) / n;
      muN = muN.map(v => v - mm);
      const nx = rankMarginalWithJacobian(muN, sd, r, fn, points);
      const rN = nx.p.map((v, i) =>
        Math.log(Math.max(v, 1e-300)) - logt[i]);
      const costN = rN.reduce((a, b) => a + b * b, 0);
      if (costN < cost) {
        mu = muN; phat = nx.p; J = nx.J; resid = rN; cost = costN;
        residMax = Math.max(...resid.map(Math.abs));
        lam = Math.max(lam / 3, 1e-10);
        accepted = true;
        break;
      }
      lam *= 8;
    }
    if (!accepted) break;
  }
  const converged = residMax < tol;
  if (!converged && !returnInfo)
    console.warn(`abilitiesFromRankMarginal did not converge: max |log ` +
                 `residual| ${residMax.toExponential(2)} after ${iters} iterations`);
  if (returnInfo)
    return { mu, info: { converged, maxLogResidual: residMax,
                         iterations: iters } };
  return mu;
}
