// The general race: min-wins, normal/gumbel bases, winner-bulk lattice,
// adaptive factor quadrature. Port of winning/factor/races.py.
import { TINY, ndtr, logndtr, npdf, hermiteNodes, mean } from "./core.mjs";

const EULER = 0.5772156649015329;

export const BASES = {
  normal: z => {
    const S = Math.max(1 - ndtr(z), 1e-300);
    const f = npdf(z);
    return [S, f, -z * f];
  },
  gumbel: z => {
    const c = Math.PI / Math.sqrt(6);
    const u = Math.min(z * c - EULER, 30);
    const eu = Math.exp(u);
    const S = Math.max(Math.exp(-eu), 1e-300);
    const f = c * eu * S;
    return [S, f, c * c * eu * S * (1 - eu)];
  },
};
const SPANS = { normal: [8, 8], gumbel: [22, 8] };

function setup(mu, V, D, F, W, base) {
  const n = mu.length;
  D = D ? D.slice() : new Array(n).fill(1);
  if (!V) {
    V = mu.map(() => [0]);
    F = [[0]]; W = [1];
  } else {
    if (!F || !W) {
      // adaptive order: sharpness rule identical to python/R
      let sharp = 0;
      for (let i = 0; i < n; i++) {
        const nv = Math.sqrt(V[i].reduce((a, b) => a + b * b, 0));
        sharp = Math.max(sharp, nv / Math.sqrt(Math.max(D[i], 1e-300)));
      }
      const r = V[0].length;
      if (r === 1 && Math.ceil(8 * sharp) > 201) {
        // rank-1 extreme sharpness (matching python/R): equal-weight
        // midpoint-quantile grid scaled with sharpness replaces GH
        const Q = Math.min(Math.ceil(8 * sharp), 4001);
        F = []; W = new Array(Q).fill(1 / Q);
        for (let q = 0; q < Q; q++) F.push([invNormalRational((q + 0.5) / Q)]);
      } else if (Math.pow(15, r) > 100000) {
        // high-rank tensor footgun (matching python/R): Halton fallback
        const Q = 8192;
        F = []; W = new Array(Q).fill(1 / Q);
        const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                        43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89];
        for (let idx = 0; idx < Q; idx++) {
          const node = [];
          for (let dim = 0; dim < r; dim++) {
            const b = primes[dim];
            let i = idx + 21, f = 1 / b, h = 0;
            while (i > 0) { h += f * (i % b); i = Math.floor(i / b); f /= b; }
            node.push(invNormalRational(Math.min(Math.max(h, 1e-12), 1 - 1e-12)));
          }
          F.push(node);
        }
      } else {
        const cap = r === 1 ? 201 : r === 2 ? 41 : 15;
        const Q = Math.min(Math.max(Math.ceil(8 * sharp), 15), cap);
        const hw = hermiteNodes(r, Q);
        F = hw.F; W = hw.W;
      }
    }
  }
  const fn = typeof base === "function" ? base : BASES[base];
  const span = typeof base === "function" ? [12, 12] : (SPANS[base] || [12, 12]);
  return { mu, V, D, F, W, fn, left: span[0], right: span[1] };
}

function condMeans(mu, V, F) {
  // M[q][i] = mu_i + V_i . F_q
  return F.map(fq => mu.map((m, i) => {
    let s = m;
    for (let r = 0; r < fq.length; r++) s += V[i][r] * fq[r];
    return s;
  }));
}


function invNormalRational(p) {
  // Acklam rational approximation, adequate for node placement
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
  if (p > 1 - pl) return -invNormalRational(1 - p);
  const q = p - 0.5, r2 = q * q;
  return (((((a[0]*r2+a[1])*r2+a[2])*r2+a[3])*r2+a[4])*r2+a[5])*q /
         (((((b[0]*r2+b[1])*r2+b[2])*r2+b[3])*r2+b[4])*r2+1);
}

function bulkWindow(Mall, sd, points, delta) {
  const n = sd.length;
  const muLo = new Array(n).fill(Infinity), muHi = new Array(n).fill(-Infinity);
  for (const row of Mall) for (let i = 0; i < n; i++) {
    if (row[i] < muLo[i]) muLo[i] = row[i];
    if (row[i] > muHi[i]) muHi[i] = row[i];
  }
  const smax = Math.max(...sd);
  const G = (x, mus) => {
    let ls = 0;
    for (let i = 0; i < n; i++) ls += Math.log(Math.max(1 - ndtr((x - mus[i]) / sd[i]), 1e-300));
    return 1 - Math.exp(ls);
  };
  const lo0 = Math.min(...muLo) - 9 * smax;
  const hi0 = Math.max(...muHi) + 9 * smax;
  let a = lo0, b = hi0;
  for (let it = 0; it < 80; it++) {
    const m = 0.5 * (a + b);
    if (G(m, muLo) < delta) a = m; else b = m;
  }
  const xlo = a;
  a = xlo; b = hi0;
  for (let it = 0; it < 80; it++) {
    const m = 0.5 * (a + b);
    if (G(m, muHi) < 1 - delta) a = m; else b = m;
  }
  const pad = 2 * smax;
  const out = new Array(points);
  const step = (b + pad - (xlo - pad)) / (points - 1);
  for (let t = 0; t < points; t++) out[t] = xlo - pad + t * step;
  return out;
}

export function raceProbabilities(mu, opts = {}) {
  const { V = null, D = null, F = null, W = null, base = "normal",
          points = 257, returnSlopes = false, window: win = "bulk",
          delta = 1e-12, structure = null, qa = 9, qf = 15 } = opts;
  if (structure) {
    return dispatchProbabilities(mu, structure, { base, points, qa, qf, returnSlopes });
  }
  const st = setup(mu, V, D, F, W, base);
  const n = st.mu.length;
  const sd = st.D.map(Math.sqrt);
  const Mall = condMeans(st.mu, st.V, st.F);
  let x;
  if (win === "bulk") {
    x = bulkWindow(Mall, sd, points, delta);
  } else {
    let mn = Infinity, mx = -Infinity;
    for (const row of Mall) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const smax = Math.max(...sd);
    x = new Array(points);
    const lo = mn - st.left * smax, hi = mx + st.right * smax;
    for (let t = 0; t < points; t++) x[t] = lo + t * (hi - lo) / (points - 1);
  }
  let dx = x[1] - x[0];
  {
    // extreme-sharpness lattice refinement (matching python/R)
    const smin = Math.min(...sd);
    let vmax = 0;
    for (const row of st.V) vmax = Math.max(vmax, Math.sqrt(row.reduce((a, b) => a + b * b, 0)));
    if (vmax / Math.max(smin, 1e-300) > 25 && dx > 0.5 * smin) {
      const span = x[x.length - 1] - x[0];
      const need = Math.ceil(span / (0.5 * smin)) + 1;
      const pts2 = Math.min(need, 8193);
      if (pts2 > x.length) {
        const x0 = x[0];
        x = new Array(pts2);
        for (let t = 0; t < pts2; t++) x[t] = x0 + t * span / (pts2 - 1);
        dx = x[1] - x[0];
      }
    }
  }
  const p = new Array(n).fill(0);
  const slope = new Array(n).fill(0);
  const logS = new Array(n), fArr = new Array(n), fpArr = new Array(n);
  for (let q = 0; q < st.F.length; q++) {
    const Mq = Mall[q], wq = st.W[q];
    const L = new Array(x.length).fill(0);
    for (let i = 0; i < n; i++) {
      const li = new Array(x.length), fi = new Array(x.length), fpi = new Array(x.length);
      for (let t = 0; t < x.length; t++) {
        const z = (x[t] - Mq[i]) / sd[i];
        const [S, f, fp] = st.fn(z);
        li[t] = Math.log(S);
        fi[t] = f / sd[i];
        fpi[t] = fp;
        L[t] += li[t];
      }
      logS[i] = li; fArr[i] = fi; fpArr[i] = fpi;
    }
    for (let i = 0; i < n; i++) {
      let si = 0, sl = 0;
      const li = logS[i], fi = fArr[i], fpi = fpArr[i];
      const sd2 = sd[i] * sd[i];
      for (let t = 0; t < x.length; t++) {
        const e = Math.min(Math.max(L[t] - li[t], -745), 0);
        const rest = Math.exp(e);
        si += fi[t] * rest;
        sl += -fpi[t] / sd2 * rest;
      }
      p[i] += wq * si * dx;
      slope[i] += wq * sl * dx;
    }
  }
  const total = p.reduce((a, b) => a + b, 0);
  const pn = p.map(v => v / total);
  if (returnSlopes) return { p: pn, slopes: slope.map(v => v / total) };
  return pn;
}

export function abilitiesFromRace(pTarget, opts = {}) {
  const { nIter = 60, tol = 1e-8, structure = null } = opts;
  if (structure) return dispatchAbilities(pTarget, structure, opts);
  let target = pTarget.slice();
  const s = target.reduce((a, b) => a + b, 0);
  target = target.map(v => v / s);
  const logt = target.map(Math.log);
  const lm = mean(logt);
  let mu = logt.map(v => -(v - lm) / 2);
  const alpha = target.length > 2 ? 1.0 : 0.7;
  for (let it = 0; it < nIter; it++) {
    const { p: phat, slopes: sl } = raceProbabilities(mu, { ...opts, returnSlopes: true, structure: null });
    const resid = phat.map((v, i) => Math.log(Math.max(v, 1e-300)) - logt[i]);
    if (Math.max(...resid.map(Math.abs)) < tol) break;
    mu = mu.map((m, i) => {
      const dlogp = Math.min(sl[i] / Math.max(phat[i], 1e-300), -1e-6);
      // residual-proportional step cap (heavy-favorite stall fix,
      // mirrored from the python engine)
      const lim = Math.min(2, 10 * Math.abs(resid[i]));
      return m - Math.min(Math.max(alpha * resid[i] / dlogp, -lim), lim);
    });
    const mm = mean(mu);
    mu = mu.map(v => v - mm);
  }
  return mu;
}

// filled in by structures.mjs to avoid a cycle
export let dispatchProbabilities = () => { throw new Error("import structures.mjs first"); };
export let dispatchAbilities = () => { throw new Error("import structures.mjs first"); };
export function _setDispatch(dp, da) { dispatchProbabilities = dp; dispatchAbilities = da; }
