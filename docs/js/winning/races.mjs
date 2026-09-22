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
  logistic(z) {
    const c = Math.PI / Math.sqrt(3);
    const u = Math.min(Math.max(c * z, -700), 700);
    const S = 1 / (1 + Math.exp(u));
    const f = c * S * (1 - S);
    return [Math.max(S, 1e-300), f, -c * f * (1 - 2 * S)];
  },
  laplace(z) {
    const b = 1 / Math.sqrt(2);
    const f = Math.exp(-Math.abs(z) / b) / (2 * b);
    const S = z < 0 ? 1 - 0.5 * Math.exp(z / b) : 0.5 * Math.exp(-z / b);
    return [Math.max(S, 1e-300), f, -Math.sign(z) * f / b];
  },
};
const SPANS = { normal: [8, 8], gumbel: [22, 8], logistic: [16, 16], laplace: [18, 18] };

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
      if (r === 1 && Math.ceil(8 * sharp) > 80) {
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
  const { nIter = 60, tol = 1e-8, structure = null, V = null, D = null,
          F = null, W = null, base = "normal" } = opts;
  if (structure) return dispatchAbilities(pTarget, structure, opts);
  let target = pTarget.slice();
  const s = target.reduce((a, b) => a + b, 0);
  target = target.map(v => v / s);
  const n = target.length;
  const logt = target.map(Math.log);
  const lm = mean(logt);
  // the field's contrast scale (matching python/R): median idiosyncratic
  // variance plus the mean factor variance under the represented nodes
  const Dn = D ? D.slice() : new Array(n).fill(1);
  const Vn = V ? V.map(row => (Array.isArray(row) ? row.slice() : [row])) : Array.from({ length: n }, () => [0]);
  const r = Vn[0].length;
  const colMean = Array.from({ length: r }, (_, c) => mean(Vn.map(row => row[c])));
  const Vc = Vn.map(row => row.map((v, c) => v - colMean[c]));
  let CovF = Array.from({ length: r }, (_, a) => Array.from({ length: r }, (_, b) => (a === b ? 1 : 0)));
  if (V && F) {
    const Q = F.length;
    const Wq = W ? W.map(w => w / W.reduce((a, b) => a + b, 0)) : new Array(Q).fill(1 / Q);
    const Fm = Array.from({ length: r }, (_, c) => F.reduce((acc, f, q) => acc + Wq[q] * f[c], 0));
    CovF = Array.from({ length: r }, (_, a) => Array.from({ length: r }, (_, b) =>
      F.reduce((acc, f, q) => acc + Wq[q] * (f[a] - Fm[a]) * (f[b] - Fm[b]), 0)));
  }
  const sigV = (i, j) => Vc[i].reduce((acc, va, a) => acc + va * CovF[a].reduce((acc2, cab, b) => acc2 + cab * Vc[j][b], 0), 0);
  const med = (arr) => { const z = arr.slice().sort((a, b) => a - b); const h = Math.floor(z.length / 2); return z.length % 2 ? z[h] : 0.5 * (z[h - 1] + z[h]); };
  const scale = Math.sqrt(med(Dn) + mean(Dn.map((_, i) => sigV(i, i))));
  if (n === 2 && base === "normal") {
    // a pair is one Gaussian contrast: closed form (matching python/R)
    const sdD = Math.sqrt(Math.max(sigV(0, 0) + sigV(1, 1) - 2 * sigV(0, 1) + Dn[0] + Dn[1], 1e-300));
    const gap = sdD * invNormalRational(target[0]);
    return [-0.5 * gap, 0.5 * gap];
  }
  let mu = logt.map(v => -(v - lm) / 2 * scale);
  // damping: a pair, or two runners holding nearly all the mass, two-cycles
  // undamped; the sweeps then adapt to the contraction they observe
  // (matching python's _jacobi_sweeps)
  const top2 = n > 2 ? target.slice().sort((a, b) => b - a).slice(0, 2).reduce((a, b) => a + b, 0) : 1;
  let alpha = (n === 2 || top2 > 0.8) ? 0.7 : 1.0;
  let prev = null;
  let prevStep = null;
  for (let it = 0; it < nIter; it++) {
    const { p: praw, slopes: sl } = raceProbabilities(mu, { ...opts, returnSlopes: true, structure: null });
    const phat = praw.map(v => Math.max(v, 1e-300));
    let resid = phat.map((v, i) => Math.log(v) - logt[i]);
    let dlogp = sl.map((v, i) => Math.min(v / phat[i], -1e-6));
    let rmax = Math.max(...resid.map(Math.abs));
    let rrms = Math.sqrt(mean(resid.map(v => v * v)));
    if (rmax < tol) break;
    if (prev && alpha > 0.1 && rmax >= prev.rmax && rrms >= prev.rrms) {
      alpha = Math.max(0.5 * alpha, 0.1);
      ({ mu, resid, dlogp, rmax, rrms } = prev);
      prevStep = null;
    }
    prev = { mu, resid, dlogp, rmax, rrms };
    // residual-proportional step cap in the field's scale
    let step = resid.map((v, i) => {
      const lim = Math.min(2, 10 * Math.abs(v)) * scale;
      return Math.min(Math.max(alpha * v / dlogp[i], -lim), lim);
    });
    const sm = mean(step);
    step = step.map(v => v - sm);
    if (prevStep) {
      const den = prevStep.reduce((a, b) => a + b * b, 0);
      const rho = den > 0 ? step.reduce((a, b, i) => a + b * prevStep[i], 0) / den : 0;
      if (rho < 0) {
        const lam = 1 - (1 - rho) / alpha;
        alpha = Math.min(Math.max(2 / (2 - lam), 0.1), 1);
      }
    }
    prevStep = step;
    mu = mu.map((m, i) => m - step[i]);
  }
  return mu;
}

// filled in by structures.mjs to avoid a cycle
export let dispatchProbabilities = () => { throw new Error("import structures.mjs first"); };
export let dispatchAbilities = () => { throw new Error("import structures.mjs first"); };
export function _setDispatch(dp, da) { dispatchProbabilities = dp; dispatchAbilities = da; }
