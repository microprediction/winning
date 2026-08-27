// Core numerics for the winning JS port. The python numpy implementation
// is the spec; parity/check.mjs pins this port to its vectors.

export const TINY = 1e-300;

/* ---- normal cdf/log-cdf: series + scaled complementary erf ---------- */
function erfSeries(x) {
  let s = x, t = x;
  for (let n = 1; n < 120; n++) {
    t *= -x * x / n;
    s += t / (2 * n + 1);
    if (Math.abs(t) < 1e-19 * Math.abs(s)) break;
  }
  return (2 / Math.sqrt(Math.PI)) * s;
}
function erfcx(x) {
  // Laplace continued fraction, accurate for x >= 2.5
  let cf = 0;
  for (let k = 60; k >= 1; k--) cf = (k / 2) / (x + cf);
  return 1 / (Math.sqrt(Math.PI) * (x + cf));
}
export function ndtr(z) {
  const x = z / Math.SQRT2;
  if (x >= 2.5) return 1 - 0.5 * erfcx(x) * Math.exp(-x * x);
  if (x <= -2.5) return 0.5 * erfcx(-x) * Math.exp(-x * x);
  return 0.5 * (1 + erfSeries(x));
}
export function logndtr(z) {
  if (z > -3.5355) return Math.log(ndtr(z));
  const x = -z / Math.SQRT2;
  return Math.log(0.5 * erfcx(x)) - x * x;
}
export function npdf(z) {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

/* ---- symmetric tridiagonal eigen (QL, implicit shifts): nodes and
   first-row eigenvector components, for Golub-Welsch ----------------- */
export function tridiagEigen(d, e) {
  const n = d.length;
  const diag = d.slice();
  const off = e.slice(); off.push(0);
  const z = new Array(n).fill(0); z[0] = 1;
  // full first-row tracking needs the rotation applied to a row vector
  const row = new Array(n).fill(0); row[0] = 1;
  for (let l = 0; l < n; l++) {
    let iter = 0;
    let m;
    do {
      for (m = l; m < n - 1; m++) {
        const dd = Math.abs(diag[m]) + Math.abs(diag[m + 1]);
        if (Math.abs(off[m]) <= 1e-16 * dd) break;
      }
      if (m !== l) {
        if (iter++ === 50) throw new Error("tridiagEigen: no convergence");
        let g = (diag[l + 1] - diag[l]) / (2 * off[l]);
        let r = Math.hypot(g, 1);
        g = diag[m] - diag[l] + off[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1, c = 1, p = 0;
        for (let i = m - 1; i >= l; i--) {
          let f = s * off[i], b = c * off[i];
          r = Math.hypot(f, g);
          off[i + 1] = r;
          if (r === 0) { diag[i + 1] -= p; off[m] = 0; break; }
          s = f / r; c = g / r;
          g = diag[i + 1] - p;
          r = (diag[i] - g) * s + 2 * c * b;
          p = s * r;
          diag[i + 1] = g + p;
          g = c * r - b;
          const ri1 = row[i + 1], ri = row[i];
          row[i + 1] = s * ri + c * ri1;
          row[i] = c * ri - s * ri1;
        }
        if (off[l] !== 0 || m - 1 >= l) { diag[l] -= p; off[l] = g; off[m] = 0; }
      }
    } while (m !== l);
  }
  const idx = diag.map((v, i) => i).sort((a, b) => diag[a] - diag[b]);
  return { values: idx.map(i => diag[i]), first: idx.map(i => row[i]) };
}

/* probabilists' Hermite nodes/weights (weights normalized to sum 1),
   matching R's .hermite1 (parity with python hermegauss to fp) */
export function hermite1(order) {
  const d = new Array(order).fill(0);
  const e = [];
  for (let i = 1; i < order; i++) e.push(Math.sqrt(i));
  const { values, first } = tridiagEigen(d, e);
  let w = first.map(v => v * v);
  const s = w.reduce((a, b) => a + b, 0);
  w = w.map(v => v / s);
  return { nodes: values, weights: w };
}

/* pruned product rule: first coordinate slowest, prune WITHOUT
   renormalizing (matching the reference exactly) */
export function hermiteNodes(k, order = 15, prune = 1e-7) {
  const h = hermite1(order);
  if (k === 1) return { F: h.nodes.map(x => [x]), W: h.weights.slice() };
  const F = [], W = [];
  const idx = new Array(k).fill(0);
  const total = Math.pow(order, k);
  for (let t = 0; t < total; t++) {
    let rem = t;
    const node = new Array(k), digits = new Array(k);
    for (let dPos = k - 1; dPos >= 0; dPos--) {   // last coordinate fastest
      digits[dPos] = rem % order;
      rem = Math.floor(rem / order);
    }
    let w = 1;
    for (let dPos = 0; dPos < k; dPos++) {
      node[dPos] = h.nodes[digits[dPos]];
      w *= h.weights[digits[dPos]];
    }
    F.push(node); W.push(w);
  }
  const wmax = Math.max(...W);
  const keepF = [], keepW = [];
  for (let i = 0; i < W.length; i++) {
    if (W[i] > prune * wmax) { keepF.push(F[i]); keepW.push(W[i]); }
  }
  return { F: keepF, W: keepW };
}

/* ---- small dense linear algebra ------------------------------------ */
export function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    const p = M[c][c];
    if (Math.abs(p) < 1e-300) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / p;
      for (let cc = c; cc <= n; cc++) M[r][cc] -= f * M[c][cc];
    }
  }
  return M.map((row, i) => (Math.abs(row[i]) > 1e-300 ? row[n] / row[i] : 0));
}

export function mean(v) { return v.reduce((a, b) => a + b, 0) / v.length; }
export function interpClamped(x, xp, fp) {
  // np.interp semantics: ascending xp, end-clamped
  if (x <= xp[0]) return fp[0];
  const last = xp.length - 1;
  if (x >= xp[last]) return fp[last];
  let lo = 0, hi = last;                 // largest j with xp[j] <= x
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  const d = xp[lo + 1] - xp[lo];
  if (d <= 0) return fp[lo];
  return fp[lo] + (x - xp[lo]) / d * (fp[lo + 1] - fp[lo]);
}
