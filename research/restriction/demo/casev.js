// Case V and Luce restriction maps, self-contained, no dependencies.
// Runs in a browser and under node. Every routine here is used by checks.js to
// verify a claim made in papers/thurstone_humans/paper.tex.

// ---------------------------------------------------------------- normal law
// Hart's algorithm for the cumulative normal, in the form given by West (2005).
// Double precision throughout, absolute error below 1e-15.
export function Phi(x) {
  const z = Math.abs(x);
  let c;
  if (z > 37) {
    c = 0;
  } else {
    const e = Math.exp(-z * z / 2);
    if (z < 7.07106781186547) {
      let b = 3.52624965998911e-2 * z + 0.700383064443688;
      b = b * z + 6.37396220353165;
      b = b * z + 33.912866078383;
      b = b * z + 112.079291497871;
      b = b * z + 221.213596169931;
      b = b * z + 220.206867912376;
      c = e * b;
      let d = 8.83883476483184e-2 * z + 1.75566716318264;
      d = d * z + 16.064177579207;
      d = d * z + 86.7807322029461;
      d = d * z + 296.564248779674;
      d = d * z + 637.333633378831;
      d = d * z + 793.826512519948;
      d = d * z + 440.413735824752;
      c = c / d;
    } else {
      let b = z + 0.65;
      b = z + 4 / b;
      b = z + 3 / b;
      b = z + 2 / b;
      b = z + 1 / b;
      c = e / b / 2.506628274631;
    }
  }
  return x > 0 ? 1 - c : c;
}

export const phi = (x) => 0.3989422804014327 * Math.exp(-0.5 * x * x);

// ---------------------------------------------------------------- the contest
// p_i = integral of phi(x - a_i) * prod_{k != i} Phi(x - a_k) dx, by Simpson.
export function winProbs(a, lo = -16, hi = 16, n = 20000) {
  const K = a.length;
  const h = (hi - lo) / n;
  const out = new Array(K).fill(0);
  for (let s = 0; s <= n; s++) {
    const x = lo + s * h;
    const w = (s === 0 || s === n) ? 1 : (s % 2 ? 4 : 2);
    const F = a.map((ak) => Phi(x - ak));
    for (let i = 0; i < K; i++) {
      let prod = phi(x - a[i]);
      if (prod === 0) continue;
      for (let k = 0; k < K; k++) if (k !== i) prod *= F[k];
      out[i] += w * prod;
    }
  }
  const scale = h / 3;
  const raw = out.map((v) => v * scale);
  const tot = raw.reduce((u, v) => u + v, 0);
  return raw.map((v) => v / tot);
}

// Invert shares to locations. Newton on the log-share residual in the K-1 free
// coordinates, with the sum-to-zero normalization the paper uses, falling back to a
// damped multiplicative update when a Newton step does not reduce the residual.
export function calibrate(p, tol = 1e-12, maxIter = 200) {
  const K = p.length;
  const mean = Math.log(p.reduce((u, v) => u + v, 0) / K);
  let a = p.map((pi) => Math.log(pi) - mean);
  const centre = (v) => { const m = v.reduce((u, w) => u + w, 0) / K; return v.map((w) => w - m); };
  a = centre(a);

  const resid = (v) => {
    const q = winProbs(v);
    return { q, r: q.map((qi, i) => Math.log(qi) - Math.log(p[i])) };
  };
  const worstOf = (r) => Math.max(...r.map(Math.abs));

  let { q, r } = resid(a);
  for (let it = 0; it < maxIter && worstOf(r) > tol; it++) {
    // numerical Jacobian of the first K-1 residuals in the first K-1 coordinates,
    // with the last coordinate pinned by the sum-to-zero constraint
    const n = K - 1, h = 1e-5;
    const J = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      const up = a.slice();
      up[j] += h;
      up[K - 1] -= h;
      const rj = resid(centre(up)).r;
      for (let i = 0; i < n; i++) J[i][j] = (rj[i] - r[i]) / h;
    }
    // solve J d = -r by Gaussian elimination with partial pivoting
    const M = J.map((row, i) => row.concat([-r[i]]));
    let singular = false;
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[piv][c])) piv = i;
      if (Math.abs(M[piv][c]) < 1e-14) { singular = true; break; }
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let i = c + 1; i < n; i++) {
        const f = M[i][c] / M[c][c];
        for (let k = c; k <= n; k++) M[i][k] -= f * M[c][k];
      }
    }
    let step = null;
    if (!singular) {
      const d = new Array(n).fill(0);
      for (let i = n - 1; i >= 0; i--) {
        let acc = M[i][n];
        for (let k = i + 1; k < n; k++) acc -= M[i][k] * d[k];
        d[i] = acc / M[i][i];
      }
      step = d;
    }
    let improved = false;
    if (step) {
      for (const t of [1, 0.5, 0.25, 0.1]) {
        const trial = a.slice();
        let tot = 0;
        for (let i = 0; i < n; i++) { trial[i] += t * step[i]; tot += t * step[i]; }
        trial[K - 1] -= tot;
        const out = resid(centre(trial));
        if (worstOf(out.r) < worstOf(r)) { a = centre(trial); q = out.q; r = out.r; improved = true; break; }
      }
    }
    if (!improved) {
      const trial = centre(a.map((ai, i) => ai + 0.5 * (Math.log(p[i]) - Math.log(q[i]))));
      const out = resid(trial);
      if (worstOf(out.r) >= worstOf(r)) break;
      a = trial; q = out.q; r = out.r;
    }
  }
  const residual = Math.max(...p.map((pi, i) => Math.abs(q[i] - pi)));
  return { a, residual };
}

// ---------------------------------------------------------------- the two maps
export const linearNormalization = (p, keep) => {
  const sub = keep.map((i) => p[i]);
  const tot = sub.reduce((u, v) => u + v, 0);
  return sub.map((v) => v / tot);
};

export const gaussianRenormalization = (a, keep) => winProbs(keep.map((i) => a[i]));

// Pair formula, Equation (3) of the paper.
export const pairProb = (ai, aj) => Phi((ai - aj) / Math.SQRT2);

// ---------------------------------------------------------------- diagnostics
// Reverse hazard of the standard normal and its log-second-derivative, which the
// paper claims equals -Var(Z | Z < x).
export const reverseHazard = (x) => phi(x) / Phi(x);
export function logRSecondDerivative(x, h = 1e-4) {
  const f = (t) => Math.log(reverseHazard(t));
  return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}
// Var(Z | Z < x) = 1 - x*r(x) - r(x)^2
export const varTruncated = (x) => {
  const r = reverseHazard(x);
  return 1 - x * r - r * r;
};

// Gumbel reverse hazard with scale s: r(x) = exp(-x/s)/s, so log r is affine.
export const gumbelLogR = (x, s = 1) => -x / s - Math.log(s);

// Contraction slope lambda, as defined in the paper: with i the higher-share
// alternative, delta_ij = logit(q_ij) - log(p_i/p_j), fitted through the origin
// as delta = -lambda * log(p_i/p_j).
export function lambdaSlope(p, q) {
  let num = 0, den = 0;
  for (let i = 0; i < p.length; i++) {
    for (let j = i + 1; j < p.length; j++) {
      const [hi, lo] = p[i] >= p[j] ? [i, j] : [j, i];
      const L = Math.log(p[hi] / p[lo]);
      const d = Math.log(q(hi, lo) / (1 - q(hi, lo))) - L;
      num += -d * L;
      den += L * L;
    }
  }
  return num / den;
}

// Within-set confusion mass, the boundary diagnostic of the paper's Getty table:
// of the errors a surviving stimulus makes on the full menu, the fraction landing
// on another survivor. rows[i][j] is the count of response j to stimulus i.
export function withinSetConfusion(rows, survivors) {
  let num = 0, den = 0;
  for (const i of survivors) {
    const total = rows[i].reduce((u, v) => u + v, 0);
    den += total - rows[i][i];
    for (const j of survivors) if (j !== i) num += rows[i][j];
  }
  return num / den;
}

export const logLoss = (pred, obs) => {
  const tot = obs.reduce((u, v) => u + v, 0);
  let s = 0;
  for (let i = 0; i < pred.length; i++) {
    if (obs[i] > 0) s -= (obs[i] / tot) * Math.log(Math.max(pred[i], 1e-12));
  }
  return s;
};
