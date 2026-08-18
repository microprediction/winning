/* Factor-correlated race transforms: JavaScript port of
 * winning.factor.core (Python canonical). Min-wins convention.
 * Parity against committed test vectors: run `node test_parity.mjs`.
 */

const SQRT2 = Math.SQRT2;
const SQRT_PI = Math.sqrt(Math.PI);
const SQRT_2PI = Math.sqrt(2 * Math.PI);
const PFLOOR = 1e-15;

/* erf by alternating series, accurate for |x| <= ~1.5 */
function erfSeries(x) {
  let term = x, sum = x;
  for (let n = 1; n < 60; n++) {
    term *= -x * x / n;
    const add = term / (2 * n + 1);
    sum += add;
    if (Math.abs(add) < 1e-18 * Math.abs(sum)) break;
  }
  return (2 / SQRT_PI) * sum;
}

/* scaled complementary error function erfcx(x) = e^{x^2} erfc(x), x >= 1,
 * by the classical continued fraction (modified Lentz). */
function erfcx(x) {
  const tiny = 1e-30;
  let f = tiny, C = f, D = 0;
  // CF: erfcx(x) = (1/sqrt(pi)) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))
  for (let n = 0; n < 400; n++) {
    const a = n === 0 ? 1.0 : n / 2.0;
    const b = n === 0 ? 0.0 : x;
    // first step: b0 = x handled by starting the wrap
    const bb = n === 0 ? x : b;
    D = bb + a * D;
    if (Math.abs(D) < tiny) D = tiny;
    C = bb + a / C;
    if (Math.abs(C) < tiny) C = tiny;
    D = 1 / D;
    const delta = C * D;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-17) break;
  }
  return f / SQRT_PI;
}

/* log of the standard normal CDF, tail-stable (port of scipy.log_ndtr use) */
export function logndtr(z) {
  if (z >= 1.0) {
    // log(1 - Phi(-z)); Phi(-z) computed via the negative branch
    return Math.log1p(-Math.exp(logndtr(-z)));
  }
  if (z > -1.0) {
    return Math.log(0.5 * (1 + erfSeries(z / SQRT2)));
  }
  // z <= -1: Phi(z) = 0.5 * erfcx(-z/sqrt2) * exp(-z^2/2)
  const x = -z / SQRT2;
  return Math.log(0.5 * erfcx(x)) - 0.5 * z * z;
}

function lattice(Mall, sd, points) {
  let mMin = Infinity, mMax = -Infinity, sMax = 0;
  for (const row of Mall) for (const v of row) { if (v < mMin) mMin = v; if (v > mMax) mMax = v; }
  for (const s of sd) if (s > sMax) sMax = s;
  const lo = mMin - 8 * sMax, hi = mMax + 8 * sMax;
  const x = new Float64Array(points);
  for (let l = 0; l < points; l++) x[l] = lo + (hi - lo) * l / (points - 1);
  return { x, dx: (hi - lo) / (points - 1) };
}

function condMeans(mu, V, F) {
  const Q = F.length, N = mu.length, K = F[0].length;
  const M = [];
  for (let q = 0; q < Q; q++) {
    const row = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = mu[i];
      for (let d = 0; d < K; d++) s += F[q][d] * V[i][d];
      row[i] = s;
    }
    M.push(row);
  }
  return M;
}

/* forward pass: shares (and optionally slopes, pairwise densities, deletions) */
export function winProbabilitiesFactor(mu, V, D, F, W, opts = {}) {
  const points = opts.points || 501;
  const N = mu.length, Q = F.length;
  const sd = D.map(Math.sqrt);
  const M = condMeans(mu, V, F);
  const { x, dx } = lattice(M, sd, points);
  const L = x.length;
  const p = new Float64Array(N);
  const slope = new Float64Array(N);
  const w = opts.pairwise ? Array.from({ length: N }, () => new Float64Array(N)) : null;
  const q = opts.deletions ? Array.from({ length: N }, () => new Float64Array(N)) : null;

  const logS = Array.from({ length: N }, () => new Float64Array(L));
  const f = Array.from({ length: N }, () => new Float64Array(L));
  const zbuf = Array.from({ length: N }, () => new Float64Array(L));
  const logSfield = new Float64Array(L);

  for (let c = 0; c < Q; c++) {
    logSfield.fill(0);
    for (let i = 0; i < N; i++) {
      for (let l = 0; l < L; l++) {
        const z = (x[l] - M[c][i]) / sd[i];
        zbuf[i][l] = z;
        const ls = logndtr(-z);
        logS[i][l] = ls;
        logSfield[l] += ls;
        f[i][l] = Math.exp(-0.5 * z * z) / (sd[i] * SQRT_2PI);
      }
    }
    const Wc = W[c];
    for (let i = 0; i < N; i++) {
      let acc = 0, accS = 0;
      for (let l = 0; l < L; l++) {
        let e = logSfield[l] - logS[i][l];
        if (e > 0) e = 0;
        const rest = e < -745 ? 0 : Math.exp(e);
        acc += f[i][l] * rest;
        accS += zbuf[i][l] * f[i][l] / sd[i] * rest;
      }
      p[i] += Wc * acc * dx;
      slope[i] += Wc * accS * dx;
    }
    if (w) {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        if (j === i) continue;
        let acc = 0;
        for (let l = 0; l < L; l++) {
          let e = logSfield[l] - logS[i][l] - logS[j][l];
          if (e > 0) e = 0;
          acc += f[i][l] * f[j][l] * (e < -745 ? 0 : Math.exp(e));
        }
        w[i][j] += Wc * acc * dx;
      }
    }
    if (q) {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (j === i) continue;
          let acc = 0;
          for (let l = 0; l < L; l++) {
            let e = logSfield[l] - logS[i][l] - logS[j][l];
            if (e > 0) e = 0;
            acc += f[j][l] * (e < -745 ? 0 : Math.exp(e));
          }
          q[i][j] += Wc * acc * dx;
        }
      }
    }
  }
  let total = 0;
  for (const v of p) total += v;
  const out = Array.from(p, (v) => v / total);
  const res = { p: out, total, slope: Array.from(slope) };
  if (w) res.w = w.map((r) => Array.from(r));
  if (q) {
    res.deletions = q.map((r) => {
      let s = 0;
      for (const v of r) s += v;
      return Array.from(r, (v) => v / s);
    });
  }
  return res;
}

/* inverse transform: damped coordinatewise Newton (port of
 * abilities_from_probabilities_factor) */
export function abilitiesFromProbabilitiesFactor(pTarget, V, D, F, W, opts = {}) {
  const nIter = opts.nIter || 50, tol = opts.tol || 1e-6, points = opts.points || 501;
  const N = pTarget.length;
  let psum = 0;
  for (const v of pTarget) { if (v <= 0) throw new Error("targets must be positive"); psum += v; }
  const p = pTarget.map((v) => v / psum);
  const logp = p.map(Math.log);
  const sd = D.map(Math.sqrt);
  const vnorm2 = V.map((row) => row.reduce((a, b) => a + b * b, 0));
  const floor = Math.max(1e-9, 1e-4 / N);
  const ident = p.map((v) => v > floor);

  let mu;
  const anyV = V.some((row) => row.some((v) => v !== 0));
  if (F[0].length >= 1 && anyV) {
    const sdTot2 = D.map((d, i) => d + vnorm2[i]);
    mu = abilitiesFromProbabilitiesFactor(
      p, V.map(() => [0]), sdTot2, [[0]], [1], { nIter, tol, points });
  } else {
    const m = logp.reduce((a, b) => a + b, 0) / N;
    mu = logp.map((v) => (v - m) / 2.0);
  }
  const stepCap = D.map((d, i) => Math.sqrt(d + vnorm2[i]));
  let prevRes = Infinity, damp = 1.0, res = Infinity;
  for (let it = 0; it < nIter; it++) {
    const fwd = winProbabilitiesFactor(mu, V, D, F, W, { points });
    const phat = fwd.p.map((v) => Math.max(v / 1.0, PFLOOR));
    const slope = fwd.slope;
    const resid = phat.map((v, i) => Math.log(v) - logp[i]);
    res = 0;
    let any = false;
    for (let i = 0; i < N; i++) if (ident[i]) { any = true; res = Math.max(res, Math.abs(resid[i])); }
    if (!any) for (let i = 0; i < N; i++) res = Math.max(res, Math.abs(resid[i]));
    if (res < tol) break;
    if (res > prevRes * 1.2) damp = Math.max(0.25, damp * 0.5);
    prevRes = res;
    let mean = 0;
    for (let i = 0; i < N; i++) {
      let dlogp = slope[i] / phat[i];
      const ceil = -1e-3 / (sd[i] + 1e-9);
      if (dlogp > ceil) dlogp = ceil;
      let delta = damp * resid[i] / dlogp;
      if (delta > stepCap[i]) delta = stepCap[i];
      if (delta < -stepCap[i]) delta = -stepCap[i];
      mu[i] -= delta;
      mean += mu[i];
    }
    mean /= N;
    for (let i = 0; i < N; i++) mu[i] -= mean;
  }
  return mu;
}
