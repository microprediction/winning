// The classic state-price lattice calibration -- port of
// winning/lattice.py + lattice_calibration.py, dead heats included.
import { ndtr, npdf, interpClamped, mean } from "./core.mjs";

export function pdfToCdf(f) {
  const c = new Array(f.length);
  let s = 0;
  for (let i = 0; i < f.length; i++) { s += f[i]; c[i] = s; }
  return c;
}
export function cdfToPdf(c) {
  const f = new Array(c.length);
  let prev = 0;
  for (let i = 0; i < c.length; i++) { f[i] = c[i] - prev; prev = c[i]; }
  return f;
}
export function impliedL(density) { return (density.length - 1) >> 1; }

function integerShift(cdf, k) {
  const m = cdf.length;
  k = Math.max(-(m - 1), Math.min(m - 1, k));
  if (k < 0) {
    const a = -k;
    const out = cdf.slice(a);
    const last = cdf[m - 1];
    for (let i = 0; i < a; i++) out.push(last);
    return out;
  }
  if (k === 0) return cdf.slice();
  const out = new Array(k).fill(0);
  for (let i = 0; i < m - k; i++) out.push(cdf[i]);
  return out;
}
function lowHigh(offset, L) {
  if (offset > -L + 2 && offset < L - 2) {
    const lo = Math.floor(offset), up = Math.ceil(offset);
    const r = offset - lo;
    return [[lo, 1 - r], [up, r]];
  }
  if (offset >= L - 2) return [[L - 2, 1], [L - 1, 0]];
  return [[-L + 1, 0], [-L + 2, 1]];
}
function shiftedCdf(cdf, offset, L) {
  const [[a, ac], [b, bc]] = lowHigh(offset, L);
  const sa = integerShift(cdf, a), sb = integerShift(cdf, b);
  return sa.map((v, i) => ac * v + bc * sb[i]);
}
function winnerOfManyCdfs(cdfs) {
  const m = cdfs[0].length;
  let cdfMin = cdfs[0].slice();
  let mult = new Array(m).fill(1);
  for (let k = 1; k < cdfs.length; k++) {
    const cb = cdfs[k];
    const fa = cdfToPdf(cdfMin), fb = cdfToPdf(cb);
    const newCdf = new Array(m), newMult = new Array(m);
    for (let t = 0; t < m; t++) {
      const win = fa[t] * (1 - cb[t]);
      const draw = fa[t] * fb[t];
      const lose = fb[t] * (1 - cdfMin[t]);
      newMult[t] = (win * mult[t] + draw * (mult[t] + 1) + lose + 1e-18)
        / (win + draw + lose + 1e-18);
      newCdf[t] = 1 - (1 - cdfMin[t]) * (1 - cb[t]);
    }
    cdfMin = newCdf; mult = newMult;
  }
  return { cdf: cdfMin, mult };
}
function expectedPayoffSum(cdf, cdfAll, multAll) {
  const m = cdf.length;
  const f1 = cdfToPdf(cdf);
  const cdfRest = new Array(m);
  for (let t = 0; t < m; t++) {
    cdfRest[t] = 1 - (1 - cdfAll[t] + 1e-18) / (1 - cdf[t] + 1e-6);
  }
  const fRest = cdfToPdf(cdfRest);
  let kmax = 0, fmax = -Infinity;
  for (let t = 0; t < m; t++) if (f1[t] > fmax) { fmax = f1[t]; kmax = t; }
  const multRest = new Array(m);
  for (let t = 0; t < m; t++) {
    const mm = multAll[t];
    const s1 = 1 - cdf[t];
    const srest = (1 - cdfAll[t] + 1e-18) / (s1 + 1e-6);
    if (t < kmax) {
      const numer = mm * f1[t] * srest + mm * (f1[t] + s1) * fRest[t]
        - f1[t] * (srest + fRest[t]);
      const denom = fRest[t] * (f1[t] + s1);
      multRest[t] = (1e-18 + numer) / (1e-18 + denom);
    } else {
      const t1 = (s1 + 1e-18) / (f1[t] + 1e-6);
      const trest = (srest + 1e-18) / (fRest[t] + 1e-6);
      multRest[t] = mm * trest / (1 + t1) + mm - (1 + trest) / (1 + t1);
    }
  }
  let run = -Infinity, total = 0, prev = 0;
  for (let t = 0; t < m; t++) {
    run = Math.max(run, cdfRest[t]);
    const fr = run - prev;
    prev = run;
    total += f1[t] * (1 - run) + f1[t] * fr / (1 + multRest[t]);
  }
  return total;
}
function implicitPrices(baseCdf, cdfAll, multAll, offsets, L) {
  return offsets.map(k => {
    if (k === Math.trunc(k))
      return expectedPayoffSum(integerShift(baseCdf, k), cdfAll, multAll);
    const [[a, ac], [b, bc]] = lowHigh(k, L);
    return ac * expectedPayoffSum(integerShift(baseCdf, a), cdfAll, multAll)
      + bc * expectedPayoffSum(integerShift(baseCdf, b), cdfAll, multAll);
  });
}

export function statePricesFromOffsets(density, offsets) {
  const L = impliedL(density);
  const baseCdf = pdfToCdf(density);
  const cdfs = offsets.map(o => shiftedCdf(baseCdf, o, L));
  const { cdf, mult } = winnerOfManyCdfs(cdfs);
  return implicitPrices(baseCdf, cdf, mult, offsets, L);
}

export function solveForImpliedOffsets(prices, density, opts = {}) {
  const L = impliedL(density);
  let { offsetSamples = null, guess = null, nIter = 3 } = opts;
  if (!offsetSamples) {
    offsetSamples = [];
    for (let k = Math.trunc(L / 2) - 1; k >= -Math.trunc(L / 2); k--) offsetSamples.push(k);
  }
  if (!guess) {
    guess = [];
    for (let k = 0; k < Math.trunc(L / 3); k++) guess.push(k);
  }
  const baseCdf = pdfToCdf(density);
  let cdfs = guess.map(o => shiftedCdf(baseCdf, o, L));
  let implied = prices.slice();
  for (let it = 0; it < nIter; it++) {
    const { cdf, mult } = winnerOfManyCdfs(cdfs);
    const table = implicitPrices(baseCdf, cdf, mult, offsetSamples, L);
    implied = prices.map(p => interpClamped(p, table, offsetSamples));
    cdfs = implied.map(o => shiftedCdf(baseCdf, o, L));
  }
  return implied;
}

export function skewNormalDensity(L, unit, { loc = 0, scale = 1.0, a = 2.0 } = {}) {
  const n = 2 * L + 1;
  const density = new Array(n);
  for (let i = 0; i < n; i++) {
    const x = unit * (i - L);
    const t = (x - loc) / scale;
    density[i] = 2 / scale * npdf(t) * ndtr(a * t);
  }
  let s = density.reduce((x, y) => x + y, 0);
  let d = density.map(v => v / s);
  // center, then apply the reference's density-vector fractional shift
  let m = 0;
  for (let i = 0; i < n; i++) m += d[i] * (i - L);
  d = cdfToPdf(shiftedCdf(pdfToCdf(d), -m, L));
  return shiftedCdf(d, loc / unit, L);       // reference quirk: cdf-machinery on the density
}

export function pricesFromDividends(dividends, nanValue = 2000) {
  const p = dividends.map(x => 1 / (Number.isFinite(x) ? x : nanValue));
  const s = p.reduce((a, b) => a + b, 0);
  return p.map(v => v / s);
}

export function dividendImpliedAbility(dividends, density, { nanValue = 2000, unit = 1.0 } = {}) {
  const p = pricesFromDividends(dividends, nanValue);
  const guess = new Array(p.length).fill(0);
  return solveForImpliedOffsets(p, density, { guess }).map(v => v * unit);
}
