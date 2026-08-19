// Empirical demos: run the two maps against what people actually did, in the browser.
// Three collections are embedded from the committed data by demo/collections.json.
// Each returns per-cell losses so the page can show the comparison rather than assert it.

import { calibrate, winProbs, pairProb, logLoss } from './casev.js';

const ALPHA = 0.5;
const FLOOR = 1e-9;

const smooth = (counts) => {
  const tot = counts.reduce((a, b) => a + b, 0);
  return counts.map((c) => (c + ALPHA) / (tot + ALPHA * counts.length));
};
const renorm = (p, keep) => {
  const sub = keep.map((i) => p[i]);
  const t = sub.reduce((a, b) => a + b, 0);
  return sub.map((v) => Math.max(v / t, FLOOR));
};

// ---------------------------------------------------------------- Getty 1979
// Eight sounds, all eight responses allowed, then only four. Stimuli never change.
export function getty(data) {
  const rows = [];
  for (const obs of Object.keys(data.master)) {
    for (const cond of Object.keys(data.restricted[obs] || {})) {
      for (const stim of Object.keys(data.restricted[obs][cond])) {
        const cell = data.restricted[obs][cond][stim];
        const counts = data.master[obs][stim];
        if (!counts) continue;
        const n = cell.obs.reduce((a, b) => a + b, 0);
        if (n < 5) continue;
        const p = smooth(counts);
        const { a } = calibrate(p);
        const keep = cell.signals.map((x) => x - 1);
        const lin = renorm(p, keep);
        const gau = winProbs(keep.map((i) => a[i]));
        rows.push({
          label: `observer ${obs}, condition ${cond}, stimulus ${stim}`,
          cond, signal: cell.signals.includes(Number(stim)),
          n, linear: logLoss(lin, cell.obs), gaussian: logLoss(gau, cell.obs),
        });
      }
    }
  }
  return rows;
}

// ---------------------------------------------------------------- tones
// Ten tones, then the middle eight or six. Matrices are participant-averaged, so
// each row is a distribution rather than counts, and the loss is weighted by it.
export function tones(data, spacing, small) {
  const big = data[`${spacing}_N10`];
  const obsM = data[`${spacing}_N${small}`];
  const off = (10 - small) / 2;
  const keep = Array.from({ length: small }, (_, k) => k + off);
  const rows = [];
  for (let i = 0; i < small; i++) {
    // an exact zero has to be floored, which sends a location far out and makes the
    // inversion ill conditioned; such rows are excluded, as in tones.py
    if (big[i + off].some((v) => v === 0)) continue;
    const p = big[i + off].map((v) => Math.max(v, 1e-6));
    const t = p.reduce((x, y) => x + y, 0);
    const pn = p.map((v) => v / t);
    const { a } = calibrate(pn);
    const lin = renorm(pn, keep);
    const gau = winProbs(keep.map((k) => a[k]));
    const o = obsM[i];
    rows.push({
      label: `${spacing} spacing, ten to ${small}, stimulus ${i + 1 + off}`,
      n: 1, linear: logLoss(lin, o), gaussian: logLoss(gau, o),
    });
  }
  return rows;
}

// ---------------------------------------------------------------- Yeon and Rahnev
// Four colours, menu revealed only after the display has gone.
export function yeonrahnev(data) {
  const rows = [];
  for (const key of Object.keys(data.pairs)) {
    const [dom, alt] = key.split('-').map(Number);
    const counts = data.rows[String(dom)];
    if (!counts) continue;
    const obs = data.pairs[key];
    const n = obs[0] + obs[1];
    if (n < 20) continue;
    const p = smooth(counts);
    const { a } = calibrate(p);
    const keep = [dom - 1, alt - 1];
    const lin = renorm(p, keep);
    const gau = [pairProb(a[keep[0]], a[keep[1]])];
    gau.push(1 - gau[0]);
    rows.push({
      label: `dominant colour ${dom} against ${alt}`,
      n, linear: logLoss(lin, obs), gaussian: logLoss(gau, obs),
    });
  }
  return rows;
}

export function summarise(rows) {
  if (!rows.length) return null;
  const mean = (f) => rows.reduce((s, r) => s + f(r), 0) / rows.length;
  const lin = mean((r) => r.linear);
  const gau = mean((r) => r.gaussian);
  return {
    cells: rows.length, linear: lin, gaussian: gau, gain: lin - gau,
    won: rows.filter((r) => r.linear > r.gaussian).length,
  };
}
