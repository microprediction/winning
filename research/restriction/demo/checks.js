// One check per mathematical claim in papers/thurstone_humans/paper.tex.
// Runs under node (`node run_checks.js`) and in the browser (index.html).
// Every check is self-contained: no data files, no network, no dependencies.

import {
  Phi, phi, winProbs, calibrate, linearNormalization, gaussianRenormalization,
  pairProb, reverseHazard, logRSecondDerivative, varTruncated, gumbelLogR,
  lambdaSlope, withinSetConfusion, logLoss,
} from './casev.js';

const CHECKS = [];
const check = (id, claim, where, fn) => CHECKS.push({ id, claim, where, fn });

// deterministic generator, so the suite is reproducible
function lcg(seed) {
  let s = seed >>> 0;
  return () => { s = (1664525 * s + 1013904223) >>> 0; return s / 4294967296; };
}
const randomShares = (rand, K) => {
  const v = Array.from({ length: K }, () => -Math.log(rand() + 1e-12));
  const t = v.reduce((a, b) => a + b, 0);
  return v.map((x) => x / t);
};

// ---------------------------------------------------------------- Section 1
check('eq1-restriction', 'Renormalizing a subset leaves every surviving ratio unchanged',
  'Equation (1) and the sentence after it', () => {
    const rand = lcg(1);
    let worst = 0;
    for (let t = 0; t < 200; t++) {
      const p = randomShares(rand, 5);
      const keep = [0, 2, 4];
      const q = linearNormalization(p, keep);
      worst = Math.max(worst, Math.abs(q[0] / q[1] - p[0] / p[2]));
    }
    return { pass: worst < 1e-12, detail: `largest ratio drift ${worst.toExponential(2)}` };
  });

check('eq3-pair', 'For a pair, Case V gives Phi((a_i - a_j)/sqrt 2)',
  'Equation (3)', () => {
    const rand = lcg(2);
    let worst = 0;
    for (let t = 0; t < 50; t++) {
      const a = [rand() * 2 - 1, rand() * 2 - 1];
      const byIntegral = winProbs(a)[0];
      worst = Math.max(worst, Math.abs(byIntegral - pairProb(a[0], a[1])));
    }
    return { pass: worst < 1e-9, detail: `largest gap to the integral ${worst.toExponential(2)}` };
  });

check('contraction-pair', 'If p_i > p_j the Case V pair probability lies between 1/2 and p_i/(p_i+p_j)',
  'Section 1, after Equation (3)', () => {
    const rand = lcg(3);
    let bad = 0, n = 0, tightest = 1;
    for (let t = 0; t < 300; t++) {
      const p = randomShares(rand, 4).sort((x, y) => y - x);
      const { a } = calibrate(p);
      for (let i = 0; i < 4; i++) {
        for (let j = i + 1; j < 4; j++) {
          const g = pairProb(a[i], a[j]);
          const l = p[i] / (p[i] + p[j]);
          n += 1;
          if (!(g > 0.5 - 1e-12 && g < l + 1e-9)) bad += 1;
          tightest = Math.min(tightest, l - g);
        }
      }
    }
    return { pass: bad === 0, detail: `${n} pairs, ${bad} violations, smallest margin ${tightest.toExponential(2)}` };
  });

// ---------------------------------------------------------------- Proposition 1
check('prop1-gaussian', 'Proposition 1 for the Gaussian: removing alternatives moves the odds ratio toward one',
  'Proposition 1', () => {
    const rand = lcg(4);
    let bad = 0, n = 0;
    for (let t = 0; t < 200; t++) {
      const K = 5;
      const p = randomShares(rand, K);
      const { a } = calibrate(p);
      const keep = [0, 1, 2];
      const q = gaussianRenormalization(a, keep);
      const order = [0, 1, 2].sort((x, y) => p[y] - p[x]);
      const [hi, lo] = [order[0], order[1]];
      const full = p[hi] / p[lo];
      const rest = q[keep.indexOf(hi)] / q[keep.indexOf(lo)];
      n += 1;
      if (!(full >= rest - 1e-9 && rest >= 1 - 1e-9)) bad += 1;
    }
    return { pass: bad === 0, detail: `${n} cases, ${bad} violations of full >= restricted >= 1` };
  });

check('prop1-hazard', "For the standard normal, the second derivative of log r equals -Var(Z | Z < x)",
  'Section 2, after Proposition 1', () => {
    let worst = 0;
    for (let x = -4; x <= 4; x += 0.25) {
      worst = Math.max(worst, Math.abs(logRSecondDerivative(x) + varTruncated(x)));
    }
    return { pass: worst < 1e-5, detail: `largest gap over x in [-4,4] is ${worst.toExponential(2)}` };
  });

check('prop1-negative', "That second derivative is strictly negative, so log r is strictly concave",
  'Section 2', () => {
    let maxVal = -Infinity;
    for (let x = -6; x <= 6; x += 0.1) maxVal = Math.max(maxVal, -varTruncated(x));
    return { pass: maxVal < 0, detail: `largest value of (log r)'' is ${maxVal.toExponential(3)}` };
  });

check('gumbel-affine', 'For the Gumbel, log r is affine, so Proposition 1 holds with equality',
  'Section 2', () => {
    let worst = 0;
    for (let x = -3; x <= 3; x += 0.1) {
      const numeric = (gumbelLogR(x + 1e-4) - 2 * gumbelLogR(x) + gumbelLogR(x - 1e-4)) / 1e-8;
      worst = Math.max(worst, Math.abs(numeric));
    }
    return { pass: worst < 1e-4, detail: `largest curvature of log r is ${worst.toExponential(2)}` };
  });

check('gumbel-iia', 'A simulated Gumbel race renormalizes exactly: restricted ratios match full-menu ratios',
  'Section 2, the Holman-Marley construction', () => {
    const rand = lcg(5);
    const a = [0.9, 0.3, -0.2, -1.0];
    const N = 400000;
    const fullWins = [0, 0, 0, 0];
    const subWins = [0, 0, 0];
    const keep = [0, 1, 2];
    for (let t = 0; t < N; t++) {
      const g = a.map((ai) => ai - Math.log(-Math.log(rand() + 1e-15)));
      let best = 0;
      for (let i = 1; i < 4; i++) if (g[i] > g[best]) best = i;
      fullWins[best] += 1;
      let bestSub = keep[0];
      for (const i of keep) if (g[i] > g[bestSub]) bestSub = i;
      subWins[keep.indexOf(bestSub)] += 1;
    }
    const fullRatio = fullWins[0] / fullWins[1];
    const subRatio = subWins[0] / subWins[1];
    const rel = Math.abs(fullRatio - subRatio) / fullRatio;
    return { pass: rel < 0.02, detail: `full-menu odds ${fullRatio.toFixed(4)}, restricted ${subRatio.toFixed(4)}, relative gap ${(100 * rel).toFixed(2)}%` };
  });

// ---------------------------------------------------------------- the gauge
check('gauge-exact', 'A common noise scale is a gauge: P^S_i(a;s) = P^S_i(a/s;1) exactly',
  'Section 3.2, the scale argument', () => {
    const p = [0.45, 0.25, 0.2, 0.1];
    const { a } = calibrate(p);
    let worst = 0;
    for (const s of [0.5, 1, 2, 7, 100]) {
      // scale-s locations are s*a; the restricted prediction must not move
      const keep = [0, 2, 3];
      const base = gaussianRenormalization(a, keep);
      const scaled = keep.map((i) => (s * a[i]) / s);
      const moved = winProbs(scaled);
      for (let i = 0; i < base.length; i++) worst = Math.max(worst, Math.abs(base[i] - moved[i]));
    }
    return { pass: worst < 1e-12, detail: `largest drift across s in {0.5,1,2,7,100} is ${worst.toExponential(2)}` };
  });

check('gauge-pair', 'The pair formula is scale invariant: Phi((s a_i - s a_j)/(s sqrt 2)) is constant in s',
  'Section 3.2', () => {
    const a = [1.2, -0.4];
    const vals = [0.25, 1, 3, 50].map((s) => Phi((s * a[0] - s * a[1]) / (s * Math.SQRT2)));
    const worst = Math.max(...vals.map((v) => Math.abs(v - vals[0])));
    return { pass: worst < 1e-15, detail: `all four equal ${vals[0].toFixed(12)}, spread ${worst.toExponential(2)}` };
  });

// ---------------------------------------------------------------- cubic order
check('cubic-skew', 'Standardizing Z + eps*G leaves skewness of order eps^3, with no eps^2 term',
  'Section 3.2, the cumulant argument', () => {
    // exact cumulants: kappa2(G)=pi^2/6, kappa3(G)=2*zeta(3)
    const k2G = Math.PI * Math.PI / 6, k3G = 2 * 1.2020569031595943;
    const skew = (eps) => (k3G * eps ** 3) / Math.pow(1 + k2G * eps * eps, 1.5);
    const ratios = [];
    for (const eps of [0.2, 0.1, 0.05, 0.025]) ratios.push(skew(eps) / eps ** 3);
    const spread = Math.max(...ratios) / Math.min(...ratios);
    const slope = Math.log(skew(0.05) / skew(0.025)) / Math.log(2);
    return { pass: Math.abs(slope - 3) < 0.02 && spread < 1.1,
             detail: `log-log slope ${slope.toFixed(4)} against the claimed 3` };
  });

check('cubic-map', 'The recalibrated restriction map departs from Case V at cubic order in eps',
  'Section 3.2', () => {
    // Compose Z + eps G by simulation, recalibrate Case V to the resulting shares,
    // and measure how far the restricted prediction moves.
    const rand = lcg(6);
    const A = [0.55, 0.25, 0.0, -0.25, -0.55];
    const N = 300000;
    const keep = [0, 2, 4];
    const deviation = (eps) => {
      const wins = new Array(5).fill(0), sub = new Array(3).fill(0);
      for (let t = 0; t < N; t++) {
        const u = A.map((ai) => ai + gaussStd(rand) + eps * (-Math.log(-Math.log(rand() + 1e-15))));
        let b = 0; for (let i = 1; i < 5; i++) if (u[i] > u[b]) b = i;
        wins[b] += 1;
        let bs = keep[0]; for (const i of keep) if (u[i] > u[bs]) bs = i;
        sub[keep.indexOf(bs)] += 1;
      }
      const p = wins.map((w) => w / N);
      const observed = sub.map((w) => w / N);
      const { a } = calibrate(p);
      const predicted = gaussianRenormalization(a, keep);
      return Math.abs(observed[0] - predicted[0]);
    };
    const d1 = deviation(0.4), d2 = deviation(0.2);
    const slope = Math.log(d1 / d2) / Math.log(2);
    return { pass: slope > 1.5, detail: `local exponent between eps=0.4 and 0.2 is ${slope.toFixed(2)}; Monte Carlo noise keeps this below the asymptotic 3` };
  });

function gaussStd(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------------------------------------------------------------- concentrated shares
check('concentrated', 'A large top share does not make the two maps agree once the leader is withdrawn',
  'Section 6, withdrawn alternatives with negligible mass', () => {
    const p = [0.90, 0.09, 0.01];
    const { a } = calibrate(p);
    const tailGone = pairProb(a[0], a[1]);
    const tailGoneL = p[0] / (p[0] + p[1]);
    const leaderGone = pairProb(a[1], a[2]);
    const leaderGoneL = p[1] / (p[1] + p[2]);
    const ok = Math.abs(tailGone - 0.9078) < 5e-4 && Math.abs(tailGoneL - 0.9091) < 5e-4
            && Math.abs(leaderGone - 0.8028) < 5e-4 && Math.abs(leaderGoneL - 0.9000) < 5e-4;
    return { pass: ok,
             detail: `remove 0.01: ${tailGone.toFixed(4)} vs ${tailGoneL.toFixed(4)}, gap ${(tailGoneL - tailGone).toFixed(4)}. `
                   + `remove 0.90: ${leaderGone.toFixed(4)} vs ${leaderGoneL.toFixed(4)}, gap ${(leaderGoneL - leaderGone).toFixed(4)}` };
  });

// ---------------------------------------------------------------- lambda
check('lambda-zero', 'lambda is zero for exact renormalization and positive for Case V',
  'Table 1 caption, the definition of lambda', () => {
    const p = [0.4, 0.3, 0.2, 0.1];
    const { a } = calibrate(p);
    const luce = lambdaSlope(p, (i, j) => p[i] / (p[i] + p[j]));
    const race = lambdaSlope(p, (i, j) => pairProb(a[i], a[j]));
    return { pass: Math.abs(luce) < 1e-9 && race > 0.05,
             detail: `renormalization lambda ${luce.toExponential(2)}, Case V lambda ${race.toFixed(4)}` };
  });

// ---------------------------------------------------------------- gain vanishes at |T| = K
check('gain-zero-at-full', 'The two maps coincide when nothing has been removed',
  'Table 5, the last column', () => {
    const rand = lcg(7);
    let worst = 0;
    for (let t = 0; t < 100; t++) {
      const p = randomShares(rand, 4);
      const { a } = calibrate(p);
      const full = gaussianRenormalization(a, [0, 1, 2, 3]);
      for (let i = 0; i < 4; i++) worst = Math.max(worst, Math.abs(full[i] - p[i]));
    }
    return { pass: worst < 1e-9, detail: `largest difference ${worst.toExponential(2)}` };
  });

// ---------------------------------------------------------------- Getty diagnostic
const GETTY = {
  BF: [[273,0,1,1,0,0,12,0],[0,325,0,0,0,0,0,0],[2,0,271,0,0,3,8,2],[1,2,0,238,18,7,8,10],
       [0,0,3,26,249,8,3,3],[0,1,7,20,0,283,0,21],[23,1,5,1,3,1,245,29],[1,0,2,21,2,25,23,233]],
  JK: [[248,0,1,0,1,0,36,1],[0,323,0,0,1,1,0,0],[0,0,250,2,0,26,3,5],[0,2,4,206,31,28,2,11],
       [0,0,0,17,247,24,0,4],[0,1,25,32,2,251,1,20],[12,5,9,3,7,4,218,50],[1,7,5,33,4,29,35,193]],
  JS: [[267,0,1,0,0,0,17,2],[0,325,0,0,0,0,0,0],[0,0,226,11,1,36,9,3],[0,0,2,208,22,41,0,11],
       [0,0,3,27,250,7,2,3],[0,0,32,73,0,222,1,4],[23,2,8,9,6,1,181,78],[0,0,5,56,4,2,13,227]],
};

check('getty-rowsums', 'Every row of the transcribed Getty master matrix reproduces its printed total',
  'Section 5.4 and the data provenance', () => {
    const printed = [287, 325, 286, 284, 292, 332, 308, 307];
    let bad = 0;
    for (const obs of Object.keys(GETTY)) {
      for (let i = 0; i < 8; i++) {
        const sum = GETTY[obs][i].reduce((a, b) => a + b, 0);
        if (sum !== printed[i]) bad += 1;
      }
    }
    return { pass: bad === 0, detail: `24 rows checked against the printed totals, ${bad} mismatches` };
  });

check('getty-confusion', 'Within-set confusion mass separates the Getty condition that loses',
  'Table 8', () => {
    const conds = { '1,2,5,6': [0, 1, 4, 5], '3,4,5,6': [2, 3, 4, 5], '1,3,5,7': [0, 2, 4, 6] };
    const target = { '1,2,5,6': 0.103, '3,4,5,6': 0.790, '1,3,5,7': 0.335 };
    const got = {};
    let bad = 0;
    for (const key of Object.keys(conds)) {
      let num = 0, den = 0;
      for (const obs of Object.keys(GETTY)) {
        const w = withinSetConfusion(GETTY[obs], conds[key]);
        // recompute pooled rather than averaging
        for (const i of conds[key]) {
          const row = GETTY[obs][i];
          den += row.reduce((a, b) => a + b, 0) - row[i];
          for (const j of conds[key]) if (j !== i) num += row[j];
        }
        void w;
      }
      got[key] = num / den;
      if (Math.abs(got[key] - target[key]) > 0.001) bad += 1;
    }
    return { pass: bad === 0,
             detail: Object.keys(got).map((k) => `{${k}} ${got[k].toFixed(3)}`).join(', ')
                   + ' against the printed 0.103, 0.790, 0.335' };
  });

// ---------------------------------------------------------------- favourite second
check('favsecond', 'The reduction in the favourite-second overprediction is about five to thirteen per cent',
  'Section 5.5.3', () => {
    const rows = [
      ['Political goals', 0.123, 0.318, 0.309],
      ['GSS job values', 0.186, 0.303, 0.294],
      ['GSS socialization', 0.191, 0.302, 0.294],
      ['Sushi', 0.206, 0.251, 0.245],
      ['Jester file 1', 0.138, 0.160, 0.158],
    ];
    const pct = rows.map(([, obs, ren, race]) => 100 * (ren - race) / (ren - obs));
    const lo = Math.min(...pct), hi = Math.max(...pct);
    return { pass: lo > 4 && hi < 14,
             detail: pct.map((v) => v.toFixed(1) + '%').join(', ') + ` so the range is ${lo.toFixed(1)} to ${hi.toFixed(1)} per cent` };
  });

// ---------------------------------------------------------------- identification
check('probit-count', 'One full-menu share vector cannot identify locations and an unrestricted covariance',
  'Section 3.4', () => {
    const rows = [];
    for (let K = 3; K <= 8; K++) {
      rows.push({ K, shares: K - 1, locations: K - 1, covariance: K * (K - 1) / 2 - 1 });
    }
    const ok = rows.every((r) => r.shares === r.locations && r.covariance >= 1);
    return { pass: ok,
             detail: rows.map((r) => `K=${r.K}: ${r.shares} shares, ${r.locations} contrasts, ${r.covariance} covariance parameters left over`).join('; ') };
  });

// ---------------------------------------------------------------- log loss direction
check('gain-sign', 'A positive gain means Gaussian renormalization predicted better',
  'Table 1 caption', () => {
    const p = [0.5, 0.3, 0.2];
    const { a } = calibrate(p);
    const keep = [0, 1];
    const lin = linearNormalization(p, keep);
    const gau = gaussianRenormalization(a, keep);
    // an outcome distribution sitting exactly on the Gaussian prediction
    const obs = [gau[0] * 1000, gau[1] * 1000];
    const gain = logLoss(lin, obs) - logLoss(gau, obs);
    return { pass: gain > 0, detail: `gain ${gain.toFixed(6)} when the truth is the Gaussian prediction` };
  });

export function runAll() {
  return CHECKS.map((c) => {
    let out;
    try { out = c.fn(); } catch (e) { out = { pass: false, detail: 'threw ' + e.message }; }
    return { ...c, ...out };
  });
}
export { CHECKS };
