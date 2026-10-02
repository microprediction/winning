/* Parity test: JavaScript port vs committed vectors from winning.factor. */
import { readFileSync } from "fs";
import { logndtr, winProbabilitiesFactor, abilitiesFromProbabilitiesFactor, skewNormalBase }
  from "./factor_race.mjs";

const T = JSON.parse(readFileSync(new URL("./test_vectors.json", import.meta.url)));
const { mu, V, D } = T.problem;
const { F, W } = T.hermite;
let failures = 0;
const check = (name, got, want, tol) => {
  let worst = 0;
  const g = got.flat(2), w = want.flat(2);
  for (let i = 0; i < g.length; i++) worst = Math.max(worst, Math.abs(g[i] - w[i]));
  const ok = worst < tol;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"} ${name}: max|diff| = ${worst.toExponential(2)} (tol ${tol})`);
};

// special function first
check("logndtr", T.logndtr.z.map(logndtr), T.logndtr.v, 1e-12);

const fwd = winProbabilitiesFactor(mu, V, D, F, W, { pairwise: true, deletions: true });
check("forward shares", fwd.p, T.expected.p, 1e-10);
check("pairwise tie densities", fwd.w, T.expected.w, 1e-10);
check("deletion ensemble", fwd.deletions, T.expected.deletions, 1e-10);

const muHat = abilitiesFromProbabilitiesFactor(T.expected.p, V, D, F, W);
check("calibrated abilities vs python", muHat, T.expected.mu_hat, 2e-6);
check("calibrated abilities vs truth", muHat, mu, 5e-6);



// gumbel base: forward, calibration, and the exact softmax special case
{
  const G = JSON.parse(readFileSync(new URL("./test_vectors_gumbel.json", import.meta.url)));
  const gp = G.problem, gh = G.hermite;
  const gf = winProbabilitiesFactor(gp.mu, gp.V, gp.D, gh.F, gh.W, { base: "gumbel" });
  check("gumbel forward shares", gf.p, G.expected.p, 1e-10);
  const gHat = abilitiesFromProbabilitiesFactor(G.expected.p, gp.V, gp.D, gh.F, gh.W, { base: "gumbel" });
  check("gumbel calibrated abilities vs python", gHat, G.expected.mu_hat, 2e-6);
  check("gumbel calibrated abilities vs truth", gHat, gp.mu, 5e-6);
  const zeroV = gp.mu.map(() => [0]);
  const ones = gp.mu.map(() => 1);
  const ind = winProbabilitiesFactor(gp.mu, zeroV, ones, [[0]], [1], { base: "gumbel" });
  check("independent gumbel = softmax", ind.p, G.expected.p_independent, 1e-9);
  const c = Math.PI / Math.sqrt(6);
  const ex = gp.mu.map((m) => Math.exp(-c * m));
  const tot = ex.reduce((a, b) => a + b, 0);
  check("independent gumbel vs closed-form Luce", ind.p, ex.map((v) => v / tot), 1e-9);
}



// tabulated bases (skew-normal alpha=3, Student-t nu=4) vs scipy exact
{
  const B = JSON.parse(readFileSync(new URL("./test_vectors_bases.json", import.meta.url)));
  const bp = B.problem, bh = B.hermite;
  for (const name of ["skew", "t4"]) {
    const fw = winProbabilitiesFactor(bp.mu, bp.V, bp.D, bh.F, bh.W, { base: name });
    check(`${name} forward shares vs scipy`, fw.p, B.expected[name], 2e-6);
    const hat = abilitiesFromProbabilitiesFactor(fw.p, bp.V, bp.D, bh.F, bh.W, { base: name });
    check(`${name} calibration roundtrip`, hat, bp.mu, 1e-3);
  }
}

// parameterized skew: alpha = 0 must reproduce the normal base exactly,
// and every alpha must be standardized (mass 1, mean 0, variance 1)
{
  const mu = [-0.6, -0.1, 0.2, 0.5], V = mu.map(() => [0.5]), D = mu.map(() => 0.75);
  const F = [[-1.7], [0], [1.7]], W = [0.25, 0.5, 0.25];
  const pn = winProbabilitiesFactor(mu, V, D, F, W, { base: "normal" });
  const p0 = winProbabilitiesFactor(mu, V, D, F, W, { base: skewNormalBase(0) });
  check("skewNormalBase(0) = normal", p0.p, pn.p, 1e-6);
  for (const alpha of [-2, 0.5, 3]) {
    const b = skewNormalBase(alpha);
    let m0 = 0, m1 = 0, m2 = 0;
    const dz = 0.01;
    for (let z = -20; z <= 20; z += dz) {
      const f = b.pdf(z);
      m0 += f * dz; m1 += z * f * dz; m2 += z * z * f * dz;
    }
    check(`skew alpha=${alpha} standardized`, [m0, m1, m2], [1, 0, 1], 1e-4);
  }
}

/* --- W and c*W are the same law (#281) ------------------------------
   The forward normalises its accumulated shares, so `p` was already
   invariant. It returns the own-slopes UNNORMALISED, and the inverse
   divides those by the normalised probabilities, so the Newton
   derivative carried the factor c and ONLY the inverse moved: a
   self-generated target repriced 0.16 away after fifty iterations at
   c = 0.1. Checked here across twelve orders of magnitude, because a
   rescaling is a spelling of the same law and a custom quadrature or
   importance rule need not arrive normalised. */
{
  const Vi = [[0.8], [-0.3], [0.1], [0.5]];
  const Di = [0.7, 1.1, 0.8, 1.2];
  const Fi = [[-1], [1]];
  const Wi = [0.5, 0.5];
  const mu0 = [-0.6, -0.1, 0.2, 0.5];
  const target = winProbabilitiesFactor(mu0, Vi, Di, Fi, Wi, { points: 1001 }).p;

  let worstFwd = 0, worstInv = 0;
  for (const c of [1e-6, 1e-3, 0.1, 1, 10, 1e3, 1e6]) {
    const Wc = Wi.map((w) => c * w);
    const fwd = winProbabilitiesFactor(mu0, Vi, Di, Fi, Wc, { points: 1001 }).p;
    worstFwd = Math.max(worstFwd, ...fwd.map((x, i) => Math.abs(x - target[i])));
    const mu2 = abilitiesFromProbabilitiesFactor(target, Vi, Di, Fi, Wc,
                                                 { points: 501, nIter: 50 });
    const rep = winProbabilitiesFactor(mu2, Vi, Di, Fi, Wi, { points: 1001 }).p;
    worstInv = Math.max(worstInv, ...rep.map((x, i) => Math.abs(x - target[i])));
  }
  check("forward is invariant to W -> cW", [worstFwd], [0], 1e-12);
  // the inverse's own convergence floor is ~2.5e-7; what must not
  // happen is that rescaling MOVES it, and at c = 0.1 it was 0.16
  check("inverse reprices to target at every W scale", [worstInv], [0], 1e-6);

  // and the contract refuses what it cannot normalise, instead of
  // returning a normalised, plausible, wrong answer
  const refuses = (W, why) => {
    let threw = false;
    try { winProbabilitiesFactor(mu0, Vi, Di, Fi, W, { points: 257 }); }
    catch (e) { threw = true; }
    check(`refuses ${why}`, [threw ? 1 : 0], [1], 0.5);
  };
  refuses([0.5, -0.5], "a negative weight");
  refuses([0, 0], "an all-zero W");
  refuses([0.5, NaN], "a non-finite weight");
  refuses([0.5, 0.3, 0.2], "a W of the wrong length");
}

// --- the boundary and the inverse (#198 #210 #254 #290 #330 #331 #349
// #358 #371 #391 #428 #443)
{
  const refuses = (name, fn) => {
    let ok = false;
    try { fn(); } catch (e) { ok = true; }
    check(name, [ok ? 0 : 1], [0], 0.5);
  };
  const mx = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  const Z3 = [[], [], []];
  // #254: D is a scalar or exactly one positive variance per runner
  refuses("refuses an extra D entry (#254)", () =>
    winProbabilitiesFactor([-0.4, 0, 0.4], [[0], [0], [0]], [1, 1, 1, 1e4], [[0]], [1]));
  for (const D of [[1, 1], [1, -1, 1], [1, 0, 1], [1, NaN, 1]])
    refuses(`refuses D=${JSON.stringify(D)} (#254)`, () =>
      winProbabilitiesFactor([-0.4, 0, 0.4], [[0], [0], [0]], D, [[0]], [1]));
  check("a scalar D broadcasts (#254)",
        winProbabilitiesFactor([-0.4, 0, 0.4], [[0], [0], [0]], 1, [[0]], [1]).p,
        winProbabilitiesFactor([-0.4, 0, 0.4], [[0], [0], [0]], [1, 1, 1], [[0]], [1]).p, 1e-15);
  // #290: F carries exactly the loadings' rank
  const V2 = [[1.2, -0.7], [-0.4, 1.1], [0.6, 0.9], [-1.0, -0.5]];
  refuses("refuses a short F (#290)", () =>
    winProbabilitiesFactor([-0.6, -0.2, 0.15, 0.7], V2, [0.5, 0.8, 0.6, 0.9], [[-1], [-1], [1], [1]], [0.25, 0.25, 0.25, 0.25]));
  refuses("refuses a ragged F (#290)", () =>
    winProbabilitiesFactor([-0.6, -0.2, 0.15, 0.7], V2, [0.5, 0.8, 0.6, 0.9], [[-1], [-1, 100], [1, -100], [1, 100]], [0.25, 0.25, 0.25, 0.25]));
  refuses("the inverse refuses a short F (#290)", () =>
    abilitiesFromProbabilitiesFactor([0.4, 0.3, 0.2, 0.1], V2, [0.5, 0.8, 0.6, 0.9], [[-1], [-1], [1], [1]], [0.25, 0.25, 0.25, 0.25]));
  // #428: the budget is a finite positive integer
  for (const nIter of [0, -1, 0.5, NaN, Infinity, "50", true])
    refuses(`refuses nIter=${JSON.stringify(nIter)} (#428)`, () =>
      abilitiesFromProbabilitiesFactor([0.6, 0.3999, 0.0001], Z3, [1, 1, 1], [[]], [1], { nIter }));
  // #198: a pair is one equation, solved against the forward
  for (const t of [[0.8, 0.2], [0.999, 0.001], [0.5, 0.5]]) {
    const o = abilitiesFromProbabilitiesFactor(t, [[0], [0]], [1, 1], [[0]], [1], { returnInfo: true, tol: 1e-10 });
    check(`pair round trip at ${t[0]} (#198)`, winProbabilitiesFactor(o.mu, [[0], [0]], [1, 1], [[0]], [1]).p, t, 1e-8);
  }
  {
    const Vp = [[0], [1]], Fp = [[-1], [1]], Wp = [0.5, 0.5], Dp = [0.05, 0.05];
    const t = winProbabilitiesFactor([-0.4, 0.4], Vp, Dp, Fp, Wp).p;
    const o = abilitiesFromProbabilitiesFactor(t, Vp, Dp, Fp, Wp, { returnInfo: true, tol: 1e-10 });
    check("pair round trip under a two-point factor law (#198)", winProbabilitiesFactor(o.mu, Vp, Dp, Fp, Wp).p, t, 1e-8);
  }
  // #358: two contenders and a longshot converge, and say so
  {
    const t = [0.6, 0.3999, 0.0001];
    const o = abilitiesFromProbabilitiesFactor(t, Z3, [1, 1, 1], [[]], [1], { returnInfo: true });
    check("near-pair target converges (#358)", [o.converged ? 0 : 1], [0], 0.5);
    check("near-pair target reprices (#358)", winProbabilitiesFactor(o.mu, Z3, [1, 1, 1], [[]], [1], { points: 4001 }).p, t, 1e-7);
  }
  // #443: V -> aV, F -> F/a is the same race, and the same inverse
  {
    const mu0 = [-1.2, -0.1, 0.4, 0.9], V0 = [[-1], [-0.2], [0.4], [0.8]], Dq = [0.3, 0.8, 0.5, 1.1];
    const F0 = [[-1], [1]], Wq = [0.5, 0.5];
    const t = winProbabilitiesFactor(mu0, V0, Dq, F0, Wq, { points: 1001 }).p;
    for (const a of [1e-4, 100, 1e4]) {
      const Va = V0.map(([x]) => [a * x]), Fa = F0.map(([x]) => [x / a]);
      const m = abilitiesFromProbabilitiesFactor(t, Va, Dq, Fa, Wq, { points: 501, nIter: 50, tol: 1e-8 });
      check(`reciprocal V/F scale ${a} reprices (#443)`, winProbabilitiesFactor(m, Va, Dq, Fa, Wq, { points: 4001 }).p, t, 1e-7);
    }
  }
  // #371: the derivative of the NORMALISED share, at a coarse lattice
  {
    const mu = [0.771, 0.170, -0.474, -0.467], Dq = [1.426, 0.379, 0.415, 0.363];
    const Vq = [[-1.946], [0.471], [-2.087], [3.562]];
    const Fq = [[-2.02018287045609], [-0.958572464613819], [0], [0.958572464613819], [2.02018287045609]];
    const Wq = [0.0112574113277207, 0.222075922005613, 0.533333333333333, 0.222075922005613, 0.0112574113277207];
    const t = winProbabilitiesFactor(mu, Vq, Dq, Fq, Wq, { points: 21 });
    const m = abilitiesFromProbabilitiesFactor(t.p, Vq, Dq, Fq, Wq, { points: 21, nIter: 50, tol: 1e-8 });
    check("coarse-lattice self round trip (#371)", winProbabilitiesFactor(m, Vq, Dq, Fq, Wq, { points: 21 }).p, t.p, 1e-7);
    const h = 1e-6, o = winProbabilitiesFactor(mu, Vq, Dq, Fq, Wq, { points: 21, ownLogSlope: true });
    const fd = mu.map((_, i) => {
      const a = mu.slice(), b = mu.slice(); a[i] += h; b[i] -= h;
      return (Math.log(winProbabilitiesFactor(a, Vq, Dq, Fq, Wq, { points: 21 }).p[i])
              - Math.log(winProbabilitiesFactor(b, Vq, Dq, Fq, Wq, { points: 21 }).p[i])) / (2 * h);
    });
    check("ownLogSlope is the derivative of log p (#371)", o.ownLogSlope, fd, 2e-3);
  }
  // #331: own slopes of a sharp skew-normal are non-positive, and invert
  {
    const mu0 = [-0.8, -0.2, 0.3, 0.9], Z4 = [[], [], [], []], D4 = [1, 1, 1, 1];
    const base = skewNormalBase(400);
    const t = winProbabilitiesFactor(mu0, Z4, D4, [[]], [1], { base, points: 1001 });
    check("skew(400) own slopes are non-positive (#331)", [Math.max(0, ...t.slope)], [0], 1e-12);
    const m = abilitiesFromProbabilitiesFactor(t.p, Z4, D4, [[]], [1], { base, points: 1001, tol: 1e-8 });
    check("skew(400) round trip (#331)", winProbabilitiesFactor(m, Z4, D4, [[]], [1], { base, points: 1001 }).p, t.p, 1e-7);
    refuses("skewNormalBase refuses a non-finite alpha (#331)", () => skewNormalBase(NaN));
  }
  // #349 / #330: an irrelevant distant runner leaves a tie density and a
  // removal row alone
  {
    const w = winProbabilitiesFactor([0, 0.1, 100], [[0], [0], [0]], [0.01, 0.01, 1], [[0]], [1],
                                     { pairwise: true, points: 501 }).w[0][1];
    const exact = Math.exp(-0.5 * 0.01 / 0.02) / Math.sqrt(2 * Math.PI * 0.02);
    check("a distant runner does not dilute a tie density (#349)", [w], [exact], 1e-9);
    const direct = winProbabilitiesFactor([0, 1], [[], []], [1, 1], [[]], [1], { points: 501 }).p;
    const del = winProbabilitiesFactor([-2000, 0, 1], Z3, [1, 1, 1], [[]], [1], { points: 501, deletions: true });
    check("removing a far favourite is the direct pair (#330)", del.deletions[0].slice(1), direct, 1e-9);
    const mu4 = [-225.6, 0, 0.1, 0.2], V4 = [[0], [0], [0], [0]], D4 = [0.01, 0.01, 0.01, 0.01];
    const d4 = winProbabilitiesFactor(mu4, V4, D4, [[0]], [1], { points: 501, deletions: true }).deletions[0].slice(1);
    check("an aliased removal row is resolved (#330)", d4,
          winProbabilitiesFactor(mu4.slice(1), V4.slice(1), D4.slice(1), [[0]], [1], { points: 501 }).p, 1e-9);
  }
  // #391: the home page's reachable 301-point t4 field
  {
    const h = [-4.1445471861258945, -2.802485861287542, -1.6365190424351082, -0.5390798113513752,
               0.5390798113513752, 1.6365190424351082, 2.802485861287542, 4.1445471861258945];
    const w = [1.126145383753679e-4, 9.635220120788263e-3, 0.117239907661759, 0.3730122576790774,
               0.3730122576790774, 0.117239907661759, 9.635220120788263e-3, 1.126145383753679e-4];
    const Fh = [], Wh = [];
    for (const a of h) for (const b of h) Fh.push([a, b]);
    for (const a of w) for (const b of w) Wh.push(a * b);
    const mu = [-1, -0.4, 0, 0.3, 0.9, 1.4], L1 = [0.7, -0.7, 0.7, -0.7, 0.7, -0.7], L2 = [0.7, 0.7, -0.7, -0.7, 0, 0];
    const Vh = mu.map((_, i) => [L1[i], L2[i]]), Dh = mu.map((_, i) => Math.max(1 - L1[i] ** 2 - L2[i] ** 2, 0.02));
    check("the home page's 301-point t4 field is resolved (#391)",
          winProbabilitiesFactor(mu, Vh, Dh, Fh, Wh, { base: "t4", points: 301 }).p,
          winProbabilitiesFactor(mu, Vh, Dh, Fh, Wh, { base: "t4", points: 4001 }).p, 2e-6);
  }
}

/* --- finite weights whose SUM overflows are the same law (#415) -------
   The normaliser divided by the raw total, so [1e308, 1e308] became
   W / Infinity = [0, 0] and every probability NaN. */
{
  const mu = [0, 0.4, 1], V = [[0.7], [0], [-0.4]], D = [1, 1, 1];
  const F = [[-1], [1]];
  const p1 = winProbabilitiesFactor(mu, V, D, F, [1, 1], { points: 257 }).p;
  const pb = winProbabilitiesFactor(mu, V, D, F, [1e308, 1e308], { points: 257 }).p;
  check("W = [1e308, 1e308] prices like [1, 1]", pb, p1, 1e-15);
  check("and matches the python reference", p1,
        [0.51529828, 0.30546137, 0.17924036], 1e-6);
  const ab = abilitiesFromProbabilitiesFactor(p1, V, D, F, [1e308, 1e308],
                                             { points: 257, nIter: 50 });
  const a1 = abilitiesFromProbabilitiesFactor(p1, V, D, F, [1, 1],
                                             { points: 257, nIter: 50 });
  check("the inverse shares the normaliser", ab, a1, 1e-12);
}

process.exit(failures ? 1 : 0);
