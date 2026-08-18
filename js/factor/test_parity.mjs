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

process.exit(failures ? 1 : 0);
