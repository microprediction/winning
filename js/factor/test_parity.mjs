/* Parity test: JavaScript port vs committed vectors from winning.factor. */
import { readFileSync } from "fs";
import { logndtr, winProbabilitiesFactor, abilitiesFromProbabilitiesFactor }
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

process.exit(failures ? 1 : 0);
