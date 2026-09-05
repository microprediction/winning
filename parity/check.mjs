// JS parity checker: rebuild the same-named scenarios as gen_vectors.py
// from the embedded inputs and compare with the python reference.
//
// Usage: node parity/check.mjs
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const here = dirname(fileURLToPath(import.meta.url));
const eng = p => import(join(here, "..", "docs", "js", "winning", p));
const [races, blocks, structures, classic, polish, topk] = await Promise.all([
  eng("races.mjs"), eng("blocks.mjs"), eng("structures.mjs"),
  eng("classic.mjs"), eng("polish.mjs"), eng("topk.mjs"),
]);

const vec = JSON.parse(readFileSync(join(here, "vectors.json"), "utf8"));
const inp = vec.inputs;
const mu = inp.mu, D = inp.D, cl = inp.cluster, ld = inp.loading;
const V1 = inp.V1, V2 = inp.V2, ld2 = inp.loading2, cp = inp.coupling;
const pa = inp.parent, stg = inp.strength, pt = inp.p_target;
const density = classic.skewNormalDensity(inp.classic_L, inp.classic_unit,
                                          { a: inp.classic_a });
const gumD = mu.map(() => Math.PI * Math.PI / 6);

let _polish = null;
const polishCached = () => (_polish ??= polish.polishRace({ p0: pt, V: V1, D, points: 257, nameCaps: 0.15 }));

const runs = {
  independent_normal: () => races.raceProbabilities(mu, { D, points: 257 }),
  factor1_normal: () => races.raceProbabilities(mu, { V: V1, D, points: 257 }),
  factor2_normal: () => races.raceProbabilities(mu, { V: V2, D, points: 257 }),
  factor2_slopes: () =>
    races.raceProbabilities(mu, { V: V2, D, points: 257, returnSlopes: true }).slopes,
  factor2_span: () =>
    races.raceProbabilities(mu, { V: V2, D, points: 501, window: "span" }),
  gumbel_independent: () =>
    races.raceProbabilities(mu, { D: gumD, base: "gumbel", points: 1001 }),
  blocks_r1: () => blocks.blockRaceProbabilities(mu, cl, ld, D, { points: 257 }),
  blocks_r2: () => blocks.blockRaceProbabilities(mu, cl, ld2, D, { points: 257 }),
  nested: () => blocks.nestedRaceProbabilities(mu, cl, ld, D,
    { coupling: cp, gamma: 0.7, points: 257 }),
  tree: () => blocks.treeRaceProbabilities(mu, cl, ld, D, pa, stg, { points: 257 }),
  jacobian_factor: () => polish.raceJacobian(mu, { V: V1, D, points: 257 }),
  jacobian_blocks: () => blocks.blockRaceJacobian(mu, cl, ld, D, { points: 257 }),
  jacobian_nested: () => blocks.nestedRaceJacobian(mu, cl, ld, D,
    { coupling: cp, gamma: 0.7, points: 257 }),
  invert_factor: () => races.abilitiesFromRace(pt, { V: V1, D, points: 257 }),
  invert_blocks: () =>
    blocks.abilitiesFromBlockRace(pt, cl, ld, D, { points: 257 }).mu,
  classic_ability: () =>
    classic.dividendImpliedAbility(inp.dividends, density),
  classic_state_prices: () =>
    classic.statePricesFromOffsets(density, vec.scenarios.classic_ability.value),
  polish_p: () => polishCached().p,
  polish_mu: () => polishCached().mu,
  jacobian_tree: () =>
    blocks.treeRaceJacobian(mu, cl, ld, D, pa, stg, { points: 257 }),
  coph_tree: () => {
    const tr = structures.treeFromLinkage(inp.linkage_Z);
    return blocks.treeRaceProbabilities(mu, tr.cluster, tr.loading, tr.D,
      tr.parent, tr.strength, { points: 257 });
  },
  polish_tree_p: () =>
    polish.polishRace({ p0: pt, structure: structures.treeFromLinkage(inp.linkage_Z),
                        points: 257, nameCaps: 0.14 }).p,
  topk2_normal: () => topk.topKProbabilities(mu, 2, { D, points: 257 }),
  topk4_gumbel: () => topk.topKProbabilities(mu, 4,
    { D: gumD, base: "gumbel", points: 1001 }),
  topk2_jacobian_mu: () =>
    topk.topKJacobians(mu, 2, { D, points: 257 }).Jmu,
  topk2_jacobian_sigma: () =>
    topk.topKJacobians(mu, 2, { D, points: 257 }).Jsigma,
  invert_topk2: () => topk.abilitiesFromTopk(
    vec.scenarios.topk2_normal.value, 2, { D, points: 257 }),
  loc_scale_win: () => topk.topKProbabilities(mu, 1,
    { D: inp.sd_true.map(s => s * s), points: 257 }),
  loc_scale_place: () => topk.topKProbabilities(mu, 3,
    { D: inp.sd_true.map(s => s * s), points: 257 }),
  loc_scale_mu: () => topk.locScaleFromTopkPair(
    vec.scenarios.loc_scale_win.value, 1,
    vec.scenarios.loc_scale_place.value, 3, { points: 257 }).mu,
  loc_scale_sd: () => topk.locScaleFromTopkPair(
    vec.scenarios.loc_scale_win.value, 1,
    vec.scenarios.loc_scale_place.value, 3, { points: 257 }).sd,
  rank_marginals: () => topk.rankProbabilities(mu, { D, points: 257 }),
  win_second_mu: () => topk.locScaleFromWinAndSecond(
    vec.scenarios.rank_marginals.value.map(r => r[0]),
    vec.scenarios.rank_marginals.value.map(r => r[1]),
    { points: 257 }).mu,
  win_second_sd: () => topk.locScaleFromWinAndSecond(
    vec.scenarios.rank_marginals.value.map(r => r[0]),
    vec.scenarios.rank_marginals.value.map(r => r[1]),
    { points: 257 }).sd,
  loc_scale_ridge_mu: () => topk.locScaleFromTopkPair(
    vec.scenarios.loc_scale_win.value, 1,
    vec.scenarios.loc_scale_place.value, 3,
    { ridge: 0.05, points: 257 }).mu,
  loc_scale_ridge_sd: () => topk.locScaleFromTopkPair(
    vec.scenarios.loc_scale_win.value, 1,
    vec.scenarios.loc_scale_place.value, 3,
    { ridge: 0.05, points: 257 }).sd,
  invert_second: () => topk.abilitiesFromRankMarginal(
    vec.scenarios.rank_marginals.value.map(r => r[1]), 2,
    { mu0: vec.scenarios.invert_topk2.value, D, points: 257 }),
  topk2_jacobian_mu_factor: () =>
    topk.topKJacobians(mu, 2, { D, V: V1, points: 257 }).Jmu,
  topk2_jacobian_sigma_factor: () =>
    topk.topKJacobians(mu, 2, { D, V: V1, points: 257 }).Jsigma,
};

let fails = 0;
for (const [name, sc] of Object.entries(vec.scenarios)) {
  const ref = [].concat(...[].concat(sc.value));   // flatten to 1-d
  let got;
  const t0 = Date.now();
  try {
    got = [].concat(...[].concat(runs[name]()));
  } catch (e) {
    console.log(`FAIL  ${name.padEnd(22)} error: ${e.message}`);
    fails++;
    continue;
  }
  let d = 0;
  for (let i = 0; i < ref.length; i++) d = Math.max(d, Math.abs(got[i] - ref[i]));
  const ok = d <= sc.tol;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name.padEnd(22)} max|diff| ${d.toExponential(3)}  (tol ${sc.tol})  ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  if (!ok) fails++;
}
if (fails > 0) { console.log(fails + " parity failures"); process.exit(1); }
console.log("all scenarios match the python reference");
