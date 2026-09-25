// Behavioural checks for the browser API guards, across every module.
//
// The pytest suite can only read the source; these CALL the functions,
// which is the point -- #186 shipped because tests inspected markers, and
// #199/#200 shipped because the guard covered two of twenty entry points.
// A javascript object silently swallows a key nobody reads, so an options
// API cannot lean on the language to reject a wrong or misspelled key.
const here = new URL(".", import.meta.url).pathname;
const eng = p => import(here + "../docs/js/winning/" + p);
const [races, topk, blocks, polish, classic] = await Promise.all(
  ["races.mjs", "topk.mjs", "blocks.mjs", "polish.mjs", "classic.mjs"].map(eng));

let fails = 0;
const rejects = (fn, args, why, expect = null) => {
  try {
    fn(...args);
    console.log(`FAIL  ${why}: accepted`);
    fails++;
  } catch (e) {
    const good = !expect || e.message.includes(expect);
    console.log(`${good ? "ok  " : "FAIL"}  ${why}: ${e.message.slice(0, 56)}`);
    if (!good) fails++;
  }
};
const holds = (label, cond, detail = "") => {
  console.log(`${cond ? "ok  " : "FAIL"}  ${label}${detail ? ": " + detail : ""}`);
  if (!cond) fails++;
};

const mu3 = [0, 1, 2], D3 = [1, 1, 1], V3 = [[1], [0], [-1]];
const I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

// --- each race API keeps to its own signature (#186)
rejects(races.raceProbabilities, [mu3, { cov: I3 }], "forward rejects cov", "cov=");
rejects(races.raceProbabilities, [mu3, { nIter: 1 }], "forward rejects nIter");
rejects(races.raceProbabilities, [mu3, { returnInfo: true }], "forward rejects returnInfo");
rejects(races.abilitiesFromRace, [[0.2, 0.3, 0.5], { returnSlopes: true }], "inverse rejects returnSlopes");
rejects(races.abilitiesFromRace, [[0.2, 0.3, 0.5], { window: "span" }], "inverse rejects window");

// --- top-k family (#199, #200)
rejects(topk.topKProbabilities, [mu3, 1, { nIter: 3 }], "topK rejects nIter");
rejects(topk.rankProbabilities, [mu3, { retrunSlopes: true }], "rank rejects a typo");
rejects(topk.abilitiesFromTopk, [[0.5, 0.3, 0.2], 1, { window: "span" }], "abilitiesFromTopk rejects window");
rejects(topk.locScaleFromTopkPair, [[0.5, 0.3, 0.2], 1, [0.8, 0.6, 0.6], 2, { V: V3 }],
        "locScale refuses V with a reason", "under-identified");
rejects(topk.locScaleFromWinAndSecond, [[0.5, 0.3, 0.2], [0.3, 0.4, 0.3], { V: V3 }],
        "locScaleWinSecond refuses V", "under-identified");

// --- the grammars and the polish
rejects(blocks.blockRaceProbabilities, [mu3, [0, 0, 1], [0.3, 0.3, 0.3], D3, { V: V3 }], "blockRace rejects V");
rejects(blocks.nestedRaceProbabilities, [mu3, [0, 0, 1], [0.3, 0.3, 0.3], D3, { nIter: 2 }], "nestedRace rejects nIter");
rejects(blocks.treeRaceProbabilities, [mu3, [0, 0, 1], [0.3, 0.3, 0.3], D3, [-1, 0], [0.5, 0.5], { cov: I3 }],
        "treeRace rejects cov", "cov=");
rejects(polish.raceJacobian, [mu3, { nIter: 2 }], "raceJacobian rejects nIter");

// --- rankProbabilities must USE the loadings, not swallow them (#199)
const rIndep = topk.rankProbabilities(mu3, { D: D3, points: 513 });
const rFactor = topk.rankProbabilities(mu3, { D: D3, V: V3, points: 513 });
const gap = Math.max(...rIndep.flat().map((v, i) => Math.abs(v - rFactor.flat()[i])));
holds("loadings change the rank matrix", gap > 1e-3, `max diff ${gap.toExponential(2)}`);
for (const [lbl, P] of [["independent", rIndep], ["factor", rFactor]]) {
  const rows = P.map(r => r.reduce((a, b) => a + b, 0));
  const cols = P[0].map((_, j) => P.reduce((a, r) => a + r[j], 0));
  holds(`${lbl} rank matrix is doubly stochastic`,
        Math.max(...rows.map(v => Math.abs(v - 1)), ...cols.map(v => Math.abs(v - 1))) < 5e-3);
}

// --- legitimate calls still work everywhere
holds("forward works", races.raceProbabilities(mu3, { D: D3 }).every(Number.isFinite));
holds("inverse works", races.abilitiesFromRace([0.5, 0.3, 0.2], { D: D3 }).every(Number.isFinite));
holds("topK works", topk.topKProbabilities(mu3, 1, { D: D3 }).every(Number.isFinite));
holds("blocks work", blocks.blockRaceProbabilities(mu3, [0, 0, 1], [0.3, 0.3, 0.3], D3, { points: 257 }).every(Number.isFinite));

// --- the inverse honours the options it advertises (#226)
rejects(races.abilitiesFromRace, [[0.5, 0.5, 0], { D: [1, 1, 1] }],
        "zero target without a floor", "no finite inverse");
rejects(races.abilitiesFromRace, [[0.5, 0.3, 0.2], { D: [1, 1, 1], targetFloor: -1 }],
        "negative targetFloor", "must be positive");
{
  const D = [1, 1, 1];
  const info = races.abilitiesFromRace([0.5, 0.3, 0.2], { D, returnInfo: true });
  holds("returnInfo returns diagnostics",
        Array.isArray(info.mu) && typeof info.converged === "boolean"
        && Number.isFinite(info.maxLogResidual) && Array.isArray(info.floored),
        `converged=${info.converged} resid=${info.maxLogResidual.toExponential(2)}`);
  const f = races.abilitiesFromRace([0.5, 0.5, 0], { D, targetFloor: 1e-4, returnInfo: true });
  holds("targetFloor floors and says which", JSON.stringify(f.floored) === "[false,false,true]",
        JSON.stringify(f.floored));
  const plain = races.abilitiesFromRace([0.5, 0.3, 0.2], { D });
  holds("returnInfo does not change the answer",
        Math.max(...plain.map((v, i) => Math.abs(v - info.mu[i]))) < 1e-12);
}

if (fails) { console.error(`${fails} browser API failures`); process.exit(1); }
console.log("browser API guards behave");
