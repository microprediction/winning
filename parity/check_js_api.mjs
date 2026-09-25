// Behavioural checks for the browser API guards, across every module.
//
// The pytest suite can only read the source; these CALL the functions,
// which is the point -- #186 shipped because tests inspected markers, and
// #199/#200 shipped because the guard covered two of twenty entry points.
// A javascript object silently swallows a key nobody reads, so an options
// API cannot lean on the language to reject a wrong or misspelled key.
const here = new URL(".", import.meta.url).pathname;
const eng = p => import(here + "../docs/js/winning/" + p);
const [races, topk, blocks, polish, classic, demo] = await Promise.all(
  ["races.mjs", "topk.mjs", "blocks.mjs", "polish.mjs", "classic.mjs",
   "demo.mjs"].map(eng));

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
// A call that is SUPPOSED to work. Without this an exception on a happy
// path kills the file, so the run fails with no named check and every
// later check is skipped -- the sabotage rehearsal found exactly that.
const accepts = (label, fn, check = v => true, detail = () => "") => {
  let v;
  try {
    v = fn();
  } catch (e) {
    console.log(`FAIL  ${label}: threw ${e.message.slice(0, 44)}`);
    fails++;
    return undefined;
  }
  holds(label, check(v), detail(v));
  return v;
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

// --- the loadings shape contract, in every module that takes V (#232)
// V is (n, rank), one ROW per contestant, and as_loadings is the one
// place that rule is decided. The browser indexed V directly instead, so
// a length-n vector threw `V[i].reduce is not a function` and a RAGGED V
// was truncated to the first row's width and answered all-NaN in silence.
const mu5 = [0, 0.3, 0.6, 0.9, 1.2], D5 = new Array(5).fill(1);
const vec = [0.5, 0.2, 0.3, 0.1, 0.0];
const spellings = {
  "flat vector": vec,
  "(n, rank)": vec.map(v => [v]),
  "(rank, n)": [vec],
};
const ref = races.raceProbabilities(mu5, { V: spellings["(n, rank)"], D: D5 });
for (const [lbl, V] of Object.entries(spellings)) {
  accepts(`forward reads ${lbl} as the same race`,
          () => races.raceProbabilities(mu5, { V, D: D5 }),
          p => Math.max(...p.map((v, i) => Math.abs(v - ref[i]))) < 1e-12,
          p => `max diff ${Math.max(...p.map((v, i) => Math.abs(v - ref[i]))).toExponential(2)}`);
  accepts(`topK reads ${lbl}`, () => topk.topKProbabilities(mu5, 2, { V, D: D5 }),
          t => t.every(Number.isFinite));
}
accepts("a scalar V is rank 1 for every contestant",
        () => races.raceProbabilities(mu5, { V: 0.4, D: D5 }),
        p => Math.max(...p.map((v, i) => Math.abs(v - races.raceProbabilities(
          mu5, { V: mu5.map(() => [0.4]), D: D5 })[i]))) < 1e-14);

const ragged = [[0.5, 0.1], [0.2], [0.3, 0.4], [0.1, 0.2], [0.0, 0.0]];
rejects(races.raceProbabilities, [mu5, { V: ragged, D: D5 }],
        "forward rejects a ragged V", "ragged");
rejects(topk.topKProbabilities, [mu5, 2, { V: ragged, D: D5 }],
        "topK rejects a ragged V", "ragged");
rejects(races.raceProbabilities, [mu5, { V: [0.5, 0.2], D: D5 }],
        "forward rejects a vector of the wrong length", "one entry per contestant");
rejects(races.raceProbabilities, [mu5, { V: [[1, 2, 3], [4, 5, 6]], D: D5 }],
        "forward rejects a shape that is neither", "neither");
rejects(races.raceProbabilities, [mu5, { V: [0.5, 0.2, NaN, 0.1, 0.0], D: D5 }],
        "forward rejects a non-finite loading", "non-finite");

// --- no literal prime table behind the low-discrepancy nodes (#233)
// Halton read its bases from a 24-entry array in races.mjs and a
// 16-entry one in demo.mjs; one factor past the end made every node NaN
// with no error, so a rank-25 race answered NaN and a rank-17 demo too.
for (const r of [8, 17, 25, 40]) {
  const V = Array.from({ length: 6 }, (_, i) =>
    Array.from({ length: r }, (_, c) => 0.3 * Math.cos(1 + i + 2 * c) / Math.sqrt(r)));
  const D = new Array(6).fill(1);
  accepts(`rank ${r} race is finite and sums to one`,
          () => races.raceProbabilities([0, 0.2, 0.4, 0.6, 0.8, 1.0], { V, D }),
          p => p.every(Number.isFinite) &&
               Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-6,
          p => `sum ${p.reduce((a, b) => a + b, 0).toFixed(9)}`);
}
accepts("demo halton nodes survive past the old 16-prime table",
        () => demo.haltonNormalNodes(20, 8),
        h => h.F.every(r => r.every(Number.isFinite)));
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

// --- an allowlist names PUBLIC keys, not destructured locals (#207)
// polishRace reads `const { mu0: mu0In = null } = opts`, and #204 put the
// LOCAL name in its allowlist. So the supported call was refused as an
// unknown option while the internal alias was accepted and ignored, and
// 21 guard checks plus 87 surface tests all passed because none of them
// made either call. tests/test_browser_option_keys.py sweeps the whole
// surface for the same slip; these two calls pin this one.
accepts("polishRace takes its public mu0",
        () => polish.polishRace({ mu0: [-0.5, 0, 0.5], D: [1, 1, 1] }),
        r => r && r.mu && r.mu.every(Number.isFinite));
rejects(polish.polishRace, [{ mu0In: [-0.5, 0, 0.5], D: [1, 1, 1] }],
        "polishRace rejects the internal alias", "unknown option 'mu0In'");

if (fails) { console.error(`${fails} browser API failures`); process.exit(1); }
console.log("browser API guards behave");
