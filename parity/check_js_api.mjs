// Behavioural checks for the browser API guards, across every module.
//
// The pytest suite can only read the source; these CALL the functions,
// which is the point -- #186 shipped because tests inspected markers, and
// #199/#200 shipped because the guard covered two of twenty entry points.
// A javascript object silently swallows a key nobody reads, so an options
// API cannot lean on the language to reject a wrong or misspelled key.
const here = new URL(".", import.meta.url).pathname;
const eng = p => import(here + "../docs/js/winning/" + p);
const [races, topk, blocks, polish, classic, demo, structures, core] =
  await Promise.all(
    ["races.mjs", "topk.mjs", "blocks.mjs", "polish.mjs", "classic.mjs",
     "demo.mjs", "structures.mjs", "core.mjs"].map(eng));

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
  // The predicate and the detail run OUTSIDE the call, and they often
  // call the API again to compare against. Wrapping only fn() meant a
  // throw in check() still killed the file: a sabotage rehearsal
  // produced ONE named failure and then aborted, hiding the other five.
  try {
    holds(label, check(v), detail(v));
  } catch (e) {
    console.log(`FAIL  ${label}: its check threw ${e.message.slice(0, 34)}`);
    fails++;
  }
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

// --- a forwarding wrapper really accepts what it forwards (#237)
// bottomKProbabilities and locScaleFromWinAndSecond validate opts and
// then hand it to another API. The source audits had nothing to compare
// their allowlists against and skipped them, so dropping a supported key
// from a wrapper's own list would have turned a working call into
// `unknown option`, silently, with every static check still green. These
// pass each option at a NON-DEFAULT value, which is the only way to see
// that the wrapper let it through.
const mu4 = [0, 0.5, 1.0, 1.5], D4 = [1, 0.8, 1.2, 0.9], V4 = [[0.6], [0.2], [-0.3], [-0.5]];
for (const [k, v] of [["V", V4], ["D", D4], ["base", "gumbel"],
                      ["points", 1025], ["qa", 21]]) {
  accepts(`bottomKProbabilities forwards ${k}`,
          () => topk.bottomKProbabilities(mu4, 1, { D: D4, [k]: v }),
          p => p.every(Number.isFinite));
}
for (const [k, v] of [["base", "gumbel"], ["points", 1025], ["nIter", 30],
                      ["tol", 1e-7], ["ridge", 1e-6], ["returnInfo", true],
                      ["D0", [1, 0.9, 1.1]], ["mu0", [0.4, 0, -0.4]]]) {
  accepts(`locScaleFromWinAndSecond forwards ${k}`,
          () => topk.locScaleFromWinAndSecond(
            [0.5, 0.3, 0.2], [0.3, 0.4, 0.3], { [k]: v }),
          r => r != null);
}

// --- the one-leaf tree (#241)
// An EMPTY linkage is the valid scipy-style spelling of a hierarchy with
// one leaf. javascript has no negative indexing, so rho[parent[i]] with
// parent[i] = -1 was undefined and the leaf variance came out NaN;
// pricing that tree failed instead of returning the certain [1]. The
// values below are python's, which the browser must reproduce.
const oneLeaf = structures.treeFromLinkage([]);
holds("one-leaf tree has finite variance",
      oneLeaf.D.every(Number.isFinite), `D = [${oneLeaf.D}]`);
holds("one-leaf tree matches python's D = [1]",
      Math.abs(oneLeaf.D[0] - 1) < 1e-15);
accepts("the one-leaf tree prices to certainty",
        () => blocks.treeRaceProbabilities([0], oneLeaf.cluster,
          oneLeaf.loading, oneLeaf.D, oneLeaf.parent, oneLeaf.strength),
        p => p.length === 1 && Math.abs(p[0] - 1) < 1e-12,
        p => `p = [${p}]`);
const threeLeaf = structures.treeFromLinkage([[0, 1, 0.5], [2, 3, 0.8]]);
holds("a real linkage is unchanged: python's D = [0.5, 0.5, 1]",
      Math.max(...threeLeaf.D.map((v, i) => Math.abs(v - [0.5, 0.5, 1][i]))) < 1e-15,
      `D = [${threeLeaf.D}]`);

// --- a caller-supplied factor law reaches the derivative too (#209)
// raceProbabilities has always accepted {F, W}. raceJacobian and
// polishRace rejected them and built their own standard-normal Hermite
// rule, so the browser could PRICE a discrete factor law and could not
// differentiate or polish it: the Jacobian returned was the derivative
// of a DIFFERENT model, and on this two-point law it is wrong by 0.2 in
// absolute terms. Dropping F/W is not a workaround, it is the bug.
const muF = [0, 0.4, 1.0], VF = [[-1], [0], [1]], DF = [1, 1, 1];
const FF = [[-3], [3]], WF = [0.5, 0.5];
const fwdF = m => races.raceProbabilities(m, { V: VF, D: DF, F: FF, W: WF });
accepts("raceJacobian takes a caller factor law",
        () => polish.raceJacobian(muF, { V: VF, D: DF, F: FF, W: WF }),
        J => J.length === 3 && J.every(r => r.every(Number.isFinite)));
{
  const J = polish.raceJacobian(muF, { V: VF, D: DF, F: FF, W: WF });
  const h = 1e-5;
  let worst = 0;
  for (let j = 0; j < 3; j++) {
    const a = muF.slice(), b = muF.slice();
    a[j] += h; b[j] -= h;
    const pa = fwdF(a), pb = fwdF(b);
    for (let i = 0; i < 3; i++)
      worst = Math.max(worst, Math.abs(J[i][j] - (pa[i] - pb[i]) / (2 * h)));
  }
  holds("the jacobian is the derivative of THAT forward",
        worst < 1e-8, `max |analytic - finite difference| ${worst.toExponential(2)}`);
  const Jg = polish.raceJacobian(muF, { V: VF, D: DF });
  let gap = 0;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) gap = Math.max(gap, Math.abs(Jg[i][j] - J[i][j]));
  holds("and the internal gaussian rule is NOT the same answer",
        gap > 1e-2, `differs by ${gap.toExponential(2)}`);
}
accepts("polishRace takes a caller factor law and honours it",
        () => polish.polishRace({ mu0: muF, V: VF, D: DF, F: FF, W: WF,
                                  nameCaps: 0.35 }),
        res => {
          const re = fwdF(res.mu);
          return Math.max(...re.map(v => v - 0.35)) < 1e-8 &&
                 Math.max(...re.map((v, i) => Math.abs(v - res.p[i]))) < 1e-8;
        });

// --- dividends that are not ordinary positive numbers (#242)
// Only a MISSING quote becomes nanValue. The browser used
// `Number.isFinite(x) ? x : nanValue`, conflating every non-finite value
// with a missing one, and divided unconditionally: an infinite-dividend
// entrant took 1/2000 of the book instead of nothing, a dividend of 0
// gave NaN, and a NEGATIVE dividend gave a negative "probability". The
// expected rows below are python's StatePricer.prices_from_dividends.
for (const [label, d, want] of [
  ["+Infinity is worthless", [2, 4, Infinity], [2 / 3, 1 / 3, 0]],
  ["zero is worthless", [2, 4, 0], [2 / 3, 1 / 3, 0]],
  ["negative is worthless", [2, 4, -5], [2 / 3, 1 / 3, 0]],
  ["-Infinity is worthless", [2, 4, -Infinity], [2 / 3, 1 / 3, 0]],
  ["an all-infinite book is zeros, not 0/0", [Infinity, Infinity], [0, 0]],
  ["a MISSING quote still takes nanValue", [2, 4, NaN],
   [0.6662225183211193, 0.33311125916055967, 0.0006662225183211193]],
  ["null is missing too", [2, 4, null],
   [0.6662225183211193, 0.33311125916055967, 0.0006662225183211193]],
]) {
  accepts(`dividends: ${label}`,
          () => classic.pricesFromDividends(d),
          p => p.every(Number.isFinite) && p.every(v => v >= 0) &&
               Math.max(...p.map((v, i) => Math.abs(v - want[i]))) < 1e-12,
          p => `[${p.map(v => v.toFixed(6))}]`);
}

// --- portfolio limits are refused when malformed, not dropped (#247)
// These encode name and sector caps. A typo that DROPS a constraint
// returns an ordinary-looking result that is simply under-constrained,
// which is worse than an error: nameCaps of length n-1 left an 80%
// position uncapped and still reported maxViolation 0, and a group index
// of n produced NaN for every runner and called it feasible.
rejects(polish.concentrationMatrix, [3, { nameCaps: [0.3, 0.3] }],
        "a short nameCaps is refused", "one entry per contestant");
rejects(polish.concentrationMatrix, [3, { nameCaps: [0.3, 0.3, 0.3, 0.3] }],
        "an overlong nameCaps is refused", "one entry per contestant");
rejects(polish.concentrationMatrix, [3, { groups: [[[0, 3], 0.5]] }],
        "a group index of n is refused", "not an integer in [0, 3)");
rejects(polish.concentrationMatrix, [3, { groups: [[[0, -1], 0.5]] }],
        "a negative group index is refused", "not an integer in [0, 3)");
rejects(polish.polishRace,
        [{ p0: [0.2, 0.3, 0.5], D: [1, 1, 1], A: [[1, 0]], b: [0.4],
           points: 129 }],
        "an A row of the wrong width is refused", "one coefficient per");
// python documents a non-finite ENTRY as "no cap for that name", so it
// stays a feature rather than becoming an error.
accepts("a NaN entry still means no cap for that name",
        () => polish.concentrationMatrix(3, { nameCaps: [0.3, NaN, 0.4] }),
        cm => cm.A.length === 2 && cm.b.length === 2);
accepts("and the constraints that are well formed still bind",
        () => polish.polishRace({ p0: [0.1, 0.1, 0.8], D: [1, 1, 1],
                                  nameCaps: [0.5, 0.5, 0.5], points: 129 }),
        r => Math.max(...r.p) <= 0.5 + 1e-8,
        r => `p = [${r.p.map(v => v.toFixed(4))}]`);
accepts("a group cap binds on its members",
        () => polish.polishRace({ p0: [0.2, 0.3, 0.5], D: [1, 1, 1],
                                  groups: [[[0, 1], 0.45]], points: 129 }),
        r => Math.abs(r.p[0] + r.p[1] - 0.45) < 1e-6,
        r => `p0 + p1 = ${(r.p[0] + r.p[1]).toFixed(6)}`);

// --- the idiosyncratic-variance contract, D's half of asLoadings (#254)
// The browser normalised V and left D alone. A scalar threw
// `D.slice is not a function`; short, zero, negative and NaN entries
// propagated NaN; and -- the dangerous one -- a D with ONE EXTRA entry
// was ACCEPTED and returned a normalised, plausible, materially wrong
// answer. The expected numbers below are python's.
const mu254 = [-0.4, 0, 0.4];
const P_FLAT = [0.506706, 0.312135, 0.181159];
accepts("a scalar D is the same variance for everyone",
        () => races.raceProbabilities(mu254, { D: 1 }),
        p => Math.max(...p.map((v, i) => Math.abs(v - P_FLAT[i]))) < 1e-6,
        p => `[${p.map(v => v.toFixed(6))}]`);
accepts("and agrees with the length-n spelling",
        () => races.raceProbabilities(mu254, { D: [1, 1, 1] }),
        p => Math.max(...p.map((v, i) =>
          Math.abs(v - races.raceProbabilities(mu254, { D: 1 })[i]))) < 1e-14);
for (const [label, D] of [
  ["one entry too many", [1, 1, 1, 1]],
  ["one entry too few", [1, 1]],
]) {
  rejects(races.raceProbabilities, [mu254, { D }],
          `forward rejects a D with ${label}`, "one idiosyncratic variance per");
  rejects(topk.topKProbabilities, [mu254, 2, { D }],
          `topK rejects a D with ${label}`, "one idiosyncratic variance per");
}
rejects(races.raceProbabilities, [mu254, { D: [1, -1, 1] }],
        "a negative variance is refused", "negative variance");
rejects(races.raceProbabilities, [mu254, { D: [1, 0, 1] }],
        "a zero variance is refused", "strictly positive");
rejects(races.raceProbabilities, [mu254, { D: [1, NaN, 1] }],
        "a non-finite variance is refused", "non-finite");
rejects(races.abilitiesFromRace, [[0.5, 0.3, 0.2], { D: [1, 1, 1, 1] }],
        "the inverse has its own copy and refuses too",
        "one idiosyncratic variance per");
accepts("the inverse takes a scalar D, matching python",
        () => races.abilitiesFromRace([0.5, 0.3, 0.2], { D: 1 }),
        mu => Math.max(...mu.map((v, i) =>
          Math.abs(v - [-0.379114, 0.039741, 0.339373][i]))) < 1e-6,
        mu => `[${mu.map(v => v.toFixed(6))}]`);
// --- the jacobian differentiates the forward's own lattice (#212)
// raceJacobian built its OWN grid: a plain span window, no adaptive
// placement, no refinement. So it differentiated a different lattice
// than raceProbabilities computed on, and the two parted company
// exactly where the lattice is coarse relative to the field. Sharing
// forwardGrid is what closes it; the fixture below is chosen for a
// variance ratio of 4005, where the gap was 1.0e-3.
{
  const muJ = [-0.8582367891716942, -1.257503986012891,
               0.40498888604575267, 0.5712404263071904];
  const DJ = [74.38440122352718, 0.06946348206585057,
              6.65916905702646, 0.018571769943793635];
  const fd = (mu, D, L) => {
    const J = polish.raceJacobian(mu, { D, points: L });
    const h = 1e-5;
    let worst = 0;
    for (let j = 0; j < mu.length; j++) {
      const a = mu.slice(), b = mu.slice();
      a[j] += h; b[j] -= h;
      const pa = races.raceProbabilities(a, { D, points: L });
      const pb = races.raceProbabilities(b, { D, points: L });
      for (let i = 0; i < mu.length; i++)
        worst = Math.max(worst, Math.abs(J[i][j] - (pa[i] - pb[i]) / (2 * h)));
    }
    return worst;
  };
  holds("the jacobian follows its forward on a coarse lattice",
        fd(muJ, DJ, 257) < 1e-4, `|J - FD| ${fd(muJ, DJ, 257).toExponential(2)}`);
  holds("and converges as the lattice refines",
        fd(muJ, DJ, 1025) < 1e-8, `|J - FD| ${fd(muJ, DJ, 1025).toExponential(2)}`);
  holds("an ordinary field is unchanged",
        fd([0, 1, 2], [0.8, 1.0, 1.2], 257) < 1e-9);
}

// --- the nested coupling is a loading matrix like any other (#264)
// `coupling` was normalised by hand in blocks.mjs and never transposed,
// so a (rank, n) spelling -- which python treats as the SAME race -- was
// read as rank n and refused. Rank 3 was refused outright, with a
// remediation that could not be followed: "supply nodes explicitly",
// when no nested verb accepts a `nodes` option and checkOpts rejects it
// before pricing. Rank >= 3 now escalates the FAMILY to Halton, as
// races.mjs does for high-rank loadings.
{
  const muN = [-0.4, -0.1, 0.2, 0.5];
  const clN = [0, 0, 1, 1], ldN = [0.3, 0.2, 0.4, 0.25];
  const DN = [1, 0.8, 0.9, 1.1];
  const c2 = [[0.3, 0.1], [0.2, -0.2], [-0.1, 0.4], [0.15, 0.05]];
  const c2T = [[0.3, 0.2, -0.1, 0.15], [0.1, -0.2, 0.4, 0.05]];
  const c3 = [[0.30, 0.10, -0.20], [0.20, -0.20, 0.10],
              [-0.10, 0.40, 0.20], [0.15, 0.05, -0.15]];
  const price = c => blocks.nestedRaceProbabilities(
    muN, clN, ldN, DN, { coupling: c, gamma: 0.3 });
  accepts("a transposed coupling is the same race",
          () => price(c2T),
          p => Math.max(...p.map((v, i) => Math.abs(v - price(c2)[i]))) < 1e-14);
  accepts("a rank-3 coupling prices at all",
          () => price(c3),
          p => p.every(Number.isFinite) &&
               Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  // python's scrambled Sobol against the browser's Halton: different QMC
  // families, so agreement is to quadrature tolerance, not to the digit
  holds("and agrees with python's rank-3 answer",
        Math.max(...price(c3).map((v, i) =>
          Math.abs(v - [0.414742, 0.263511, 0.192839, 0.128909][i]))) < 5e-4,
        `max diff ${Math.max(...price(c3).map((v, i) =>
          Math.abs(v - [0.414742, 0.263511, 0.192839, 0.128909][i]))).toExponential(2)}`);
  accepts("the rank-3 jacobian follows its own forward",
          () => blocks.nestedRaceJacobian(muN, clN, ldN, DN,
                                          { coupling: c3, gamma: 0.3 }),
          J => {
            const h = 1e-5;
            let worst = 0;
            for (let j = 0; j < 4; j++) {
              const a = muN.slice(), b2 = muN.slice();
              a[j] += h; b2[j] -= h;
              const pa = blocks.nestedRaceProbabilities(a, clN, ldN, DN,
                { coupling: c3, gamma: 0.3 });
              const pb = blocks.nestedRaceProbabilities(b2, clN, ldN, DN,
                { coupling: c3, gamma: 0.3 });
              for (let i = 0; i < 4; i++)
                worst = Math.max(worst,
                  Math.abs(J[i][j] - (pa[i] - pb[i]) / (2 * h)));
            }
            return worst < 1e-8;
          });
}

// --- the product rule is never materialised (#268)
// hermiteNodes built the whole order**rank tensor and pruned it
// after. `Math.max(...W)` spreads every weight as an argument list, so
// rank 5 at order 15 -- 759,375 nodes -- died with a RangeError before
// pruning ran; rank 5 at order 41 would have been 115,856,201 nodes
// built and thrown away. python closed this in #155 and the browser
// kept it. The expected counts below are python's.
{
  const expect = [[1, 15, 15], [2, 15, 145], [3, 15, 1317], [4, 15, 10929],
                  [5, 15, 83757], [5, 7, 10295], [3, 41, 6283]];
  for (const [k, q, n] of expect)
    accepts(`hermiteNodes(${k}, ${q}) is python's rule`,
            () => core.hermiteNodes(k, q),
            r => r.F.length === n
                 && Math.abs(r.W.reduce((a, b) => a + b, 0) - 1) < 1e-12
                 && r.F.every(row => row.length === k),
            r => `${r.F.length} nodes (python ${n})`);

  // the one from the report, which used to be a RangeError, and the
  // one that was never going to fit in memory at all
  accepts("rank 5 order 15 returns instead of overflowing the stack",
          () => core.hermiteNodes(5, 15),
          r => r.F.length === 83757);
  accepts("rank 5 order 41 prunes 115,856,201 down to python's count",
          () => core.hermiteNodes(5, 41),
          r => r.F.length === 1061871,
          r => `${r.F.length} nodes from a 1.16e8 tensor`);

  // pruning must not reorder: the kept set is the full tensor's, in
  // the tensor's order, so a node's neighbours still mean what the
  // reference means by them
  accepts("the kept nodes are in tensor order (first coordinate slowest)",
          () => core.hermiteNodes(2, 15),
          r => {
            for (let i = 1; i < r.F.length; i++) {
              const a = r.F[i - 1], b2 = r.F[i];
              if (a[0] > b2[0] + 1e-12) return false;          // never decreases
              if (Math.abs(a[0] - b2[0]) < 1e-12 && a[1] > b2[1] + 1e-12)
                return false;                                  // ties ascend in x1
            }
            return true;
          });

  rejects(core.hermiteNodes, [0, 15], "hermiteNodes refuses rank 0",
          "positive integer");
  rejects(core.hermiteNodes, [2.5, 15], "hermiteNodes refuses a fractional rank",
          "positive integer");
  rejects(core.hermiteNodes, [2, 0], "hermiteNodes refuses order 0",
          "positive integer");
}

// --- classic calibration: offsetSamples must descend (#274)
// The interpolation table is built by mapping over offsetSamples and
// read back with interpClamped, whose xp must ascend -- descending
// offsets are what make the prices ascend. An ascending array runs the
// binary search on a descending table, so every lookup clamps to an
// end: three distinct prices came back as the lattice boundary
// [24, -25, -25], repriced 0.53 away from the target, with no error.
// python raises ValueError("Not descending") and R stops with
// "offset_samples must be descending" for the very same input.
{
  const density = classic.skewNormalDensity(50, 0.1);
  const target = classic.statePricesFromOffsets(density, [-3, 0.5, 2]);
  const descending = Array.from({ length: 50 }, (_, i) => 24 - i);
  const solve = os => classic.solveForImpliedOffsets(
    target, density, { offsetSamples: os, guess: [0, 0, 0], nIter: 3 });

  rejects(solve, [descending.slice().reverse()],
          "classic refuses ascending offsetSamples", "must be descending");
  rejects(solve, [[]], "classic refuses empty offsetSamples", "is empty");
  rejects(solve, [[5, NaN, -5]],
          "classic refuses a non-finite offset", "not a finite offset");
  rejects(solve, [7], "classic refuses a non-array offsetSamples",
          "must be an array");
  rejects(classic.asDescendingOffsets, [[0, 1, 2]],
          "the contract names the first ascent", "[1] = 1 is above");

  // ties are LEGAL -- a repeated offset is a flat step in the table,
  // not an ascent -- and python's `d > 0` allows them too
  accepts("a repeated offset is not an ascent",
          () => classic.asDescendingOffsets([5, 5, 0, -5]),
          v => v.length === 4 && v[1] === 5);

  // and the descending path is untouched: the fix must not move the
  // answer it was already getting right
  accepts("descending offsetSamples still reprice to the target",
          () => classic.statePricesFromOffsets(density, solve(descending)),
          back => Math.max(...back.map((b, i) => Math.abs(b - target[i]))) < 2e-3,
          back => `max reprice error ${Math.max(...back.map((b, i) =>
            Math.abs(b - target[i]))).toExponential(2)}`);

  // the DEFAULT is built internally and must still be accepted by the
  // contract it now has to satisfy
  accepts("the internally built default is descending",
          () => classic.solveForImpliedOffsets(target, density, { guess: [0, 0, 0] }),
          a => a.length === 3 && a.every(Number.isFinite),
          a => `[${a.map(v => v.toFixed(4))}]`);
  accepts("dividendImpliedAbility still prices",
          () => classic.dividendImpliedAbility([2.5, 4.0, 6.0], density),
          a => a.length === 3 && a.every(Number.isFinite),
          a => `[${a.map(v => v.toFixed(4))}]`);
}

if (fails) { console.error(`${fails} browser API failures`); process.exit(1); }
console.log("browser API guards behave");
