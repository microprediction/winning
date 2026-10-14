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

// --- rare exact places are not deconvolution residue (#511)
{
  // independent min-Gumbels are Plackett-Luce with weights exp(-pi/sqrt6 mu)
  const mu = [-9, -2, 2, 10], c = Math.PI / Math.sqrt(6);
  const w = mu.map(v => Math.exp(-c * v));
  const E = mu.map(() => mu.map(() => 0));
  const perm = (a, k = 0) => {
    if (k === a.length) {
      let p = 1;
      for (let r = 0; r < a.length - 1; r++) {
        let d = 0;
        for (let s = r; s < a.length; s++) d += w[a[s]];
        p *= w[a[r]] / d;
      }
      a.forEach((i, r) => { E[i][r] += p; });
      return;
    }
    for (let j = k; j < a.length; j++) {
      [a[k], a[j]] = [a[j], a[k]]; perm(a, k + 1); [a[k], a[j]] = [a[j], a[k]];
    }
  };
  perm([0, 1, 2, 3]);
  for (const points of [129, 513, 8193]) {
    const P = topk.rankProbabilities(mu, { D: 1, base: "gumbel", points });
    const worst = Math.max(...P.flatMap((row, i) => row.map((v, r) => Math.abs(v / E[i][r] - 1))));
    holds(`Gumbel rank matrix matches Plackett-Luce entrywise at ${points} points (#511)`,
          worst < 1e-5, `last place ${P[0][3].toExponential(4)} vs ${E[0][3].toExponential(4)}`);
  }
  for (const points of [513, 8193]) {
    const last = topk.rankProbabilities([-12, -4, 4, 12], { points })[1][3];
    holds(`normal last place is the cavity product at ${points} points (#511)`,
          Math.abs(last / 2.9183334444853456e-30 - 1) < 1e-9, last.toExponential(6));
    const q = topk.topKProbabilities([-14, 0, 0], 1, { points })[1];
    const r = races.raceProbabilities([-14, 0, 0], { points })[1];
    holds(`top-1 equals the win market in the tail at ${points} points (#511)`,
          Math.abs(q / 2.091912795347403e-23 - 1) < 1e-9 && Math.abs(q / r - 1) < 3e-3,
          `${q.toExponential(6)} vs race ${r.toExponential(6)}`);
  }
}

// --- a callable base may carry its own tails: cdf / logCdf / logSf
// (#511 follow-up). A wrapper of a symmetric named base lost the lower
// tail (identity-keyed reflection): [-20, 0, 0]'s second place 99% low.
{
  for (const name of ["normal", "logistic", "laplace"]) {
    const b = races.BASES[name];
    const plain = z => b(z);
    const tailed = z => b(z);
    tailed.cdf = name === "normal" ? core.ndtr : z => b(-z)[0];
    tailed.logSf = b.logSf;
    for (const lead of [-20, -30]) {
      const mu = [lead, 0, 0];
      const ref = [topk.rankProbabilities(mu, { base: name })[0][1],
                   topk.bottomKProbabilities(mu, 1, { base: name })[0]];
      const got = [topk.rankProbabilities(mu, { base: tailed })[0][1],
                   topk.bottomKProbabilities(mu, 1, { base: tailed })[0]];
      const worst = Math.max(...got.map((g, j) => Math.abs(g / ref[j] - 1)));
      holds(`${name} wrapper with cdf/logSf matches the named base at ${lead} (#511)`,
            worst < 1e-12, `worst rel ${worst.toExponential(2)}`);
    }
    if (name === "normal") {
      const lost = topk.rankProbabilities([-30, 0, 0], { base: plain })[0][1];
      holds("a plain normal wrapper still takes 1 - S (#511)",
            lost < 1e-10 * 7.212994e-100, lost.toExponential(3));
    }
  }
  const lc = z => races.BASES.normal(z);
  lc.logCdf = core.logndtr;
  const r = topk.rankProbabilities([-20, 0, 0], { base: lc })[0][1];
  holds("logCdf serves when cdf is absent (#511)",
        Math.abs(r / topk.rankProbabilities([-20, 0, 0])[0][1] - 1) < 1e-12,
        r.toExponential(6));
  const bad = extra => Object.assign(z => races.BASES.normal(z), extra);
  rejects(topk.rankProbabilities, [[0, 1, 2], { base: bad({ cdf: 0.5 }) }],
          "a non-function cdf", "must be a function");
  rejects(topk.rankProbabilities, [[0, 1, 2], { base: bad({ cdf: z => core.ndtr(-z) }) }],
          "a survival passed as cdf", "non-decreasing");
  rejects(topk.rankProbabilities,
          [[0, 1, 2], { base: bad({ cdf: z => 1 / (1 + Math.exp(-z)) }) }],
          "a cdf of another law", "disagrees");
  rejects(races.raceProbabilities, [[0, 1, 2], { base: bad({ logSf: z => -core.logndtr(-z) }) }],
          "an increasing logSf", "non-increasing");
}

// --- #511 residuals: pair cavities, bottom-k CDFs, floored survivals (#502)
{
  // off-diagonals against python's, whose pair cavities match a directly
  // multiplied pair term to 1e-15; the one-direction rule was 1e5..1e75 off
  const ref = {
    1: [[0, 1.47899073950077e-22, 1.0488416889543e-44, 1.01368725593504e-58, 4.71442776496118e-93],
        [0, 0, 1.44874122289024e-48, 3.38899276549204e-61, 2.87678936277052e-94],
        [0, 0, 0, 3.20946024779509e-71, 2.59194254794805e-100],
        [0, 0, 0, 0, 1.8451583059137e-107]],
    3: [[0, 7.51220871032102e-72, 8.38483810428965e-71, 1.28051609107831e-70, 4.80864530065274e-98],
        [0, 0, 1.67688004459801e-11, 4.99629173629526e-11, 8.25460739746445e-26],
        [0, 0, 0, 0.029732572278728, 4.99629173629526e-11],
        [0, 0, 0, 0, 1.67688004459801e-11]],
  };
  for (const k of [1, 3]) {
    const J = topk.topKJacobians([-20, -6, 0, 3, 9], k).Jmu;
    let worst = 0;
    for (let i = 0; i < 4; i++)
      for (let j = i + 1; j < 5; j++)
        worst = Math.max(worst, Math.abs(J[i][j] / ref[k][i][j] - 1),
                         Math.abs(J[j][i] / ref[k][i][j] - 1));
    holds(`top-${k} Jacobian pair terms are not deconvolution residue (#511)`,
          worst < 1e-8, `worst rel ${worst.toExponential(2)}`);
  }
  const last = topk.bottomKProbabilities([-30, 0, 0], 1)[0];
  holds("bottom-k takes the CDFs directly, not 1 - S (#511)",
        Math.abs(last / 4.684196986668009e-134 - 1) < 1e-9, last.toExponential(6));
  const t1 = topk.topKProbabilities([0, 53, 106], 1)[1];
  holds("top-1 below 1e-300 is not the survival floor (#502)",
        Math.abs(t1 / 1.1054538321318671e-307 - 1) < 1e-6, t1.toExponential(6));
  const pn = races.raceProbabilities([0, 53, 106], { window: "span", points: 8193 })[1];
  holds("race below 1e-300 is not the survival floor: normal (#502)",
        Math.abs(pn / 1.1054538321318671e-307 - 1) < 1e-6, pn.toExponential(6));
  const pl = races.raceProbabilities([0, 500], { base: "laplace", window: "span", points: 65537 })[1];
  holds("race below 1e-300 is not the survival floor: laplace (#502)",
        Math.abs(pl / 1.4324466679622696e-305 - 1) < 1e-5, pl.toExponential(6));
  const pg = races.raceProbabilities([0, 25, 50], { base: "logistic", window: "span", points: 65537 });
  holds("logistic left-tail density is not zeroed (#467)",
        Math.abs(pg[1] / 8.990174560131736e-19 - 1) < 1e-9 &&
        Math.abs(pg[2] / 1.8637021047561597e-38 - 1) < 1e-9,
        `${pg[1].toExponential(6)} ${pg[2].toExponential(6)}`);
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
// --- the pair inverse keeps a tiny share whatever the label order (#412)
// [1, 1e-16] normalizes to [1, 1e-16] (1 + 1e-16 == 1), the closed form
// inverted the first share, invNormalRational(1) = NaN, certified exact.
// Referee: scipy ndtri(t / (1 + t)), the contrast at unit sd.
for (const [t, z] of [[1e-12, -7.034483825301272], [1e-16, -8.222082216130435],
                      [1e-20, -9.262340089798409], [1e-50, -14.933337534788487]]) {
  const D = [0.5, 0.5];
  const a = races.abilitiesFromRace([t, 1], { D, returnInfo: true });
  const b = races.abilitiesFromRace([1, t], { D, returnInfo: true });
  holds(`pair inverse [1, ${t}] is finite, converged and mirrors [${t}, 1]`,
        b.mu.every(Number.isFinite) && a.converged && b.converged
        && Math.abs(b.mu[0] - a.mu[1]) <= 1e-12 * Math.abs(a.mu[1])
        && Math.abs((b.mu[0] - b.mu[1]) - z) <= 1e-8 * Math.abs(z),
        `a=${a.mu} b=${b.mu}`);
}
// --- a larger budget never returns a worse race inverse (#646); python
// returns 0.70704802702675 at n_iter 2, 3 and 4
{
  const D = [4.109124193642786, 1.1933390070735221, 0.22399089188231003,
             0.14226868263033934, 0.156512843473103];
  const t = [0.14786642573393072, 0.04868820314808961, 0.015147330702447054,
             0.01374172467779407, 0.7745563157377385];
  const r = [2, 3, 4].map(nIter => {
    const o = races.abilitiesFromRace(t, { D, points: 2049, nIter, tol: 1e-12, returnInfo: true });
    const p = races.raceProbabilities(o.mu, { D, points: 2049 });
    return [o.maxLogResidual, Math.max(...p.map((x, i) => Math.abs(Math.log(x / t[i]))))];
  });
  holds("race inverse keeps the better of its last two iterates (#646)",
        r.every(([rep, act]) => Math.abs(rep - 0.70704802702675) < 1e-9 && Math.abs(act - rep) < 1e-12),
        JSON.stringify(r));
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

// --- the automatic node rule depends on V V' alone (#130) and the top-k
// node budget is met with a whole rule, not a prefix (#340)
{
  const mx = (p, q) => Math.max(...p.map((x, i) => Math.abs(x - q[i])));
  const warn = console.warn;
  console.warn = () => {};
  try {
    const mu = [-0.5, 0.1, 0.7], D = [0.19, 0.19, 1.0], b = [[0.9, 0], [0, 0.9], [0, 0]];
    const pad = (a, c) => b.map(r => { const o = Array(80).fill(0); o[a] = r[0]; o[c] = r[1]; return o; });
    const p0 = races.raceProbabilities(mu, { V: pad(0, 1), D, points: 257 });
    holds("80-column padding: placement and swap of the active columns (#130)",
          mx(p0, races.raceProbabilities(mu, { V: pad(68, 69), D, points: 257 })) < 1e-12
          && mx(p0, races.raceProbabilities(mu, { V: pad(69, 68), D, points: 257 })) < 1e-12
          // the unpadded spelling keeps python's rule in its own axes: 1.1e-5
          && mx(p0, races.raceProbabilities(mu, { V: b, D, points: 257 })) < 5e-5);
    const n = 4, r = 12;
    const m4 = Array.from({ length: n }, (_, i) => 0.7 * Math.sin((i + 1) * 9 * 0.37) + 0.2 * i);
    const V12 = Array.from({ length: n }, (_, i) =>
      Array.from({ length: r }, (_, j) => 0.25 * Math.sin((i + 1) * (j + 2) + 8 * 0.31)));
    const D4 = [0.03, 0.045, 0.06, 0.075];
    holds("rank-12 column reversal prices the same race (#130)",
          mx(races.raceProbabilities(m4, { V: V12, D: D4 }),
             races.raceProbabilities(m4, { V: V12.map(row => row.slice().reverse()), D: D4 })) < 1e-12);
    const bm = [-0.7, -0.1, 0.35, 0.9], cl = [0, 0, 0, 0], bD = [0.08, 0.08, 0.08, 0.08];
    const bb = [[2.5, 0], [0, 2.5], [-2.5, 0], [0, -2.5]];
    const bpad = (a, c) => bb.map(x => Array.from({ length: 80 }, (_, j) => j === a ? x[0] : j === c ? x[1] : 0));
    const q0 = blocks.blockRaceProbabilities(bm, cl, bb, bD, { points: 257, qa: 9 });
    holds("block loading padding and column swap price the same race (#130)",
          [[0, 1], [68, 69], [69, 68]].every(([a, c]) =>
            mx(q0, blocks.blockRaceProbabilities(bm, cl, bpad(a, c), bD, { points: 257, qa: 9 })) < 1e-12));
    const V1 = [[-3], [-1], [1], [3]];
    for (const d of [1e-3, 1e-4]) {
      const m = [-0.015, 0.005, 0.010, 0.0], Dd = [d, d, d, d];
      holds(`sharp rank-one top-1 is the race at D=${d} (#340)`,
            mx(races.raceProbabilities(m, { V: V1, D: Dd, points: 257 }),
               topk.topKProbabilities(m, 1, { V: V1, D: Dd, points: 257 })) < 1e-9);
    }
    const tm = [0.15270157009363175, -0.2301855193451047, 0.23726646788418293, 0.15630929786711933];
    const tD = [0.009629986205992744, 0.0018727096736087688, 0.00503595103127878, 0.010563698968609512];
    const tV = [[1.2288664504885674, 0.8170432653278112], [-1.1702364422380924, 2.074527421966195],
                [0.64559206366539, -1.9816835839301348], [0.021002475172281265, 1.0335299242287874]];
    const th = 0.47, c = Math.cos(th), s = Math.sin(th);
    const tVr = tV.map(([a, bq]) => [a * c + bq * s, -a * s + bq * c]);
    holds("a sharp rank-two rotation leaves top-3 alone (#340)",
          mx(topk.topKProbabilities(tm, 3, { V: tV, D: tD, points: 513 }),
             topk.topKProbabilities(tm, 3, { V: tVr, D: tD, points: 513 })) < 1e-12);
  } finally { console.warn = warn; }
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

// --- an inverted linkage is refused, not clipped (#133)
// python and R refuse a merge below its own parent; the browser clipped
// the negative increment and returned D = [1, 1, 0.5] for the second
// fixture, breaking unit variance. Same fixtures, message, node and size
// as tests/test_tree_linkage_monotonic.py and test-structures.R.
rejects(structures.treeFromLinkage, [[[0, 1, 0.5, 2], [2, 3, 0.3, 3]]],
        "treeFromLinkage refuses an inversion",
        "node 3 merges 0.32 BELOW its parent, and 1 node(s)");
rejects(structures.treeFromLinkage, [[[0, 1, 0.8, 2], [2, 3, 0.5, 3]]],
        "treeFromLinkage refuses an inversion above rho = 0.5",
        "not monotonic");

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

  // rank 0 is the EMPTY PRODUCT, not an error: #68 settled that a
  // zero-rank loading matrix prices the independent race, and python, R
  // and julia all answer it. This used to assert a refusal, written
  // when this port was the only one that refused.
  accepts("hermiteNodes: rank 0 is one node with no columns",
          () => core.hermiteNodes(0, 15),
          h => h.F.length === 1 && h.F[0].length === 0 && h.W[0] === 1);
  rejects(core.hermiteNodes, [-1, 15], "hermiteNodes refuses a negative rank",
          "k >= 0");
  rejects(core.hermiteNodes, [2.5, 15], "hermiteNodes refuses a fractional rank",
          "integer k");
  rejects(core.hermiteNodes, [2, 0], "hermiteNodes refuses order 0",
          "order >= 1");
}

// --- the block Jacobian says rank-one, or says so (#271)
// The FORWARD block kernel prices rank-r cluster loadings. The
// Jacobian is rank-one only: it takes each loading row as a number,
// and javascript coerces a ONE-element array to its number, so an
// (n, 1) column kept working while a rank-2 row went NaN at both the
// quadrature amplitude and the shift. The NaN reached every cell, so
// the call returned an all-NaN matrix and said nothing. python's
// block_race_jacobian raises for exactly this input.
{
  const muB = [-0.4, 0.1, 0.2, 0.5];
  const clB = [0, 0, 1, 1];
  const DB = [0.7, 0.8, 0.9, 1.0];
  const rank2 = [[0.5, 0.2], [0.3, -0.4], [-0.2, 0.6], [0.1, 0.5]];
  const column = [[0.5], [0.3], [-0.2], [0.1]];
  const flat = [0.5, 0.3, -0.2, 0.1];
  const want = "rank-one cluster loadings only";

  // every Jacobian door funnels through blockRaceJacobian, so all of
  // them have to refuse -- that is the point of guarding there
  rejects(blocks.blockRaceJacobian, [muB, clB, rank2, DB],
          "block jacobian refuses rank-2 loadings", want);
  rejects(blocks.nestedRaceJacobian,
          [muB, clB, rank2, DB, { coupling: [0.2, -0.1, 0.3, 0.0] }],
          "nested jacobian refuses rank-2 loadings", want);
  rejects(polish.raceJacobian,
          [muB, { structure: { kind: "Blocks", cluster: clB, loading: rank2, D: DB } }],
          "the structured front door refuses rank-2 too", want);
  rejects(blocks.abilitiesFromBlockRace,
          [[0.4, 0.25, 0.2, 0.15], clB, rank2, DB],
          "the block inverse refuses rank-2 too", want);

  // and the message points somewhere: the forward DOES price it
  accepts("the forward still prices the rank-2 field it refuses to differentiate",
          () => blocks.blockRaceProbabilities(muB, clB, rank2, DB),
          p2 => Math.abs(p2.reduce((a, b) => a + b, 0) - 1) < 1e-9
                && p2.every(v => v > 0),
          p2 => `[${p2.map(v => v.toFixed(5))}]`);

  // the rank-one spellings are untouched, INCLUDING the (n, 1) column
  // that only ever worked by javascript's number coercion
  accepts("an (n,1) column is the same jacobian as the flat vector",
          () => [blocks.blockRaceJacobian(muB, clB, column, DB),
                 blocks.blockRaceJacobian(muB, clB, flat, DB)],
          ([a, b2]) => JSON.stringify(a) === JSON.stringify(b2));
  accepts("the rank-one jacobian still follows its forward",
          () => {
            const J = blocks.blockRaceJacobian(muB, clB, flat, DB);
            const h = 1e-5;
            let worst = 0;
            for (let j = 0; j < 4; j++) {
              const a = muB.slice(), c = muB.slice();
              a[j] += h; c[j] -= h;
              const pa = blocks.blockRaceProbabilities(a, clB, flat, DB);
              const pc = blocks.blockRaceProbabilities(c, clB, flat, DB);
              for (let i = 0; i < 4; i++)
                worst = Math.max(worst, Math.abs(J[i][j] - (pa[i] - pc[i]) / (2 * h)));
            }
            return worst < 1e-6;
          });
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

// --- Mendell-Elston does not depend on the labels (#287)
// Sequential moment matching conditions one coordinate at a time and
// pretends the remainder is still normal, so the ORDER decides the
// answer. This applied it in raw contestant order while python's
// _order_variables sorts hardest-first, so permuting a four-runner
// race, evaluating, and undoing the permutation moved a share by 3.9
// percentage points.
{
  const muME = [-0.5, 1.5, 1.0, -1.5];
  const CME = [[5.5, 1.0, 0.0, -2.0], [1.0, 1.5, 2.0, 2.0],
               [0.0, 2.0, 5.5, 6.0], [-2.0, 2.0, 6.0, 8.5]];
  const shares = (m, C) => {
    const raw = m.map((_, i) => demo.mendellElstonOne(m, C, i));
    const t = raw.reduce((a, b) => a + b, 0);
    return raw.map(x => x / t);
  };
  const permuted = (m, C, perm) => {
    const pp = shares(perm.map(i => m[i]), perm.map(i => perm.map(j => C[i][j])));
    const back = new Array(m.length);
    perm.forEach((old, now) => { back[old] = pp[now]; });
    return back;
  };
  const base = shares(muME, CME);

  for (const perm of [[3, 1, 0, 2], [1, 0, 3, 2], [2, 3, 1, 0], [3, 2, 1, 0]])
    accepts(`mendell-elston is invariant under [${perm}]`,
            () => permuted(muME, CME, perm),
            b => Math.max(...b.map((x, i) => Math.abs(x - base[i]))) < 1e-12,
            b => `max |diff| ${Math.max(...b.map((x, i) =>
              Math.abs(x - base[i]))).toExponential(2)}`);

  // and it is the PYTHON answer, not merely a self-consistent one:
  // winning.methods.orthant_extra.mendell_elston(-mu, B, D) for
  // B = [[1,-2],[1,0],[2,1],[2,2]], D = 0.5 (this port is min-wins)
  const REF = [0.4041262077118515, 0.0007992933998171157,
               0.0012330295143568694, 0.5938414693739745];
  accepts("mendell-elston matches the python reference",
          () => base,
          b => Math.max(...b.map((x, i) => Math.abs(x - REF[i]))) < 1e-12,
          b => `max |diff| ${Math.max(...b.map((x, i) =>
            Math.abs(x - REF[i]))).toExponential(2)}`);

  // --- the baselines are scale free (#101): (mu, C) -> (sqrt(c) mu, c C)
  // is the same race. The absolute pivot floors moved the GHK and ME
  // leaders by 0.266 at c = 1e-14, and cholesky(1e-12 I) returned
  // 1e-5 I (covariance 1e-10 I).
  const scaled = (c) => [muME.map(x => Math.sqrt(c) * x),
                         CME.map(r => r.map(x => c * x))];
  for (const c of [1e-10, 1e-14, 1e-16]) {
    const [m2, C2] = scaled(c);
    accepts(`mendell-elston is scale free at c = ${c}`,
            () => shares(m2, C2),
            b => Math.max(...b.map((x, i) => Math.abs(x - base[i]))) < 1e-9,
            b => `max |diff| ${Math.max(...b.map((x, i) =>
              Math.abs(x - base[i]))).toExponential(2)}`);
    accepts(`ghk preparation is scale free at c = ${c}`,
            () => [0, 1, 2, 3].map(i => [demo.ghkPrepareOne(muME, CME, i),
                                         demo.ghkPrepareOne(m2, C2, i)]),
            v => v.every(([a, b]) => a.L.every((x, k) =>
              Math.abs(b.L[k] / Math.sqrt(c) - x) <= 1e-9 * Math.abs(x) + 1e-300)));
  }
  // #409: the relative floor raised EVERY pivot below it, so near-twins
  // with variance 2^-43 (a valid small contrast) priced [0.5, 0.158,
  // 0.342] for the exact [0.49999992, 0.07932763, 0.42067245]
  {
    const delta = 2 ** -43, a = 0.5 * Math.sqrt(2 * delta);
    const muT = [0, a, -a];
    const CT = [[1, 0, 0], [0, 1 + delta, 1], [0, 1, 1 + delta]];
    accepts("ghk keeps a valid small pivot (#409)",
            () => {
              const raw = muT.map((_, i) => demo.ghkSampleOne(
                demo.ghkPrepareOne(muT, CT, i), 100000, demo.rng(1234 + i)));
              const t = raw.reduce((x, y) => x + y, 0);
              return raw.map(x => x / t);
            },
            p => Math.abs(p[1] - 0.07932763) < 3e-3 &&
                 Math.abs(p[2] - 0.42067245) < 3e-3,
            p => `shares ${p.map(x => x.toFixed(5))}`);
  }
  // #666: PSD in correlation units, at every scale
  for (const [C, label] of [[[[1, 0], [0, -1e-9]], "negative variance"],
                            [[[1, 2, 0], [2, 1, 0], [0, 0, 1e12]],
                             "correlation 2 beside a 1e12 variance"],
                            [[[0, 1e-309], [1e-309, 0]], "[[0,c],[c,0]] at 1e-309"]])
    rejects(demo.validateCovariance, [C],
            `validateCovariance refuses ${label} (#666)`, "positive semidefinite");
  accepts("validateCovariance keeps a valid heterogeneous-scale matrix (#666)",
          () => demo.validateCovariance([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1e12]]),
          S => S.length === 3, () => "refused");
  accepts("cholesky keeps the covariance scale at 1e-12 I",
          () => demo.cholesky([[1e-12, 0], [0, 1e-12]]),
          L => Math.abs(L[0][0] - 1e-6) < 1e-18 && Math.abs(L[1][1] - 1e-6) < 1e-18,
          L => `L00 ${L[0][0]}`);
}

{
  // Common-loading invariance, the property that says the engine is
  // pricing a race and not a coordinate system. Adding the same loading
  // to every contestant adds one common Gaussian shock, which cannot
  // move an argmin. Uncentered, it moved a share by 0.0141 and a
  // Jacobian entry by 0.0199 (#303). The fixture is the issue's, whose
  // loadings are FAR from centered -- a near-centered V would have
  // passed this test against the broken code.
  const gmu = [-0.8342822144132106, 1.0225266393018637, 1.1413101763571138,
    -0.22521847232046474, 0.465466912363422, -1.2022915735825428];
  const gv = [-4.4130421891327085, -4.259934330177284, -1.3890276719160926,
    -4.409152054880935, 3.770280643971603, -3.2829086131210348];
  const gD = [0.43056642860174177, 0.30260230067651717, 0.31256756619550286,
    0.24615555584896356, 0.29463348248973487, 0.5154752123868093];
  const V0 = gv.map(x => [x]);
  const shifted = c => gv.map(x => [x + c]);
  accepts("the race is invariant to a common loading",
          () => {
            const p0 = races.raceProbabilities(gmu, {V: V0, D: gD, points: 257});
            let worst = 0;
            for (const c of [1, -3, 100]) {
              const p = races.raceProbabilities(
                gmu, {V: shifted(c), D: gD, points: 257});
              worst = Math.max(worst, ...p0.map((x, i) => Math.abs(x - p[i])));
            }
            return worst;
          },
          w => w < 1e-12,
          w => `worst shift ${w.toExponential(2)}`);
  accepts("the Jacobian is invariant to a common loading",
          () => {
            const J0 = polish.raceJacobian(gmu, {V: V0, D: gD, points: 257});
            const J1 = polish.raceJacobian(gmu, {V: shifted(1), D: gD,
                                                 points: 257});
            return Math.max(...J0.flatMap(
              (row, i) => row.map((x, j) => Math.abs(x - J1[i][j]))));
          },
          w => w < 1e-12,
          w => `worst shift ${w.toExponential(2)}`);
  // and the fixture must MOVE the answer, or the invariance above is
  // just two ways of computing the independent race
  accepts("the fixture's loadings actually matter",
          () => {
            const withV = races.raceProbabilities(gmu, {V: V0, D: gD, points: 257});
            const without = races.raceProbabilities(gmu, {D: gD, points: 257});
            return Math.max(...withV.map((x, i) => Math.abs(x - without[i])));
          },
          w => w > 0.01,
          w => `loadings move the race by ${w.toFixed(4)}`);
  // a CONSTANT loading column is the independent race, gauge-fixed
  accepts("a constant loading column is the independent race",
          () => {
            const flat = races.raceProbabilities(
              gmu, {V: gmu.map(() => [2.5]), D: gD, points: 257});
            const ind = races.raceProbabilities(gmu, {D: gD, points: 257});
            return Math.max(...flat.map((x, i) => Math.abs(x - ind[i])));
          },
          w => w < 1e-12,
          w => `gap ${w.toExponential(2)}`);
}

// --- caller-supplied factor nodes carry the loadings' rank (#290)
// condMeans dotted each node row against the loadings over the ROW's
// own length, so the rank was whatever each row happened to be: a
// uniformly short F priced a LOWER-RANK model, a ragged F priced a
// different rank at each quadrature node, extra weights were ignored
// and too few produced NaN -- all of it finite and normalised.
{
  const muF = [-0.6, -0.2, 0.15, 0.7];
  const VF = [[1.2, -0.7], [-0.4, 1.1], [0.6, 0.9], [-1.0, -0.5]];
  const DF = [0.5, 0.8, 0.6, 0.9];
  const FF = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
  const WF = [0.25, 0.25, 0.25, 0.25];
  const fwd = o => races.raceProbabilities(muF, { V: VF, D: DF, ...o });

  rejects(fwd, [{ F: FF.map(r => [r[0]]), W: WF }],
          "forward refuses a uniformly short F", "rank 2");
  rejects(fwd, [{ F: [[-1, -1], [-1], [1, -1], [1, 1]], W: WF }],
          "forward refuses a ragged F", "coordinate");
  rejects(fwd, [{ F: FF, W: [...WF, 0.9] }],
          "forward refuses too many weights", "one weight per factor node");
  rejects(fwd, [{ F: FF, W: WF.slice(0, 3) }],
          "forward refuses too few weights", "one weight per factor node");
  rejects(fwd, [{ F: [], W: [] }], "forward refuses an empty F", "empty");
  rejects(fwd, [{ F: FF, W: [0.25, -0.25, 0.5, 0.5] }],
          "forward refuses a negative weight", "negative weight");

  // the jacobian door and the inverse share the contract
  rejects(polish.raceJacobian,
          [muF, { V: VF, D: DF, F: [[-1, -1], [-1], [1, -1], [1, 1]], W: WF }],
          "the jacobian refuses a ragged F too", "coordinate");
  rejects(races.abilitiesFromRace,
          [[0.4, 0.3, 0.2, 0.1], { V: VF, D: DF, F: FF, W: [0.5, 0.5] }],
          "the inverse refuses mismatched weights", "one weight per factor node");

  // and the spellings that ARE the same race agree
  const base = accepts("explicit nodes still price",
                       () => fwd({ F: FF, W: WF }),
                       p2 => Math.abs(p2.reduce((a, b) => a + b, 0) - 1) < 1e-12,
                       p2 => `[${p2.map(v => v.toFixed(5))}]`);
  accepts("W and c*W are the same law",
          () => fwd({ F: FF, W: WF.map(w => 10 * w) }),
          p2 => base && Math.max(...p2.map((v, i) => Math.abs(v - base[i]))) < 1e-15);
  accepts("the internally built nodes are unaffected",
          () => fwd({}),
          p2 => Math.abs(p2.reduce((a, b) => a + b, 0) - 1) < 1e-12);

  // The jacobian was scale-dependent too, which nobody had reported:
  // the same defect as #281 but in this tree rather than the
  // standalone js/factor module. On main J(W) and J(10W) differ by
  // 1.473; normalising W at the door makes them identical, and leaves
  // the already-normalised answer bit-identical.
  const jac = w => polish.raceJacobian(muF, { V: VF, D: DF, F: FF, W: w });
  const jbase = accepts("the jacobian prices with explicit nodes",
                        () => jac(WF),
                        J => J.length === 4 && J.every(r => r.length === 4));
  accepts("the jacobian is invariant to W -> cW",
          () => jac(WF.map(w => 10 * w)),
          J => jbase && Math.max(...J.map((r, i) =>
            Math.max(...r.map((v, j) => Math.abs(v - jbase[i][j]))))) === 0,
          J => jbase ? `max |diff| ${Math.max(...J.map((r, i) =>
            Math.max(...r.map((v, j) => Math.abs(v - jbase[i][j]))))).toExponential(2)}` : "");
}

// --- a top-k depth is a count, so it is an integer (#272's neighbour)
// Every guard truncated first -- Math.trunc(k) here, int(k) in python,
// as.integer in R -- and then range-checked the TRUNCATED value. So
// k=0, k=n and k>n were all refused and only a non-integer slipped
// through, silently floored: topKProbabilities(mu, 1.5) returned the
// top-1 curve with mass 1. The caller asked for a curve that does not
// exist and got a different one, and the mass they can check is 1, not
// the 1.5 they asked about.
{
  const muK = [-0.4, 0.1, 0.2, 0.5];
  const DK = [0.7, 0.8, 0.9, 1.0];
  const tkp = (k) => topk.topKProbabilities(muK, k, { D: DK });

  for (const k of [1, 2, 3])
    accepts(`topK depth ${k} prices`, () => tkp(k),
            p2 => Math.abs(p2.reduce((a, b) => a + b, 0) - k) < 1e-9,
            p2 => `mass ${p2.reduce((a, b) => a + b, 0).toFixed(6)}`);

  for (const k of [1.5, 2.5, 0.5, 2.0001])
    rejects(tkp, [k], `topK refuses the fractional depth ${k}`,
            "whole number of places");
  for (const k of [0, 4, 5, -1])
    rejects(tkp, [k], `topK refuses the depth ${k}`, "[1, n-1]");

  // the two depths around 1.5 are genuinely different curves, which is
  // why flooring it left nothing odd-looking to notice
  accepts("the neighbouring depths differ",
          () => [tkp(1), tkp(2)],
          ([a, b2]) => Math.max(...a.map((v, i) => Math.abs(v - b2[i]))) > 0.1);

  // the pair door checks BOTH depths
  {
    const q1 = tkp(1), q2 = tkp(2);
    const pair = (a, b2) => topk.locScaleFromTopkPair(q1, a, q2, b2);
    accepts("the pair door takes whole depths", () => pair(1, 2),
            r => r && (r.mu || r.sd || Array.isArray(r)));
    rejects(pair, [1.5, 2], "the pair door refuses a fractional k1",
            "k1 must be a whole number");
    rejects(pair, [1, 2.5], "the pair door refuses a fractional k2",
            "k2 must be a whole number");
    rejects(pair, [1, 1], "the pair door still refuses k1 == k2", "k1 == k2");
  }
}

// --- a lattice size is a whole number >= 2 at every modular door (#444)
{
  const mu = [-0.5, 0, 0.5];
  const doors = {
    raceProbabilities: p => races.raceProbabilities(mu, { points: p }),
    raceJacobian: p => polish.raceJacobian(mu, { points: p }),
    abilitiesFromRace: p => races.abilitiesFromRace([0.5, 0.3, 0.2], { points: p }),
    topKProbabilities: p => topk.topKProbabilities(mu, 1, { points: p }),
    rankProbabilities: p => topk.rankProbabilities(mu, { points: p }),
    blockRaceProbabilities: p => blocks.blockRaceProbabilities(mu, [0, 0, 1], [0.5, 0.5, 0.5], 1, { points: p }),
  };
  for (const [name, f] of Object.entries(doors))
    for (const p of [0, 1, true, null, -1, 2.5, "257", Infinity, NaN])
      rejects(f, [p], `${name} refuses points=${JSON.stringify(p)} (#444)`, "lattice");
  const p257 = races.raceProbabilities(mu, { points: 257 });
  const want = [0.5487437165504273, 0.3009257008897869, 0.15033058255978574];
  holds("points=257 still prices (#444)",
        Math.max(...p257.map((v, i) => Math.abs(v - want[i]))) < 1e-12);
}
// --- the classic input contract (#339, #377)
{
  const dV = classic.skewNormalDensity(50, 0.1);
  const offs = [-3, 0.5, 2];
  const p0 = classic.statePricesFromOffsets(dV, offs);
  accepts("raw counts price as their frequencies (#339)",
          () => classic.statePricesFromOffsets(dV.map(x => 10 * x), offs),
          q => q.every((x, i) => Math.abs(x - p0[i]) < 1e-15) &&
               Math.abs(q.reduce((a, b) => a + b, 0) - 1) < 1e-13,
          q => `[${q.map(x => x.toFixed(10)).join(", ")}]`);
  for (const [bad, why, msg] of [
    [[0, 0, 0, 0, 0, 0, 0], "zero mass", "no positive mass"],
    [[0.2, -0.1, 0.9, 0, 0, 0, 0], "a negative atom", "negative atom"],
    [[0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0], "even length", "odd length"],
    [[0.1, NaN, 0.9, 0, 0, 0, 0], "a NaN atom", "non-finite"],
    [[0.1, Infinity, 0.9, 0, 0, 0, 0], "an infinite atom", "non-finite"],
    [[0.25, 0.5, 0.25], "L = 1", "L >= 3"],
  ]) {
    rejects(classic.statePricesFromOffsets, [bad, [0, 0]],
            `statePricesFromOffsets refuses ${why} (#339)`, msg);
    rejects(classic.solveForImpliedOffsets, [[0.7, 0.3], bad],
            `solveForImpliedOffsets refuses ${why} (#339)`, msg);
  }
  const d7 = [0, 0, 0.25, 0.5, 0.25, 0, 0];
  const t7 = classic.statePricesFromOffsets(d7, [0, 0]);
  accepts("the smallest lattice round-trips (#339)",
          () => classic.solveForImpliedOffsets(t7, d7, { guess: [0, 0] }),
          a => a.every(Number.isFinite) &&
               classic.statePricesFromOffsets(d7, a).every((x, i) => Math.abs(x - t7[i]) < 1e-12),
          a => `[${a.join(", ")}]`);
  const dS = classic.skewNormalDensity(500, 0.01, { a: 1.5 });
  const tgt = [0.4, 0.3, 0.2, 0.1];
  const centre = a => { const m = a.reduce((x, y) => x + y, 0) / a.length; return a.map(x => x - m); };
  const a1 = centre(classic.solveForImpliedOffsets(tgt, dS, { guess: [0, 0, 0, 0] }));
  for (const c of [1.1, 2, 10])
    accepts(`the inverse is invariant to target scale ${c} (#377)`,
            () => centre(classic.solveForImpliedOffsets(tgt.map(x => c * x), dS,
                                                        { guess: [0, 0, 0, 0] })),
            a => a.every((x, i) => Math.abs(x - a1[i]) < 1e-12),
            a => `[${a.map(x => x.toFixed(6)).join(", ")}]`);
  rejects(classic.solveForImpliedOffsets, [[0.5, -0.1, 0.6], dS],
          "the inverse refuses a negative target (#377)", "negative");
  rejects(classic.solveForImpliedOffsets, [[0, 0], dS],
          "the inverse refuses a zero-mass target (#377)", "no positive mass");
  rejects(classic.solveForImpliedOffsets, [[0.5, NaN], dS],
          "the inverse refuses a NaN target (#377)", "non-finite");
}

// --- exact dead heats (#418, #362, #348, #373, #369)
// Closed forms or brute-force enumeration of every joint atom outcome
// with a tied minimum split equally; python's
// tests/test_classic_exact_dead_heats.py checks the same fixtures.
{
  const close = (a, b, tol) => a.length === b.length &&
    a.every((x, i) => Math.abs(x - b[i]) < tol);
  const fmt = p => `[${Array.from(p, x => x.toFixed(10)).join(", ")}]`;
  const three = Array(17).fill(0); three[7] = 0.2; three[8] = 0.3; three[9] = 0.5;
  accepts("a compact-support pair is exact (#418)",
          () => classic.statePricesFromOffsets(three, [-1, 1]),
          p => close(p, [0.95, 0.05], 1e-14), fmt);
  accepts("disjoint support pays the certain winner (#418)",
          () => classic.statePricesFromOffsets(three, [-2, 2]),
          p => close(p, [1, 0], 1e-14), fmt);
  const d3 = [0, 0, 0, 0, 0.25, 0.5, 0.25, 0, 0, 0, 0];
  accepts("twenty iid entrants split the claim equally (#362)",
          () => classic.statePricesFromOffsets(d3, Array(20).fill(0)),
          p => close(p, Array(20).fill(0.05), 1e-14),
          p => `sum ${p.reduce((a, b) => a + b, 0)}`);
  accepts("unequal eight-runner field matches enumeration (#362)",
          () => classic.statePricesFromOffsets(d3, [0, -1, 1, 0, 0, 0, 2, 0]),
          p => close(p, [0.0918114072, 0.5380292620, 0.0029137021, 0.0918114072,
                         0.0918114072, 0.0918114072, 0, 0.0918114072], 1e-10), fmt);
  let g = Array.from({ length: 83 }, (_, i) => Math.exp(-0.5 * (i - 41) ** 2));
  const gz = g.reduce((a, b) => a + b, 0); g = g.map(x => x / gz);
  accepts("identical fractional runners are exchangeable (#348)",
          () => classic.statePricesFromOffsets(g, [0.5, 0.5]),
          p => close(p, [0.5, 0.5], 1e-15), fmt);
  accepts("a fractional three-runner field is exhaustive (#348)",
          () => classic.statePricesFromOffsets(g, [-2.5, 0.5, 3.5]),
          p => Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-13, fmt);
  const u = Array(83).fill(1 / 83);
  for (const k of [10, 19, 39, -19, 19.5])
    accepts(`a translated runner keeps its mass at ${k} (#373)`,
            () => [classic.statePricesFromOffsets(u, [k]),
                   classic.statePricesFromOffsets(u, [k, k])],
            ([s, pr]) => Math.abs(s[0] - 1) < 1e-14 && close(pr, [0.5, 0.5], 1e-14),
            ([s, pr]) => `${s[0]} ${fmt(pr)}`);
  const dS = classic.skewNormalDensity(15, 0.2, { a: 1.5 });
  const tS = classic.statePricesFromOffsets(dS, [6, -2]);
  accepts("the default guess is one zero per price (#369)",
          () => [classic.solveForImpliedOffsets(tS, dS),
                 classic.solveForImpliedOffsets(tS, dS, { guess: [0, 0] })],
          ([a, b]) => close(a, b, 1e-15), ([a, b]) => `${fmt(a)} vs ${fmt(b)}`);
  rejects(() => classic.solveForImpliedOffsets(tS, dS, { guess: [0, 1, 2, 3, 4] }),
          [], "a guess of the wrong length is refused (#369)",
          "one starting offset per price");
  const tC = classic.statePricesFromOffsets(three, [-1.5, 0, 0.7, 2]);
  accepts("the inverse converges to the exact forward map",
          () => classic.statePricesFromOffsets(three,
            classic.solveForImpliedOffsets(tC, three, { guess: [0, 0, 0, 0], nIter: 10 })),
          p => close(p, tC, 1e-6), fmt);
}

// --- state prices stay exhaustive at boundary offsets (#292)
// statePricesFromOffsets builds the field with shiftedCdf, which goes
// through lowHigh and PINS any offset at or past L-2 to the boundary.
// implicitPrices took a fast path for exact integers and called
// integerShift(baseCdf, k) with the RAW k, whose own clamp is the much
// wider +/-(m-1). A runner could sit in the field at offset L-2 and be
// PAID at 60, so the prices summed to 1.8835 -- while an epsilon off
// the integer gave 0.99999, because the non-integer path had been
// clamped correctly all along.
{
  const dB = classic.skewNormalDensity(50, 0.1);
  const ORD = 1;   // exhaustive; 0.99998675 was the old engine's loss (#418, #362)
  const total = a => classic.statePricesFromOffsets(dB, [a, -a])
    .reduce((x, y) => x + y, 0);

  for (const a of [48, 48.5, 49, 49.000001, 50, 60, 60.1, 100, 1000])
    accepts(`state prices sum to one at offset ${a}`, () => total(a),
            t => Math.abs(t - ORD) < 5e-9,
            t => `sum ${t.toFixed(10)}`);

  for (const k of [49, 60, -49, -60])
    accepts(`offset ${k} is continuous in its neighbourhood`,
            () => [total(k), total(k - 1e-6), total(k + 1e-6)],
            ([a, b2, c]) => Math.abs(a - b2) < 5e-9 && Math.abs(a - c) < 5e-9,
            ([a, b2, c]) => `${a.toFixed(9)} / ${b2.toFixed(9)} / ${c.toFixed(9)}`);

  accepts("past the clamp every offset is the same distribution",
          () => [classic.statePricesFromOffsets(dB, [60, -60]),
                 classic.statePricesFromOffsets(dB, [80, -80])],
          ([a, b2]) => Math.max(...a.map((v, i) => Math.abs(v - b2[i]))) < 1e-15);

  accepts("interior integers are untouched",
          () => [0, 1, 5, 20, 40, 47, -40].map(k =>
            classic.statePricesFromOffsets(dB, [k, -k, k / 3])
              .reduce((x, y) => x + y, 0)),
          ts => ts.every(t => Math.abs(t - 1) < 1e-3),
          ts => `worst |sum - 1| ${Math.max(...ts.map(t => Math.abs(t - 1))).toExponential(2)}`);
}

{
  // The node-rule counts. k = 0 is the EMPTY PRODUCT -- one node of
  // weight 1 with no columns -- which the browser already fell through
  // to while python returned the RANK-1 rule and R raised. Pin it on
  // the side that was right, so it cannot drift to the other two.
  const core = await eng("core.mjs");
  accepts("k = 0 is one node with no columns",
          () => core.hermiteNodes(0, 5),
          h => h.F.length === 1 && h.F[0].length === 0
               && Math.abs(h.W[0] - 1) < 1e-12,
          h => `rows ${h.F.length} cols ${h.F[0].length} W ${h.W}`);
  accepts("k = 1 is still the rank-one rule",
          () => core.hermiteNodes(1, 5),
          h => h.F.length === 5 && h.F[0].length === 1);
  const mu = [0, 0.3, -0.2, 0.5], D = [1, 1, 1, 1];
  accepts("a zero-rank loading matrix prices the independent race",
          () => races.raceProbabilities(mu, { V: [[], [], [], []], D }),
          p => {
            const q = races.raceProbabilities(mu, { D });
            return p.every((v, i) => Math.abs(v - q[i]) < 1e-9);
          },
          p => `[${p.map(v => v.toFixed(6))}]`);
  for (const k of [-1, 1.5, NaN])
    rejects(core.hermiteNodes, [k, 5], `k = ${k} is refused`, "k");
  // order 0 built a rule with NO NODES, which normalises 0/0
  for (const o of [0, -3, 2.5])
    rejects(core.hermiteNodes, [1, o], `order = ${o} is refused`, "order");
}

{
  // Rank zero reaches top-k through its OWN node builder, which
  // special-cased rank 1 and sent everything else through a rank-2
  // tensor -- so an (n, 0) matrix became "RangeError: Invalid array
  // length" (#309). The race path was fixed in #68; this one was not.
  const zmu = [0, 0.3, -0.2, 0.5], zD = [1, 1, 1, 1];
  const Z = [[], [], [], []];
  const same = (a, b) => Math.max(...a.flat(9).map(
    (x, i) => Math.abs(x - b.flat(9)[i])));
  accepts("zero-rank top-k is the independent top-k",
          () => same(topk.topKProbabilities(zmu, 2, { V: Z, D: zD }),
                     topk.topKProbabilities(zmu, 2, { D: zD })),
          w => w < 1e-12, w => `gap ${w.toExponential(2)}`);
  accepts("zero-rank bottom-k is the independent bottom-k",
          () => same(topk.bottomKProbabilities(zmu, 2, { V: Z, D: zD }),
                     topk.bottomKProbabilities(zmu, 2, { D: zD })),
          w => w < 1e-12, w => `gap ${w.toExponential(2)}`);
  accepts("zero-rank rank marginals are the independent ones",
          () => same(topk.rankProbabilities(zmu, { V: Z, D: zD }),
                     topk.rankProbabilities(zmu, { D: zD })),
          w => w < 1e-12, w => `gap ${w.toExponential(2)}`);
  // finite weights whose raw SUM overflows are the same law (#415):
  // asWeights divided by that total, [1e308, 1e308] -> [0, 0] -> NaN
  accepts("asWeights normalises [1e308, 1e308] to [0.5, 0.5]",
          () => core.asWeights([1e308, 1e308], 2),
          w => w[0] === 0.5 && w[1] === 0.5);
  {
    const wo = { V: [[-0.7], [0.7]], D: [1, 1], F: [[-1], [1]], points: 1001 };
    accepts("raceProbabilities with W = [1e308, 1e308] prices like [1, 1]",
            () => same(races.raceProbabilities([0, 0.4], { ...wo, W: [1e308, 1e308] }),
                       races.raceProbabilities([0, 0.4], { ...wo, W: [1, 1] })),
            w => w < 1e-15, w => `gap ${w.toExponential(2)}`);
    accepts("raceJacobian with W = [1e308, 1e308] is the [1, 1] Jacobian",
            () => same(polish.raceJacobian([0, 0.4], { ...wo, W: [1e308, 1e308] }),
                       polish.raceJacobian([0, 0.4], { ...wo, W: [1, 1] })),
            w => w < 1e-15, w => `gap ${w.toExponential(2)}`);
  }
  // the exported Jacobian and its polishRace caller: the independent
  // default node was the RANK-ONE [[0]], so (n, 0) loadings read
  // V[i][0] = undefined and returned an all-NaN Jacobian (#68)
  accepts("zero-rank race Jacobian is the independent Jacobian",
          () => same(polish.raceJacobian(zmu, { V: Z, D: zD }),
                     polish.raceJacobian(zmu, { D: zD })),
          w => w < 1e-12, w => `gap ${w.toExponential(2)}`);
  accepts("zero-rank race Jacobian is finite",
          () => polish.raceJacobian(zmu, { V: Z, D: zD }),
          J => J.flat().every(Number.isFinite));
  accepts("a zero rank-one column still gives the independent Jacobian",
          () => same(polish.raceJacobian(zmu, { V: [[0], [0], [0], [0]], D: zD }),
                     polish.raceJacobian(zmu, { D: zD })),
          w => w < 1e-12, w => `gap ${w.toExponential(2)}`);
  accepts("zero-rank polishRace matches its independent spelling",
          () => {
            const opts = { p0: [0.1, 0.1, 0.8], D: [1, 1, 1],
                           nameCaps: [0.5, 0.5, 0.5], points: 129 };
            const a = polish.polishRace({ ...opts, V: [[], [], []] });
            const b = polish.polishRace(opts);
            return [a, b];
          },
          ([a, b]) => {
            const pa = a.p || a.probabilities || a.prices;
            const pb = b.p || b.probabilities || b.prices;
            return Array.isArray(pa) && pa.every(Number.isFinite) &&
              Math.max(...pa.map((x, i) => Math.abs(x - pb[i]))) < 1e-9 &&
              Math.max(...pa) <= 0.5 + 1e-6;
          });
  // and a rank-one loading must still MOVE top-k, or the three above
  // are just two ways of computing the independent answer
  accepts("a rank-one loading still moves top-k",
          () => same(topk.topKProbabilities(
                       zmu, 2, { V: [[0.9], [-0.4], [0.2], [-0.7]], D: zD }),
                     topk.topKProbabilities(zmu, 2, { D: zD })),
          w => w > 0.005, w => `moves by ${w.toFixed(4)}`);
}

{
  // structureCov must describe the race the kernels PRICE, whatever the
  // cluster labels are called. It read labels as tree-node indices, so
  // relabelling 0,1,2 -> 10,20,30 sent every lookup past the end of
  // `parent`: an empty ancestor set, every shared factor silently gone,
  // and a covariance 0.52 out. Negative and string labels threw (#306).
  const tmu = [-0.5, -0.1, 0.2, 0.4], tD = [0.8, 0.9, 1.0, 1.1];
  const tload = [0.2, 0.3, -0.1, 0.4];
  const tparent = [3, 3, 4, 4, -1], tstrength = [0, 0, 0, 0.6, 0.4];
  const covFor = cl => demo.structureCov({
    kind: "Tree", cluster: cl, loading: tload, D: tD,
    parent: tparent, strength: tstrength });
  const REFC = covFor([0, 0, 1, 2]);
  for (const cl of [[10, 10, 20, 30], [-5, -5, 0, 7], ["a", "a", "b", "c"]])
    accepts(`structureCov is the same under ${JSON.stringify(cl)}`,
            () => Math.max(...covFor(cl).flat().map(
              (x, i) => Math.abs(x - REFC.flat()[i]))),
            w => w < 1e-12, w => `gap ${w.toExponential(2)}`);
  // the labels must actually be non-trivial: a tree whose ancestors
  // contribute nothing would pass the above however it was indexed
  accepts("the fixture's ancestor factors are not zero",
          () => {
            const flat = demo.structureCov({
              kind: "Tree", cluster: [0, 0, 1, 2], loading: tload, D: tD,
              parent: tparent, strength: [0, 0, 0, 0, 0] });
            return Math.max(...REFC.flat().map(
              (x, i) => Math.abs(x - flat.flat()[i])));
          },
          w => w > 0.1, w => `ancestors contribute ${w.toFixed(4)}`);
}

// #430: Tree.from_linkage leaf variances are 2 h^2 EXACTLY -- the shared
// fixture python and R read -- and a merge at height 0 is refused
{
  const { readFileSync } = await import("node:fs");
  const fx = JSON.parse(readFileSync(
    here + "../tests/golden/linkage_leaf_variance.json", "utf8"));
  for (const c of fx.cases) {
    const t = structures.treeFromLinkage(c.Z);
    const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
    let worst = 0;
    c.D.forEach((v, i) => { worst = Math.max(worst, rel(t.D[i], v)); });
    for (let j = c.D.length; j < c.strength.length; j++)
      worst = Math.max(worst, rel(t.strength[j], c.strength[j]));
    holds(`treeFromLinkage keeps 2h^2 (${c.name})`, worst <= fx.rtol,
          `max rel err ${worst.toExponential(2)}`);
  }
  rejects(structures.treeFromLinkage, [fx.refuse_zero_height],
          "treeFromLinkage refuses a zero-height merge", "height 0");
}

// #350: the tree Jacobian is the derivative of the tree forward; a common
// root shock cancels from both
{
  const exact = Math.exp(-0.5 * 0.8 * 0.8 / 1.85) / Math.sqrt(2 * Math.PI * 1.85);
  for (const r of [0, 0.8, 2]) {
    const J = blocks.treeRaceJacobian([-0.4, 0.4], [0, 1], [0.3, 0.4], [0.7, 0.9],
                                      [2, 2, -1], [0, 0, r]);
    holds(`treeRaceJacobian is root-invariant (strength ${r})`,
          Math.abs(J[0][1] - exact) < 1e-8, `J01 ${J[0][1]}`);
  }
}

// factor-core top-k batch, browser parity with python/R
{
  const mu = [-.5, .2, .8, -.1], sd = [.7, 1.1, .9, 1.3];
  const a = topk.topKProbabilities(mu, 2, { D: sd.map(v => v * v) });
  const b = topk.topKProbabilities(mu.map(v => v * 1e-18), 2,
                                   { D: sd.map(v => (v * 1e-18) ** 2) });
  holds("top-k window is scale-free (#370)",
        Math.max(...a.map((x, i) => Math.abs(x - b[i]))) < 1e-9);
  const bk = topk.bottomKProbabilities([-12, 0, 0], 1, { D: [1, 1, 1] });
  holds("bottom-k keeps a 7.7e-24 last place (#365)",
        Math.abs(bk[0] / 7.726967753938468796e-24 - 1) < 1e-6, `${bk[0]}`);
  const warn = console.warn; console.warn = () => {};
  const ls = topk.locScaleFromTopkPair([.8, .1, .1], 1, [.2, .9, .9], 2,
                                       { points: 1025, ridge: 1e-4, returnInfo: true });
  console.warn = warn;
  holds("a ridge stall on an impossible board is not convergence (#353)",
        !ls.info.converged && !ls.info.nested);
  rejects(topk.abilitiesFromRankMarginal, [[.4, .3, .2, .1], 1.5],
          "fractional rank refused (#317)", "whole-number");
  const P = topk.rankProbabilities([-1, -.4, .05, .45, .9],
                                   { D: [1, 1, 1, 1, 1], points: 257 }).map(r => r[2]);
  rejects(topk.abilitiesFromRankMarginal, [P, 3, { D: [1, 1, 1, 1, 1], points: 257 }],
          "odd-field middle rank needs mu0 (#378)", "mu0");
  const m = [-3, -1, 1, 3], D = [4, 9, 16, 25];
  const q1 = topk.topKProbabilities(m, 1, { D, points: 1025 });
  const q2 = topk.topKProbabilities(m, 2, { D, points: 1025 });
  const w = topk.locScaleFromTopkPair(q1, 1, q2, 2, { D0: D, mu0: m, points: 1025, returnInfo: true });
  const gm = Math.exp(w.sd.reduce((x, v) => x + Math.log(v), 0) / 4);
  holds("exact warm start returns the canonical gauge (#360)", Math.abs(gm - 1) < 1e-12);
  const q = topk.topKProbabilities([-.8, -.3, 0, .4, .7], 2);
  const f = topk.abilitiesFromTopk(q, 2, { D: new Array(5).fill(1e-6), returnInfo: true });
  holds("abilitiesFromTopk is unit-equivariant (#100)",
        f.info.converged && Math.abs(f.mu[0] / 1e-3 + 0.8) < 1e-6);
  rejects(topk.abilitiesFromTopk, [[.5, NaN, .5], 1], "non-finite top-k target refused (#110)",
          "not a finite");
}

// factor-core race batch: #263 weights, #110 targets, #89 structured
// controls, #100 structured inverses in the field's units
{
  const mu = [0, 0.3, 1], V = [[0], [1], [-0.5]], F = [[-1], [1]];
  const a = races.raceProbabilities(mu, { V, D: [1, 1, 1], F, W: [1e307, 1e307] });
  const b = races.raceProbabilities(mu, { V, D: [1, 1, 1], F, W: [1e308, 1e308] });
  holds("weights normalise without overflow (#263)",
        Math.max(...a.map((x, i) => Math.abs(x - b[i]))) < 1e-12);
  rejects(races.abilitiesFromRace, [[0.8, NaN], { returnInfo: true }],
          "NaN target refused (#110)", "not a finite");
  rejects(races.abilitiesFromRace, [[Infinity, 1], { returnInfo: true }],
          "Infinite target refused (#110)", "not a finite");
  const pr = races.abilitiesFromRace([0.9999999999999999, 1e-20], { D: [1, 1], returnInfo: true });
  holds("pair shortcut uses the longshot tail (#110)",
        pr.mu.every(Number.isFinite) && Math.abs(pr.mu[1] - pr.mu[0] - 13.0989269761827) < 1e-6,
        `${pr.mu}`);
  const muI = [-1.88, 3.24, -0.27, 0.063], DI = [0.0557, 1.70, 14.8, 3.60];
  const d1 = races.raceProbabilities(muI, { D: DI, points: 65, window: "span" });
  const d2 = races.raceProbabilities(muI, { structure: structures.Independent(DI), points: 65, window: "span" });
  holds("structured window reaches the race (#89)",
        Math.max(...d1.map((x, i) => Math.abs(x - d2[i]))) === 0);
  const bl = structures.Blocks([0, 0, 1, 1], [0.3, 0.2, 0.4, 0.1], [1, 1, 1, 1]);
  rejects(races.raceProbabilities, [[-0.4, -0.1, 0.2, 0.5], { structure: bl, base: "logistic" }],
          "structured hierarchical base refused (#89)", "base");
  rejects(races.abilitiesFromRace, [[0.4, 0.3, 0.2, 0.1], { structure: bl, base: "logistic" }],
          "structured hierarchical inverse base refused (#89)", "base");
  rejects(races.raceProbabilities, [[-0.4, -0.1, 0.2, 0.5], { structure: bl, D: [1, 1, 1, 1] }],
          "structure plus D refused (#89)", "structure");
  const tr = structures.Tree([0, 0, 1, 1], [4, 3.2, 3.6, 2.8], [0.1, 0.12, 0.08, 0.15],
                             [2, 2, -1], [0, 0, 0.9]);
  const pt = races.raceProbabilities([-1.5, -0.2, 0.3, 1.4], { structure: tr, points: 257, qa: 9 });
  const it = races.abilitiesFromRace(pt, { structure: tr, points: 257, qa: 9, nIter: 3,
                                           returnInfo: true });
  holds("structured inverse honours nIter and reports convergence (#89)",
        it.iterations <= 3 && it.converged === false);
  const target = [0.6, 0.25, 0.1, 0.05], cl = [0, 0, 1, 1];
  const L0 = [0.9, 0.6, 0.8, 0.3], D0 = [0.19, 0.64, 0.36, 0.91];
  let worst = 0;
  for (const c of [1e-6, 1e-3, 1e3, 1e6]) {
    for (const s of [structures.Blocks(cl, L0.map(x => c * x), D0.map(x => c * c * x)),
                     structures.Nested(cl, L0.map(x => c * x), D0.map(x => c * c * x),
                                       [[0.2 * c], [-0.1 * c], [0.3 * c], [-0.2 * c]], 0.7)]) {
      const m = races.abilitiesFromRace(target, { structure: s });
      const back = races.raceProbabilities(m, { structure: s });
      worst = Math.max(worst, ...back.map((x, i) => Math.abs(x - target[i])));
    }
  }
  holds("structured inverses are unit-equivariant (#100)", worst < 1e-8,
        `worst share error ${worst.toExponential(2)}`);
}

// #340: top-k factor nodes follow the race's sharpness rule
{
  const mu = [-0.15, 0.05, 0.10, 0.0], V = [[-3], [-1], [1], [3]], D = [0.01, 0.01, 0.01, 0.01];
  const race = races.raceProbabilities(mu, { V, D, points: 1025 });
  const top1 = topk.topKProbabilities(mu, 1, { V, D, points: 1025 });
  const R = topk.rankProbabilities(mu, { V, D, points: 1025 }).map(r => r[0]);
  holds("sharp rank-one top-1 equals the race (#340)",
        Math.max(...race.map((x, i) => Math.abs(x - top1[i]))) < 1e-6 &&
        Math.max(...race.map((x, i) => Math.abs(x - R[i]))) < 1e-6);
}


// --- quadrature orders are counts: "15" from a form control used to be
// new Array("15") -- ONE node -- and priced a different model (#442)
{
  const muQ = [0.67, 0.76, -1.62, -0.01], VQ = [[-0.65], [0.69], [-1.36], [-1.30]];
  for (const bad of ["15", true, null, 0, 1.5, NaN, Infinity])
    rejects(topk.topKProbabilities, [muQ, 1, { V: VQ, qa: bad, points: 129 }],
            `topKProbabilities refuses qa=${JSON.stringify(bad)}`);
  rejects(blocks.blockRaceProbabilities,
          [[-0.4, -0.1, 0.2, 0.5], [0, 0, 1, 1], [1.4, 0.7, 1.1, 0.3], [0.2, 0.3, 0.25, 0.4],
           { qa: "9", points: 129 }], "blockRaceProbabilities refuses qa='9'");
  rejects(blocks.nestedRaceProbabilities,
          [[-0.4, -0.1, 0.2, 0.5], [0, 0, 1, 1], [1.4, 0.7, 1.1, 0.3], [0.2, 0.3, 0.25, 0.4],
           { coupling: [[-0.7], [0.4], [-0.2], [0.6]], qf: "15", points: 129 }],
          "nestedRaceProbabilities refuses qf='15'");
  rejects(core.hermite1, ["15"], "hermite1 refuses a string order");
}

// --- classic: offsets and warm starts are finite numbers (#438); typed
// vectors are the same input as plain ones (#329); nIter is a finite
// positive integer (#401)
{
  const dens = classic.skewNormalDensity(50, 0.1);
  for (const bad of [NaN, undefined, "bad", Infinity, null, true])
    rejects(classic.statePricesFromOffsets, [dens, [0, bad]],
            `statePricesFromOffsets refuses offset ${String(bad)}`);
  rejects(classic.solveForImpliedOffsets, [[0.8, 0.2], dens, { guess: [0, NaN] }],
          "solveForImpliedOffsets refuses a NaN guess");
  const plain = classic.statePricesFromOffsets(dens, [-2, 0, 2]);
  accepts("typed offsets price the same race (#329)",
          () => classic.statePricesFromOffsets(dens, new Float64Array([-2, 0, 2])),
          v => Math.max(...v.map((x, i) => Math.abs(x - plain[i]))) === 0);
  const inv = classic.solveForImpliedOffsets([0.5, 0.3, 0.2], dens, { guess: [0, 0, 0] });
  accepts("typed prices and guess invert the same (#329)",
          () => classic.solveForImpliedOffsets(new Float64Array([0.5, 0.3, 0.2]), dens,
                                               { guess: new Float64Array([0, 0, 0]) }),
          v => Math.max(...v.map((x, i) => Math.abs(x - inv[i]))) === 0);
  const div = classic.dividendImpliedAbility([2, 4, 8], dens);
  accepts("dividendImpliedAbility takes a typed book (#329)",
          () => classic.dividendImpliedAbility(new Float64Array([2, 4, 8]), dens),
          v => Math.max(...v.map((x, i) => Math.abs(x - div[i]))) === 0);
  for (const bad of [0, -1, 0.5, NaN, Infinity, "3", true, null])
    rejects(classic.solveForImpliedOffsets,
            [[0.6, 0.3, 0.1], dens, { guess: [0, 0, 0], nIter: bad }],
            `solveForImpliedOffsets refuses nIter=${JSON.stringify(bad)}`, "nIter");
  accepts("solveForImpliedOffsets keeps integer budgets",
          () => classic.solveForImpliedOffsets([0.6, 0.3, 0.1], dens, { guess: [0, 0, 0], nIter: 2 }),
          v => v.every(Number.isFinite));
}

// --- top-k/rank doors: finite nonempty abilities (#440), typed vectors
// (#334), finite positive budgets (#413), the effective-pair damping
// python got for #151 (#408), and D0 through asIdio (#254)
{
  const D3 = [1, 1, 1];
  for (const bad of [[0, NaN, 1], [0, Infinity, 1], [0, -Infinity, 1]]) {
    rejects(topk.topKProbabilities, [bad, 1, { D: D3, points: 65 }], "topKProbabilities refuses non-finite mu", "not finite");
    rejects(topk.bottomKProbabilities, [bad, 1, { D: D3, points: 65 }], "bottomKProbabilities refuses non-finite mu", "not finite");
    rejects(topk.topKJacobians, [bad, 1, { D: D3, points: 65 }], "topKJacobians refuses non-finite mu", "not finite");
    rejects(topk.rankProbabilities, [bad, { D: D3, points: 65 }], "rankProbabilities refuses non-finite mu", "not finite");
  }
  rejects(topk.rankProbabilities, [[], { D: [], points: 65 }], "rankProbabilities refuses an empty field", "empty");
  const mu = [-0.5, 0, 0.5], D = [0.7, 1, 1.3], V = [-0.4, 0.1, 0.6];
  const ref = topk.topKProbabilities(mu, 2, { D, V });
  accepts("typed mu/D/V give the same top-k (#334)",
          () => topk.topKProbabilities(new Float64Array(mu), 2,
                  { D: new Float64Array(D), V: new Float64Array(V) }),
          v => Math.max(...v.map((x, i) => Math.abs(x - ref[i]))) === 0);
  for (const bad of [0, -1, 0.5, NaN, Infinity, "80", true])
    rejects(topk.abilitiesFromTopk, [[0.5, 0.3, 0.2], 1, { nIter: bad }],
            `abilitiesFromTopk refuses nIter=${JSON.stringify(bad)}`, "nIter");
  for (const [n, eps] of [[3, 6e-4], [10, 1e-4]]) {
    const p = [0.6, 0.4, ...Array(n - 2).fill(eps)];
    const t = p.reduce((a, b) => a + b, 0);
    const pp = p.map(v => v / t);
    const Dn = Array(n).fill(3.2 ** 2);
    accepts(`abilitiesFromTopk converges on #151's effective pair, n=${n} (#408)`,
            () => topk.abilitiesFromTopk(pp, 1, { D: Dn, points: 500, returnInfo: true }),
            o => o.info.converged && o.info.maxLogitResidual < 1e-8,
            o => `residual ${o.info.maxLogitResidual.toExponential(2)}`);
  }
  const q1 = topk.topKProbabilities([-1.1, -0.2, 0.4, 0.9], 1, { D: [0.5, 0.8, 1.2, 1.5], points: 513 });
  const q2 = topk.topKProbabilities([-1.1, -0.2, 0.4, 0.9], 2, { D: [0.5, 0.8, 1.2, 1.5], points: 513 });
  for (const bad of [0, NaN, -1, [1, 1, 1]])
    rejects(topk.locScaleFromTopkPair, [q1, 1, q2, 2, { D0: bad, points: 513, nIter: 2 }],
            `locScaleFromTopkPair refuses D0=${JSON.stringify(bad)} (#254)`);
  const a1 = topk.locScaleFromTopkPair(q1, 1, q2, 2, { D0: [1, 1, 1, 1], points: 513, nIter: 3, returnInfo: true });
  accepts("a scalar D0 is the broadcast vector (#254)",
          () => topk.locScaleFromTopkPair(q1, 1, q2, 2, { D0: 1, points: 513, nIter: 3, returnInfo: true }),
          o => Math.max(...o.mu.map((x, i) => Math.abs(x - a1.mu[i]))) === 0);
}

// --- block/tree doors: loadings through asLoadings (#335), tree topology
// validated before any parent walk (#328), finite positive maxIter and a
// no-progress break (#421), caller nodes validated (#290)
{
  const mu = [-0.8, -0.1, 0.3, 0.9], cl = [0, 0, 1, 1], D = [0.7, 1.0, 0.8, 1.1];
  const v = [0.8, 0.4, 0.3, -0.2];
  const VV = [[0.8, 0.2], [0.4, -0.5], [0.3, 0.7], [-0.2, 0.4]];
  const VT = [[0.8, 0.4, 0.3, -0.2], [0.2, -0.5, 0.7, 0.4]];
  const flat = blocks.blockRaceProbabilities(mu, cl, v, D);
  accepts("a (1, n) rank-one loading is the flat one (#335)",
          () => blocks.blockRaceProbabilities(mu, cl, [v], D),
          p => Math.max(...p.map((x, i) => Math.abs(x - flat[i]))) < 1e-15);
  const Jf = blocks.blockRaceJacobian(mu, cl, v, D);
  accepts("a (1, n) rank-one loading has the flat Jacobian (#335)",
          () => blocks.blockRaceJacobian(mu, cl, [v], D),
          J => Math.max(...J.flat().map((x, i) => Math.abs(x - Jf.flat()[i]))) < 1e-15);
  const r2 = blocks.blockRaceProbabilities(mu, cl, VV, D);
  accepts("an (r, n) loading is the (n, r) one (#335)",
          () => blocks.blockRaceProbabilities(mu, cl, VT, D),
          p => Math.max(...p.map((x, i) => Math.abs(x - r2[i]))) < 1e-15);
  rejects(blocks.blockRaceProbabilities, [mu, cl, [[0.8, 0.2], [0.4], [0.3, 0.7], [-0.2, 0.4]], D],
          "a ragged block loading is refused, not zero-padded (#335)", "ragged");
  rejects(blocks.nestedRaceProbabilities, [mu, cl, [[0.8, 0.2], [0.4], [0.3, 0.7], [-0.2, 0.4]], D,
          { coupling: [0.2, -0.1, 0.3, 0.0] }], "nested refuses a ragged loading (#335)", "ragged");
  // a cycle used to spin forever; each of these must come back at once
  const tr = (parent, strength, label, expect) =>
    rejects(blocks.treeRaceProbabilities, [[0, 0], [0, 1], [0, 0], [1, 1], parent, strength,
            { points: 33, qa: 3 }], label, expect);
  tr([1, 0], [0.2, 0.2], "a two-node parent cycle is refused (#328)");
  tr([2, 2, 2], [0, 0, 0.3], "a self-parent is refused (#328)", "cycle");
  tr([2, 2, -1, 3], [0, 0, 0.3, 0.1], "a disconnected self-loop is refused (#328)", "cycle");
  tr([2, 2, 5], [0, 0, 0.3], "an out-of-range parent is refused (#328)");
  tr([2, 2, -1], [0, 0], "parent/strength length mismatch is refused (#328)", "strength");
  tr([2, -1, -1], [0, 0, 0.3], "two roots are refused (#328)", "one root");
  rejects(blocks.treeRaceJacobian, [[0, 0], [0, 1], [0, 0], [1, 1], [1, 0], [0.2, 0.2], { points: 33, qa: 3 }],
          "treeRaceJacobian refuses the cycle too (#328)");
  accepts("a valid tree still prices", () => blocks.treeRaceProbabilities(
            [0, 0.3], [0, 1], [0.2, 0.2], [1, 1], [2, 2, -1], [0, 0, 0.5], { points: 129, qa: 7 }),
          p => Math.abs(p[0] + p[1] - 1) < 1e-12 && p[0] > p[1]);
  const pT = [0.6, 0.25, 0.1, 0.05], clB = [0, 0, 1, 1], ldB = [0.9, 0.6, 0.8, 0.3], DB = [0.19, 0.64, 0.36, 0.91];
  for (const bad of [0, -1, 0.5, NaN, Infinity])
    rejects(blocks.abilitiesFromBlockRace, [pT, clB, ldB, DB, { maxIter: bad, tol: Number.MIN_VALUE }],
            `abilitiesFromBlockRace refuses maxIter=${bad} (#421)`, "maxIter");
  accepts("abilitiesFromBlockRace reports the Newton steps it took (#421)",
          () => blocks.abilitiesFromBlockRace(pT, clB, ldB, DB),
          o => o.iterations <= 5 && o.residual < 1e-10, o => `${o.iterations} steps`);
  accepts("a sub-machine tol stops when no step can be accepted (#421)",
          () => blocks.abilitiesFromBlockRace(pT, clB, ldB, DB, { maxIter: 1000, tol: Number.MIN_VALUE }),
          o => o.iterations < 1000, o => `${o.iterations} steps`);
  for (const w of [[1.01, -0.01], [0.5, 0.5, 100], [0.5, NaN]])
    rejects(blocks.blockRaceProbabilities, [mu, cl, v, D, { points: 257, nodes: { nodes: [[-1], [1]], w } }],
            `caller block nodes refuse w=${JSON.stringify(w)} (#290)`);
}

// --- the winner race: typed vectors (#334), whole factor rules (#290),
// the target contract before structure dispatch (#387), honest inverse
// diagnostics (#354), the pair shortcut only for the automatic rule
// (#374), one factor-rule chooser for forward and Jacobian (#325), a
// Factor structure keeps the caller's rule (#209), scalar D in the
// Jacobian (#254), the base-aware bulk window (#380), a KKT stop in
// polishRace (#379), typed name caps (#334)
{
  const mu = [-0.5, 0, 0.5], D = [0.7, 1, 1.3], V = [-0.4, 0.1, 0.6];
  const ref = races.raceProbabilities(mu, { D, V });
  accepts("typed mu/D/V price the same race (#334)",
          () => races.raceProbabilities(new Float64Array(mu), { D: new Float64Array(D), V: new Float64Array(V) }),
          p => Math.max(...p.map((x, i) => Math.abs(x - ref[i]))) === 0);
  const inv = races.abilitiesFromRace([0.5, 0.3, 0.2], { D });
  accepts("a typed target inverts the same (#334)",
          () => races.abilitiesFromRace(new Float64Array([0.5, 0.3, 0.2]), { D }),
          m => Math.max(...m.map((x, i) => Math.abs(x - inv[i]))) === 0);

  const V2 = [[1], [0.2], [-0.8]];
  rejects(races.raceProbabilities, [[-0.4, 0.2, 0.8], { V: V2, D: [1, 1, 1], F: [[1000]] }],
          "a lone F is refused, not replaced (#290)", "BOTH");
  rejects(races.raceProbabilities, [[-0.4, 0.2, 0.8], { V: V2, D: [1, 1, 1], W: [12345] }],
          "a lone W is refused, not replaced (#290)", "BOTH");
  rejects(polish.raceJacobian, [[-0.4, 0.2, 0.8], { V: V2, D: [1, 1, 1], F: [[1]] }],
          "raceJacobian refuses a lone F (#290)", "BOTH");
  rejects(races.abilitiesFromRace, [[0.8, 0.2], { V: [[1], [-1]], D: [1, 1], F: [[-1], [1]], W: [0.25, 0.25, 0.5] }],
          "the pair inverse validates W before any shortcut (#290)", "one weight per");

  const nested = structures.Nested([0, 0, 1], [0.2, 0.3, 0.4], [1, 1, 1], [0.1, 0.2, 0.3], 0.5);
  for (const s of [structures.Independent([1, 1, 1]), nested]) {
    rejects(races.abilitiesFromRace, [[0.6, 0.4, 0], { structure: s, points: 129, qa: 7, qf: 7 }],
            `${s.kind}: a zero share is refused before dispatch (#387)`, "positive");
    rejects(races.abilitiesFromRace, [[0.6, 0.5, -0.1], { structure: s, points: 129, qa: 7, qf: 7 }],
            `${s.kind}: a negative share is refused (#387)`, "positive");
    accepts(`${s.kind}: targetFloor and returnInfo reach the grammar (#387)`,
            () => races.abilitiesFromRace([0.6, 0.4, 0], { structure: s, points: 129, qa: 7, qf: 7,
                                                           targetFloor: 1e-4, returnInfo: true }),
            o => o.floored[2] === true && !o.floored[0] && o.converged && o.mu.every(Number.isFinite));
  }

  // #354: a starved solve warns; iterations are the ones actually run
  const warned = [];
  const ow = console.warn;
  console.warn = (...x) => warned.push(x.join(" "));
  races.abilitiesFromRace([0.9, 0.09, 0.01], { nIter: 1, tol: 1e-12, points: 513 });
  console.warn = ow;
  holds("a starved inverse warns (#354)", warned.some(w => w.includes("did not converge")));
  accepts("a uniform target reports one iteration (#354)",
          () => races.abilitiesFromRace([1 / 3, 1 / 3, 1 / 3], { points: 513, returnInfo: true }),
          o => o.iterations === 1 && o.converged);
  accepts("an ordinary target reports fewer than nIter (#354)",
          () => races.abilitiesFromRace([0.5, 0.3, 0.2], { points: 513, returnInfo: true }),
          o => o.iterations < 60 && o.converged, o => `${o.iterations}`);
  for (const bad of [0, -1, 0.5, NaN, Infinity])
    rejects(races.abilitiesFromRace, [[0.5, 0.3, 0.2], { nIter: bad }],
            `abilitiesFromRace refuses nIter=${bad}`, "nIter");

  // #374: a translated Hermite rule and a two-point law, both through
  // the actual forward rather than the Gaussian pair closed form
  const { F: hF, W: hW } = core.hermiteNodes(1, 31);
  for (const [label, opts, truth] of [
    ["translated Hermite", { V: [[0], [1]], D: [1, 1], F: hF.map(([x]) => [x + 1]), W: hW, points: 257 }, [0, 1]],
    ["two-point law", { V: [[0], [1]], D: [0.05, 0.05], F: [[-1], [1]], W: [0.5, 0.5], points: 1025 }, [-0.4, 0.4]],
  ]) {
    const target = races.raceProbabilities(truth, opts);
    accepts(`the pair inverse reprices a ${label} rule (#374)`,
            () => races.abilitiesFromRace(target, { ...opts, returnInfo: true }),
            o => {
              const back = races.raceProbabilities(o.mu, opts);
              return o.converged && Math.max(...back.map((x, i) => Math.abs(x - target[i]))) < 1e-7;
            });
  }

  // #325: the Jacobian differentiates the forward's own rule, on both
  // sides of the rank-one midpoint handoff
  const fdGap = (m, opts) => {
    const J = polish.raceJacobian(m, opts);
    let worst = 0;
    for (let j = 0; j < m.length; j++) {
      const a = m.slice(), b = m.slice(); a[j] += 1e-5; b[j] -= 1e-5;
      const pa = races.raceProbabilities(a, opts), pb = races.raceProbabilities(b, opts);
      for (let i = 0; i < m.length; i++)
        worst = Math.max(worst, Math.abs(J[i][j] - (pa[i] - pb[i]) / 2e-5));
    }
    return worst;
  };
  for (const Dk of [[0.01, 0.01, 0.01], [0.5, 0.5, 0.5]])
    accepts(`raceJacobian matches the forward's rule at D=${Dk[0]} (#325)`,
            () => fdGap([-0.2, 0, 0.3], { V: [[-1], [0], [1]], D: Dk, points: 513 }),
            w => w < 1e-6, w => w.toExponential(2));
  accepts("raceJacobian matches the forward's escalated rank-2 rule (#325)",
          () => fdGap([-0.2, 0, 0.3], { V: [[-1, 0.5], [0, -1], [1, 0.4]], D: [0.02, 0.02, 0.02], points: 257 }),
          w => w < 1e-5, w => w.toExponential(2));

  // #209: the Factor structure keeps the caller's rule in forward,
  // Jacobian and inverse, and a rule's scale is irrelevant
  const fo = { F: [[-3], [3]], W: [0.5, 0.5], points: 257 };
  const sF = structures.Factor([[-1], [0], [1]], [1, 1, 1]);
  const direct = races.raceProbabilities([0, 0.4, 1], { V: [[-1], [0], [1]], D: [1, 1, 1], ...fo });
  accepts("structure: Factor prices with the caller's F/W (#209)",
          () => races.raceProbabilities([0, 0.4, 1], { structure: sF, ...fo }),
          p => Math.max(...p.map((x, i) => Math.abs(x - direct[i]))) === 0);
  accepts("the structured Factor Jacobian is the derivative of its forward (#209)",
          () => fdGap([0, 0.4, 1], { structure: sF, ...fo }), w => w < 1e-6, w => w.toExponential(2));
  const J1 = polish.raceJacobian([0, 0.4, 1], { V: [[-1], [0], [1]], D: [1, 1, 1], ...fo });
  accepts("raceJacobian is invariant to W -> cW (#209)",
          () => polish.raceJacobian([0, 0.4, 1], { V: [[-1], [0], [1]], D: [1, 1, 1], ...fo, W: [5, 5] }),
          J => Math.max(...J.flat().map((x, i) => Math.abs(x - J1.flat()[i]))) < 1e-15);
  rejects(races.raceProbabilities, [[0, 0.4, 1], { structure: sF, V: [[1], [0], [-1]] }],
          "structure= and V= together are refused", "not both");
  rejects(races.raceProbabilities, [[0, 0.4, 1], { structure: structures.Blocks([0, 0, 1], [0.5, 0.5, 0.5], [1, 1, 1]), ...fo }],
          "a Blocks structure refuses F/W it would ignore");

  // #254: the Jacobian's D goes through asIdio like the forward's
  const Jv = polish.raceJacobian([-0.4, 0, 0.4], { D: [1, 1, 1] });
  accepts("raceJacobian broadcasts a scalar D (#254)",
          () => polish.raceJacobian([-0.4, 0, 0.4], { D: 1 }),
          J => Math.max(...J.flat().map((x, i) => Math.abs(x - Jv.flat()[i]))) === 0);

  // #380: a callable base's own tail sets the bulk window; Student-t(3)
  // at 4001 points was stuck 1.05e-3 TV from the improper integral
  const n40 = 40;
  const m40 = Array.from({ length: n40 }, (_, i) => 1.7 * Math.sin(1.37 * i) + 0.7 * Math.cos(0.43 * i));
  const D40 = Array.from({ length: n40 }, (_, i) => Math.exp(1.1 * Math.sin(0.91 * i) - 0.2 * Math.cos(0.37 * i)));
  const st3 = z => [Math.max(0.5 - (Math.atan(z) + z / (1 + z * z)) / Math.PI, 1e-300),
                    2 / (Math.PI * (1 + z * z) ** 2), -8 * z / (Math.PI * (1 + z * z) ** 3)];
  const owarn = console.warn; console.warn = () => {};
  const p40 = races.raceProbabilities(m40, { D: D40, base: st3, points: 4001 });
  console.warn = owarn;
  holds("a Student-t(3) base gets a Student-t window (#380)",
        Math.abs(p40[8] - 0.22863979140410515) < 2e-6 && Math.abs(p40[22] - 0.21438380857304878) < 2e-6,
        `${p40[8]} ${p40[22]}`);

  // #379: the nearest feasible race, not the first feasible iterate
  accepts("polishRace stops at a KKT point, cap binding (#379)",
          () => polish.polishRace({ mu0: [-1.5, -1.5, 3], D: [1, 1, 1], groups: [[[0, 1], 0.9]], points: 257 }),
          o => Math.abs(o.p[0] + o.p[1] - 0.9) < 1e-6 && o.info.active.includes(0)
               && Math.abs(o.info.muDistance - 2.7827022862) < 1e-5,
          o => `group ${(o.p[0] + o.p[1]).toFixed(6)}, distance ${o.info.muDistance}`);
  // #334: typed name caps are caps, not "no cap"
  const capsP = polish.polishRace({ p0: [0.1, 0.1, 0.8], D: [1, 1, 1], nameCaps: [0.4, 0.4, 0.4], points: 257 });
  accepts("typed nameCaps constrain like plain ones (#334)",
          () => polish.polishRace({ p0: [0.1, 0.1, 0.8], D: [1, 1, 1], nameCaps: new Float64Array([0.4, 0.4, 0.4]), points: 257 }),
          o => o.info.active.length === 1 && Math.max(...o.p.map((x, i) => Math.abs(x - capsP.p[i]))) < 1e-12);
  const cm = polish.concentrationMatrix(3, { nameCaps: new Float64Array([0.4, 0.4, 0.4]) });
  holds("concentrationMatrix reads typed caps (#334)", cm.A.length === 3 && cm.b.every(v => v === 0.4));
}

// --- linkage helpers: a distance matrix is symmetric (#361); maxclust
// cuts at a distance level and keeps tied merges together (#346)
{
  const A = [[0, 0.1, 0.5, 0.5], [0.5, 0, 0.5, 0.5], [0.1, 0.5, 0, 0.1], [0.5, 0.1, 0.5, 0]];
  const AT = A.map((_, i) => A.map(r => r[i]));
  rejects(structures.averageLinkage, [A], "averageLinkage refuses an asymmetric matrix (#361)", "symmetric");
  rejects(structures.averageLinkage, [AT], "...and its transpose (#361)", "symmetric");
  rejects(structures.averageLinkage, [[[0, 1], [1, 0], [1, 1]]], "a non-square matrix is refused (#361)", "square");
  rejects(structures.averageLinkage, [[[0, -1], [-1, 0]]], "a negative distance is refused (#361)", "negative");
  rejects(structures.averageLinkage, [[[1, 1], [1, 0]]], "a nonzero diagonal is refused (#361)", "diagonal");
  rejects(structures.averageLinkage, [[[0, NaN], [NaN, 0]]], "a NaN distance is refused (#361)", "finite");
  accepts("tolerance-level asymmetry is averaged (#361)",
          () => structures.averageLinkage([[0, 1, 4], [1 + 1e-12, 0, 2], [4, 2, 0]]),
          Z => Z.length === 2 && Z[0][0] === 0 && Z[0][1] === 1);
  const Z = [[0, 1, 0, 2], [2, 3, 0, 2], [4, 5, 1, 4]];
  const want = { 1: [0, 0, 0, 0], 2: [0, 0, 1, 1], 3: [0, 0, 1, 1], 4: [0, 1, 2, 3], 5: [0, 1, 2, 3] };
  for (const k of [1, 2, 3, 4, 5])
    holds(`cutLinkage matches scipy maxclust at k=${k} (#346)`,
          JSON.stringify(structures.cutLinkage(Z, 4, k)) === JSON.stringify(want[k]));
  const Zs = [[2, 3, 0, 2], [0, 1, 0, 2], [4, 5, 1, 4]];
  holds("swapping tied rows does not change the partition (#346)",
        JSON.stringify(structures.cutLinkage(Zs, 4, 3)) === JSON.stringify([0, 0, 1, 1]));
  const Ze = [[0, 1, 0.5, 2], [2, 3, 0.5, 2], [4, 5, 0.5, 4]];
  holds("an all-equal linkage is indivisible below n (#346)",
        JSON.stringify(structures.cutLinkage(Ze, 4, 3)) === JSON.stringify([0, 0, 0, 0]));
  for (const bad of [0, 1.5, NaN])
    rejects(structures.cutLinkage, [Z, 4, bad], `cutLinkage refuses k=${bad} (#346)`, "k must");
}

// --- cutLinkage refuses an inverted linkage (#667): scipy's maxclust on
// one is an artifact of binary-searching unsorted row heights (it gives
// [1,1,1,1] at k=3 here; the browser gave three clusters while claiming
// parity). R's cutree and treeFromLinkage (#133) refuse it too.
{
  rejects(structures.cutLinkage, [[[0, 1, 3, 2], [2, 3, 1, 2], [4, 5, 2, 4]], 4, 3],
          "cutLinkage refuses an inverted linkage (#667)", "not monotonic");
  rejects(structures.cutLinkage, [[[0, 1, 0.5, 2], [2, 3, 0.3, 3]], 3, 2],
          "cutLinkage refuses a parent below its child (#667)", "not monotonic");
  // computed average linkages are monotone up to round-off: never refused
  const g = demo.rng(667);
  let refused = 0;
  for (let rep = 0; rep < 50; rep++) {
    const m = 3 + (rep % 9);
    const P = Array.from({ length: m }, () => [g.n(), g.n(), g.n()]);
    const Dm = P.map(a => P.map(b => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])));
    const Zr = structures.averageLinkage(Dm);
    for (let k = 1; k <= m; k++) {
      try { structures.cutLinkage(Zr, m, k); } catch (e) { refused++; }
    }
  }
  holds("average linkages are never refused as inverted (#667)", refused === 0, String(refused));
}

// --- demo helpers: structureCov for every grammar (#337); fitGrammar
// validates its input (#357), keeps a pair's contrast and is scale
// equivariant (#341), and does not cut a tied eigenspace (#383); the
// comparison arms are scale- and common-mode-free (#375, #394); and
// Mendell-Elston is label-free on ties (#287)
{
  const dsum = (A, B) => Math.max(...A.flat().map((x, i) => Math.abs(x - B.flat()[i])));
  const Dd = [1, 4, 0.25];
  holds("structureCov(Independent) is diag(D) (#337)",
        dsum(demo.structureCov(structures.Independent(Dd)), Dd.map((d, i) => Dd.map((_, j) => (i === j ? d : 0)))) === 0);
  const nest = structures.Nested([0, 0, 1, 1], [0.5, 0.2, 0.4, -0.3], [0.8, 1.1, 0.9, 1.2], [0.6, -0.4, 0.3, -0.5], 0.7);
  const wantN = nest.D.map((d, i) => nest.D.map((_, j) => (i === j ? d : 0)
    + (nest.cluster[i] === nest.cluster[j] ? nest.loading[i] * nest.loading[j] : 0)
    + 0.49 * nest.coupling[i] * nest.coupling[j]));
  holds("structureCov(Nested) adds the coupling (#337)", dsum(demo.structureCov(nest), wantN) < 1e-15);
  const L2 = [[0.6, 0.2], [-0.2, 0.5], [0.4, -0.3], [-0.5, -0.1]], DB = [0.6, 0.7, 0.8, 0.9], clB = [0, 0, 1, 1];
  const wantB = DB.map((d, i) => DB.map((_, j) => (i === j ? d : 0)
    + (clB[i] === clB[j] ? L2[i][0] * L2[j][0] + L2[i][1] * L2[j][1] : 0)));
  holds("structureCov(Blocks) dots rank-r rows (#337)",
        dsum(demo.structureCov({ kind: "Blocks", cluster: clB, loading: L2, D: DB }), wantB) < 1e-15);
  const Vf = [1.4, -0.8, 0.9, -1.2], Df = [0.2, 0.3, 0.25, 0.35];
  holds("structureCov(Factor) reads a flat V as rank one (#337)",
        dsum(demo.structureCov(structures.Factor(Vf, Df)),
             Df.map((d, i) => Df.map((_, j) => Vf[i] * Vf[j] + (i === j ? d : 0)))) < 1e-15);
  rejects(demo.structureCov, [{ kind: "Factr", D: [1, 1] }], "structureCov refuses an unknown kind (#337)", "unknown");

  rejects(demo.fitGrammar, [[[1, 0.9, -0.6], [0, 1, 0.8], [0, 0, 1]], 2, 2], "fitGrammar refuses an asymmetric C (#357)", "symmetric");
  rejects(demo.fitGrammar, [[[1, 1.4, 0], [1.4, 1, 0], [0, 0, 1]], 2, 2], "fitGrammar refuses an indefinite C (#357)", "semidefinite");
  rejects(demo.fitGrammar, [[[1, 0], [0]]], "fitGrammar refuses a ragged C (#357)", "square");
  rejects(demo.fitGrammar, [[[1, NaN], [NaN, 1]]], "fitGrammar refuses NaN (#357)", "non-finite");
  for (const rho of [0, 0.9, 0.99, 0.999]) {
    const f = demo.fitGrammar([[1, rho], [rho, 1]]);
    const C2 = demo.structureCov(structures.Factor(f.V, f.D));
    holds(`fitGrammar keeps a pair's contrast at rho=${rho} (#341)`,
          Math.abs(C2[0][0] + C2[1][1] - 2 * C2[0][1] - 2 * (1 - rho)) < 1e-12);
  }
  const B = [[0.8, 0.1], [0.4, -0.3], [-0.2, 0.6], [-0.5, -0.1]], DD = [0.4, 0.7, 0.5, 0.6], m4 = [-0.5, -0.1, 0.2, 0.4];
  const C4 = B.map((bi, i) => B.map((bj, j) => bi[0] * bj[0] + bi[1] * bj[1] + (i === j ? DD[i] : 0)));
  const price = c => {
    const f = demo.fitGrammar(C4.map(r => r.map(x => c * x)));
    const q = demo.haltonNormalNodes(f.V[0].length, 4096);
    return races.raceProbabilities(m4.map(x => Math.sqrt(c) * x), { V: f.V, D: f.D, F: q.F, W: q.W, points: 513 });
  };
  const p1 = price(1);
  for (const c of [1e-2, 1e-4])
    accepts(`fitGrammar is scale equivariant at c=${c} (#341)`, () => price(c),
            p => Math.max(...p.map((x, i) => Math.abs(x - p1[i]))) < 1e-9);
  const n10 = 10;
  const Ce = Array.from({ length: n10 }, (_, i) => Array.from({ length: n10 }, (_, j) => (i === j ? 1 : 0.5)));
  const me = Array.from({ length: n10 }, (_, i) => 1.3 * Math.sin(1.7 * i) + 0.4 * Math.cos(0.31 * i));
  const perm = Array.from({ length: n10 }, (_, i) => (7 * i + 3) % n10);
  const owarn = console.warn; console.warn = () => {};
  const fe = demo.fitGrammar(Ce), fp = demo.fitGrammar(perm.map(i => perm.map(j => Ce[i][j])));
  console.warn = owarn;
  const o = races.raceProbabilities(me, { ...fe, points: 129 });
  const pp = races.raceProbabilities(perm.map(i => me[i]), { ...fp, points: 129 });
  const back = new Array(n10); perm.forEach((oi, ni) => { back[oi] = pp[ni]; });
  holds("fitGrammar of an exchangeable C is label free (#383)",
        Math.max(...o.map((x, i) => Math.abs(x - back[i]))) < 1e-12);

  // #505: opt-in iterated principal-axis factoring recovers an exact
  // v v' + diag(D) with heterogeneous D (the default eigen fit got its
  // decisive contrast 0.22 as 2.58); the default stays bit for bit.
  {
    const v5 = [0, 0, 0, 1, 0, 0.7], D5 = [18, 15, 12, 0.08, 6, 0.05];
    const C5 = v5.map((x, i) => v5.map((y, j) => x * y + (i === j ? D5[i] : 0)));
    const cmax = (A, B) => { let e = 0;
      for (let i = 0; i < A.length; i++) for (let j = 0; j < A.length; j++)
        e = Math.max(e, Math.abs((A[i][i] + A[j][j] - 2 * A[i][j]) - (B[i][i] + B[j][j] - 2 * B[i][j])));
      return e; };
    for (const [k, m] of [[1, 0], [3, 5]]) {
      const g = demo.fitGrammar(C5, k, m, { method: "ipaf" });
      holds(`fitGrammar ipaf recovers the #505 rank-one + diagonal at k=${k}, m=${m}`,
            g.ipaf.converged && cmax(C5, demo.structureCov({ kind: "Factor", V: g.V, D: g.D })) < 1e-10,
            JSON.stringify(g.ipaf));
      const d0 = demo.fitGrammar(C5, k, m);
      holds(`fitGrammar default is unchanged by the ipaf option (k=${k}, m=${m}; #505)`,
            JSON.stringify(d0) === JSON.stringify(demo.fitGrammar(C5, { k, m })) &&
            JSON.stringify(d0) === JSON.stringify(demo.fitGrammar(C5, k, m, { method: "eig" })) &&
            !("ipaf" in d0));
    }
    const g4 = demo.fitGrammar(C4, { k: 2, m: 0, method: "ipaf" });
    holds("fitGrammar ipaf recovers an ordinary rank-two + diagonal (#505)",
          g4.ipaf.converged && cmax(C4, demo.structureCov({ kind: "Factor", V: g4.V, D: g4.D })) < 1e-10);
    const ow2 = console.warn; let warned = 0; console.warn = () => { warned++; };
    const gc = demo.fitGrammar(C5, 1, 0, { method: "ipaf", maxIter: 2 });
    console.warn = ow2;
    holds("fitGrammar ipaf honours maxIter and reports non-convergence (#505)",
          !gc.ipaf.converged && gc.ipaf.iterations <= 4 && warned === 1, JSON.stringify(gc.ipaf));
    rejects(demo.fitGrammar, [C5, 1, 0, { method: "paf" }], "fitGrammar refuses an unknown method (#505)", "method");
    rejects(demo.fitGrammar, [C5, 1, 0, { method: "ipaf", maxIter: 0 }], "fitGrammar refuses maxIter < 1 (#505)", "maxIter");
  }

  // frequency MC in the race quotient: a huge common mode costs nothing
  const a = 2e15, Cc = [[a + 1, a], [a, a + 1]];
  const Lc = demo.raceFactor(Cc);
  holds("raceFactor keeps the contrast variance under a common mode (#394)",
        Math.abs(Lc[1][1] ** 2 - 2) < 1e-12, `${Lc[1][1] ** 2}`);
  const cnt = [0, 0];
  demo.mcBatch([0, 1], Lc, demo.rng(123), 200000, cnt);
  holds("frequency MC converges to Phi(1/sqrt 2) under a common mode (#394)",
        Math.abs(cnt[0] / 200000 - 0.7602499389065233) < 4e-3);
  // tiny-scale race: every arm unchanged by the units (#375)
  for (const d of [1e-14]) {
    const s = Math.sqrt(d), mu3 = [-s, 0, s];
    const C3 = [0, 1, 2].map(i => [0, 1, 2].map(j => 1 + (i === j ? d : 0)));
    const exact = [0.7287510153, 0.2240983048, 0.0471506799];
    let mE = mu3.map((_, i) => demo.mendellElstonOne(mu3, C3, i));
    const t = mE.reduce((x, y) => x + y, 0); mE = mE.map(x => x / t);
    holds("Mendell-Elston is unit free (#375)", Math.max(...mE.map((x, i) => Math.abs(x - exact[i]))) < 5e-3);
    const g = demo.rng(7);
    let gh = mu3.map((_, i) => demo.ghkSampleOne(demo.ghkPrepareOne(mu3, C3, i), 20000, g));
    const tg = gh.reduce((x, y) => x + y, 0); gh = gh.map(x => x / tg);
    holds("GHK is unit free (#375)", Math.max(...gh.map((x, i) => Math.abs(x - exact[i]))) < 5e-3);
    const c3 = [0, 0, 0];
    demo.mcBatch(mu3, demo.raceFactor(C3), demo.rng(9), 200000, c3);
    holds("frequency MC is unit free (#375)", Math.max(...c3.map((x, i) => Math.abs(x / 200000 - exact[i]))) < 5e-3);
  }
  // Mendell-Elston: relabelling relabels, ties included
  const meP = (m, C) => { const r = m.map((_, i) => demo.mendellElstonOne(m, C, i)); const t = r.reduce((x, y) => x + y, 0); return r.map(x => x / t); };
  const relabel = (m, C, pm) => {
    const p = meP(m, C), q = meP(pm.map(i => m[i]), pm.map(i => pm.map(j => C[i][j])));
    const b = new Array(m.length); pm.forEach((oi, ni) => { b[oi] = q[ni]; });
    return Math.max(...p.map((x, i) => Math.abs(x - b[i])));
  };
  const BB = [[1, -2], [1, 0], [2, 1], [2, 2]];
  const CB = BB.map((x, i) => BB.map((y, j) => x[0] * y[0] + x[1] * y[1] + (i === j ? 0.5 : 0)));
  for (const pm of [[3, 1, 0, 2], [1, 0, 3, 2], [2, 3, 1, 0]])
    holds(`Mendell-Elston with tied abilities is label free, perm ${pm} (#287)`,
          relabel([0, 0, 0, 0], CB, pm) < 1e-14);
}

// --- converge.html's lattice reference runs in a worker (#647): the
// default rank-2 Factor fixture really does take the 8,192-node rule (so
// it must stay off the main thread), and the worker's reference is the
// ordinary race on the field the page builds from (generator, n, seed)
{
  const cw = await import(here + "../docs/js/converge_worker.mjs");
  const dflt = Object.keys(demo.GENERATORS)[0];
  const { g } = cw.demoField(dflt, 60, 7);
  holds("the default converge fixture reports its 8,192-node reference (#647)",
        cw.referenceNodes(g) === 8192, String(cw.referenceNodes(g)));
  const { g: gb, mu: mb } = cw.demoField("sectors (blocks)", 12, 3);
  accepts("the worker's reference is the page's race (#647)",
          () => cw.latticeReference("sectors (blocks)", 12, 3),
          r => Number.isFinite(r.tLat)
               && Math.max(...r.pLat.map((x, i) => Math.abs(x - races.raceProbabilities(
                    mb, { structure: gb.structure, points: 257 })[i]))) === 0);
  rejects(cw.demoField, ["factr", 10, 1], "the worker refuses an unknown generator (#647)", "unknown generator");
}

// --- callable-base central resolution reaches every lattice (#468, #600)
{
  const student3 = z => {
    const q = 1 + z * z, f = 2 / (Math.PI * q * q);
    return [1 - (0.5 + (Math.atan(z) + z / q) / Math.PI), f, -8 * z / (Math.PI * q * q * q)];
  };
  student3.resolution = 1 / Math.sqrt(3);
  const quad = 0.6881209595067086;        // P(X0 < X1), direct quadrature
  accepts("Student-t(3) top-k prices at the default budget (#468)",
          () => topk.topKProbabilities([0, 0.5], 1, { D: [1, 1], base: student3 }),
          q => Math.abs(q[0] - quad) < 1e-7, q => `${q[0]}`);
  accepts("Student-t(3) bottom-k is the reflection (#468)",
          () => topk.bottomKProbabilities([0, 0.5], 1, { D: [1, 1], base: student3 }),
          q => Math.abs(q[1] - quad) < 1e-7, q => `${q[1]}`);
  // python's rank_probabilities returns the same matrix to 1e-8 (row
  // 0: 0.68814385, 0.31185615 since the dyadic point count, #515): the
  // rank route is 2.3e-5 off quadrature
  accepts("Student-t(3) rank matrix prices at the default budget (#468)",
          () => topk.rankProbabilities([0, 0.5], { D: [1, 1], base: student3 }),
          P => Math.abs(P[0][0] - 0.68814385) < 1e-7 && Math.abs(P[0][0] + P[1][0] - 1) < 1e-4,
          P => `${P[0][0]}`);
  accepts("Student-t(3) top-k Jacobian is finite (#468)",
          () => topk.topKJacobians([0, 0.5], 1, { D: [1, 1], base: student3 }),
          J => J.Jmu.flat().every(Number.isFinite));
  // a unit-variance mixture whose centre has scale 0.1
  const w = 0.99, s = 0.1, t = Math.sqrt((1 - w * s * s) / (1 - w));
  const narrow = z => {
    const ps = core.npdf(z / s), pt = core.npdf(z / t);
    return [Math.max(w * core.ndtr(-z / s) + (1 - w) * core.ndtr(-z / t), 1e-300),
            w * ps / s + (1 - w) * pt / t,
            -w * z * ps / s ** 3 - (1 - w) * z * pt / t ** 3];
  };
  narrow.span = [30, 30];
  narrow.resolution = s;
  const exact = w * w * core.ndtr(-0.2 / Math.sqrt(2 * s * s)) +
    2 * w * (1 - w) * core.ndtr(-0.2 / Math.sqrt(s * s + t * t)) +
    (1 - w) ** 2 * core.ndtr(-0.2 / Math.sqrt(2 * t * t));
  accepts("race honours base.resolution at 257 points (python: 0.0887283) (#600)",
          () => races.raceProbabilities([0, 0.2], { D: [1, 1], base: narrow }),
          p => Math.abs(p[1] - 0.08872825429) < 1e-8, p => `${p[1]}`);
  accepts("race reaches the analytic mixture by 2001 points (#600)",
          () => races.raceProbabilities([0, 0.2], { D: [1, 1], base: narrow, points: 2001 }),
          p => Math.abs(p[1] - exact) < 1e-8);
  accepts("top-k and bottom-k honour base.resolution at the default budget (#468, #600)",
          () => [topk.topKProbabilities([0, 0.2], 1, { D: [1, 1], base: narrow }),
                 topk.bottomKProbabilities([0, 0.2], 1, { D: [1, 1], base: narrow })],
          ([a, b]) => Math.abs(a[1] - exact) < 1e-8 && Math.abs(b[0] - exact) < 1e-8);
  const bad = z => narrow(z);
  bad.resolution = "0.1";
  rejects(races.raceProbabilities, [[0, 0.2], { D: [1, 1], base: bad }],
          "a malformed base.resolution is refused (#600)", "resolution");
  rejects(topk.topKProbabilities, [[0, 0.2], 1, { D: [1, 1], base: bad }],
          "top-k refuses a malformed base.resolution (#600)", "resolution");
}

// --- polishRace certifies a solved KKT point, without the FD re-solve (#547)
{
  const o = polish.polishRace({ mu0: [-1, 0, 1], D: [1, 1, 1], points: 129,
                                nameCaps: [0.55, NaN, NaN], fdFallback: false });
  const pyMu = [-0.6167863985024332, -0.3124412243042889, 0.9292276228067221];
  holds("a binding name cap reports converged (#547)", o.info.converged === true);
  holds("...with the right active set and no violation (#547)",
        o.info.active.length === 1 && o.info.active[0] === 0 && o.info.maxViolation <= 1e-8);
  holds("...at python's SLSQP optimum (#547)",
        Math.max(...o.mu.map((v, i) => Math.abs(v - pyMu[i]))) < 1e-7 && Math.abs(o.p[0] - 0.55) < 1e-8);
}

// --- polishRace is equivariant under a change of ability units (#593)
for (const c of [1e-8, 100]) {
  const o = polish.polishRace({ mu0: [-1.5 * c, -1.5 * c, 3 * c], D: [c * c, c * c, c * c],
                                groups: [[[0, 1], 0.9]], points: 257 });
  holds(`polishRace at units x${c} is the unit-scale optimum (#593)`,
        o.info.converged && Math.abs(o.p[0] + o.p[1] - 0.9) < 1e-7 &&
        Math.abs(o.mu[2] / c - 0.7279331) < 1e-5, `${o.mu.map(x => x / c)}`);
}

// --- a constraint row means the same thing at any positive scale (#653)
{
  const base = { mu0: [-1.5, -1.5, 3], D: [1, 1, 1], points: 257 };
  const ref = polish.polishRace({ ...base, A: [[1, 1, 0]], b: [0.9] });
  const refL = polish.polishRace({ ...base, A: [[0, 0, -1]], b: [-0.2] });
  for (const c of [1e-12, 1e-8, 1, 1e8]) {
    const o = polish.polishRace({ ...base, A: [[c, c, 0]], b: [0.9 * c] });
    holds(`a cap row scaled by ${c} is the unit-row optimum (#653)`,
          o.info.converged && o.info.active.length === 1 && o.info.active[0] === 0 &&
          Math.abs(o.p[0] + o.p[1] - 0.9) < 1e-7 &&
          Math.max(...o.mu.map((v, i) => Math.abs(v - ref.mu[i]))) < 1e-6,
          `${o.p} ${o.info.converged} [${o.info.active}]`);
    const l = polish.polishRace({ ...base, A: [[0, 0, -c]], b: [-0.2 * c] });
    holds(`a lower-bound row scaled by ${c} is the unit-row optimum (#653)`,
          l.info.converged && l.info.active.length === 1 && Math.abs(l.p[2] - 0.2) < 1e-7 &&
          Math.max(...l.mu.map((v, i) => Math.abs(v - refL.mu[i]))) < 1e-6,
          `${l.p} ${l.info.converged} [${l.info.active}]`);
  }
  const r = polish.polishRace({ ...base, A: [[1, 1, 0], [1e-8, 1e-8, 0]], b: [0.9, 9e-9] });
  holds("a redundant scaled copy of a row changes nothing (#653)",
        r.info.converged && Math.abs(r.p[0] + r.p[1] - 0.9) < 1e-7 &&
        Math.max(...r.mu.map((v, i) => Math.abs(v - ref.mu[i]))) < 1e-6, `${r.p}`);
}

// --- a saturated capped pair lands on the exact boundary, not the
// opposite deterministic winner (#683). Exact: mu = -+Phi^-1(0.9)/sqrt 2.
{
  const xs = 0.9061938024368232;
  // starting longshot shares 2.1e-12 (above the old 1e-12 cut), 7.7e-13
  // (below it) and 6.2e-16 (the issue's fixture)
  for (const a of [4.9, 5, 8 / Math.SQRT2])
    for (const points of [129, 257, 1025])
      for (const fdFallback of [false, true]) {
        const o = polish.polishRace({ mu0: [-a, a], D: [1, 1], nameCaps: [0.9, NaN],
                                      points, fdFallback });
        holds(`saturated pair a=${a.toFixed(3)} points=${points} fd=${fdFallback} is the exact cap (#683)`,
              o.info.converged && o.info.active.length === 1 && o.info.active[0] === 0 &&
              Math.abs(o.p[0] - 0.9) < 1e-8 && Math.abs(o.mu[0] + xs) < 1e-8 &&
              Math.abs(o.mu[1] - xs) < 1e-8, `${o.p} ${o.mu} ${o.info.converged}`);
      }
}

// --- a structured Jacobian refuses a base its forward refuses (#89)
for (const s of [structures.Blocks([0, 0, 1], 0.8, 0.2),
                 structures.Tree([0, 0, 1], 0.8, 0.2, [2, 2, -1], [0, 0, 0.5])])
  rejects(polish.raceJacobian, [[0, 0.3, 0.8], { structure: s, base: "logistic" }],
          `${s.kind} raceJacobian refuses base "logistic" like its forward (#89)`, "support base");

// --- typed controls are not coerced, and falsy is not "omitted"
{
  const q = [0.6, 0.4, 0];
  for (const tf of ["0.1", true, Infinity, NaN])
    rejects(topk.abilitiesFromTopk, [q, 1, { D: [1, 1, 1], points: 257, targetFloor: tf }],
            `top-k targetFloor ${String(tf)} is refused (#525)`, "targetFloor");
  const mu = [0, 0.3, 0.8], cl = [0, 0, 1], ld = [0.4, 0.6, 0.5], D = [1, 1, 1];
  const cp = [0.7, -0.2, 0.4];
  for (const g of ["2", true, false, null, NaN]) {
    rejects(blocks.nestedRaceJacobian, [mu, cl, ld, D, { coupling: cp, gamma: g, points: 129 }],
            `nested Jacobian refuses gamma ${String(g)} (#540)`, "gamma");
    rejects(polish.raceJacobian, [mu, { structure: structures.Nested(cl, ld, D, cp, g), points: 129 }],
            `structured nested Jacobian refuses gamma ${String(g)} (#540)`, "gamma");
  }
  for (const c of [NaN, false, ""]) {
    rejects(blocks.nestedRaceProbabilities, [mu, cl, ld, D, { coupling: c, points: 129 }],
            `nested forward refuses coupling ${JSON.stringify(c)} (#586)`, "coupling");
    rejects(blocks.nestedRaceJacobian, [mu, cl, ld, D, { coupling: c, points: 129 }],
            `nested Jacobian refuses coupling ${JSON.stringify(c)} (#586)`, "coupling");
    rejects(races.raceProbabilities, [mu, { structure: structures.Nested(cl, ld, D, c), points: 129 }],
            `structured nested forward refuses coupling ${JSON.stringify(c)} (#586)`, "coupling");
  }
  accepts("an omitted coupling is still the block race (#586)",
          () => blocks.nestedRaceProbabilities(mu, cl, ld, D, { points: 129 }),
          p => Math.abs(p[0] - blocks.blockRaceProbabilities(mu, cl, ld, D, { points: 129 })[0]) < 1e-15);
  const m4 = [-0.8, -0.1, 0.3, 0.9], D4 = [0.7, 1.1, 0.9, 1.3];
  for (const V of [NaN, false, ""]) {
    for (const [name, f, args] of [
      ["topKProbabilities", topk.topKProbabilities, [m4, 2, { D: D4, V }]],
      ["bottomKProbabilities", topk.bottomKProbabilities, [m4, 2, { D: D4, V }]],
      ["topKJacobians", topk.topKJacobians, [m4, 2, { D: D4, V }]],
      ["rankProbabilities", topk.rankProbabilities, [m4, { D: D4, V }]],
      ["abilitiesFromTopk", topk.abilitiesFromTopk, [[0.85, 0.57, 0.38, 0.2], 2, { D: D4, V }]]])
      rejects(f, args, `${name} refuses falsy V ${JSON.stringify(V)} (#549)`, "V");
  }
  for (const win of ["bulkk", "", null, 3, true])
    rejects(races.raceProbabilities, [[0, 1, 2], { D: [0.01, 4, 100], window: win }],
            `raceProbabilities refuses window ${JSON.stringify(win)} (#582)`, "window");
  rejects(classic.skewNormalDensity, [50, 0.1, { sacle: 0.3 }],
          "skewNormalDensity refuses a misspelled key (#561)", "sacle");
  rejects(classic.dividendImpliedAbility, [[2, 4, 8], classic.skewNormalDensity(50, 0.1, { a: 0 }), { unti: 0.1 }],
          "dividendImpliedAbility refuses a misspelled key (#561)", "unti");
  rejects(polish.concentrationMatrix, [3, { namecaps: [0.5] }],
          "concentrationMatrix refuses a misspelled key (#561)", "namecaps");
}

// --- a relative target is scale-free past the double range of its SUM
//     (#461, #463, #483, #484)
{
  const q = [0.8, 0.6, 0.4, 0.2];
  const same = (label, f) => accepts(label, () => [f(1), f(1e308)],
    ([a, b]) => a.every(Number.isFinite) && Math.max(...a.map((v, i) => Math.abs(v - b[i]))) < 1e-9);
  same("abilitiesFromTopk at 1e308 (#461)",
       c => topk.abilitiesFromTopk(q.map(x => x * c), 2, { points: 257 }));
  same("abilitiesFromRankMarginal at 1e308 (#483)",
       c => topk.abilitiesFromRankMarginal(q.map(x => x * c), 1, { points: 257 }));
  same("locScaleFromWinAndSecond at 1e308 (#483)",
       c => topk.locScaleFromWinAndSecond([0.8, 0.6, 0.4, 0.2].map(x => x * c),
                                          [0.4, 0.6, 0.6, 0.4].map(x => x * c),
                                          { points: 257, nIter: 20 }).mu);
  same("abilitiesFromBlockRace at 1e308 (#463)",
       c => blocks.abilitiesFromBlockRace(q.map(x => x * c), [0, 0, 1, 1],
                                          [0.25, -0.15, 0.35, -0.05], [0.8, 0.9, 1, 1.1],
                                          { points: 129 }).mu);
  same("asClassicPrices at 1e308 (#484)",
       c => classic.asClassicPrices([0.5, 0.25, 0.125, 0.125].map(x => x * c)));
  same("asClassicDensity at 1e308 (#484)",
       c => classic.asClassicDensity(new Array(7).fill(c)));
}

// --- hierarchical Jacobian / inverse scale read canonical shapes (#596, #597)
{
  const mu = [-0.4, 0.4], c = [0, 1], v = [0.3, 0.4], pa = [2, 2, -1], st = [0, 0, 0.8];
  const J0 = blocks.treeRaceJacobian(mu, c, v, [0.8, 0.8], pa, st)[0][1];
  for (const [label, L, D] of [["typed loading", new Float64Array(v), [0.8, 0.8]],
                               ["scalar D", v, 0.8]])
    accepts(`ancestor treeRaceJacobian takes a ${label} (#596)`,
            () => blocks.treeRaceJacobian(mu, c, L, D, pa, st)[0][1],
            j => Math.abs(j - J0) < 1e-12);
  for (const [label, S] of [
    ["Blocks scalar D", structures.Blocks(c, v, 0.8)],
    ["Blocks scalar loading", structures.Blocks(c, 0.3, [0.8, 0.8])],
    ["Blocks (1, n) loading", structures.Blocks(c, [v], [0.7, 0.9])],
    ["Nested scalar coupling", structures.Nested(c, v, [0.7, 0.9], 0.1, 0.5)],
    ["Tree scalar D", structures.Tree(c, v, 0.8, pa, st)]])
    accepts(`${label} inverts to its generating abilities (#597)`,
            () => races.abilitiesFromRace(races.raceProbabilities(mu, { structure: S }),
                                          { structure: S, returnInfo: true }),
            o => o.converged && Math.max(...o.mu.map((x, i) => Math.abs(x - mu[i]))) < 1e-6);
}

// --- counts are counts; a linkage merges each cluster once (#557, #603)
for (const r of [NaN, -1, 2.5, Infinity])
  rejects(demo.haltonNormalNodes, [r, 3], `haltonNormalNodes refuses rank ${r} (#557)`, "rank");
rejects(core.firstPrimes, [Infinity], "firstPrimes refuses Infinity rather than hanging (#557)");
rejects(structures.cutLinkage, [[[0, 0, 0.1, 2], [1, 3, 0.2, 3]], 3, 1],
        "cutLinkage refuses a self-merge (#603)", "same cluster");
rejects(structures.cutLinkage, [[[0, 1, 0.1, 2], [0, 3, 0.2, 3]], 3, 1],
        "cutLinkage refuses a reused cluster (#603)", "same cluster");

// --- demo covariance helpers keep the shape and PSD contracts (#563, #564)
{
  const eq = (A, B) => A.length === B.length &&
    A.every((r, i) => r.every((v, j) => Math.abs(v - B[i][j]) < 1e-15));
  for (const [kind, mk] of [
    ["Factor", D => structures.Factor([-0.2, 0, 0.2], D)],
    ["Blocks", D => structures.Blocks([0, 0, 1], [0.3, 0.2, 0.4], D)],
    ["Nested", D => structures.Nested([0, 0, 1], [0.3, 0.2, 0.4], D, [0.1, 0.2, 0.3])],
    ["Tree", D => structures.Tree([0, 1, 1], [0.3, 0.2, 0.4], D, [2, 2, -1], [0, 0, 0.8])]])
    accepts(`structureCov(${kind}) is the same for scalar and vector D (#563)`,
            () => [demo.structureCov(mk(1)), demo.structureCov(mk([1, 1, 1]))],
            ([a, b]) => a.length === 3 && eq(a, b));
  rejects(demo.structureCov, [structures.Independent(1)],
          "structureCov(Independent(scalar)) refuses rather than returning [] (#563)", "field size");
  rejects(demo.raceFactor, [[[1, 0.9, -0.6], [0, 1, 0.8], [0, 0, 1]]],
          "raceFactor refuses an asymmetric C (#564)", "symmetric");
  rejects(demo.raceFactor, [[[1, 1.4, 0], [1.4, 1, 0], [0, 0, 1]]],
          "raceFactor refuses an indefinite C (#564)", "positive semidefinite");
}

// --- classic: a scratched entrant stays scratched (#589); the default
// table covers what the forward represents (#498)
{
  const maxAbs = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  for (const [L, unit] of [[3, 1.0], [15, 0.2]]) {
    const d = classic.skewNormalDensity(L, unit, { a: 0 });
    const live = classic.dividendImpliedAbility([2, 4], d);
    for (const scratch of [Infinity, 0, -1]) {
      const a = classic.dividendImpliedAbility([2, 4, scratch], d);
      holds(`L=${L} dividend ${scratch} is ability +Infinity, the rest the reduced race (#589)`,
            a[2] === Infinity && maxAbs(a.slice(0, 2), live) < 1e-12, String(a));
    }
  }
  const d15 = classic.skewNormalDensity(15, 0.1, { a: 0 });
  const t = classic.statePricesFromOffsets(d15, [-10, 10]);
  accepts("a 97/3 book round-trips at defaults (#498)",
          () => classic.solveForImpliedOffsets(t, d15),
          mu => maxAbs(classic.statePricesFromOffsets(d15, mu), t) < 1e-4,
          mu => String(classic.statePricesFromOffsets(d15, mu)));
  accepts("the adaptive default table equals the full table (#498)",
          () => [classic.solveForImpliedOffsets([0.5, 0.3, 0.15, 0.05], d15),
                 classic.solveForImpliedOffsets([0.5, 0.3, 0.15, 0.05], d15,
                   { offsetSamples: classic.defaultOffsetSamples(15) })],
          ([a, b]) => maxAbs(a, b) === 0);
  const warned = [];
  const keep = console.warn;
  console.warn = m => warned.push(m);
  try {
    classic.solveForImpliedOffsets([0.5, 0.5, 0], classic.skewNormalDensity(3, 1.0, { a: 0 }));
  } finally { console.warn = keep; }
  holds("an unrepresentable target warns (#498)",
        warned.length === 1 && warned[0].includes("did not reach the target"));
}

// --- classic: the ability forward takes the scratch sentinel (#662);
// statePricesFromOffsets still refuses it (#438). Reference values are
// python's ability_implied_state_prices on the same inputs.
{
  const maxAbs = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  const d = classic.skewNormalDensity(15, 0.2, { a: 0 });
  for (const scratch of [Infinity, 0, -1]) {
    const a = classic.dividendImpliedAbility([2, 4, scratch], d);
    accepts(`dividends [2, 4, ${scratch}] round-trip through abilityImpliedStatePrices (#662)`,
            () => classic.abilityImpliedStatePrices(a, d),
            p => p[2] === 0 && maxAbs(p, [0.6553529940305943, 0.3446470059694057, 0]) < 1e-12,
            p => String(p));
  }
  accepts("one live runner and a scratch price exactly [1, 0] (#662)",
          () => classic.abilityImpliedStatePrices(classic.dividendImpliedAbility([2, Infinity], d), d),
          p => p[0] === 1 && p[1] === 0, p => String(p));
  // python: ability_implied_state_prices([-0.4, 0.3], d) (the support is
  // wider than L here, so this is the clustered branch, not the plain one)
  const live = [0.5367196595728523, 0.4632803404271477];
  for (const at of [0, 1, 2]) {
    const a = [-0.4, 0.3]; a.splice(at, 0, Infinity);
    accepts(`a scratch at index ${at} keeps shape and order (#662)`,
            () => classic.abilityImpliedStatePrices(a, d),
            p => p.length === 3 && p[at] === 0
                 && maxAbs(p.filter((_, i) => i !== at), live) < 1e-14, p => String(p));
  }
  const d200 = classic.skewNormalDensity(200, 0.1, { a: 0 });
  accepts("an all-live in-window field equals statePricesFromOffsets (#662)",
          () => classic.abilityImpliedStatePrices([0, 0.7, 1.3, 2.1], d200),
          p => maxAbs(p, classic.statePricesFromOffsets(d200, [0, 0.7, 1.3, 2.1])) < 1e-15);
  accepts("an all-live field wider than the lattice matches python (#662)",
          () => classic.abilityImpliedStatePrices([0, 0.7, 1.3, 2.1], classic.skewNormalDensity(40, 0.1, { a: 0 })),
          p => maxAbs(p, [0.28573759854465663, 0.2577847581738416, 0.23941631079393647,
                          0.2170613324875653]) < 1e-14, p => String(p));
  accepts("-Infinity runners split the pot, +Infinity alone is uniform (#662)",
          () => [classic.abilityImpliedStatePrices([-Infinity, 0.3, -Infinity], d),
                 classic.abilityImpliedStatePrices([Infinity, Infinity], d)],
          ([a, b]) => maxAbs(a, [0.5, 0, 0.5]) === 0 && maxAbs(b, [0.5, 0.5]) === 0);
  const d50 = classic.skewNormalDensity(50, 0.1, { a: 0 });
  accepts("a far-separated field is clustered as python (#662)",
          () => classic.abilityImpliedStatePrices([-40, -39, 40, 41, 42.5], d50),
          p => maxAbs(p, [0.5277153991928714, 0.472284600798019, 3.4847031497493502e-12,
                          3.072512437648325e-12, 2.5524467876884387e-12]) < 1e-14, p => String(p));
  accepts("a no-chance runner far behind prices 0 (#662)",
          () => classic.abilityImpliedStatePrices([0, 0.5, 400], d50),
          p => p[2] === 0 && maxAbs(p, [0.5138576995988393, 0.4861423004011607, 0]) < 1e-14);
  accepts("unit rescales the abilities (#662)",
          () => classic.abilityImpliedStatePrices([-0.08, 0.06, Infinity], d, { unit: 0.2 }),
          p => maxAbs(p, classic.abilityImpliedStatePrices([-0.4, 0.3, Infinity], d)) === 0);
  rejects(classic.statePricesFromOffsets, [d, [0, Infinity]],
          "statePricesFromOffsets still refuses the sentinel (#438, #662)", "finite");
  rejects(classic.abilityImpliedStatePrices, [[0, NaN], d], "abilityImpliedStatePrices refuses NaN (#662)", "not a number");
  rejects(classic.abilityImpliedStatePrices, [[0, "1"], d], "abilityImpliedStatePrices refuses a string (#662)", "not a number");
  rejects(classic.abilityImpliedStatePrices, [[0, 1], d, { unti: 1 }], "abilityImpliedStatePrices refuses a misspelt option (#662)", "unknown option 'unti'");
  rejects(classic.abilityImpliedStatePrices, [[0, 1], d, { unit: 0 }], "abilityImpliedStatePrices refuses unit 0 (#662)", "finite positive");
}

// --- classic: skewNormalDensity's domain (#509) and option typos (#561)
for (const [args, why] of [[[50, 0], "unit 0"], [[50, -0.1], "unit < 0"],
                           [[50, 0.1, { scale: -1 }], "scale < 0"], [[50, 0.1, { scale: 0 }], "scale 0"]])
  rejects(classic.skewNormalDensity, args, `skewNormalDensity refuses ${why} (#509)`, "finite positive");
rejects(classic.skewNormalDensity, [50, 0.1, { sacle: 0.3 }],
        "skewNormalDensity refuses a misspelt option (#561)", "unknown option 'sacle'");
rejects(classic.dividendImpliedAbility, [[2, 4, 8], classic.skewNormalDensity(50, 0.1), { unti: 0.1 }],
        "dividendImpliedAbility refuses a misspelt option (#561)", "unknown option 'unti'");
accepts("dividendImpliedAbility still takes unit (#561)",
        () => classic.dividendImpliedAbility([2, 4, 8], classic.skewNormalDensity(50, 0.1), { unit: 0.1 }),
        a => a.every(Number.isFinite));

// --- the bulk window's delta: (0, 0.5), and -1 used to hang (#445)
for (const d of [-1, -1e-9, 0, 0.5, 1, 2, NaN, Infinity, "1e-2", null, true])
  rejects(races.raceProbabilities, [[-1, 0, 1], { D: [1, 1, 1], delta: d, points: 257 }],
          `raceProbabilities refuses delta=${String(d)} (#445)`, "delta must be");
accepts("raceProbabilities takes delta=1e-3 (#445)",
        () => races.raceProbabilities([-1, 0, 1], { D: [1, 1, 1], delta: 1e-3 }),
        p => p.every(Number.isFinite));

// --- null and negligible factor nodes are no-ops for the price (#416)
{
  const mu = [-0.8, -0.1, 0.5, 1.0], V = [[-1], [-0.2], [0.4], [0.8]], D = [1, 1, 1, 1];
  const md = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  const base = races.raceProbabilities(mu, { V, D, F: [[-1], [1]], W: [0.5, 0.5], points: 257 });
  for (const w of [0, 1e-300, 1e-17])
    accepts(`a weight-${w} node far out leaves the race unchanged (#416)`,
            () => races.raceProbabilities(mu, { V, D, F: [[-1], [1], [1000]], W: [0.5, 0.5, w], points: 257 }),
            p => md(p, base) < 1e-12);
  const m2 = [0.1, 0.2, 0.4], V2 = [[2], [0], [-2]], D2 = [1e-6, 1e-6, 1e-6];
  const c = blocks.blockRaceProbabilities(m2, [0, 1, 2], V2, D2, { points: 501, qa: 7, nodes: { nodes: [[0]], w: [1] } });
  accepts("zero-weight block node padding is a no-op (#416)",
          () => blocks.blockRaceProbabilities(m2, [0, 1, 2], V2, D2, { points: 501, qa: 7, nodes: { nodes: [[-100], [0], [100]], w: [0, 1, 0] } }),
          p => md(p, c) < 1e-12);
}

// --- treeFromLinkage checks topology and heights itself (#328, #459)
for (const [Z, why] of [
  [[[0, 1, -0.5, 2], [2, 3, 0.8, 3]], "a negative height"],
  [[[0, 1, NaN, 2], [2, 3, 0.8, 3]], "a NaN height"],
  [[[3, 0, 0.5, 2], [1, 2, 0.8, 3]], "a future cluster"],
  [[[0, 0, 0.5, 2], [1, 2, 0.8, 3]], "a self-merge"],
  [[[0, 1, 0.5, 2], [0, 2, 0.8, 3]], "a reused cluster"],
  [[[0, 1.5, 0.5, 2], [2, 3, 0.8, 3]], "a fractional id"]])
  rejects(structures.treeFromLinkage, [Z], `treeFromLinkage refuses ${why} (#328/#459)`, "linkage row");
accepts("treeFromLinkage builds a valid linkage (#328)",
        () => structures.treeFromLinkage([[0, 1, 0.5, 2], [2, 3, 0.8, 3]]),
        t => t.parent.join() === "3,3,4,4,-1");

// --- an explicit Tree leaf map makes relabeling invariant (#372)
{
  const mu = [-0.5, -0.1, 0.2, 0.4], D = [0.8, 0.9, 1.0, 1.1], L = [0.2, 0.3, -0.1, 0.4];
  const pa = [3, 3, 4, 4, -1], st = [0, 0, 0, 0.6, 0.4];
  const md = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  const a = blocks.treeRaceProbabilities(mu, [0, 0, 1, 2], L, D, pa, st);
  accepts("treeRaceProbabilities honours leaves (#372)",
          () => blocks.treeRaceProbabilities(mu, [30, 30, 10, 20], L, D, pa, st, { leaves: [30, 10, 20] }),
          p => md(p, a) === 0);
  accepts("Tree(..., leaves) prices the canonical tree (#372)",
          () => races.raceProbabilities(mu, { structure: structures.Tree([30, 30, 10, 20], L, D, pa, st, [30, 10, 20]) }),
          p => md(p, races.raceProbabilities(mu, { structure: structures.Tree([0, 0, 1, 2], L, D, pa, st) })) === 0);
  rejects(blocks.treeRaceProbabilities, [mu, [30, 30, 10, 20], L, D, pa, st, { leaves: [30, 10] }],
          "leaves must map every label (#372)", "does not map");
}

// --- targetFloor holds in the final normalised target (#592)
for (const c of [1e-3, 1, 1e3])
  accepts(`top-k targetFloor holds at scale ${c} (#592)`,
          () => topk.topKProbabilities(topk.abilitiesFromTopk([0.7 * c, 0.3 * c, 0], 1,
                  { D: [1, 1, 1], targetFloor: 0.2 }), 1, { D: [1, 1, 1] }),
          q => Math.abs(q[2] - 0.2) < 1e-6 && Math.abs(q[0] - 0.56) < 1e-6);
rejects(races.abilitiesFromRace, [[0.5, 0.5, 0], { targetFloor: 0.4 }],
        "an infeasible targetFloor is refused (#592)", "infeasible");

// --- an exactly common loading column is inert at any scale (#471)
{
  const n = 9, mu = Array.from({ length: n }, (_, i) => (i - 4) * 0.1), D = Array(n).fill(1);
  const md = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
  const ind = races.raceProbabilities(mu, { D, points: 257 });
  const tk = topk.topKProbabilities(mu, 3, { D, points: 257 });
  const J0 = topk.topKJacobians(mu, 3, { D, points: 257 }).Jmu.flat();
  for (const c of [1e20, 1e307, 1e308]) {
    const V = Array.from({ length: n }, () => [c]);
    accepts(`a common [${c}] column leaves the race unchanged (#471)`,
            () => races.raceProbabilities(mu, { V, D, F: [[-1], [1]], W: [0.5, 0.5], points: 257 }),
            p => md(p, ind) < 1e-15);
    accepts(`a common [${c}] column leaves top-k unchanged (#471)`,
            () => topk.topKProbabilities(mu, 3, { V, D, points: 257, qa: 3 }),
            q => md(q, tk) < 1e-12);
    accepts(`a common [${c}] column leaves the top-k Jacobian unchanged (#471)`,
            () => topk.topKJacobians(mu, 3, { V, D, points: 257, qa: 3 }).Jmu.flat(),
            J => md(J, J0) < 1e-12);
  }
  const mu3 = [0, 0.3, -0.2], c = 1e308;
  accepts("an overflowing loading with compensating nodes prices the same race (#471)",
          () => races.raceProbabilities(mu3, { V: [[c], [c], [-c]], D: [1, 1, 1], F: [[-1 / c], [1 / c]], W: [0.5, 0.5] }),
          p => md(p, races.raceProbabilities(mu3, { V: [[1], [1], [-1]], D: [1, 1, 1], F: [[-1], [1]], W: [0.5, 0.5] })) < 1e-12);
}

// --- gamma = 0 Nested ignores its coupling in the inverse too (#644)
{
  const s0 = structures.Nested([0, 0, 1], [0.4, 0.6, 0.5], [1, 1, 1], [0.2, 0.3], 0);
  const p0 = races.raceProbabilities([-0.4, 0.1, 0.3], { structure: s0 });
  accepts("gamma 0 Nested with a stale coupling inverts (#644)",
          () => races.abilitiesFromRace(p0, { structure: s0, returnInfo: true }),
          o => o.converged && Math.abs(o.mu[0] + 0.4) < 1e-6);
  rejects(races.raceProbabilities, [[-0.4, 0.1, 0.3], { structure: structures.Nested([0, 0, 1], [0.4, 0.6, 0.5], [1, 1, 1], [0.2, 0.3], 0.5) }],
          "an active wrong-length coupling is still refused (#644)", "coupling");
}

// --- a default hierarchical target inverts with default controls (#656)
{
  const mu = [0.36, -0.92, 0.30, 0.26], cl = [0, 0, 1, 2];
  const L = [0.022, 1.28, -0.083, -0.04], D = [0.035, 25, 0.062, 0.015];
  const keep = console.warn; console.warn = () => {};
  try {
    const t = blocks.blockRaceProbabilities(mu, cl, L, D);
    accepts("abilitiesFromBlockRace round-trips a default target (#656)",
            () => blocks.abilitiesFromBlockRace(t, cl, L, D, { tol: 1e-8 }),
            o => o.residual < 1e-8 && Math.abs(o.mu[1] + 0.92) < 1e-6);
    const tr = structures.Tree(cl, L, D, [3, 3, 4, 4, -1], [0, 0, 0, 0.2, 0]);
    const p = races.raceProbabilities(mu, { structure: tr });
    accepts("structured Tree inverse round-trips a default target (#656)",
            () => races.abilitiesFromRace(p, { structure: tr, returnInfo: true }),
            o => o.converged && Math.abs(o.mu[1] + 0.92) < 1e-6);
  } finally { console.warn = keep; }
}

// --- fitGrammar reads C through its contrasts: a common shift cancels (#465)
{
  const n = 4;
  const contr = M => { const o = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) o.push(M[i][i] + M[j][j] - 2 * M[i][j]); return o; };
  const fitAt = a => demo.fitGrammar(Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => a + (i === j ? 1 : 0))), 3, 4);
  const f0 = fitAt(0);
  for (const a of [2 ** 20, 2 ** 51])
    accepts(`fitGrammar(C + ${a} 11') fits C (#465)`, () => fitAt(a),
            f => JSON.stringify(f) === JSON.stringify(f0) &&
                 contr(demo.structureCov({ kind: "Factor", ...f })).every(c => Math.abs(c - 2) < 1e-5));
}

// --- classic: dividendImpliedAbility's output unit is a lattice spacing (#649)
for (const unit of [0, -0.1, NaN, Infinity, -Infinity, "0.1"])
  rejects(classic.dividendImpliedAbility, [[2, 4], classic.skewNormalDensity(50, 0.1, { a: 0 }), { unit }],
          `dividendImpliedAbility refuses unit ${String(unit)} (#649)`, "finite positive");
accepts("dividendImpliedAbility keeps the favourite first at unit 0.1 (#649)",
        () => classic.dividendImpliedAbility([2, 4], classic.skewNormalDensity(50, 0.1, { a: 0 }), { unit: 0.1 }),
        a => a[0] < a[1] && Math.abs(a[0] + 0.2646016516) < 1e-8 && Math.abs(a[1] - 0.3455146431) < 1e-8,
        a => JSON.stringify(a));

// --- classic: a common translation across the lattice edge keeps the contrast (#585)
{
  const d = classic.skewNormalDensity(15, 0.5, { a: 0 });
  const base = classic.statePricesFromOffsets(d, [-3, 3, 1]);
  for (const c of [11, 20, -20, 40, -55])
    accepts(`statePricesFromOffsets is translation-free at shift ${c} (#585)`,
            () => classic.statePricesFromOffsets(d, [-3 + c, 3 + c, 1 + c]),
            p => p.every((x, i) => Math.abs(x - base[i]) < 1e-12) && base[0] > 0.9,
            p => JSON.stringify(p));
}

// --- laplace inverses on exact self-generated targets (#267): the uniform
// lattice's slopes are first order across each runner's kink, so the
// inverses take their derivatives from the split lattice
{
  const quiet = console.warn; console.warn = () => {};
  const mx = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  {
    const mu = [-1.5, 1.5, 0], o = { D: [0.1, 6, 2], base: "laplace" };
    for (const points of [257, 513]) {
      const t = races.raceProbabilities(mu, { ...o, points });
      const f = races.abilitiesFromRace(t, { ...o, points, returnInfo: true });
      holds(`laplace race inverse round-trips at ${points} points (#267)`,
            f.converged && mx(races.raceProbabilities(f.mu, { ...o, points }), t) < 1e-8);
    }
  }
  {
    const mu = [-1.0758897613734006, 0.8221703173592687, 1.2747228138148785,
                -1.7811144122853875, 0.7601110424846411];
    const sd = [0.5289511696798439, 1.2097101565229562, 0.9433803641440728,
                1.3242591310799288, 1.2509595951816537];
    const D = sd.map(v => v * v);
    const q1 = topk.topKProbabilities(mu, 1, { D, base: "laplace" });
    const q2 = topk.topKProbabilities(mu, 2, { D, base: "laplace" });
    const f = topk.locScaleFromTopkPair(q1, 1, q2, 2, { base: "laplace", returnInfo: true });
    const Df = f.sd.map(v => v * v);
    holds("laplace loc/scale pair round-trips at 513 points (#267)",
          f.info.converged &&
          mx(topk.topKProbabilities(f.mu, 2, { D: Df, base: "laplace" }), q2) < 1e-8);
  }
  {
    const mu = [-1.7151211049140525, 1.6647723741888125, 0.5038679121443425,
                -0.5829490360565223, 0.12942985463742007];
    const D = [0.7813762250327798, 0.7548860941612756, 1.0407350057755838,
               0.5488380588062226, 2.0439199583301906];
    const q = topk.topKProbabilities(mu, 3, { D, base: "laplace" });
    const f = topk.abilitiesFromTopk(q, 3, { D, base: "laplace", returnInfo: true });
    holds("laplace top-3 inverse round-trips at 513 points (#267)",
          f.info.converged &&
          mx(topk.topKProbabilities(f.mu, 3, { D, base: "laplace" }), q) < 1e-8);
  }
  {
    const mu = [-0.32582296524196863, 0.47305250354111195, 0.6627286607399583,
                1.3682900704443455, 1.9737782655283809];
    const sd = [1.1240951818414033, 1.6630497237201780, 0.37187049128115174,
                0.4599949211347848, 0.7765970396809280];
    const mu0 = [-0.34953938962426034, 0.5153643666766584, 0.6403211534721777,
                 1.3358619652222843, 2.0055403112666683];
    const D = sd.map(v => v * v);
    const t = topk.rankProbabilities(mu, { D, base: "laplace" }).map(r => r[1]);
    const f = topk.abilitiesFromRankMarginal(t, 2, { D, base: "laplace", mu0, returnInfo: true });
    const c = v => v.map(x => x - v.reduce((a, b) => a + b, 0) / v.length);
    holds("laplace exact-rank inverse recovers its generating field (#267)",
          f.info.converged && mx(c(f.mu), c(mu)) < 1e-6);
  }
  console.warn = quiet;
}

{
  // averageLinkage is positively homogeneous; the size-weighted numerator
  // overflowed 9e307 + 9e307 into an Infinity height that treeFromLinkage
  // then refused (#695)
  const unit = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]];
  const Z1 = structures.averageLinkage(unit);
  let ok = true;
  for (const a of [1e-100, 1e300, 9e307, 1.7e308]) {
    const Z = structures.averageLinkage(unit.map(r => r.map(x => a * x)));
    ok = ok && Z.every((row, k) => row[0] === Z1[k][0] && row[1] === Z1[k][1] &&
                                   row[3] === Z1[k][3] && row[2] === a * Z1[k][2]);
    try { structures.treeFromLinkage(Z); } catch (e) { ok = false; }
  }
  holds("averageLinkage scales its heights and stays finite to 1.7e308 (#695)", ok);
}
{
  // modular logndtr keeps the upper tail: log Phi(z) is representable to
  // z ~ 38 and log(ndtr(z)) was exactly 0 from z ~ 8.3 (#696). scipy
  // log_ndtr values.
  const ref = [[8, -6.220960574271742e-16], [8.3, -5.2055697448902465e-17],
               [10, -7.619853024160474e-24], [20, -2.753624118606155e-89],
               [37, -5.725571222523923e-300], [2, -0.02301290932896348],
               [1, -0.1727537790234499]];
  const worst = Math.max(...ref.map(([z, v]) => Math.abs(core.logndtr(z) / v - 1)));
  holds("core logndtr keeps the upper tail to z = 37 (#696)", worst < 1e-12,
        `worst relative error ${worst.toExponential(2)}`);
  holds("core logndtr at +-Infinity (#696)",
        core.logndtr(Infinity) === 0 && core.logndtr(-Infinity) === -Infinity);
}
{
  // g -> g + 1c' is one common shock: the same race. Canonicalizing the
  // raw rank-12 coupling moved a share by 0.0015 and the Jacobian by
  // 0.008 (#698); the rank-2 tensor rule is the control.
  const n = 6, ph = 8;
  const mu = Array.from({ length: n }, (_, i) => 0.7 * Math.sin((i + 1) * (ph + 1) * 0.37) + 0.2 * i);
  const cl = [...Array(n).keys()], ld = new Array(n).fill(0);
  const D = Array.from({ length: n }, (_, i) => 0.03 + 0.015 * (i % 4));
  const mxm = (a, b) => Math.max(...a.flat().map((v, i) => Math.abs(v - b.flat()[i])));
  for (const r of [2, 12]) {
    const g = Array.from({ length: n }, (_, i) =>
      Array.from({ length: r }, (_, j) => 0.7 * Math.sin((i + 1) * (j + 2) + ph * 0.31)));
    const gs = g.map(row => row.map((x, j) => x + Math.cos((j + 1) * 0.43)));
    const o = { gamma: 1, points: 65 };
    const p = blocks.nestedRaceProbabilities(mu, cl, ld, D, { ...o, coupling: g });
    const q = blocks.nestedRaceProbabilities(mu, cl, ld, D, { ...o, coupling: gs });
    const J = blocks.nestedRaceJacobian(mu, cl, ld, D, { ...o, coupling: g });
    const Js = blocks.nestedRaceJacobian(mu, cl, ld, D, { ...o, coupling: gs });
    // the inverse (its preconditioner reads the coupling norm) at rank 2
    // only: a rank-12 Halton inverse is minutes of CI for the same check
    let inv = 0;
    if (r === 2) {
      const m1 = races.abilitiesFromRace(p, { structure: structures.Nested(cl, ld, D, g), points: 65 });
      const m2 = races.abilitiesFromRace(p, { structure: structures.Nested(cl, ld, D, gs), points: 65 });
      const c = v => v.map(x => x - v.reduce((a, b) => a + b, 0) / v.length);
      inv = mxm(c(m1), c(m2));
    }
    holds(`Nested rank-${r} forward/Jacobian/inverse ignore a common coupling shift (#698)`,
          mxm(p, q) < 1e-12 && mxm(J, Js) < 1e-12 && inv < 1e-7,
          `${mxm(p, q).toExponential(1)} / ${mxm(J, Js).toExponential(1)} / ${inv.toExponential(1)}`);
  }
}

{
  // a two-runner GHK weight is the exact Phi(-gap/sqrt2) with no sampling
  // error; 1 - ndtr(a) and a 1e-14 cutoff zeroed it from gap 11 (#555)
  let worst = 0;
  for (const gap of [8, 11, 12, 20, 50]) {
    const got = demo.ghkSampleOne(demo.ghkPrepareOne([gap, 0], [[1, 0], [0, 1]], 0), 1, demo.rng(7));
    worst = Math.max(worst, Math.abs(got / core.ndtr(-gap / Math.SQRT2) - 1));
  }
  holds("GHK keeps representable upper tails to gap 50 (#555)", worst < 1e-12,
        `worst relative error ${worst.toExponential(2)}`);
  // a rare first truncation followed by nontrivial correlated steps
  const C = [[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]];
  const w = demo.ghkSampleOne(demo.ghkPrepareOne([15, 0, 0.5], C, 0), 200, demo.rng(3)) / 200;
  holds("GHK prices a 15-unit longshot in a correlated three-runner field (#555)",
        w > 1e-46 && w < 1e-41, w.toExponential(3));
}

{
  // jacobiEigh and fitGrammar squared raw entries: past 1e154 no rotation
  // ran, and fitGrammar(1e160 C) moved normalized contrasts by 0.53 and
  // returned NaN D at 1e308 (#536). Both now run on an exact power-of-four
  // rescale.
  let ok = true;
  for (const a of [1e-200, 1e154, 1e308]) {
    const v = demo.jacobiEigh([[0, a], [a, 0]], 50).values.slice().sort((x, y) => x - y);
    ok = ok && Math.abs(v[0] / -a - 1) < 1e-12 && Math.abs(v[1] / a - 1) < 1e-12;
  }
  holds("jacobiEigh finds [-a, a] at every scale (#536)", ok);
  const B = [[0.8, 0.1], [0.4, -0.3], [-0.2, 0.6], [-0.5, -0.1]], Dd = [0.4, 0.7, 0.5, 0.6];
  const C = B.map((bi, i) => B.map((bj, j) => bi[0] * bj[0] + bi[1] * bj[1] + (i === j ? Dd[i] : 0)));
  const contrast = (M, i, j) => M[i][i] + M[j][j] - 2 * M[i][j];
  const nf = c => {
    const f = demo.fitGrammar(C.map(r => r.map(x => c * x)));
    return demo.structureCov(structures.Factor(f.V.map(r => r.map(x => x / Math.sqrt(c))),
                                               f.D.map(x => x / c)));
  };
  const o = nf(1);
  let drift = 0;
  for (const c of [1e-300, 1e160, 1e307]) {
    const f = nf(c);
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++)
      drift = Math.max(drift, Math.abs(contrast(f, i, j) - contrast(o, i, j)));
  }
  holds("fitGrammar is scale-equivariant from 1e-300 to 1e307 (#536)", drift < 1e-12,
        `max normalized contrast drift ${drift.toExponential(2)}`);
  const big = demo.fitGrammar([[1e308, 0, 0], [0, 1e308, 0], [0, 0, 1e308]]);
  holds("fitGrammar(1e308 I_3) is finite (#536)",
        big.D.every(Number.isFinite) && big.V.flat().every(Number.isFinite));
}

{
  // canonicalBasis formed a * b of squared column norms: at loadings near
  // 1e80 it overflowed, no rotation ran, and V and VQ (the same law)
  // priced 7e-4 apart (#130 residual)
  const V = [[-2, 1, .5], [1, -2, 1], [.5, 1.5, -2], [.5, -.5, .5]];
  const a = .61, b = Math.sqrt(1 - a * a);
  const Q = [[a, -b, 0], [b, a, 0], [0, 0, 1]];
  const Vr = V.map(row => Q[0].map((_, j) => Q.reduce((z, r, k) => z + row[k] * r[j], 0)));
  let worst = 0;
  for (const s of [1e-80, 1e80, 1e100]) {
    const mu = [-.7, .1, .4, 1].map(x => x * s), D = [.03, .05, .04, .02].map(x => x * s * s);
    const p = races.raceProbabilities(mu, { V: V.map(r => r.map(x => x * s)), D, points: 257, delta: 1e-4 });
    const q = races.raceProbabilities(mu, { V: Vr.map(r => r.map(x => x * s)), D, points: 257, delta: 1e-4 });
    worst = Math.max(worst, ...p.map((x, i) => Math.abs(x - q[i])));
  }
  holds("a rotated factor basis prices alike at 1e-80..1e100 (#130)", worst < 1e-12,
        worst.toExponential(2));
}

{
  // Math.max(...D) spread a runner-sized array into call arguments: past
  // the engine's argument limit (~125k on node 24) a valid field threw
  // "Maximum call stack size exceeded" in setup (#729)
  const n = 150000, keep = console.warn;
  console.warn = () => {};
  try {
    accepts("a 150,000-runner equal field prices uniformly (#729)",
            () => races.raceProbabilities(new Array(n).fill(0), { D: new Array(n).fill(1), points: 3, delta: 1e-4 }),
            p => p.length === n && p.every(v => Math.abs(v * n - 1) < 1e-6) &&
                 Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-9);
    accepts("translatedHome takes a 150,000-entry D (#729)",
            () => races.translatedHome(new Array(n).fill(1e15), new Array(n).fill(1)),
            m => m.length === n && m.every(v => v === 0));
  } finally { console.warn = keep; }
}

{
  // the bulk window bracketed only the winner envelope: the middle runner
  // of [0, 20, 40] wins near x = 10, past its right edge, and came back at
  // 0.099 of Phi(-20/sqrt 2) at any point count (#723, python's #97)
  const exact = 1.0442437918812723e-45;
  let worst = 0;
  for (const points of [257, 4097])
    worst = Math.max(worst, Math.abs(races.raceProbabilities([0, 20, 40], { D: [1, 1, 1], points })[1] / exact - 1));
  holds("default bulk window keeps a 1e-45 middle runner (#723)", worst < 1e-6, worst.toExponential(2));
}

{
  // Gumbel's lower-tail CDF was 1 - S, exactly 0 once S rounds to one: the
  // Plackett-Luce last place 2.0e-39 of [-35, 0, 0] came back 0 (#710)
  const c = Math.PI / Math.sqrt(6);
  let worst = 0;
  for (const gap of [20, 30, 35]) {
    const w0 = Math.exp(c * gap), exact = 2 / ((w0 + 2) * (w0 + 1));
    const mu = [-gap, 0, 0];
    const b = topk.bottomKProbabilities(mu, 1, { base: "gumbel", points: 513 })[0];
    const r = topk.rankProbabilities(mu, { base: "gumbel", points: 513 })[0][2];
    worst = Math.max(worst, Math.abs(b / exact - 1), Math.abs(r / exact - 1));
  }
  holds("Gumbel last place matches Plackett-Luce at gaps 20..35 (#710)", worst < 1e-6, worst.toExponential(2));
}

{
  // a first truncation with q = 2.9e-316 floored its draw at 1e-300, which
  // put e[0] inside the truncation and priced the path 8.6e-322 (#555
  // residual). ndtr(-38) ndtr(5) <= P <= ndtr(-38).
  const mu = [0, -38, -37.5], C = [[0, 0, 0], [0, 1, 1], [0, 1, 1.01]];
  const lo = core.ndtr(-38) * core.ndtr(5), hi = core.ndtr(-38);
  let ok = true, got = [];
  for (const seed of [1, 2, 3, 7, 19]) {
    const w = demo.ghkSampleOne(demo.ghkPrepareOne(mu, C, 0), 1, demo.rng(seed));
    got.push(w.toExponential(3));
    ok = ok && w >= lo * (1 - 1e-6) && w <= hi * (1 + 1e-6);
  }
  holds("GHK keeps a 2.9e-316 correlated path (#555)", ok, got.join(", "));
}

if (fails) { console.error(`${fails} browser API failures`); process.exit(1); }
console.log("browser API guards behave");
