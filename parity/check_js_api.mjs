// Behavioural checks for the browser API guards. The pytest suite can only
// read the source; these actually call the functions, which is the whole
// point -- #186 shipped because the tests inspected markers rather than
// behaviour, and a shared allowlist let each API accept the other's keys.
const here = new URL(".", import.meta.url).pathname;
const races = await import(here + "../docs/js/winning/races.mjs");

let fails = 0;
const rejects = (fn, args, why) => {
  try { fn(...args); console.log(`FAIL  ${why}: accepted`); fails++; }
  catch (e) { console.log(`ok    ${why}: ${e.message.slice(0, 58)}`); }
};
const works = (label, got) => {
  const bad = !Array.isArray(got) || got.some(v => !Number.isFinite(v));
  console.log(`${bad ? "FAIL " : "ok   "} ${label}: ${got.map(v => v.toFixed(4)).join(" ")}`);
  if (bad) fails++;
};

rejects(races.raceProbabilities, [[0, 1, 2], { cov: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }], "forward rejects cov");
rejects(races.raceProbabilities, [[0, 1, 2], { nIter: 1 }], "forward rejects inverse-only nIter");
rejects(races.raceProbabilities, [[0, 1, 2], { tol: 1e-8 }], "forward rejects inverse-only tol");
rejects(races.raceProbabilities, [[0, 1, 2], { returnInfo: true }], "forward rejects inverse-only returnInfo");
rejects(races.raceProbabilities, [[0, 1, 2], { retrunSlopes: true }], "forward rejects a typo");
rejects(races.abilitiesFromRace, [[0.2, 0.3, 0.5], { returnSlopes: true }], "inverse rejects forward-only returnSlopes");
rejects(races.abilitiesFromRace, [[0.2, 0.3, 0.5], { window: "span" }], "inverse rejects forward-only window");
rejects(races.abilitiesFromRace, [[0.2, 0.3, 0.5], { cov: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }], "inverse rejects cov");

works("forward still works", races.raceProbabilities([0, 1, 2], { D: [1, 1, 1] }));
works("inverse still works", races.abilitiesFromRace([0.5, 0.3, 0.2], { D: [1, 1, 1] }));
works("inverse honours points", races.abilitiesFromRace([0.5, 0.3, 0.2], { D: [1, 1, 1], points: 513 }));

if (fails) { console.error(`${fails} browser API failures`); process.exit(1); }
console.log("browser API guards behave");
