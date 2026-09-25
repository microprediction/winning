// Emit one ACCEPT/REFUSE verdict per case, for the browser port.
import { readFileSync } from "fs";
const here = new URL(".", import.meta.url).pathname;
const races = await import(here + "../docs/js/winning/races.mjs");
const topk = await import(here + "../docs/js/winning/topk.mjs");
const core = await import(here + "../docs/js/winning/core.mjs");
const num = x => Array.isArray(x) ? x.map(num)
  : (typeof x === "string" ? Number(x) : x);
const cases = JSON.parse(readFileSync(process.argv[2], "utf8")).cases;
const out = {};
for (const c of cases) {
  try {
    const mu = num(c.mu), D = num(c.D);
    const V = c.Vt !== undefined ? num(c.Vt) : num(c.V);
    let p;
    if (c.verb === "hermite") {
      const h = core.hermiteNodes(c.r, c.order);
      // SHAPE, not the nodes: the count AND the width
      out[c.id] = { verdict: "ACCEPT",
                    value: [h.F.length, h.F.length ? h.F[0].length : 0] };
      continue;
    }
    if (c.verb === "rank") p = topk.rankProbabilities(mu, { D });
    else if (c.verb === "bottomk") p = topk.bottomKProbabilities(mu, c.k, { D });
    else if (c.verb === "race") p = races.raceProbabilities(mu, { V, D });
    else if (c.verb === "inverse") p = races.abilitiesFromRace(num(c.p), { D });
    else p = topk.topKProbabilities(mu, c.k, { V, D });
    const a = Array.from(p.flat ? p.flat(9) : p, Number);
    out[c.id] = {
      verdict: a.every(Number.isFinite) ? "ACCEPT" : "ACCEPT_NONFINITE",
      value: a.slice(0, 6),
    };
  } catch (e) {
    out[c.id] = { verdict: "REFUSE", error: String(e.message).slice(0, 40) };
  }
}
console.log(JSON.stringify(out));
