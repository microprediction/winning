// The convergence demo's lattice reference, off the page's main thread.
//
// A sharp rank-2 factor field escalates to the 8,192-node quasi-Monte
// Carlo rule (as the python engine does, and as accuracy needs: #534
// showed an under-resolved reference corrupts the chart). That costs
// seconds, so converge.html computes it here, in a module worker, and
// the page stays responsive while it runs (#647). The worker rebuilds
// the field from (generator, n, seed) rather than receiving it, so the
// main thread and the worker cannot price different fields.
import { raceProbabilities, _factorNodeRule } from "./winning/races.mjs";
import "./winning/structures.mjs";
import { GENERATORS, rng, fitGrammar, haltonNormalNodes } from "./winning/demo.mjs";

export function demoField(genName, n, seed) {
  const gen = GENERATORS[genName];
  if (!gen) throw new Error(`unknown generator '${genName}'`);
  const g = gen(n, rng(seed));
  const rmu = rng(seed + 1000);
  const mu = Array.from({ length: n }, () => rmu.n() * 0.9);
  return { g, mu };
}

// The node budget the reference will use, so the page can say up front.
export function referenceNodes(g) {
  if (g.structure && g.structure.kind === "Factor")
    return _factorNodeRule(g.structure.V, g.structure.D).F.length;
  return g.structure ? 0 : 1024;
}

// The timed lattice reference (includes the grammar fit for dense C).
export function latticeReference(genName, n, seed) {
  const { g, mu } = demoField(genName, n, seed);
  const t0 = performance.now();
  let pLat;
  if (g.structure) {
    pLat = raceProbabilities(mu, { structure: g.structure, points: 257 });
  } else {
    const fit = fitGrammar(g.C, 3, 5);
    const nodes = haltonNormalNodes(fit.V[0].length, 1024);
    pLat = raceProbabilities(mu, { V: fit.V, D: fit.D,
                                   F: nodes.F, W: nodes.W, points: 129 });
  }
  return { pLat, tLat: performance.now() - t0 };
}

// Only when loaded as a worker: a page or node import just gets the
// functions above.
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) {
  self.onmessage = ({ data }) => {
    try {
      self.postMessage({ token: data.token, ...latticeReference(data.gen, data.n, data.seed) });
    } catch (e) {
      self.postMessage({ token: data.token, error: String(e && e.message || e) });
    }
  };
}
