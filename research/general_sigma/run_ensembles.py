"""The named-ensemble battery: one-call estimator vs MC across every
randomcov generator. 'General Sigma' means nothing without naming the
measure; this is the operational version -- one row per named ensemble."""
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
import importlib.util
spec = importlib.util.spec_from_file_location("b", "run_large_n2.py")
b = importlib.util.module_from_spec(spec)
import sys; sys.modules["b"] = b
src = open("run_large_n2.py").read()
exec(src[:src.index("rng = np.random.default_rng(21)")], b.__dict__)

from randomcov.corrgens.allcorrgens import CORR_GENERATORS, CorrMethod

n, M = 300, 2_000_000
mu = np.sort(np.random.default_rng(5).normal(size=n)) * 1.2
KW = {CorrMethod.SPECTRUM: [{"kind": "marchenko_pastur"}, {"kind": "spiked"}],
      CorrMethod.FACTOR: [{"sparse_links": 30}]}
rows = []
for method, gen in CORR_GENERATORS.items():
    for kw in KW.get(method, [{}]):
        tag = method.value + ("" if not kw else ":" + str(list(kw.values())[0]))
        try:
            try:
                C = np.asarray(gen(n=n, rng=11, **kw))
            except TypeError:
                np.random.seed(11)
                C = np.asarray(gen(n=n, **kw))
            p1 = b.one_call(mu, C, n_blocks=20)
            pmc, counts = b.big_mc(mu, C, M, np.random.default_rng(4))
            sd = np.sqrt(np.maximum(pmc * (1 - pmc), 1e-300) / M)
            seen = counts >= 25
            z = (p1[seen] - pmc[seen]) / sd[seen]
            abs_err = np.abs(p1[seen] - pmc[seen])
            print(f"{tag:24s} resolvable={seen.sum():3d}  rms z={np.sqrt((z**2).mean()):6.2f}  "
                  f"median abs err={np.median(abs_err):.1e}  "
                  f"max abs err={abs_err.max():.1e}", flush=True)
        except Exception as e:
            print(f"{tag:24s} ERROR: {type(e).__name__}: {e}", flush=True)
