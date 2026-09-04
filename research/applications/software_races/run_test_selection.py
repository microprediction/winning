"""Correlated test selection as an extremal race, simulated.

A bug lives in some code modules. A test detects it if it covers a
buggy module (with detection probability d). Tests that cover the
same modules FAIL TOGETHER -- their failures are correlated, and the
coverage matrix IS the factor loading. "The suite catches the bug"
is P(at least one selected test fails) = 1 - P(all selected pass),
an extremal-race object. Selecting k tests to maximize it is exactly
the group-selection this package prices: submodular, and correlation
is the whole game.

We do not need public data to make the point: the ground truth is
generated from the coverage structure, which in reality is just the
coverage matrix of the test suite (coverage.py, or Defects4J for real
bugs). Here we generate realistic coverage (modules of varying
popularity, tests covering a few modules each) and compare three
selection rules at a fixed test budget, scored by fault-detection
rate over many random bug placements:

  independent  greedy by each test's MARGINAL detection rate -- the
               incumbent that ignores overlap and piles onto tests
               covering the same popular modules.
  correlated   greedy on the ACTUAL catch probability P(at least one
               selected fails), which the shared-field / cavity
               machinery evaluates and whose marginal gain is
               submodular -- it spreads coverage, dropping redundant
               tests (the duplicates-vs-specialist insight).
  random       control.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def make_suite(n_tests, n_modules, rng):
    """Coverage matrix C[test, module] in {0,1}; module popularity
    is heavy-tailed (a few modules covered by many tests)."""
    pop = rng.pareto(1.5, n_modules) + 0.2
    pop /= pop.sum()
    C = np.zeros((n_tests, n_modules), bool)
    for t in range(n_tests):
        deg = 1 + rng.poisson(3)                 # test covers a few
        mods = rng.choice(n_modules, size=min(deg, n_modules),
                          replace=False, p=pop)
        C[t, mods] = True
    return C


def catch_prob_mc(C, selected, bug_rate, detect, n_mc, rng):
    """True P(selected set catches a random bug), Monte Carlo over
    bug placements and detection noise. Ground truth."""
    n_mod = C.shape[1]
    caught = 0
    S = C[selected]
    for _ in range(n_mc):
        buggy = rng.random(n_mod) < bug_rate
        if not buggy.any():
            continue
        # each selected test detects if it covers a buggy module
        covers_bug = (S & buggy[None, :]).any(1)
        det = covers_bug & (rng.random(len(selected)) < detect)
        caught += det.any()
    return caught / n_mc


def select_independent(C, k, bug_rate, detect):
    """Top-k by marginal detection rate (ignores overlap)."""
    p_cover = 1 - np.prod(1 - bug_rate * C, axis=1)   # P(covers a bug)
    marg = detect * p_cover
    return list(np.argsort(marg)[::-1][:k])


def select_correlated(C, k, bug_rate, detect):
    """Greedy on the catch probability: add the test that most raises
    P(at least one selected fails), factor-conditioned on the bug.
    Conditional on the bug vector the tests are independent, so
    P(none catch) = E_bug prod_{i in S} (1 - detect * covers_i(bug)).
    Approximate the expectation analytically per module: for module m,
    a selected test covering m fails to catch a bug there w.p.
    (1 - detect); the miss probability factorizes over modules."""
    n_mod = C.shape[1]
    chosen = []
    # log P(no selected test catches a bug in module m), per module
    log_miss = np.zeros(n_mod)          # starts at 0 (no tests)
    def gain(t):
        # adding test t multiplies miss prob in covered modules by
        # (1 - detect); overall catch = sum_m bug_rate * (1 - miss_m)
        new = log_miss.copy()
        new[C[t]] += np.log(1 - detect)
        catch = np.sum(bug_rate * (1 - np.exp(new)))
        return catch
    for _ in range(k):
        best = max((t for t in range(C.shape[0]) if t not in chosen),
                   key=gain, default=None)
        if best is None:
            break
        chosen.append(best)
        log_miss[C[best]] += np.log(1 - detect)
    return chosen


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N_TESTS, N_MOD = 800, 200
    BUG_RATE, DETECT = 0.02, 0.8
    C = make_suite(N_TESTS, N_MOD, rng)
    print(f"{N_TESTS} tests, {N_MOD} modules, mean coverage "
          f"{C.sum(1).mean():.1f} modules/test; bug_rate {BUG_RATE}, "
          f"detect {DETECT}")
    results = {}
    for k in (5, 10, 20, 40, 80):
        ind = select_independent(C, k, BUG_RATE, DETECT)
        cor = select_correlated(C, k, BUG_RATE, DETECT)
        rnd = list(rng.choice(N_TESTS, k, replace=False))
        ev = lambda sel: catch_prob_mc(C, sel, BUG_RATE, DETECT,
                                       40000, np.random.default_rng(7))
        di, dc, dr = ev(ind), ev(cor), ev(rnd)
        # distinct modules covered (the diversity the correlated rule buys)
        mods = lambda sel: int(C[sel].any(0).sum())
        results[k] = dict(independent=di, correlated=dc, random=dr,
                          mod_ind=mods(ind), mod_cor=mods(cor))
        print(f"k={k:3d}: catch  correlated {dc:.3f}  independent "
              f"{di:.3f}  random {dr:.3f}   | distinct modules "
              f"cor {mods(cor)} vs ind {mods(ind)}  "
              f"| lift {(dc-di)/max(di,1e-9)*100:+.0f}%")
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"),
              indent=2)
    print("wrote results.json")
