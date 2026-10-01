"""The factor-node rule: one per-rank table, the same in three languages,
and live at every rank it names (GH_RULE in winning/factor/races.py).

A single sharpness threshold of 3.0 used to serve every rank, and at rank
3 it was defending against a Gauss-Hermite cap of 15 that was itself 3x
WORSE than the Sobol rule it escalated to. The table pairs each rank's cap
with the sharpness past which even that order loses, both measured; these
tests pin the dispatch those numbers imply, and the accuracy claim that
justifies the rank-3 entry.
"""
import pathlib
import re

import numpy as np
import pytest

from winning.factor.core import hermite_nodes, qmc_nodes
from winning.factor.races import GH_RULE, GH_RULE_DEFAULT, _setup

ROOT = pathlib.Path(__file__).resolve().parents[1]
R_SRC = (ROOT / "r" / "winning" / "R" / "races.R").read_text()
JS_SRC = (ROOT / "docs" / "js" / "winning" / "races.mjs").read_text()
SOBOL = len(qmc_nodes(2, m=13)[0])          # 8192, the escalation target


def field(n, r, target, seed=0):
    """A field whose post-gauge-fix sharpness is near `target`; the test
    reads the sharpness back rather than trusting the construction, since
    centering V moves it."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, r))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ratio = rng.uniform(0.3, 1.0, size=n)
    ratio = ratio * (target / (np.sqrt(2) * ratio.max()))
    h2 = ratio ** 2 / (1 + ratio ** 2)
    V, D = U * np.sqrt(h2)[:, None], 1 - h2
    Vc = V - V.mean(axis=0)
    sharp = float(np.sqrt(2) * np.max(np.sqrt((Vc ** 2).sum(1)) / np.sqrt(D)))
    return rng.normal(size=n) * 0.5, V, D, sharp


def chosen(mu, V, D):
    return len(_setup(mu, V, D, None, None, "normal")[3])


def expected(r, sharp):
    cap, sharp_max = GH_RULE.get(r, GH_RULE_DEFAULT)
    if r >= 2 and sharp > sharp_max:
        return SOBOL
    if cap ** r > 100_000:
        return SOBOL
    return len(hermite_nodes(r, Q=int(np.clip(np.ceil(8.0 * sharp), 15, cap)))[0])


@pytest.mark.parametrize("r,target", [(2, 2.0), (2, 3.0), (2, 5.0),
                                      (3, 2.0), (3, 4.0), (3, 7.0),
                                      (4, 2.0), (4, 4.0), (5, 2.0)])
def test_dispatch_follows_the_table(r, target):
    mu, V, D, sharp = field(60, r, target)
    assert chosen(mu, V, D) == expected(r, sharp), (r, sharp)


def test_rank_three_cap_is_live():
    """The entry that matters: a realistic correlated field at rank 3 takes
    the raised cap, neither the old Q=15 tensor nor the Sobol escalation."""
    mu, V, D, sharp = field(60, 3, 4.0)
    assert 3.0 < sharp < 6.0
    assert chosen(mu, V, D) == len(hermite_nodes(3, Q=31)[0]) == 4067
    assert chosen(mu, V, D) not in (len(hermite_nodes(3, Q=15)[0]), SOBOL)


def test_rank_two_keeps_its_cap_and_gains_its_band():
    """Sharpness 3-4 was escalating to 8192 nodes; 413 are better there."""
    mu, V, D, sharp = field(60, 2, 3.0)
    assert 3.0 < sharp < GH_RULE[2][1]
    assert chosen(mu, V, D) < 500


def test_rank_four_and_up_still_escalate_at_three():
    """No cheap side to reach for: the rank-4 tensor is already dearer than
    Sobol at Q=15 (10929 nodes against 8192), so the old threshold stands."""
    assert GH_RULE.get(4, GH_RULE_DEFAULT) == GH_RULE_DEFAULT == (15, 3.0)
    assert len(hermite_nodes(4, Q=15)[0]) > SOBOL
    mu, V, D, sharp = field(60, 4, 4.0)
    assert sharp > 3.0 and chosen(mu, V, D) == SOBOL


def test_every_cap_fits_the_tensor_budget():
    """Raising a cap must not quietly breach the 1e5-node budget: the guard
    is measured on the order actually reachable, so this is a live check."""
    for r, (cap, _) in GH_RULE.items():
        if r == 1:
            continue                      # rank 1 hands over to a 1-D grid
        assert cap ** r <= 100_000, (r, cap)


def test_rank_one_is_untouched_by_the_sharpness_escalation():
    """The escalation is guarded r >= 2; rank 1 hands over to the
    midpoint-quantile grid at Q > 80 instead."""
    mu, V, D, sharp = field(60, 1, 6.0)
    assert sharp > 3.0
    Q = int(min(np.ceil(8.0 * sharp), 4001))
    assert chosen(mu, V, D) == (Q if Q > 80 else len(hermite_nodes(1, Q=max(Q, 15))[0]))


def test_table_agrees_across_the_three_ports():
    r_caps = re.search(r"cap <- if \(r == 1\) (\d+) else if \(r == 2\) (\d+) "
                       r"else if \(r == 3\) (\d+) else (\d+)", R_SRC)
    js_caps = re.search(r"cap = r === 1 \? (\d+) : r === 2 \? (\d+) : "
                        r"r === 3 \? (\d+) : (\d+)", JS_SRC)
    assert r_caps and js_caps
    py_caps = [GH_RULE[1][0], GH_RULE[2][0], GH_RULE[3][0], GH_RULE_DEFAULT[0]]
    assert [int(x) for x in r_caps.groups()] == py_caps
    assert [int(x) for x in js_caps.groups()] == py_caps

    r_sm = re.search(r"sharp_max <- if \(r == 2\) ([\d.]+) else if \(r == 3\) "
                     r"([\d.]+) else ([\d.]+)", R_SRC)
    js_sm = re.search(r"sharpMax = r === 2 \? ([\d.]+) : r === 3 \? "
                      r"([\d.]+) : ([\d.]+)", JS_SRC)
    assert r_sm and js_sm
    py_sm = [GH_RULE[2][1], GH_RULE[3][1], GH_RULE_DEFAULT[1]]
    assert [float(x) for x in r_sm.groups()] == py_sm
    assert [float(x) for x in js_sm.groups()] == py_sm


def test_ports_measure_the_budget_on_the_cap():
    """15^r would let a raised cap breach the budget silently."""
    assert "cap^r > 1e5" in R_SRC and "Math.pow(cap, r) > 100000" in JS_SRC


@pytest.mark.parametrize("target", [3.2, 4.0])
def test_rank_three_choice_beats_the_rule_it_replaced(target):
    """The load-bearing measurement, re-run small: at the sharpness where
    the old rule escalated, its 8192 Sobol nodes are less accurate than the
    4067 the table now picks. Reference: two 2^15 scrambles."""
    from winning.factor import race_probabilities
    mu, V, D, sharp = field(60, 3, target)
    assert 3.0 < sharp < GH_RULE[3][1]
    # the worst field at this sharpness, not the median one, is what the
    # threshold has to survive; see GH_RULE for the 8-seed win rates
    refs = np.array([np.asarray(race_probabilities(mu, V=V, D=D, F=F, W=W))
                     for F, W in (qmc_nodes(3, m=15, seed=q) for q in range(1, 5))])
    ref = refs.mean(0)
    sem = 0.5 * float(refs.std(0, ddof=1).sum()) / np.sqrt(len(refs))

    def tv(F, W):
        p = np.asarray(race_probabilities(mu, V=V, D=D, F=F, W=W))
        return 0.5 * float(np.abs(p - ref).sum())

    new = tv(*hermite_nodes(3, Q=GH_RULE[3][0]))
    old = tv(*qmc_nodes(3, m=13))
    assert new < old, (new, old)
    # the gap is resolved by the reference, not noise in it
    assert old > 3 * sem, (old, sem)
