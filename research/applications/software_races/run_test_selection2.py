"""Honest rerun: does the FACTOR model beat additional-greedy COVERAGE,
and only when there is latent co-failure structure beyond coverage?

The agent's correct objection: additional-greedy coverage (the TCP
incumbent) already spreads coverage and is submodular, so beating
'independent top-k' proves little. The real question is whether a
factor model of TEST CO-FAILURE -- which can see correlation NOT in
the observed coverage matrix -- beats coverage-greedy.

Failure model with a LATENT factor beyond coverage: a bug activates
(a) some code modules (captured by the coverage matrix, which
coverage-greedy sees) and (b) a hidden failure factor (shared
environment / assumption / upstream dependency -- the Eckhardt-Lee
'difficulty', NOT in the coverage matrix). Test i detects the bug if
it covers an activated module OR its hidden loading aligns with the
bug's hidden factor. Coverage-greedy is blind to (b); a factor model
fit to the historical test co-failure covariance sees it.

Baselines: independent top-k; additional-greedy COVERAGE (incumbent,
coverage matrix only); factor-correlated (selects on P(catch) under a
model fit to the co-failure covariance, which includes the latent
factor). Swept over latent-strength alpha: alpha=0 (no hidden
structure -> factor should TIE coverage-greedy) to alpha large (hidden
structure -> factor should WIN).
"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def make_world(n_tests, n_mod, n_hidden, alpha, rng):
    pop = rng.pareto(1.5, n_mod) + 0.2; pop /= pop.sum()
    C = np.zeros((n_tests, n_mod), bool)
    for t in range(n_tests):
        deg = 1 + rng.poisson(3)
        C[t, rng.choice(n_mod, min(deg, n_mod), replace=False, p=pop)] = True
    # hidden loadings: each test loads on one hidden factor
    W = np.zeros((n_tests, n_hidden))
    W[np.arange(n_tests), rng.integers(0, n_hidden, n_tests)] = alpha
    return C, W


def sample_fail(C, W, bug_rate, detect, rng):
    """One bug; return per-test failure (bool)."""
    n_mod, n_hidden = C.shape[1], W.shape[1]
    buggy = rng.random(n_mod) < bug_rate
    bug_hidden = np.zeros(n_hidden); bug_hidden[rng.integers(0, n_hidden)] = 1.0
    if not buggy.any():
        return None
    covers = (C & buggy[None, :]).any(1)                 # coverage hit
    hid = (W @ bug_hidden) > 0                            # hidden-factor hit
    hit = covers | hid
    return hit & (rng.random(len(hit)) < detect)


def cofail_cov(C, W, bug_rate, detect, rng, m=4000):
    """Empirical test co-failure covariance -- what a factor model
    would be fit to from history. Includes the latent factor."""
    n = C.shape[0]; acc = np.zeros((n, n)); mu = np.zeros(n); cnt = 0
    for _ in range(m):
        f = sample_fail(C, W, bug_rate, detect, rng)
        if f is None: continue
        x = f.astype(float); mu += x; acc += np.outer(x, x); cnt += 1
    mu /= cnt; cov = acc / cnt - np.outer(mu, mu)
    return mu, cov


def greedy_catch(scorefn, n, k):
    chosen = []
    for _ in range(k):
        best = max((t for t in range(n) if t not in chosen),
                   key=lambda t: scorefn(chosen + [t]), default=None)
        if best is None: break
        chosen.append(best)
    return chosen


def eval_catch(C, W, sel, bug_rate, detect, rng, m=30000):
    c = 0; tot = 0
    for _ in range(m):
        f = sample_fail(C, W, bug_rate, detect, rng)
        if f is None: continue
        tot += 1; c += f[sel].any()
    return c / tot


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    N, M, H = 600, 150, 12
    BR, DET, K = 0.02, 0.8, 10
    print(f"{N} tests, {M} modules, {H} hidden factors; k={K}")
    out = {}
    for alpha in (0.0, 0.5, 1.0):
        C, W = make_world(N, M, H, alpha, rng)
        mu, cov = cofail_cov(C, W, BR, DET, np.random.default_rng(9))
        # independent top-k: by marginal failure rate mu
        ind = list(np.argsort(mu)[::-1][:K])
        # additional-greedy COVERAGE: maximize distinct covered modules
        cov_sel = greedy_catch(lambda s: C[s].any(0).sum(), N, K)
        # factor-correlated: maximize P(>=1 fail) approx from the
        # co-failure covariance -- 1 - prod over a Gaussian-copula
        # style catch; use the empirical P(all pass) via mu and pairwise
        # (second-order incl.-excl. on the fitted moments)
        def cat(s):
            s = list(s)
            p = mu[s]
            # inclusion-exclusion to 2nd order with covariance
            first = p.sum()
            pij = 0.0
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    pij += p[i]*p[j] + cov[s[i], s[j]]
            return first - pij            # approx P(at least one)
        fac = greedy_catch(cat, N, K)
        e_ind = eval_catch(C, W, ind, BR, DET, np.random.default_rng(11))
        e_cov = eval_catch(C, W, cov_sel, BR, DET, np.random.default_rng(11))
        e_fac = eval_catch(C, W, fac, BR, DET, np.random.default_rng(11))
        out[alpha] = dict(independent=e_ind, coverage_greedy=e_cov,
                          factor=e_fac)
        print(f"alpha={alpha}: catch  factor {e_fac:.3f}  "
              f"coverage-greedy {e_cov:.3f}  independent {e_ind:.3f}   "
              f"| factor vs coverage {(e_fac-e_cov)/max(e_cov,1e-9)*100:+.0f}%")
    json.dump(out, open(os.path.join(HERE, "results2.json"), "w"), indent=2)
    print("wrote results2.json")
