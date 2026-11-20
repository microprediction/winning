"""Stopping on exact vs marginal-only probability of best, under CRN.

The setting is ranking & selection with common random numbers. Each
replicate r scores every system on one shared scenario:

    Y_ir = mu_i + v_i' F_r + eps_ir,   F_r ~ N(0, I_rho) shared,
                                       eps_ir ~ N(0, sigma_i^2).

The unknown is mu (prior N(0, s0^2 I)); V and sigma are the declared
simulation structure. Because F_r is shared, the posterior covariance
of mu given n replicates is EXACTLY factor-plus-diagonal:

    precision = I/s0^2 + n Lambda^{-1},  Lambda = V V' + diag(sigma^2),

and Woodbury turns the inverse back into  Sigma_n = diag(d) + W W'.
So the posterior probability that system i is truly best is an exact
factor race -- the quantity VAPOR (arXiv:2311.13294, p.4) calls
"several complicated integrals ... intractable in most cases" and
ToSFiT/VBOS (arXiv:2510.13328, Eq. 1-2) approximates from marginals
alone.

Three estimates of P(i best | data), all from the same posterior:
  exact  -- winning race_probabilities on (mu_n, W, d), max-race by
            negation; certified against Monte Carlo argmax counts.
  indep  -- the F-LITE construction (threshold kappa* with
            sum_i Phi((m_i - kappa*)/s_i) = 1), marginals only.
  vbos   -- ToSFiT Eq. 2: pi_x = v((m_x - kappa*)/s_x),
            v(c) = exp(-(sqrt(c^2+4) - c)^2 / 8), sum pi = 1.
            (An optimistic policy, not a PoM estimate -- included
            because it is what the fine-tuning target actually is.)

The stopping experiment: each rule stops at the first n where its own
max estimate >= 1 - delta and selects its argmax. Measured over many
replications: realized probability of correct selection, and
replicates consumed. The two correlation regimes are chosen to bend
the marginal-only rule in both directions:
  aligned  -- the top contenders load on the SAME factor direction
              (their difference variance is small, so the true leader
              probability exceeds the independence estimate: indep is
              conservative and overspends replicates);
  opposed  -- the top contenders load on OPPOSING directions (their
              difference variance is large: indep is overconfident
              and its realized PCS falls below nominal).
Both distortions vanish at c = 0 (no shared factor), which is the
control row.

Three further rules separate the effect of correlation from the
threshold bias of the marginal surrogates (added 2026-11-19):
  marg   -- the EXACT independent-marginal probabilities: the race on
            the marginal sds with no factor (Menet et al. 2025, Prop. 1
            without the threshold step). Ignores correlation only.
  bonf   -- a correlation-aware lower bound from the exact pairwise
            contrast variances: max_i 1 - sum_j Phi(-(m_i-m_j)/s_ij),
            s_ij^2 = d_i + d_j + |w_i - w_j|^2. Stops when >= 1 - delta.
  mc     -- counting the argmax of MC_DRAWS draws from the correlated
            posterior (its own random stream).
The original three rules see exactly the random numbers they saw
before: data are drawn from the experiment stream for as long as the
original loop would have drawn them, and from a separate stream after
that, so the original rows reproduce bit for bit.
"""
import json
import os
import sys

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from winning.factor import race_probabilities          # noqa: E402
from scipy.stats import norm                            # noqa: E402

DELTA = 0.05
S0 = 1.0            # prior sd of true means
K = 100             # systems
RHO = 2             # shared factors
N_MAX = 400         # replicate budget
REPS = 200          # experiment replications per configuration
CERT_EVERY = 50     # Monte Carlo certification cadence (per config)
MC_CERT = 200_000
MC_DRAWS = 2_000    # draws per step for the counting rule
OLD_RULES = ("exact", "indep", "vbos")
RULES = OLD_RULES + ("marg", "bonf", "mc")
# One fixed seed per regime (was hash((kind, c)), which is randomized per
# process by PYTHONHASHSEED and so not reproducible).
SEEDS = {"independent": 20261101, "aligned": 20261102, "opposed": 20261103}


def make_config(kind, c):
    """Loadings V (K x RHO) and idiosyncratic sd for one regime.

    c is the factor share of replicate variance; total replicate
    variance is 1 for every system so the marginal information per
    replicate is identical across regimes -- only the correlation
    geometry differs.
    """
    u = np.zeros((K, RHO))
    if kind == "aligned":
        u[:, 0] = 1.0
    elif kind == "opposed":
        u[: K // 2, 0] = 1.0
        u[K // 2:, 0] = -1.0
    elif kind == "independent":
        pass
    else:
        raise ValueError(kind)
    V = np.sqrt(c) * u
    sigma = np.sqrt(np.maximum(1.0 - c * (u ** 2).sum(1), 1e-9))
    return V, sigma


def posterior_factor_form(n, ybar, V, sigma):
    """Exact posterior N(mu_n, diag(d) + W W') after n CRN replicates.

    Lambda = V V' + diag(sigma^2); precision A = I/s0^2 + n Lambda^-1.
    Woodbury twice keeps everything K x RHO. Returns mu_n, W, d and,
    for the marginal-only rules, the exact marginal sds.
    """
    s2 = sigma ** 2
    # Lambda^{-1} = D^{-1} - D^{-1} V M^{-1} V' D^{-1},  M = I + V'D^{-1}V
    Dinv = 1.0 / s2
    M = np.eye(RHO) + (V.T * Dinv) @ V
    G = np.linalg.solve(M, (V * Dinv[:, None]).T)      # RHO x K
    # A = diag(a) - U S U' with a = 1/s0^2 + n/s2, U = D^{-1}V, S = n M^{-1}
    a = 1.0 / S0 ** 2 + n * Dinv
    U = V * Dinv[:, None]
    S = n * np.linalg.inv(M)
    # Sigma = A^{-1} = diag(1/a) + (U/a) (S^{-1} - U' diag(1/a) U)^{-1} (U/a)'
    Ua = U / a[:, None]
    core = np.linalg.inv(S) - U.T @ Ua
    C = np.linalg.inv(core)
    # C is symmetric PSD here (A PSD); factor it for the race grammar
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 0.0)
    W = Ua @ (evecs * np.sqrt(evals))
    d = 1.0 / a
    # posterior mean: mu_n = Sigma_n (n Lambda^{-1} ybar)
    h = n * (Dinv * ybar - (V * Dinv[:, None]) @ (G @ ybar))
    mu_n = h / a + Ua @ (C @ (Ua.T @ h))
    sd = np.sqrt(d + (W ** 2).sum(1))
    return mu_n, W, d, sd


def exact_pom(mu_n, W, d):
    """Max-race by negation of the min-race engine."""
    return race_probabilities(-mu_n, V=-W, D=d)


def flite_pom(m, s):
    """F-LITE / independence PoM: sum_i Phi((m_i - kappa)/s_i) = 1."""
    lo, hi = m.max() - 40 * s.max(), m.max() + 40 * s.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        tot = norm.cdf((m - mid) / s).sum()
        if tot > 1.0:
            lo = mid
        else:
            hi = mid
    q = norm.cdf((m - 0.5 * (lo + hi)) / s)
    return q / q.sum()


def vbos_policy(m, s):
    """ToSFiT Eq. 2: pi = v((m - kappa*)/s), v(c)=exp(-(sqrt(c^2+4)-c)^2/8)."""
    def v(c):
        return np.exp(-(np.sqrt(c ** 2 + 4.0) - c) ** 2 / 8.0)
    lo, hi = m.max() - 40 * s.max(), m.max() + 40 * s.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if v((m - mid) / s).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    p = v((m - 0.5 * (lo + hi)) / s)
    return p / p.sum()


def marg_pom(m, s):
    """Exact probabilities if the systems were independent with these
    marginal sds: the race with no factor."""
    return race_probabilities(-m, D=s ** 2)


def bonferroni_lcb(m, W, d):
    """Per-system lower bound on P(i best) from exact pairwise contrasts."""
    var = (d[:, None] + d[None, :]
           + ((W[:, None, :] - W[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(var, 1.0)
    z = (m[:, None] - m[None, :]) / np.sqrt(var)
    tail = norm.cdf(-z)
    np.fill_diagonal(tail, 0.0)
    return 1.0 - tail.sum(1)


def mc_pom(m, W, d, rng_mc):
    draws = (m[None, :] + rng_mc.normal(size=(MC_DRAWS, W.shape[1])) @ W.T
             + rng_mc.normal(size=(MC_DRAWS, len(m))) * np.sqrt(d))
    return np.bincount(draws.argmax(1), minlength=len(m)) / MC_DRAWS


def run_config(kind, c, rng, log=None, run_id=None, rng_extra=None,
               rng_mc=None):
    """One regime. If `log` is an open file, every replication appends one
    JSON line (regime, seed, run id, the raw stopped_n / correct / TV
    values) as it finishes, so an interrupted run keeps what it measured.
    The logging consumes no random numbers and changes no statistic."""
    if rng_extra is None:
        rng_extra = np.random.default_rng(SEEDS[kind] + 1000)
    if rng_mc is None:
        rng_mc = np.random.default_rng(SEEDS[kind] + 2000)
    V, sigma = make_config(kind, c)
    rules = RULES
    out = {r: {"stopped_n": [], "correct": [], "tv_vs_exact": []}
           for r in rules}
    cert_worst = 0.0
    for rep in range(REPS):
        mu_true = rng.normal(0.0, S0, K)
        best = int(np.argmax(mu_true))
        ysum = np.zeros(K)
        live = {r: True for r in rules}
        certify = (rep % CERT_EVERY == 0)
        rep_tv = {r: [] for r in rules}
        rep_cert = None
        old_drawing = True       # would the original loop still draw here?
        for n in range(1, N_MAX + 1):
            src = rng if old_drawing else rng_extra
            F = src.normal(size=RHO)
            eps = src.normal(0.0, sigma)
            ysum += mu_true + V @ F + eps
            old_live = any(live[r] for r in OLD_RULES)
            if not old_live:
                old_drawing = False   # the original loop broke at this step
            if not any(live.values()):
                break
            mu_n, W, d, sd = posterior_factor_form(n, ysum / n, V, sigma)
            # the original three are evaluated exactly while the original
            # loop would have run; the new rules need none of them
            if old_live:
                p_exact = exact_pom(mu_n, W, d)
                p_indep = flite_pom(mu_n, sd)
                p_vbos = vbos_policy(mu_n, sd)
            else:
                p_exact = p_indep = p_vbos = None
            p_marg = marg_pom(mu_n, sd) if live["marg"] else None
            p_bonf = bonferroni_lcb(mu_n, W, d) if live["bonf"] else None
            p_mc = mc_pom(mu_n, W, d, rng_mc) if live["mc"] else None
            if certify and n == 20 and old_live:
                draws = (mu_n[None, :]
                         + rng.normal(size=(MC_CERT, RHO)) @ W.T
                         + rng.normal(size=(MC_CERT, K)) * np.sqrt(d))
                counts = np.bincount(draws.argmax(1), minlength=K) / MC_CERT
                rep_cert = float(0.5 * np.abs(counts - p_exact).sum())
                cert_worst = max(cert_worst, rep_cert)
            for r, p in (("exact", p_exact), ("indep", p_indep),
                         ("vbos", p_vbos), ("marg", p_marg),
                         ("bonf", p_bonf), ("mc", p_mc)):
                if live[r]:
                    if r in ("indep", "vbos"):
                        tv_n = 0.5 * np.abs(p - p_exact).sum()
                        out[r]["tv_vs_exact"].append(tv_n)
                        rep_tv[r].append(float(tv_n))
                    if p.max() >= 1.0 - DELTA:
                        live[r] = False
                        out[r]["stopped_n"].append(n)
                        out[r]["correct"].append(
                            int(int(np.argmax(p)) == best))
        for r in rules:
            if live[r]:
                out[r]["stopped_n"].append(N_MAX)
                out[r]["correct"].append(-1)      # censored, never stopped
        if log is not None:
            row = dict(run_id=run_id, regime=kind, c=c, seed=SEEDS[kind],
                       rep=rep, best=best, mc_certificate_tv=rep_cert)
            for r in rules:
                row[r] = dict(stopped_n=int(out[r]["stopped_n"][-1]),
                              correct=int(out[r]["correct"][-1]),
                              tv_vs_exact=rep_tv[r])
            log.write(json.dumps(row) + "\n")
            log.flush()
    summary = {}
    for r in rules:
        stopped = np.array(out[r]["stopped_n"], float)
        corr = np.array(out[r]["correct"])
        dec = corr >= 0
        summary[r] = dict(
            mean_n=float(stopped.mean()),
            censored=int((~dec).sum()),
            stopped=int(dec.sum()),
            mean_n_stopped=(float(stopped[dec].mean()) if dec.any()
                            else None),
            pcs=float(corr[dec].mean()) if dec.any() else None,
            # TV against the exact rule is measured only for indep and
            # vbos; it is zero for exact itself and unmeasured (None) for
            # the rules added later.
            tv_median=(float(np.median(out[r]["tv_vs_exact"]))
                       if out[r]["tv_vs_exact"]
                       else (0.0 if r == "exact" else None)),
            tv_p90=(float(np.quantile(out[r]["tv_vs_exact"], 0.9))
                    if out[r]["tv_vs_exact"]
                    else (0.0 if r == "exact" else None)),
        )
    summary["mc_certificate_worst_tv"] = float(cert_worst)
    return summary


def provenance(seeds):
    """Commit, package source/version, numpy/scipy versions and seeds."""
    import subprocess
    import platform
    import scipy
    import winning
    root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        "..", "..", ".."))
    try:
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--",
             os.path.dirname(os.path.abspath(__file__)), "winning"],
            text=True).strip()
    except Exception as e:                       # pragma: no cover
        commit, dirty = f"unavailable: {e}", ""
    return dict(
        git_commit=commit,
        uncommitted_changes_in_script_dir_or_package=dirty.splitlines(),
        winning_source=os.path.dirname(os.path.abspath(winning.__file__)),
        winning_version=getattr(winning, "__version__", None),
        winning_imported_from_worktree=os.path.abspath(
            winning.__file__).startswith(root + os.sep),
        numpy=np.__version__,
        scipy=scipy.__version__,
        python=platform.python_version(),
        seeds=seeds,
        script=os.path.basename(__file__),
    )


def sanity():
    """Two independent systems: analytic PoM check for the negation."""
    mu = np.array([0.3, 0.0])
    W = np.zeros((2, 1))
    d = np.array([1.0, 1.0])
    p = exact_pom(mu, W, d)
    p_true = norm.cdf(0.3 / np.sqrt(2.0))
    assert abs(p[0] - p_true) < 1e-3, (p, p_true)


def write_json_atomic(path, obj):
    """Write to a sibling temp file and rename, so a crash mid-write never
    leaves a truncated file at `path`."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    import argparse
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    canonical = os.path.join(here, "results.json")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=tuple(SEEDS),
                    help="run a single regime (reproducibility checks); "
                         "the output then goes to a stamped file, never "
                         "to results.json")
    ap.add_argument("--out", default=None,
                    help="aggregate JSON (default results.json, or "
                         "results_<regime>_<stamp>.json under --only)")
    args = ap.parse_args()
    t0 = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime(t0))
    if args.out is None:
        args.out = (canonical if args.only is None else
                    os.path.join(here, f"results_{args.only}_{run_id}.json"))
    elif args.only and os.path.abspath(args.out) == canonical:
        raise SystemExit("--only runs a single regime; refusing to overwrite "
                         "the full results.json (pass another --out)")
    sanity()
    results = {"provenance": provenance(SEEDS)}
    results["provenance"]["run_id"] = run_id
    # raw per-replication rows, appended and flushed as they are produced
    log_path = os.path.join(here, f"replications_{run_id}_{os.getpid()}.jsonl")
    results["provenance"]["raw_log"] = os.path.basename(log_path)
    log = open(log_path, "a")
    for kind, c in [("independent", 0.0), ("aligned", 0.6),
                    ("opposed", 0.6)]:
        if args.only and kind != args.only:
            continue
        rng = np.random.default_rng(SEEDS[kind])
        rng_extra = np.random.default_rng(SEEDS[kind] + 1000)
        rng_mc = np.random.default_rng(SEEDS[kind] + 2000)
        results[f"{kind}_c{c}"] = run_config(kind, c, rng, log=log,
                                             run_id=run_id,
                                             rng_extra=rng_extra,
                                             rng_mc=rng_mc)
        s = results[f"{kind}_c{c}"]
        print(f"[{kind} c={c}] cert TV {s['mc_certificate_worst_tv']:.4f}")
        for r in RULES:
            row = s[r]
            print(f"  {r:6s} mean n {row['mean_n']:6.1f}  "
                  f"PCS {row['pcs'] if row['pcs'] is not None else 'NA'}  "
                  f"censored {row['censored']:3d}  "
                  + ("TV med/p90 n/a" if row['tv_median'] is None else
                     f"TV med/p90 {row['tv_median']:.3f}/{row['tv_p90']:.3f}"))
    log.close()
    results["provenance"]["wall_seconds"] = round(time.time() - t0, 1)
    write_json_atomic(args.out, results)
    print(f"total {time.time() - t0:.0f}s; wrote {args.out} "
          f"(raw rows in {log_path})")
