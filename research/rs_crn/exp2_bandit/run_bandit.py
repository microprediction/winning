"""Correlated Gaussian bandit: exact PoM vs its variational surrogates.

Max-reward convention. Arm means theta ~ N(0, Sigma0) with
Sigma0 = V V' + diag(D0), rank 2; pulling arm a returns
theta_a + N(0, sigma_n^2). After per-arm counts c and sums s, the
posterior is N(m_t, Sigma_t) with

    Sigma_t^{-1} = Sigma0^{-1} + diag(c)/sigma_n^2,

and the same double-Woodbury as exp1_stopping keeps Sigma_t EXACTLY
factor-plus-diagonal: Sigma0^{-1} = D0^{-1} - U S0 U' (U = D0^{-1}V),
so the posterior precision is diag(a) - U S0 U' with
a = d0^{-1} + c/sigma_n^2, and inverting back gives
Sigma_t = diag(1/a) + W W'.

Policies, all acting on the same posterior law:
  ts      Thompson: sample theta-tilde, pull its argmax.
  exact   probability matching on the EXACT PoM vector (engine race,
          max by negation). VAPOR Lemma 8 says E[TS occupancy] = PoM,
          so ts and exact have identical expected behavior -- their
          regret curves coinciding is a MEASURED CHECK of Lemma 8,
          not a win.
  vbos    sample from the ToSFiT Eq. 2 policy (marginals only).
  flite   sample from the F-LITE independence PoM (marginals only).

Measured:
  * cumulative Bayes regret over T rounds (mean over replications);
  * the approximation gap TV(pi_policy, PoM_exact) at checkpoints
    along each policy's own trajectory (ts's is 0 by Lemma 8; vbos
    and flite are the exhibit);
  * terminal identification: P(posterior-mean argmax = true best)
    and expected shortfall theta_max - theta_rec at T.
Engine certified against Monte Carlo argmax counts at checkpoints.
"""
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scipy.stats import norm                       # noqa: E402
from winning.factor import race_probabilities      # noqa: E402

N = 50
RHO = 2
T = 200
REPS = 100
SIGMA_N = 1.0
GAP_EVERY = 10
# One fixed seed per regime (was hash((kind, 99)), which is randomized per
# process by PYTHONHASHSEED and so not reproducible).
SEEDS = {"independent": 20261111, "clusters": 20261112}
# Paired bootstrap for the regret ratio (its own stream, so it cannot
# perturb the experiment's draws).
BOOT = 10_000
BOOT_SEED = 20261120


def make_prior(kind, c):
    u = np.zeros((N, RHO))
    if kind == "clusters":
        u[: N // 2, 0] = 1.0
        u[N // 2:, 1] = 1.0
    V = np.sqrt(c) * u
    d0 = np.maximum(1.0 - (V ** 2).sum(1), 1e-9)
    return V, d0


def posterior(V, d0, counts, sums):
    """N(m, diag(d) + W W') after per-arm counts/sums, prior mean 0."""
    Dinv = 1.0 / d0
    M0 = np.eye(RHO) + (V.T * Dinv) @ V
    S0 = np.linalg.inv(M0)                       # Sigma0^-1 = D^-1 - U S0 U'
    U = V * Dinv[:, None]
    a = Dinv + counts / SIGMA_N ** 2
    Ua = U / a[:, None]
    C = np.linalg.inv(np.linalg.inv(S0) - U.T @ Ua)
    evals, evecs = np.linalg.eigh(C)
    W = Ua @ (evecs * np.sqrt(np.maximum(evals, 0.0)))
    d = 1.0 / a
    h = sums / SIGMA_N ** 2                      # precision-weighted data
    m = h / a + Ua @ (C @ (Ua.T @ h))
    return m, W, d


def exact_pom(m, W, d):
    return race_probabilities(-m, V=-W, D=d)     # max race by negation


def flite_pom(m, s):
    lo, hi = m.max() - 40 * s.max(), m.max() + 40 * s.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if norm.cdf((m - mid) / s).sum() > 1 else \
            (lo, mid)
    q = norm.cdf((m - 0.5 * (lo + hi)) / s)
    return q / q.sum()


def vbos_policy(m, s):
    def v(cv):
        return np.exp(-(np.sqrt(cv ** 2 + 4.0) - cv) ** 2 / 8.0)
    lo, hi = m.max() - 40 * s.max(), m.max() + 40 * s.max()
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if v((m - mid) / s).sum() > 1 else (lo, mid)
    p = v((m - 0.5 * (lo + hi)) / s)
    return p / p.sum()


def run_config(kind, c, rng, log=None, run_id=None):
    """One regime. If `log` is an open file, every replication appends one
    JSON line (regime, seed, run id, each policy's cumulative regret,
    identification, shortfall and checkpoint TV) as it finishes. The
    logging consumes no random numbers and changes no statistic."""
    V, d0 = make_prior(kind, c)
    policies = ("ts", "exact", "vbos", "flite")
    regret = {p: np.zeros(T) for p in policies}
    rep_regret = {p: [] for p in policies}     # cumulative, per replication
    tv = {p: {} for p in policies}
    ident = {p: [] for p in policies}
    shortfall = {p: [] for p in policies}
    cert_worst = 0.0
    for rep in range(REPS):
        z0 = rng.normal(size=RHO)
        theta = V @ z0 + rng.normal(0, np.sqrt(d0))
        best_val = theta.max()
        rep_tv = {p: {} for p in policies}
        rep_cert = None
        for pol in policies:
            counts = np.zeros(N)
            sums = np.zeros(N)
            rep_cum = 0.0
            for t in range(T):
                m, W, d = posterior(V, d0, counts, sums)
                sd = np.sqrt(d + (W ** 2).sum(1))
                need_gap = (t % GAP_EVERY == 0)
                if pol == "ts":
                    draw = (m + W @ rng.normal(size=W.shape[1])
                            + rng.normal(0, np.sqrt(d)))
                    a = int(draw.argmax())
                    pi = None
                elif pol == "exact":
                    p = exact_pom(m, W, d)
                    a = int(rng.choice(N, p=np.maximum(p, 0)
                                       / np.maximum(p, 0).sum()))
                    pi = p
                elif pol == "vbos":
                    pi = vbos_policy(m, sd)
                    a = int(rng.choice(N, p=pi))
                else:
                    pi = flite_pom(m, sd)
                    a = int(rng.choice(N, p=pi))
                if need_gap and pol in ("vbos", "flite"):
                    p_ex = exact_pom(m, W, d)
                    tv_t = 0.5 * np.abs(pi - p_ex).sum()
                    tv[pol].setdefault(t, []).append(tv_t)
                    rep_tv[pol][str(t)] = float(tv_t)
                    if rep == 0 and t == 50:
                        draws = (m[None, :]
                                 + rng.normal(size=(200_000, W.shape[1]))
                                 @ W.T
                                 + rng.normal(size=(200_000, N))
                                 * np.sqrt(d))
                        counts_mc = np.bincount(draws.argmax(1),
                                                minlength=N) / 200_000
                        cert_tv = float(0.5 * np.abs(counts_mc - p_ex).sum())
                        cert_worst = max(cert_worst, cert_tv)
                        rep_cert = (cert_tv if rep_cert is None
                                    else max(rep_cert, cert_tv))
                regret[pol][t] += best_val - theta[a]
                rep_cum += best_val - theta[a]
                counts[a] += 1
                sums[a] += theta[a] + rng.normal(0, SIGMA_N)
            rep_regret[pol].append(float(rep_cum))
            m, W, d = posterior(V, d0, counts, sums)
            rec = int(m.argmax())
            ident[pol].append(int(rec == int(theta.argmax())))
            shortfall[pol].append(best_val - theta[rec])
        if log is not None:
            row = dict(run_id=run_id, regime=kind, c=c, seed=SEEDS[kind],
                       rep=rep, mc_certificate_tv=rep_cert)
            for p in policies:
                row[p] = dict(cum_regret=rep_regret[p][-1],
                              ident=int(ident[p][-1]),
                              shortfall=float(shortfall[p][-1]),
                              tv_by_t=rep_tv[p])
            log.write(json.dumps(row) + "\n")
            log.flush()
    out = {}
    for p in policies:
        out[p] = dict(
            cum_regret=float(regret[p].sum() / REPS),
            cum_regret_per_rep=rep_regret[p],
            ident=float(np.mean(ident[p])),
            shortfall=float(np.mean(shortfall[p])),
            tv_by_t={str(t): float(np.mean(v))
                     for t, v in sorted(tv[p].items())} if tv[p] else {},
        )
    out["mc_certificate_worst_tv"] = float(cert_worst)
    # Paired comparisons vs Thompson: theta is shared within a replication
    # (each policy then draws its own noise from the common stream).
    ts = np.array(rep_regret["ts"])
    # Interval for the ratio mean(diff)/mean(ts): a paired percentile
    # bootstrap over replications, the SAME resampled replications in the
    # numerator and the denominator. (Dividing the endpoints of the
    # difference interval by the observed ts mean treated a random
    # denominator as fixed and was not a valid interval for the ratio.)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(ts),
                                                    size=(BOOT, len(ts)))
    out["paired_vs_ts"] = {}
    for p in ("exact", "vbos", "flite"):
        diff = np.array(rep_regret[p]) - ts
        mean = float(diff.mean())
        half = float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))
        ratio = 100.0 * diff[idx].mean(1) / ts[idx].mean(1)
        out["paired_vs_ts"][p] = dict(
            mean_diff=mean, ci95=[mean - half, mean + half],
            sd_diff=float(diff.std(ddof=1)), reps=int(len(diff)),
            pct_of_ts=100.0 * mean / float(ts.mean()),
            pct_ci95_bootstrap=[float(np.percentile(ratio, 2.5)),
                                float(np.percentile(ratio, 97.5))],
            pct_ci95_method=(f"paired percentile bootstrap of "
                             f"100*mean(diff)/mean(ts) over replications, "
                             f"{BOOT} resamples, seed {BOOT_SEED}"))
    return out


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


def write_json_atomic(path, obj):
    """Write to a sibling temp file and rename, so a crash mid-write never
    leaves a truncated file at `path`."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    import argparse
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
    results = {"provenance": provenance(SEEDS)}
    results["provenance"]["run_id"] = run_id
    # raw per-replication rows, appended and flushed as they are produced
    log_path = os.path.join(here, f"replications_{run_id}_{os.getpid()}.jsonl")
    results["provenance"]["raw_log"] = os.path.basename(log_path)
    log = open(log_path, "a")
    for kind, c in (("independent", 0.0), ("clusters", 0.75)):
        if args.only and kind != args.only:
            continue
        rng = np.random.default_rng(SEEDS[kind])
        results[f"{kind}"] = run_config(kind, c, rng, log=log, run_id=run_id)
        r = results[f"{kind}"]
        print(f"[{kind} c={c}] cert TV {r['mc_certificate_worst_tv']:.4f}")
        for p in ("ts", "exact", "vbos", "flite"):
            row = r[p]
            tvs = row["tv_by_t"]
            tvtxt = (" tv@10/100/190 "
                     + "/".join(f"{tvs.get(str(t), float('nan')):.3f}"
                                for t in (10, 100, 190))) if tvs else ""
            print(f"  {p:6s} regret {row['cum_regret']:7.2f}  "
                  f"ident {row['ident']:.3f}  "
                  f"shortfall {row['shortfall']:.4f}{tvtxt}")
        for p, q in r["paired_vs_ts"].items():
            print(f"  {p:6s} - ts  {q['mean_diff']:+7.2f} "
                  f"[{q['ci95'][0]:+.2f}, {q['ci95'][1]:+.2f}]  "
                  f"{q['pct_of_ts']:+.1f}% "
                  f"[{q['pct_ci95_bootstrap'][0]:+.1f}, "
                  f"{q['pct_ci95_bootstrap'][1]:+.1f}] (paired bootstrap)")
    log.close()
    results["provenance"]["wall_seconds"] = round(time.time() - t0, 1)
    print(f"total {time.time() - t0:.0f}s")
    write_json_atomic(args.out, results)
    print(f"wrote {args.out} (raw rows in {log_path})")
