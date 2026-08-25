"""Phases IX and X: the full Bayesian optimization loop.

The authors' setup, unchanged: 100 random molecules to start, batches of 100,
twenty acquisition rounds, a Tanimoto GP refitted by marginal likelihood every
round. What changes is the acquisition step.

    qPO-MC-10k        their pipeline exactly -- prefilter to the top 10,000 by
                      UCB, build that 10,000 x 10,000 covariance, draw 10,000
                      joint samples, count winners
    qPO-fast-r4-10k   same prefilter, so that the prefilter and the estimator
                      can be blamed separately
    qPO-fast-r4-full  no prefilter: the factor GP hands over (mu, V, D) for
                      every remaining molecule and the probit runs on all of them
    F-LITE-full       full library, independence approximation
    UCB, Greedy       full library, no probability of maximality at all

Hyperparameters are fitted with the authors' gpytorch code; prediction runs
through the numpy posterior in factorgp.py, which test_factorgp.py checks
against gpytorch's own output to 1e-11. Doing it that way keeps the model
identical across methods while making the linear algebra explicit.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
QPO_DIR = Path(os.environ.get("QPO_DIR", Path.home() / "github" / "qPO")).expanduser()
if str(QPO_DIR) not in sys.path:
    sys.path.insert(0, str(QPO_DIR))

from factorgp import TanimotoPosterior, factor_posterior_streaming  # noqa: E402
from metrics import diversity_stats, select_batch  # noqa: E402
from pom import pom_fast, pom_flite, pom_full_mc, sobol_nodes  # noqa: E402


def fit_hyperparameters(X, y):
    """The authors' marginal-likelihood fit, returning plain numbers."""
    import torch
    from gp import TanimotoGP, fit_gp_hyperparameters
    model = TanimotoGP(train_x=torch.as_tensor(np.asarray(X, dtype=np.float64)),
                       train_y=torch.as_tensor(np.asarray(y, dtype=np.float64)))
    fit_gp_hyperparameters(model)
    return model.hparam_dict


def load_library(dataset, objective, c):
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from run import smiles_to_fingerprint_arr

    all_data = pd.read_csv(QPO_DIR / "data" / f"{dataset}.csv").sample(
        frac=1, random_state=0)
    smiles = list(all_data["smiles"])
    values = np.asarray(all_data[objective], dtype=float)
    ok = [Chem.MolFromSmiles(s) is not None for s in smiles]
    if not all(ok):
        smiles = [s for s, k in zip(smiles, ok) if k]
        values = values[np.asarray(ok)]
    cache = HERE / "snapshots" / f"{dataset}_fps.npy"
    if cache.exists():
        fps = np.load(cache, mmap_mode="r")
    else:
        fps = smiles_to_fingerprint_arr(smiles).astype(np.float32)
        np.save(cache, fps)
    return smiles, c * values, np.asarray(fps)


def acquire(method, post, X, cand, mu_c, var_c, batch, rank, sobol_m, points,
            inducing, prefilter, mc_samples, seed, max_elements):
    """Return positions within `cand` of the chosen batch, plus diagnostics."""
    info = {"rank": np.nan, "n_scored": len(cand), "sobol_nodes": np.nan,
            "n_mc_samples": np.nan}
    if method == "UCB":
        return select_batch(mu_c + np.sqrt(var_c), batch), info
    if method == "Greedy":
        return select_batch(mu_c, batch), info
    if method == "F-LITE-full":
        return select_batch(pom_flite(mu_c, var_c), batch, mu_c), info

    prefiltered = method.endswith("-10k")
    if prefiltered:
        keep = select_batch(mu_c + np.sqrt(var_c), min(prefilter, len(cand)))
        sub = cand[keep]
        info["n_scored"] = len(sub)
    else:
        keep = np.arange(len(cand))
        sub = cand

    Xs = np.ascontiguousarray(X[sub])
    if method.startswith("qPO-MC"):
        Sig = post.covariance(Xs)
        p = pom_full_mc(mu_c[keep], Sig, M=mc_samples, seed=seed, chunk=4000)
        info["n_mc_samples"] = mc_samples
        del Sig
        gc.collect()
        return keep[select_batch(p, batch, mu_c[keep])], info

    if method.startswith("qPO-fast"):
        _, V, d = factor_posterior_streaming(post, Xs, rank=rank,
                                             inducing=inducing, seed=seed)
        F, W = sobol_nodes(rank, m=sobol_m, seed=0)
        p = pom_fast(mu_c[keep], V, d, F, W, points=points,
                     max_elements=max_elements)
        info.update({"rank": rank, "sobol_nodes": len(F)})
        return keep[select_batch(p, batch, mu_c[keep])], info

    raise ValueError(method)


def dump_snapshot(tag, post, fps, cand, mu_c, var_c, oracle, iteration,
                  sizes=(500, 1000, 2000)):
    """Freeze a mid-run posterior so the rank ladder can be re-measured there.

    The iteration-0 posterior sits on 100 observations. If low rank were an
    artefact of a barely-trained GP, it would not survive to iteration 10 with
    a thousand observations and a candidate pool the model has already shaped.
    """
    out = HERE / "snapshots" / f"{tag}_iter{iteration}"
    ucb = mu_c + np.sqrt(var_c)
    order = np.argsort(-ucb)
    meta = {"dataset": tag, "seed": -1, "objective": "replayed",
            "iteration": iteration, "n_train": int(len(post.X)), "sizes": list(sizes)}
    for N in sizes:
        sel = order[:N]
        d = out / f"N{N}"
        d.mkdir(parents=True, exist_ok=True)
        Xs = np.ascontiguousarray(fps[cand[sel]])
        np.save(d / "mu.npy", mu_c[sel])
        np.save(d / "Sigma.npy", post.covariance(Xs))
        np.save(d / "fps.npy", Xs.astype(np.float32))
        np.save(d / "oracle.npy", oracle[cand[sel]])
    np.save(out / "gp_train_x.npy", np.asarray(post.X, dtype=np.float32))
    np.save(out / "gp_train_y.npy", post.y)
    (out / "meta.json").write_text(json.dumps(
        {**meta, "gp_hyperparameters": {
            "covar_module.outputscale": post.s2,
            "likelihood.noise": post.sn2,
            "mean_module.constant": post.m0}}, indent=2))
    print(f"    dumped mid-run snapshot to {out}", flush=True)


def run(dataset, objective, c, method, seed, n_iter, batch, initial, rank,
        sobol_m, points, inducing, prefilter, mc_samples, max_elements, out_rows,
        dump_at=()):
    smiles, oracle, fps = load_library(dataset, objective, c)
    N = len(smiles)
    order = np.argsort(-oracle)
    top1 = set(order[:max(1, N // 100)].tolist())
    top001 = set(order[:max(1, N // 10000)].tolist())
    best_possible = float(oracle.max())

    pos = {s: i for i, s in enumerate(smiles)}
    random.seed(seed)
    init = random.sample(sorted(set(smiles)), initial)
    acquired = np.array(sorted({pos[s] for s in init}))
    mask = np.zeros(N, dtype=bool)
    mask[acquired] = True

    t_start = time.time()
    for it in range(1, n_iter + 1):
        # `oracle` already carries the sign convention (c = -1 negates it on
        # load), so everything downstream is max-wins and trains on it directly.
        hp = fit_hyperparameters(fps[acquired], oracle[acquired])
        post = TanimotoPosterior(fps[acquired], oracle[acquired],
                                 hp["covar_module.outputscale"],
                                 hp["likelihood.noise"],
                                 hp["mean_module.constant"])
        cand = np.flatnonzero(~mask)
        t_score = time.perf_counter()
        mu_c, var_c = post.mean_and_variance(fps[cand])
        t_score = time.perf_counter() - t_score

        if it in dump_at:
            dump_snapshot(f"{dataset}_{objective}_bo{seed}", post, fps, cand,
                          mu_c, var_c, oracle, it)

        tracemalloc.start()
        t0 = time.perf_counter()
        sel_local, info = acquire(method, post, fps, cand, mu_c, var_c, batch,
                                  rank, sobol_m, points, inducing, prefilter,
                                  mc_samples, seed, max_elements)
        secs = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()

        chosen = cand[np.asarray(sel_local)]
        div = diversity_stats(fps[chosen], np.arange(len(chosen)))
        mask[chosen] = True
        acquired = np.flatnonzero(mask)
        vals = np.sort(oracle[acquired])

        row = {"dataset": dataset, "objective": objective, "method": method,
               "seed": seed, "iteration": it, "n_acquired": int(mask.sum()),
               "acquisition_seconds": secs, "peak_memory_mb": peak,
               "cumulative_seconds": time.time() - t_start,
               "library_scoring_seconds": t_score,
               "top1_ave": float(vals[-1]), "top10_ave": float(vals[-10:].mean()),
               "top100_ave": float(vals[-100:].mean()),
               "frac_top1pct": len(top1 & set(acquired.tolist())) / len(top1),
               "frac_top0.01pct": len(top001 & set(acquired.tolist())) / len(top001),
               "simple_regret": best_possible - float(vals[-1]),
               "gp_noise": hp["likelihood.noise"],
               "gp_outputscale": hp["covar_module.outputscale"],
               **{k: v for k, v in info.items()},
               **div}
        out_rows.append(row)
        print(f"  [{method} seed{seed}] iter {it:2d} "
              f"n={row['n_acquired']:5d} top10={row['top10_ave']:.4f} "
              f"top100={row['top100_ave']:.4f} "
              f"top1%={row['frac_top1pct']:.3f} "
              f"top0.01%={row['frac_top0.01pct']:.3f} "
              f"acq={secs:6.1f}s score={t_score:5.1f}s peak={peak:6.0f}MB "
              f"tan={row['mean_tanimoto']:.3f}", flush=True)
        del post
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="qm9")
    ap.add_argument("--objective", default="gap")
    ap.add_argument("--c", type=int, default=1)
    ap.add_argument("--methods", nargs="+",
                    default=["qPO-MC-10k", "qPO-fast-r4-10k", "qPO-fast-r4-full",
                             "F-LITE-full", "UCB", "Greedy"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--initial", type=int, default=100)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--sobol-m", type=int, default=6)
    ap.add_argument("--points", type=int, default=129)
    ap.add_argument("--inducing", type=int, default=256)
    ap.add_argument("--prefilter", type=int, default=10000)
    ap.add_argument("--mc-samples", type=int, default=10000)
    ap.add_argument("--max-elements", type=float, default=2.4e7)
    ap.add_argument("--out", default="closed_loop.csv")
    ap.add_argument("--dump-at", type=int, nargs="*", default=[])
    args = ap.parse_args()

    rows = []
    dest = HERE / "results" / args.out
    for method in args.methods:
        for seed in args.seeds:
            print(f"\n=== {method} seed {seed} ===", flush=True)
            run(args.dataset, args.objective, args.c, method, seed,
                args.n_iter, args.batch, args.initial, args.rank, args.sobol_m,
                args.points, args.inducing, args.prefilter, args.mc_samples,
                args.max_elements, rows, dump_at=set(args.dump_at))
            pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
