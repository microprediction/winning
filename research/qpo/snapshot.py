"""Freeze one qPO Gaussian-process posterior so every estimator sees the same
(mu, Sigma).

The point of the snapshot is that no method gets to see a different model. We
reproduce the authors' initialization exactly -- their shuffle, their seeded
random draw of 100 molecules, their Tanimoto GP, their marginal-likelihood fit,
their UCB prefilter -- and then write mu and Sigma to disk. Everything
downstream reads those arrays.

Run inside the fastqpo environment with the qPO checkout on the path:

    QPO_DIR=~/github/qPO conda run -n fastqpo python snapshot.py \
        --dataset qm9 --objective gap --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QPO_DIR = Path(os.environ.get("QPO_DIR", Path.home() / "github" / "qPO")).expanduser()
if str(QPO_DIR) not in sys.path:
    sys.path.insert(0, str(QPO_DIR))


def _load_qpo():
    """Import the authors' code. Kept in a function so --help works without torch."""
    import torch  # noqa: F401
    from acquisition_functions import mean_cov_from_gp
    from gp import TanimotoGP, fit_gp_hyperparameters
    from run import smiles_to_fingerprint_arr

    return mean_cov_from_gp, TanimotoGP, fit_gp_hyperparameters, smiles_to_fingerprint_arr


# --------------------------------------------------------------------------
# fingerprints
# --------------------------------------------------------------------------

def fingerprint_array(smiles_list, cache: Path | None, fp_fn) -> np.ndarray:
    """Count Morgan fingerprints (radius 3, 2048 bits) for the whole library.

    Stored as float32: the entries are small integer counts, so the widening
    cast back to float64 is exact and the cache is half the size.
    """
    if cache is not None and cache.exists():
        return np.load(cache)
    t0 = time.time()
    fps = fp_fn(smiles_list).astype(np.float32)
    print(f"  fingerprints: {fps.shape} in {time.time() - t0:.1f}s")
    if cache is not None:
        np.save(cache, fps)
    return fps


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------

def build_snapshot(dataset: str, objective: str, seed: int, sizes: list[int],
                   initial_batch_size: int = 100, beta: float = 1.0,
                   c: int = 1, selection: str = "ucb",
                   out_root: Path = HERE / "snapshots") -> Path:
    """c = +1 maximises the objective, c = -1 minimises it (the antibiotic
    screen). Everything downstream is max-wins, so for c = -1 the stored mean
    and oracle are negated and the covariance is left alone -- which is exactly
    the transformation the authors' own c*mean scoring applies."""
    import pandas as pd
    import torch

    (mean_cov_from_gp, TanimotoGP, fit_gp_hyperparameters,
     smiles_to_fingerprint_arr) = _load_qpo()

    tag = (f"{dataset}_{objective}_seed{seed}" + ("_min" if c == -1 else "")
           + ("" if selection == "ucb" else f"_{selection}"))
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)

    # ---- data, in the authors' order -------------------------------------
    all_data = pd.read_csv(QPO_DIR / "data" / f"{dataset}.csv").sample(frac=1, random_state=0)
    all_smiles = list(all_data["smiles"])
    values = np.asarray(all_data[objective], dtype=float)

    # A handful of the antibiotic-screen SMILES do not parse under current
    # RDKit (hypervalent Al, odd charges). The authors' featurizer would raise
    # on them, so they are dropped and the count is recorded rather than
    # silently patched.
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    ok = [Chem.MolFromSmiles(s) is not None for s in all_smiles]
    n_dropped = int(len(ok) - sum(ok))
    if n_dropped:
        print(f"  dropping {n_dropped} unparseable SMILES")
        all_smiles = [s for s, k in zip(all_smiles, ok) if k]
        values = values[np.asarray(ok)]
    test_data = {smi: v for smi, v in zip(all_smiles, values)}
    print(f"library: {len(all_smiles)} rows, {len(test_data)} unique smiles")

    fps = fingerprint_array(all_smiles, out_root / f"{dataset}_fps.npy",
                            smiles_to_fingerprint_arr)
    index = {smi: i for i, smi in enumerate(all_smiles)}   # last row wins on dupes

    # ---- the authors' initial random acquisition -------------------------
    unacquired = list(set(all_smiles))
    random.seed(seed)
    acquired_smiles = random.sample(sorted(unacquired), initial_batch_size)
    acquired = {smi: test_data[smi] for smi in acquired_smiles}
    unacquired = sorted(set(unacquired) - set(acquired_smiles))
    print(f"acquired {len(acquired)} molecules; {len(unacquired)} candidates remain")

    # ---- fit the Tanimoto GP --------------------------------------------
    X_train = np.asarray([fps[index[smi]] for smi in acquired], dtype=np.float64)
    y_train = np.asarray(list(acquired.values()), dtype=np.float64)
    t0 = time.time()
    model = TanimotoGP(train_x=torch.as_tensor(X_train), train_y=torch.as_tensor(y_train))
    fit_gp_hyperparameters(model)
    fit_seconds = time.time() - t0
    hp = model.hparam_dict
    print(f"GP fit in {fit_seconds:.1f}s: {hp}")

    # ---- UCB prefilter over the whole remaining library ------------------
    model.eval()
    model.likelihood.eval()
    t0 = time.time()
    mu_all = np.empty(len(unacquired))
    var_all = np.empty(len(unacquired))
    B = 4096
    with torch.no_grad():
        for a in range(0, len(unacquired), B):
            chunk = unacquired[a:a + B]
            Xc = np.asarray([fps[index[smi]] for smi in chunk], dtype=np.float64)
            pred = model.likelihood(model(torch.as_tensor(Xc)))
            mu_all[a:a + B] = pred.mean.detach().numpy()
            var_all[a:a + B] = pred.variance.detach().numpy()
    ucb_seconds = time.time() - t0
    ucb = c * mu_all + beta * np.sqrt(var_all)   # the authors' c*mean + beta*sd
    if selection == "ucb":
        order = np.argsort(-ucb)                   # the released prefilter
    elif selection == "random":
        # A UCB-prefiltered pool is chemically homogeneous by construction, so
        # a rank result measured only there might be an artefact of the
        # prefilter rather than a property of molecular posteriors. A uniform
        # random subset has the heterogeneity of the library itself, which is
        # what a full-library run actually faces.
        order = np.random.default_rng(seed).permutation(len(unacquired))
    else:
        raise ValueError(selection)
    print(f"UCB over {len(unacquired)} candidates in {ucb_seconds:.1f}s")

    # ---- one nested family of candidate universes ------------------------
    meta = {
        "dataset": dataset, "objective": objective, "seed": seed,
        "initial_batch_size": initial_batch_size, "beta": beta, "c": c,
        "selection": selection,
        "library_size": len(all_smiles), "n_unacquired": len(unacquired),
        "n_unparseable_dropped": n_dropped,
        "gp_hyperparameters": hp, "gp_fit_seconds": fit_seconds,
        "ucb_seconds": ucb_seconds, "sizes": sizes,
        "acquired_smiles": acquired_smiles,
        "numpy": np.__version__, "torch": torch.__version__,
    }

    for N in sizes:
        sel = order[:N]
        smis = [unacquired[i] for i in sel]
        Xc = np.asarray([fps[index[smi]] for smi in smis], dtype=np.float64)
        t0 = time.time()
        with torch.no_grad():
            mu, cov = mean_cov_from_gp(model=model, smiles=smis,
                                       featurizer={s: fps[index[s]].astype(np.float64)
                                                   for s in smis},
                                       full_cov=True, gpu=False)
        secs = time.time() - t0
        d = out / f"N{N}"
        d.mkdir(exist_ok=True)
        np.save(d / "mu.npy", c * np.asarray(mu, dtype=np.float64))
        np.save(d / "Sigma.npy", np.asarray(cov, dtype=np.float64))
        np.save(d / "fps.npy", Xc.astype(np.float32))
        np.save(d / "oracle.npy", c * np.asarray([test_data[s] for s in smis]))
        (d / "smiles.txt").write_text("\n".join(smis))
        cond = float(np.linalg.cond(cov)) if N <= 2000 else float("nan")
        print(f"  N={N:6d}: mu[{mu.min():.4f},{mu.max():.4f}] "
              f"var[{np.diag(cov).min():.3e},{np.diag(cov).max():.3e}] "
              f"cov in {secs:.1f}s cond={cond:.3e}")
        meta[f"N{N}_cov_seconds"] = secs

    # everything needed to rebuild the posterior in numpy (see factorgp.py)
    np.save(out / "gp_train_x.npy", X_train.astype(np.float32))
    np.save(out / "gp_train_y.npy", y_train)
    # and the whole remaining library, for the un-prefiltered runs
    np.save(out / "full_mu.npy", c * mu_all)
    np.save(out / "full_var.npy", var_all)
    np.save(out / "full_fp_index.npy",
            np.asarray([index[smi] for smi in unacquired], dtype=np.int64))
    np.save(out / "full_oracle.npy",
            c * np.asarray([test_data[smi] for smi in unacquired]))
    meta["fps_cache"] = str(out_root / f"{dataset}_fps.npy")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="qm9")
    ap.add_argument("--objective", default="gap")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 5000, 10000])
    ap.add_argument("--initial-batch-size", type=int, default=100)
    ap.add_argument("--c", type=int, default=1, choices=[1, -1])
    ap.add_argument("--selection", default="ucb", choices=["ucb", "random"])
    args = ap.parse_args()
    build_snapshot(args.dataset, args.objective, args.seed, args.sizes,
                   initial_batch_size=args.initial_batch_size, c=args.c,
                   selection=args.selection)


if __name__ == "__main__":
    main()
