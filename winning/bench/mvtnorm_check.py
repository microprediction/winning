"""Benchmark the R mvtnorm package (the reference MVN software) on the
arena's standing problems: each win probability is an (N-1)-dimensional
MVN orthant probability of differences, priced by pmvnorm(GenzBretz),
N calls per share vector.

    python -m winning.bench.mvtnorm_check

Writes bench_results/MVTNORM.md. Requires Rscript + mvtnorm.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .runner import problem, GRID

RESULTS = Path(__file__).resolve().parents[2] / "bench_results"

RSCRIPT = """
suppressMessages(library(mvtnorm))
args <- commandArgs(trailingOnly = TRUE)
inp <- read.csv(args[1])
n <- nrow(inp)
k <- sum(grepl("^v", names(inp)))
mu <- inp$mu
V <- as.matrix(inp[, grepl("^v", names(inp))])
D <- inp$D
maxpts <- as.integer(args[3])
Sigma <- V %*% t(V) + diag(D)
p <- numeric(n)
for (i in 1:n) {
  idx <- setdiff(1:n, i)
  A <- -diag(n)[idx, , drop = FALSE]
  A[, i] <- A[, i] + 1            # row j: X_i - X_j
  m <- as.vector(A %*% mu)
  S <- A %*% Sigma %*% t(A)
  p[i] <- pmvnorm(lower = rep(0, n - 1), upper = rep(Inf, n - 1),
                  mean = m, sigma = S,
                  algorithm = GenzBretz(maxpts = maxpts, abseps = 1e-6))[1]
}
write.csv(data.frame(p = p / sum(p)), args[2], row.names = FALSE)
"""


def main():
    lines = ["# mvtnorm (R) on the arena problems",
             "",
             "Each share vector = N calls to pmvnorm(GenzBretz), the",
             "standard MVN software, on the (N-1)-dim difference orthant.",
             "Scored against the same cached references as the arena.",
             ""]
    lines.append("| problem | maxpts | seconds | max abs err | max log-odds err |")
    lines.append("|---|---|---|---|---|")
    with tempfile.TemporaryDirectory() as td:
        rfile = Path(td) / "run.R"
        rfile.write_text(RSCRIPT)
        for pid, n, k, spread, seed in GRID:
            if n > 200:
                lines.append(f"| {pid} | - | (skipped: {n} calls of "
                             f"{n-1}-dim pmvnorm is hours of wall time) | | |")
                continue
            mu, V, D = problem(pid, n, k, spread, seed)
            truth = np.load(RESULTS / f"ref_{pid}.npy")
            inp = Path(td) / f"{pid}.csv"
            out = Path(td) / f"{pid}_out.csv"
            hdr = "mu," + ",".join(f"v{j}" for j in range(k)) + ",D"
            rows = [hdr] + [
                f"{mu[i]}," + ",".join(str(V[i, j]) for j in range(k))
                + f",{D[i]}" for i in range(n)]
            inp.write_text("\n".join(rows) + "\n")
            for maxpts in (25_000, 250_000):
                t0 = time.perf_counter()
                subprocess.run(["Rscript", str(rfile), str(inp), str(out),
                                str(maxpts)], check=True,
                               capture_output=True)
                dt = time.perf_counter() - t0
                p = np.loadtxt(out, skiprows=1)
                err = float(np.abs(p - truth).max())
                res = truth > 1.25e-3
                lerr = float(np.abs(np.log(np.maximum(p[res], 1e-300))
                                    - np.log(truth[res])).max())
                print(f"{pid} maxpts={maxpts}: {dt:.1f}s abs {err:.1e} "
                      f"logodds {lerr:.1e}")
                lines.append(f"| {pid} | {maxpts} | {dt:.1f} | {err:.1e} "
                             f"| {lerr:.1e} |")
    (RESULTS / "MVTNORM.md").write_text("\n".join(lines) + "\n")
    print("wrote", RESULTS / "MVTNORM.md")


if __name__ == "__main__":
    main()
