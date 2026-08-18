# Experiment 20: the Chebyshev-separated pass

Both stages of the per-node O(NL) lattice pass are smooth-kernel sums, and
the kernel matrices are numerically low-rank. Separating the kernels on a
tensor Chebyshev grid in (location, scale) turns the pass into O(r(N+L))
per node, with exponential convergence in r (analytic kernels).

Measured against the exact pass (k=2, GH-15, L=1501, NumPy both sides):

| N | r=384 | r=672 | r=1024 |
|---|---|---|---|
| 1000 | 2.7e-5 @ 39x | 1.6e-7 @ 20x | 1.0e-9 @ 11x |
| 5000 | 6.1e-5 @ 45x | 6.3e-7 @ 29x | 9.1e-9 @ 22x |

The speedup grows with N (r(N+L) vs NL). At the accuracy the paper's
benchmarks actually resolve (simulation references ~3e-4), the pass is
~40-45x faster before any compiled code; it composes with the fastrace
Rust kernel and applies to slopes and JVPs (same kernels), hence to
calibration.

Caveats still open: per-entry tail accuracy at extreme D heterogeneity;
k>4 RQMC node sets (wider location range -> larger rm); integration into
the inversion loop. See paper/fast-kernel-notes.md for the literature
(FGT, black-box FMM, reduced basis).

Run: `python run_separated.py` (~2 min).
