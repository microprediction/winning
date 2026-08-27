# Compression as a race: status (Aug 27, 2026)

The v3 decomposition stands: PQ's ADC error = per-candidate constant
(one stored float, exact debias) + zero-mean query-linear term with
variance kappa * rho_i (second stored float). Retrieval is then a race:
debiased score = ability, kappa*rho = variance, and shortlist depth
becomes a per-query calibrated quantity.

v4 (run_pq_race4.py) reruns v3 through the package's rust-backed
race_probabilities: identical coverage tables (implementation parity),
and the matched-achieved-coverage verdict:

    achieved 0.992 recall: adaptive mean depth  58 vs fixed 143 (2.5x)
    achieved 0.998 recall: adaptive mean depth 107 vs fixed 316 (2.9x)

(The 0.99-target arm over-provisions -- coverage saturates at 0.998; use
the 0.95 rule.) Overhead: 42 ms/query (v3 python) -> ~5 ms (span window,
65 points). Remaining path to ~1 ms is mechanical: the rust kernel
parallelizes over quadrature nodes, so the independent race (Q=1) runs
single-threaded; tile-parallelism for small Q fixes it.

Open extension: IVF probe-list sizing is the same coverage question at
the coarse level, and coarse-then-fine quantization is tree-race
structure (now built and Botev-validated in the package). Candidate
claim: calibrated coverage guarantees at every level of the hierarchy.
