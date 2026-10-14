# Julia MNP sharp-branch cloud options (#730)

- question (#730): can the Julia sharp-branch cloud (plain Halton 2^10,
  6.3 nats off at python's Fishing optimum) be replaced by one that is
  measurably better at <= 1.2x cost?
- tree: origin/main 5949b983 (julia/MultinomialProbit unchanged); variants
  were scratch copies of src/MultinomialProbit.jl with only line 362 (the
  sharp cloud) changed: `_halton_normal(r + 1, 2^k)` for k = 12, 13, or the
  R ports' scrambled Sobol (`.sobol_normal(3, 1024, seed)` from
  r/rprobitfast/R/sobol.R, written to CSV and read inside
  nodes_for_likelihood -- so the rs* fit times include file reads).
- cloud_error.txt: `cloud_error.py ../2026-10-14-plain-sobol` -- each
  cloud's log-likelihood minus the Gauss-Hermite 151^3 truth at the three
  fitted thetas of ../2026-10-14-plain-sobol (main python, branch julia,
  branch python), with python's likelihood; Halton 2^10..2^13, R Sobol
  2^10..2^12 seeds 0-4, python (scipy) Sobol 2^10 and 2^12 seeds 0-4.
- fits.txt: julia exact fits with each cloud (`fit_julia.jl`), thetas in
  thetas/; fit_truth.txt: `score_thetas.py` (sharpness bound and GH 151^3
  truth at each fitted theta).
- machine: Apple M3 Ultra, 28 cores; julia 1.12.6; python 3.14.6;
  JULIA_NUM_THREADS=4, nice -n 10. Data: fishing.csv as ../2026-10-13.

## Findings

| cloud (sharp branch) | rms error at the 3 points (nats) | julia fit: truth at its optimum | sharpness | fit time |
|---|---|---|---|---|
| Halton 2^10 (main) | 3.70 | -1216.28 | 2.89 | 13.5 s |
| Halton 2^12 | 1.05 | -1216.26 | 2.89 | 46.2 s (3.4x) |
| Halton 2^13 | 0.88 | -1213.70 | 5.1e5 | 227.9 s (17x) |
| R Sobol 2^10 seed 0/1/2 | 3.79 / 2.26 / 3.68 | -1211.62 / -1214.93 / -1210.14 | 295 / 4.3 / 20 | ~14-20 s |

- At 2^10 nodes the R scrambled Sobol is not measurably better than the
  Halton cloud (rms 2.3-3.8 nats across five seeds against 3.7), and where
  the fit lands is a seed lottery: the three seeds end at sharpness 4.3,
  20 and 295, at GH truths spread over 4.8 nats. At sharpness 20-295 the
  GH 151^3 "truth" itself is not a reliable reference.
- More Halton nodes do shrink the error (rms 1.05 at 2^12, 0.88 at 2^13)
  but cost 3.4x and 17x on this fit, and 2^12 lands in the same basin.
- No option meets "measurably better at <= 1.2x cost"; #730 left open.

The R Sobol CSVs that cloud_error.py and the rs* variants read were made
in the working directory with
`Rscript -e 'source("r/rprobitfast/R/sobol.R"); for (n in c(1024, 2048, 4096)) for (s in 0:4) write.csv(.sobol_normal(3, n, s)$F, sprintf("rsob_%d_%d.csv", n, s), row.names = FALSE)'`.
