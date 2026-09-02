# Track A: the LITE head-to-head
(Adjudicated PURSUE 2026-09-01; report in ../adjudications/lite.md.
Working code on our side: ../qpo/ already ports LITE to numpy,
matches the JAX original to 1e-6, and measures recall 0.61 vs 0.94
on QM9.)

## The claim under attack, in their words (to verify before quoting)
LITE's guarantees bound convergence to the INDEPENDENCE
approximation of PoM (their Assumption 2), and their exact reference
(TS-MC) took 21 days on the 10k-point 1000-dim linear-kernel task.
Both statements must be read in the PDF, not the summarized fetch,
before a paper cites them.

## Decisive experiment
Their Table-1 protocol, Sigma restricted to factor/tree form
(inducing-point GP posteriors; their linear-kernel task at modest
rank). Report every method's TV to OUR exact answer. Two exhibits:
winning as the ground-truth oracle their 21-day baseline
approximates, and LITE's independence bias per structure.

## Status
Not started as its own experiment; qpo results reusable. Next
action: clone lasgroup/LITE, reproduce one of their tables as
published, then swap in the grammar-form Sigma.

## Decisive experiment run (2026-09-02, exp1_headtohead/)
Grammar-form ensembles, exact vector as ground truth (certified
against 5e5-draw MC every cell, worst 0.014 = the certificate's own
MC noise):
- F-LITE total variation from exact: 0.055-0.073 at factor share
  0.3; 0.18-0.36 at share 0.7, GROWING with n. A-LITE worse
  (0.10-0.41). The independence bias is first-order under real
  correlation, exactly as positioned.
- THE TAIL IS THE HEADLINE: under share 0.7 the marginal-only
  methods assign ~1e-3-scale probabilities to alternatives whose
  exact probability is astronomically small -- measured relative
  errors of 1e16 to 1e47 at n = 1e4. Any tail-sensitive use
  (elimination, routing, risk) inherits that. TS-MC at 10k draws is
  honest but blunt: TV 0.02-0.08 and tail relative errors 3-260
  (the 1/sqrt(Mp) wall).
- Cost: exact rank-1 at n=1e4 in 0.8s; rank-4 at n=1e4 in 100s
  (tensor Hermite Q=7^4 -- the rank-vs-quadrature tradeoff to state
  plainly in any writeup; sparse grids or QMC nodes are the known
  fix and are not used here).
Remaining for the benchmark section: run LITE's own repo tasks
(their Table-1 protocol verbatim) rather than our ensembles alone,
and quote their Assumption 2 from the PDF, not the summarized fetch.
