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
