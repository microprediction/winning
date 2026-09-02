# Track B: SSTA criticality (kill test first)
(Adjudicated PURSUE one experiment deep 2026-09-01; report in
../adjudications/ssta.md. Background: ../applications/circuit-timing.md.)

## The kill test IS the contribution
No published measurement of path-covariance effective rank exists.
Take near-critical path bundles from EDA-Schema-V2 (deterministic
STA only -- add a declared Chang-Sapatnekar grid + per-gate random
variation model), build the exact incidence-Gram covariance, measure
(a) residual rank after k spatial factors, (b) exact criticality vs
1e5-sample MC vs Clark. Kill if residual rank routinely exceeds
interactive-cost grammar; publish the rank measurement either way.

## Status
Not started. Blocked behind A/E/D by sequence. First action when
opened: download one EDA-Schema-V2 design and confirm the timing-path
schema supports incidence extraction.
