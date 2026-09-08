# Port winning 1.x tests as regression fixtures

**Type:** question / maintenance

**STATUS (2026-09-08): SUBSTANTIALLY REALIZED.** `tests/` now carries
47 modules including the classic lattice, copula, event-density,
Harville and golden-value suites, plus a cross-language parity
harness the original never had. Do not post as written.

winning 1.x had 17 test modules exercising the original implementation, including
edge cases thurstone's suite does not yet cover directly:

- `test_state_prices_from_extended_offsets.py` — extensive +/-inf ability,
  walkover and heterogeneous-race cases (thurstone covers some of this via
  `ClusterSplitter` tests, but not the same matrix of cases)
- `test_lattice_five*.py` — rank-1..5 probabilities including the SIAM-paper
  values (`test_lattice_five_paper.py` pins numbers from the published tables —
  valuable golden data regardless of implementation)
- `test_densities_from_events.py` — the golf/cumulative-scoring path (see the
  scoring-events issue)
- `test_lattice_from_samples.py` — densities built from empirical samples

Since thurstone already cross-checks Python against a JS port with golden JSON
fixtures, the cheapest value here is: extract the *numbers* from winning's paper
tests into `docs/fixtures/` style goldens, so the published results stay pinned
forever. The winning 2.x repo no longer runs these tests (its scope moved to rating
systems); they live in winning git history at v1.0.6.
