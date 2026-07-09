# Gaussian copula: correlated contestants via a common factor

**Type:** enhancement (or `winning` application module — discuss)

The independence assumption is the lattice model's main restriction. winning 1.x had
a scipy-based extension (`lattice_copula.py`) integrating over a common Gaussian
factor with `quad_vec`:

- `gaussian_copula_win` — win probabilities with equicorrelated performances
- `gaussian_copula_five` — correlated rank-1..5 probabilities (used for M6, where
  asset quintile ranks are strongly cross-correlated)
- `gaussian_copula_conditional_cdf`, `gaussian_copula_functional` — the plumbing

Design question for thurstone: scipy is not a welcome dependency. The conditional
one-factor trick only needs 1-D quadrature over the common factor; a fixed
Gauss-Hermite rule in numpy would keep thurstone pure. Conditional on the factor,
contestants are independent and every existing lattice routine applies unchanged —
so this is a wrapper layer, not core surgery.

Reference implementation: `attic/lattice_copula.py` in the winning repo.
