\name{block_race_probabilities}
\alias{block_race_probabilities}
\alias{nested_race_probabilities}
\alias{tree_race_probabilities}
\alias{block_race_jacobian}
\alias{nested_race_jacobian}
\alias{abilities_from_block_race}
\title{Block, nested and tree races}
\usage{
block_race_probabilities(mu, cluster, loading, D, points = 257, qa = 9)

nested_race_probabilities(mu, cluster, loading, D, coupling = NULL,
  gamma = 1, points = 257, qa = 9, qf = 15)

tree_race_probabilities(mu, cluster, loading, D, parent, strength,
  points = 257, qa = 9)

block_race_jacobian(mu, cluster, loading, D, points = 257, qa = 9)

nested_race_jacobian(mu, cluster, loading, D, coupling = NULL, gamma = 1,
  points = 257, qa = 9, qf = 15)

abilities_from_block_race(p, cluster, loading, D, points = 257, qa = 9,
  tol = 1e-10, max_iter = 25)
}
\arguments{
\item{mu}{abilities (min-wins), length n}
\item{p}{positive target probabilities}
\item{cluster}{cluster labels, length n}
\item{loading}{numeric length n (rank 1) or n x r matrix (r <= 2 in
pure R; supply quadrature nodes for higher rank)}
\item{D}{idiosyncratic variances}
\item{coupling}{per-runner loading on the global factor}
\item{gamma}{coupling strength: 0 = independent blocks, 1 = full}
\item{parent}{1-based parent index per tree node; root marked 0 or NA}
\item{strength}{per-node uniform shared-effect strength}
\item{points}{lattice size}
\item{qa, qf}{quadrature orders (cluster effect, global factor)}
\item{tol}{inversion tolerance}
\item{max_iter}{Newton iterations after the fixed-point globalizer}
}
\value{
Probability vectors summing to one; Jacobians as n x n matrices with
zero row sums (off-diagonals are symmetric tie densities);
\code{abilities_from_block_race} a list(mu, residual, iterations).
}
\description{
Clustered covariance grammars of the general race: independent
cluster effects (block), block plus one global factor (nested,
interpolated by gamma), and a hierarchy of uniform shared effects
(tree, two message passes). Across-cluster independence factorizes the
survival field, so cost is O(n x lattice x nodes) regardless of the
number of blocks. Base-R ports of the python reference, parity-locked.
}
