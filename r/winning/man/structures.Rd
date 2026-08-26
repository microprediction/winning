\name{structures}
\alias{structures}
\alias{Independent}
\alias{Factor}
\alias{Blocks}
\alias{Nested}
\alias{Tree}
\title{One race, five covariance grammars}
\usage{
Independent(D)

Factor(V, D)

Blocks(cluster, loading, D)

Nested(cluster, loading, D, coupling, gamma = 1)

Tree(cluster, loading, D, parent, strength)
}
\arguments{
\item{D}{idiosyncratic variances (always the VARIANCE)}
\item{V}{N x k loading matrix}
\item{cluster}{cluster labels}
\item{loading}{within-cluster loadings}
\item{coupling}{loadings on the global factor}
\item{gamma}{coupling strength in [0, 1]}
\item{parent}{1-based parent index per node (root: 0 or NA)}
\item{strength}{per-node uniform shared-effect strength}
}
\value{A structure object for the structure= argument of
\code{\link{race_probabilities}}, \code{\link{abilities_from_race}},
\code{\link{race_jacobian}} and \code{\link{polish_race}}.}
\description{
Every model in this package is the SAME Gaussian min-race, Y = mu +
noise; these constructors are declarative descriptions of the noise
covariance admitting O(N)-per-lattice-point evaluation:
Sigma = diag(D); V V' + diag(D); block-diagonal rank-1 + diag; blocks
plus a global factor dialed by gamma; a hierarchy of uniform shared
effects. Independent = Blocks with zero loadings; Blocks = Tree of
depth 1.
}
