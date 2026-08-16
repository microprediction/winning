\name{win_probabilities_factor}
\alias{win_probabilities_factor}
\title{All win probabilities of a factor Gaussian race}
\usage{win_probabilities_factor(mu, V, D, nodes = NULL, points = 501)}
\arguments{
  \item{mu}{vector of locations (length N); the minimum wins. Negate for
    argmax (utility) races.}
  \item{V}{N by k matrix of factor loadings.}
  \item{D}{vector of idiosyncratic variances.}
  \item{nodes}{optional list(F, W) of factor nodes; defaults to
    \code{hermite_nodes(ncol(V))}.}
  \item{points}{lattice size.}
}
\value{Vector of win probabilities summing to one.}
\description{
Computes every P(X_i = min_j X_j) for X = mu + V f + sqrt(D) eps in one
shared-lattice pass per factor node: conditional on f the alternatives
are independent, the product of all survival functions is built once,
and each alternative's own factor is divided back out.
}
\examples{
p <- win_probabilities_factor(c(0, .5, 1), matrix(rnorm(6, 0, .3), 3), rep(1, 3))
}
