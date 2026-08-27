\name{factorize_covariance}
\alias{factorize_covariance}
\title{Exact factor-plus-diagonal decomposition of a covariance, if one exists}
\description{
Iterated principal-factor fit for ranks 1..max_rank, accepted only when
the reconstruction \eqn{VV' + diag(D)} matches \code{sigma} to \code{tol}
relative to its largest entry, verified against a recomputed factor.
}
\usage{
factorize_covariance(sigma, max_rank = 6L, tol = 1e-11, n_iter = 300L)
}
\arguments{
  \item{sigma}{covariance matrix.}
  \item{max_rank}{largest factor rank tried.}
  \item{tol}{relative reconstruction tolerance for acceptance.}
  \item{n_iter}{iteration cap per rank.}
}
\value{\code{list(V, D)} on success, \code{NULL} when no exact
decomposition within tolerance exists.}
\examples{
V <- matrix(rnorm(8), 8, 1)
D <- 0.3 + runif(8)
fd <- factorize_covariance(V \%*\% t(V) + diag(D))
}
