\name{pmvnorm_fast}
\alias{pmvnorm_fast}
\title{Fast multivariate normal rectangle probability for factor-structured covariance}
\description{
Computes \eqn{P(lower \le X \le upper)} for \eqn{X \sim N(mean, VV' + diag(D))}.
Conditional on the low-dimensional factor the coordinates are independent, so
the rectangle probability is a small smooth integral of a product of
univariate normal CDFs: deterministic, no simulation, milliseconds at
dimensions where general-purpose quadrature needs seconds to minutes.
Supply \code{V} and \code{D}, or supply \code{sigma} and an exact
decomposition is searched for; a covariance that is not exactly
factor-plus-diagonal falls back to \code{mvtnorm::pmvnorm()} unchanged.
}
\usage{
pmvnorm_fast(lower = -Inf, upper = Inf, mean = NULL, sigma = NULL,
             V = NULL, D = NULL, ...)
}
\arguments{
  \item{lower}{numeric vector of lower limits, recycled to dimension n; -Inf allowed.}
  \item{upper}{numeric vector of upper limits, recycled to dimension n; Inf allowed.}
  \item{mean}{numeric mean vector; defaults to zero.}
  \item{sigma}{covariance matrix; used when V and D are not supplied.}
  \item{V}{n-by-r factor loading matrix, so that sigma = VV' + diag(D).}
  \item{D}{length-n positive idiosyncratic variances.}
  \item{...}{passed to \code{mvtnorm::pmvnorm()} on the fallback path.}
}
\value{
A single probability. Attribute \code{method} records the path taken:
\code{"factor"}, \code{"factor-recentered"} (deep tails, below 1e-8,
evaluated by Laplace-recentered importance reweighting), or
\code{"mvtnorm-fallback"}.
}
\details{
Node selection is automatic: adaptive-order Gauss-Hermite for mild
loadings, low-discrepancy nodes past sharpness 3 or factor rank 2, and
Laplace recentering in deep tails. The exact-decomposition check used
with \code{sigma} is strict (reconstruction to 1e-11 relative, verified
against a recomputed factor), so an inexact factorization never
masquerades as the structured case.
}
\examples{
V <- matrix(rnorm(20), 10, 2)
D <- 0.5 + runif(10)
pmvnorm_fast(upper = rnorm(10, 1), V = V, D = D)
}
