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
  \item{lower}{a scalar, broadcast to every coordinate, or one limit per coordinate; -Inf allowed. Any other length is an error: R recycles silently whenever the short length divides n, which returns a plausible probability for a different rectangle (#285). mvtnorm::pmvnorm, which this is a drop-in for, refuses the same input.}
  \item{upper}{a scalar, broadcast to every coordinate, or one limit per coordinate; Inf allowed. Any other length is an error, as for \code{lower}.}
  \item{mean}{numeric mean vector; defaults to zero.}
  \item{sigma}{covariance matrix. Supply either \code{sigma} alone or
    \code{V} and \code{D} together; any other combination is an error.}
  \item{V}{n-by-r factor loading matrix, so that sigma = VV' + diag(D).
    A scalar (one common loading), a length-n vector (rank one) and the
    transposed r-by-n matrix are accepted.}
  \item{D}{idiosyncratic variances: a scalar or one non-negative value per
    coordinate.}
  \item{...}{passed to \code{mvtnorm::pmvnorm()} on the fallback path.
    An accuracy request tighter than the fast rule's 1e-5 (an
    \code{abseps} below 1e-5, a non-zero \code{releps} below 1e-5, or
    the same inside \code{algorithm = GenzBretz(...)} or
    \code{TVPACK(...)}) is honoured by sending the call to
    \code{mvtnorm::pmvnorm()}; \code{maxpts} alone is an effort cap and
    does not. mvtnorm's defaults take the fast path.}
}
\value{
A single probability. Attribute \code{method} records the path taken:
\code{"factor"}, \code{"factor-recentered"} (deep tails, below 1e-8 or
resolved by too few low-discrepancy nodes, evaluated by recentered
importance reweighting), \code{"degenerate-rectangle"},
\code{"outside-support"}, or \code{"mvtnorm-fallback"}. A fallback
taken because of the requested tolerance also carries attribute
\code{reason}.
}
\details{
Node selection is automatic: Gauss-Hermite for mild loadings (order
15 x sharpness squared, at least 61 at rank one and 15 at rank two, at
most 201 and 41), low-discrepancy nodes past sharpness 3 or factor rank 2
(rank 3 and above always uses them, by design), and
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
