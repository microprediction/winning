\name{race_probabilities}
\alias{race_probabilities}
\alias{abilities_from_race}
\alias{calibrate_abilities}
\title{The general race: one API, distributions and correlation as parameters}
\usage{
race_probabilities(mu, V = NULL, D = NULL, F = NULL, W = NULL,
  base = "normal", points = 257, return_slopes = FALSE, structure = NULL,
  window = "bulk", delta = 1e-12, qa = 9, qf = 15, nodes = NULL)

abilities_from_race(p, V = NULL, D = NULL, F = NULL, W = NULL,
  base = "normal", points = 257, n_iter = 60, tol = 1e-08,
  structure = NULL, qa = 9, qf = 15)

calibrate_abilities(p, V = NULL, D = NULL, F = NULL, W = NULL,
  base = "normal", points = 257, n_iter = 60, tol = 1e-08,
  structure = NULL, qa = 9, qf = 15)
}
\arguments{
\item{mu}{numeric abilities; performances are times, so lower is better
(min-wins convention)}
\item{p}{positive target probabilities (normalized internally)}
\item{V}{optional N x k factor loading matrix}
\item{D}{idiosyncratic variances (default 1)}
\item{F, W}{optional factor quadrature nodes and weights, e.g. from
\code{\link{hermite_nodes}}}
\item{base}{"normal", "gumbel", or a function z -> list(S, f, fp) of a
mean-zero unit-variance law}
\item{points}{lattice size}
\item{return_slopes}{also return the inversion preconditioner
d p_i / d mu_i}
\item{structure}{a covariance grammar (\code{\link{Independent}},
\code{\link{Factor}}, \code{\link{Blocks}}, \code{\link{Nested}},
\code{\link{Tree}}); overrides V/D}
\item{window}{"bulk" (winner-bulk lattice, default) or "span"}
\item{delta}{omitted winner mass bound for the bulk window}
\item{qa, qf}{quadrature orders used by structure dispatch}
\item{nodes}{deprecated alias for list(F, W)}
\item{n_iter}{maximum damped-Newton iterations}
\item{tol}{convergence tolerance on max |log p - log target|}
}
\value{
\code{race_probabilities}: probabilities summing to one (or
list(p, slopes)). \code{abilities_from_race} and its alias
\code{calibrate_abilities}: the mean-zero ability vector reproducing p.
}
\description{
Win probabilities of the general Gaussian (or gumbel, or custom-base)
race, all N in one shared-field lattice pass, and the inverse map from
probabilities to abilities. With \code{base = "gumbel"} and
D = pi^2/6 the race is exactly softmax. Base-R port of the python
reference; parity/vectors.json pins the two to machine precision.
}
\examples{
p <- race_probabilities(c(-0.5, 0, 0.8))
mu <- calibrate_abilities(c(0.5, 0.3, 0.2))
s <- Blocks(cluster = c(1, 1, 2, 2), loading = rep(0.4, 4), D = rep(1, 4))
pb <- race_probabilities(c(-0.3, 0, 0.1, 0.2), structure = s)
}
