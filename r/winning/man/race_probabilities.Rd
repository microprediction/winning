\name{race_probabilities}
\alias{race_probabilities}
\alias{calibrate_abilities}
\title{The general race: win probabilities and ability calibration}
\usage{
race_probabilities(mu, V = NULL, D = NULL, nodes = NULL, points = 501)

calibrate_abilities(p, V = NULL, D = NULL, nodes = NULL, n_iter = 50,
  tol = 1e-06, points = 501)
}
\arguments{
\item{mu}{numeric locations (length N); performances are times, lower
is better}
\item{p}{numeric positive target shares (normalized internally)}
\item{V}{optional N x k matrix of factor loadings (omit for the
independent race)}
\item{D}{optional idiosyncratic variances (default 1)}
\item{nodes}{optional list(F, W) of factor quadrature nodes, e.g. from
\code{\link{hermite_nodes}}}
\item{n_iter}{maximum Newton iterations}
\item{tol}{convergence tolerance}
\item{points}{lattice size (default 501)}
}
\value{
\code{race_probabilities} returns win probabilities summing to one;
\code{calibrate_abilities} the mean-zero ability vector reproducing the
shares (min-wins convention).
}
\description{
Front door of the factor Gaussian race, mirroring the python package's
\code{race_probabilities} / \code{abilities_from_race}: zero factors is
the independent race, and the factor case prices all N alternatives in
one shared-lattice pass. Aliases over
\code{\link{win_probabilities_factor}} and
\code{\link{abilities_from_probabilities_factor}}.
}
\examples{
p <- race_probabilities(c(-0.5, 0, 0.8))
mu <- calibrate_abilities(c(0.5, 0.3, 0.2))
}
