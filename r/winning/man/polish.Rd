\name{polish_race}
\alias{polish_race}
\alias{race_jacobian}
\alias{concentration_matrix}
\title{Polish a race onto linear constraints}
\usage{
polish_race(p0 = NULL, mu0 = NULL, V = NULL, D = NULL, F = NULL,
  W = NULL, base = "normal", points = 257, name_caps = NULL,
  groups = NULL, A = NULL, b = NULL, tol = 1e-09, max_iter = 60,
  structure = NULL)

race_jacobian(mu, V = NULL, D = NULL, F = NULL, W = NULL,
  base = "normal", points = 501, structure = NULL, qa = 9, qf = 15)

concentration_matrix(n, name_caps = NULL, groups = NULL)
}
\arguments{
\item{p0}{current weights (inverted to abilities internally)}
\item{mu0}{abilities directly (min-wins, mean-zero)}
\item{mu}{abilities at which to evaluate the Jacobian}
\item{V, D, F, W, base, points, qa, qf}{as in
\code{\link{race_probabilities}}}
\item{name_caps}{scalar or per-name weight caps (NA skipped)}
\item{groups}{list of list(indices, cap) group caps}
\item{A, b}{explicit constraint rows, A p <= b}
\item{tol}{optimizer tolerance}
\item{max_iter}{outer iterations}
\item{structure}{covariance grammar (Tree not yet supported here)}
\item{n}{number of contestants}
}
\value{
\code{polish_race}: list(p, mu, info) -- a probability vector that IS
the race at mu, satisfying the caps, with mu as close to mu0 as the
constraints allow. \code{race_jacobian}: the exact n x n d p / d mu.
\code{concentration_matrix}: list(A, b).
}
\description{
Weights ARE race probabilities, and finance imposes linear constraints
on them; clipping and renormalizing breaks model-consistency. The right
object is the nearest race satisfying the constraints, found by an
augmented-Lagrangian on the exact race Jacobian (the python reference
uses SLSQP; the two agree to optimizer tolerance).
}
\examples{
p0 <- c(0.4, 0.25, 0.2, 0.15)
out <- polish_race(p0 = p0, D = rep(1, 4), name_caps = 0.3)
max(out$p)  # <= 0.3
}
