\name{rprobit_fast}
\alias{rprobit_fast}
\alias{print.rprobit_fast}
\title{Exact multinomial probit where MACML approximates}
\description{
Multinomial probit estimation with an exact factor-conditional
likelihood and analytic score, for the model class served by the
'Rprobit' package, whose MACML engine rests on Mendell-Elston /
Solow-Joe analytic approximations. Those are fast and centrally
accurate but drift silently in the tails; this fits the same
likelihoods exactly, to quadrature accuracy, and reports
boundary-seeking covariance estimates instead of hiding them.
}
\usage{
rprobit_fast(df, covariates, r = 2L, Qf = 7L, Qz = 7L, maxit = 400L)
}
\arguments{
  \item{df}{long-format data frame with columns \code{id} (chooser),
    \code{alt} (alternative), \code{chosen} (logical), and the
    covariate columns; one row per chooser-alternative pair.}
  \item{covariates}{character vector of covariate column names.}
  \item{r}{factor rank of the utility covariance (zero reference
    loadings, unit idiosyncratic variance).}
  \item{Qf, Qz}{Gauss-Hermite orders for the factor and winning-margin
    integrals, with sharpness escalation to Halton nodes.}
  \item{maxit}{BFGS iteration cap.}
}
\value{
An object of class \code{rprobit_fast}: a list with
\code{coefficients}, \code{covariance_par}, \code{boundary} (TRUE when
the loading norm runs to the boundary, a real feature of exact MNP
likelihoods that simulation noise masks), \code{logLik}, \code{time},
\code{convergence}, \code{J} and \code{r}. A \code{print} method
summarizes the fit.
}
\examples{
set.seed(1)
J <- 3; Tn <- 60
df <- data.frame(
  id = rep(seq_len(Tn), each = J),
  alt = rep(letters[1:J], Tn),
  x = rnorm(Tn * J))
u <- 0.8 * df$x + rnorm(Tn * J)
df$chosen <- as.logical(stats::ave(u, df$id, FUN = function(v) v == max(v)))
fit <- rprobit_fast(df, "x", r = 1L, Qf = 5L, Qz = 5L, maxit = 50L)
print(fit)
}
