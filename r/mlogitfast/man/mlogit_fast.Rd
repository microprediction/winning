\name{mlogit_fast}
\alias{mlogit_fast}
\alias{print.mlogit_fast}
\title{Exact multinomial probit behind the mlogit interface}
\description{
Deterministic maximum likelihood for the multinomial probit choice
model: the GHK simulator is replaced by an exact factor-conditional
product integral with analytic score. Accepts \code{dfidx}-style long
data and a formula for alternative-specific covariates, as used by the
\pkg{mlogit} package.
}
\usage{
mlogit_fast(formula, data, r = 2L, Qf = 7L, Qz = 7L, maxit = 400L)
}
\arguments{
  \item{formula}{choice ~ covariates, with the response a logical
    chosen-alternative column and alternative-specific covariates on
    the right-hand side.}
  \item{data}{a long-format data set carrying a \code{dfidx}-style
    index (chooser, alternative), e.g. as produced by
    \code{dfidx::dfidx}.}
  \item{r}{factor rank of the utility covariance; the parameterization
    (rank-\code{r} loadings with zero reference loadings, unit
    idiosyncratic variance) spans the same identified space as
    \pkg{mlogit}'s differenced-Cholesky parameterization with the same
    degrees of freedom.}
  \item{Qf, Qz}{Gauss-Hermite orders for the factor and winning-margin
    integrals; a sharpness rule escalates to Halton nodes when the
    conditional races become nearly deterministic.}
  \item{maxit}{BFGS iteration cap.}
}
\value{
An object of class \code{mlogit_fast}: a list with
\code{coefficients} (named covariate effects),
\code{covariance_par} (loading parameters), \code{logLik},
\code{time} (seconds), \code{convergence} (0 for success), \code{J}
and \code{r}. A \code{print} method summarizes the fit.
}
\examples{
set.seed(1)
J <- 3; Tn <- 60
d <- data.frame(
  id = rep(seq_len(Tn), each = J),
  alt = factor(rep(letters[1:J], Tn)),
  x = rnorm(Tn * J))
u <- 0.8 * d$x + rnorm(Tn * J)
d$choice <- as.logical(stats::ave(u, d$id, FUN = function(v) v == max(v)))
idx <- data.frame(id = d$id, alt = d$alt)
dat <- d
dat$idx <- idx
fit <- mlogit_fast(choice ~ x, dat, r = 1L, Qf = 5L, Qz = 5L, maxit = 50L)
print(fit)
}
