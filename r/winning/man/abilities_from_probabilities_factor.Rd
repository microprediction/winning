\name{abilities_from_probabilities_factor}
\alias{abilities_from_probabilities_factor}
\title{Calibrate abilities from observed shares}
\usage{abilities_from_probabilities_factor(p, V, D, nodes = NULL,
  n_iter = 50, tol = 1e-6, points = 501)}
\arguments{
  \item{p}{vector of positive target shares (normalized internally).}
  \item{V}{N by k matrix of factor loadings.}
  \item{D}{vector of idiosyncratic variances.}
  \item{nodes}{optional list(F, W) of factor nodes.}
  \item{n_iter}{maximum Newton iterations.}
  \item{tol}{tolerance on the max log-share residual over identified
    alternatives.}
  \item{points}{lattice size.}
}
\value{Mean-zero ability vector (min-wins convention).}
\description{
Inverts the factor-race share map by damped coordinate Newton against
the frozen shared field, with analytic slopes and an independent-race
warm start. Shares determine abilities uniquely up to a common shift.
}
