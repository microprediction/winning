\name{fit_covariance}
\alias{fit_covariance}
\title{Fit a dense covariance to the race grammar}
\usage{
fit_covariance(C, k = 3L, m = 5L, blocks = NULL, nodes = 2048L)
}
\arguments{
\item{C}{covariance or correlation matrix}
\item{k}{global factor rank}
\item{m}{residual eigencolumns to promote}
\item{blocks}{cluster count for the block stage (default
\code{max(2, min(n/5, 20))})}
\item{nodes}{count of Halton nodes for the returned rule}
}
\value{
list with V, D, F, W ready for \code{\link{race_probabilities}}.
}
\description{
The paper's dense pipeline: certified quotient-space factor fit (only
\code{P Sigma P} is choice-relevant), average-linkage blocks and
residual promotion on the projected residual, and a closing diagonal
solve of the identified problem's normal equations. Grammar-member
inputs are returned undistorted; \code{race_probabilities(cov = C)}
calls this on the way in.
}
\seealso{\code{\link{race_probabilities}}}
