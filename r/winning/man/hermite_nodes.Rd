\name{hermite_nodes}
\alias{hermite_nodes}
\title{Pruned product Gauss-Hermite nodes}
\usage{hermite_nodes(k, order = 15, prune = 1e-7)}
\arguments{
  \item{k}{factor dimension.}
  \item{order}{univariate quadrature order.}
  \item{prune}{drop nodes with weight below \code{prune} times the maximum.}
}
\value{A list with matrix \code{F} (nodes by \code{k}) and weight vector \code{W}.}
\description{
Golub-Welsch nodes of the probabilists' Hermite rule for expectations
over a standard k-variate normal, as a pruned product grid.
}
