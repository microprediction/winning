\name{tree_from_linkage}
\alias{tree_from_linkage}
\alias{tree_from_hclust}
\title{The tree race implied by a hierarchical clustering}
\usage{
tree_from_linkage(Z)

tree_from_hclust(hc)
}
\arguments{
\item{Z}{scipy-style linkage matrix (n-1 rows: merged node ids 0-based,
cophenetic distance, size)}
\item{hc}{an \code{\link[stats]{hclust}} object}
}
\value{A \code{\link{Tree}} structure with unit total variance per
runner.}
\description{
Builds the tree race whose implied correlation is EXACTLY the
cophenetic correlation matrix 1 - 2 d^2 of the clustering (HRP's
implicit covariance): each merge at height h contributes
lam^2 = rho - rho_parent with rho = 1 - 2 h^2. Pass the result as
structure= to \code{\link{race_probabilities}} or
\code{\link{polish_race}} to price or polish along the dendrogram.
}
\examples{
hc <- hclust(dist(matrix(rnorm(40), 8, 5)) / 10, method = "average")
tr <- tree_from_hclust(hc)
p <- race_probabilities(rnorm(8), structure = tr)
}
