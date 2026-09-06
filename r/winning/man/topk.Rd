\name{top_k_probabilities}
\alias{top_k_probabilities}
\alias{bottom_k_probabilities}
\alias{rank_probabilities}
\alias{top_k_jacobians}
\title{Top-k memberships, rank marginals, and their Jacobians}
\usage{
top_k_probabilities(mu, k, V = NULL, D = NULL, base = "normal",
  points = 513, qa = 15)

bottom_k_probabilities(mu, k, V = NULL, D = NULL, base = "normal",
  points = 513, qa = 15)

rank_probabilities(mu, D = NULL, base = "normal", points = 513)

top_k_jacobians(mu, k, D = NULL, base = "normal", points = 513,
  V = NULL, qa = 15)
}
\arguments{
\item{mu}{numeric abilities; performances are times, so lower is better
(min-wins convention)}
\item{k}{membership depth, in [1, n-1] (k = 1 is the win probability)}
\item{V}{optional factor loadings, a vector (rank 1) or n x 2 matrix;
correlation enters as the Gauss-Hermite node mixture of conditionally
independent races}
\item{D}{idiosyncratic variances (default 1)}
\item{base}{"normal", "gumbel", "logistic", "laplace", or a function
z -> list(S, f, fp) of a mean-zero unit-variance law}
\item{points}{lattice size for the count-program quadrature}
\item{qa}{Gauss-Hermite order for the factor mixture}
}
\value{
\code{top_k_probabilities}: q with q[i] = P(runner i among the k
smallest); the slot identity sum(q) = k is enforced and a material
defect stops. \code{bottom_k_probabilities}: P(among the k largest),
by the complement identity. \code{rank_probabilities}: an n x n matrix
with entry [i, r] = P(runner i finishes r-th); rows and columns sum to
one. \code{top_k_jacobians}: list(Jmu, Jsigma) of n x n matrices
dq/dmu (minus a graph Laplacian on the rank-k boundary: symmetric
nonnegative off-diagonals, zero row sums) and dq/dsigma (satisfying
the scale-gauge Euler identity mu . row_mu + sigma . row_sigma = 0).
}
\description{
Exact membership probabilities at any depth by one shared
Poisson-binomial count program over the survival field, each runner
removed by stable-direction deconvolution. Port of
winning/factor/topk.py, refereed by the embedded parity vectors.
}
\seealso{\code{\link{abilities_from_topk}},
\code{\link{loc_scale_from_topk_pair}},
\code{\link{race_probabilities}}}
