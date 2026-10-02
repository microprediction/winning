# One race, five covariance grammars -- port of winning/factor/structures.py.
# Constructors return classed lists accepted by race_probabilities,
# abilities_from_race / calibrate_abilities, race_jacobian and polish_race
# via their structure= argument. D is always the idiosyncratic VARIANCE.

.structure <- function(cls, ...) structure(list(...), class = c(cls, "winning_structure"))

#' Covariance grammars for the general race
#'
#' \code{Independent(D)}: Sigma = diag(D). \code{Factor(V, D)}:
#' Sigma = V V' + diag(D). \code{Blocks(cluster, loading, D)}:
#' block-diagonal rank-1 (or rank-r) plus diagonal.
#' \code{Nested(cluster, loading, D, coupling, gamma)}: blocks plus one
#' global factor, gamma dialing the coupling from 0 (independent blocks)
#' to 1 (fully coupled). \code{Tree(cluster, loading, D, parent,
#' strength)}: a hierarchy of uniform shared effects (parent is 1-based,
#' root marked 0 or NA).
#'
#' @param D idiosyncratic variances
#' @param V loading matrix
#' @param cluster cluster labels
#' @param loading within-cluster loadings
#' @param coupling loadings on the global factor
#' @param gamma coupling strength in [0, 1]
#' @param parent 1-based parent index per node (root: 0/NA)
#' @param strength per-node shared-effect strength
#' @return a structure object for the structure= argument
#' @name structures
NULL

#' @rdname structures
#' @export
Independent <- function(D) .structure("Independent", D = D)

#' @rdname structures
#' @export
Factor <- function(V, D) .structure("Factor", V = V, D = D)

#' @rdname structures
#' @export
Blocks <- function(cluster, loading, D)
  .structure("Blocks", cluster = cluster, loading = loading, D = D)

#' @rdname structures
#' @export
Nested <- function(cluster, loading, D, coupling, gamma = 1.0)
  .structure("Nested", cluster = cluster, loading = loading, D = D,
             coupling = coupling, gamma = gamma)

#' @rdname structures
#' @export
Tree <- function(cluster, loading, D, parent, strength)
  .structure("Tree", cluster = cluster, loading = loading, D = D,
             parent = parent, strength = strength)

.dispatch_probabilities <- function(mu, s, base = "normal", points = 257,
                                    qa = 9, qf = 15, return_slopes = FALSE,
                                    window = "bulk", delta = 1e-12) {
  cls <- class(s)[1]
  if (cls == "Independent") {
    return(race_probabilities(mu, V = NULL, D = s$D, base = base,
                              points = points, return_slopes = return_slopes,
                              window = window, delta = delta))
  }
  if (cls == "Factor") {
    return(race_probabilities(mu, V = s$V, D = s$D, base = base,
                              points = points, return_slopes = return_slopes,
                              window = window, delta = delta))
  }
  # the hierarchical kernels are Gaussian hard races on their own
  # windows: a non-normal base, a window or a delta was dropped silently
  # and a different race priced (#89); refuse them, as python does
  .refuse_hierarchical_controls(cls, base, window, delta)
  if (return_slopes) stop("return_slopes is available for Independent/Factor only")
  if (cls == "Blocks") {
    return(block_race_probabilities(mu, s$cluster, s$loading, s$D,
                                    points = points, qa = qa))
  }
  if (cls == "Nested") {
    return(nested_race_probabilities(mu, s$cluster, s$loading, s$D,
                                     coupling = s$coupling, gamma = s$gamma,
                                     points = points, qa = qa, qf = qf))
  }
  if (cls == "Tree") {
    return(tree_race_probabilities(mu, s$cluster, s$loading, s$D,
                                   s$parent, s$strength,
                                   points = points, qa = qa))
  }
  stop("unknown structure ", cls)
}

.refuse_hierarchical_controls <- function(cls, base, window = "bulk",
                                          delta = 1e-12) {
  bad <- c(base = !identical(base, "normal"),
           window = !identical(window, "bulk"),
           delta = !isTRUE(all.equal(delta, 1e-12)))
  if (any(bad))
    stop(sprintf(paste("%s races do not support %s: the block/nested/tree",
                       "kernels are Gaussian hard races on their own",
                       "lattice windows. Use structure=Factor (or V=/D=)."),
                 cls, paste(names(bad)[bad], collapse = ", ")), call. = FALSE)
}

.dispatch_abilities <- function(p, s, base = "normal", points = 257,
                                qa = 9, qf = 15, n_iter = NULL, tol = NULL) {
  # n_iter and tol are the CALLER'S controls: every branch used to run
  # its own hidden defaults (#89). NULL keeps the branch's default.
  cls <- class(s)[1]
  ctl <- function(v, default) if (is.null(v)) default else v
  if (cls == "Independent") {
    return(abilities_from_race(p, V = NULL, D = s$D, base = base,
                               points = points, n_iter = ctl(n_iter, 60),
                               tol = ctl(tol, 1e-8)))
  }
  if (cls == "Factor") {
    return(abilities_from_race(p, V = s$V, D = s$D, base = base,
                               points = points, n_iter = ctl(n_iter, 60),
                               tol = ctl(tol, 1e-8)))
  }
  .refuse_hierarchical_controls(cls, base)
  if (cls == "Blocks") {
    return(abilities_from_block_race(p, s$cluster, s$loading, s$D,
                                     points = points, qa = qa,
                                     max_iter = ctl(n_iter, 25),
                                     tol = ctl(tol, 1e-10))$mu)
  }
  if (cls == "Nested") {
    return(.invert_generic(p, function(m)
      nested_race_probabilities(m, s$cluster, s$loading, s$D,
                                coupling = s$coupling, gamma = s$gamma,
                                points = points, qa = qa, qf = qf),
      tol = ctl(tol, 1e-9), max_iter = ctl(n_iter, 400)))
  }
  if (cls == "Tree") {
    return(.invert_generic(p, function(m)
      tree_race_probabilities(m, s$cluster, s$loading, s$D,
                              s$parent, s$strength,
                              points = points, qa = qa),
      tol = ctl(tol, 1e-9), max_iter = ctl(n_iter, 400)))
  }
  stop("unknown structure ", cls)
}

# Adaptive fixed-point inversion for structures without a Jacobian --
# port of polish._invert_generic.
.invert_generic <- function(p, forward, tol = 1e-9, max_iter = 400) {
  p <- as.numeric(p); p <- p / sum(p)
  lt <- log(pmax(p, 1e-300))
  mu <- -(lt - mean(lt))
  eta <- 1.0
  lp <- log(pmax(forward(mu), 1e-300))
  err <- max(abs(lp - lt))
  for (i in seq_len(max_iter)) {
    if (err < tol) break
    mu_n <- mu - eta * (lt - lp); mu_n <- mu_n - mean(mu_n)
    lp_n <- log(pmax(forward(mu_n), 1e-300))
    e <- max(abs(lp_n - lt))
    if (e < err) {
      mu <- mu_n; lp <- lp_n; err <- e
      eta <- min(eta * 1.2, 1.5)
    } else {
      eta <- eta * 0.5
      if (eta < 1e-4) break
    }
  }
  mu
}

#' The tree race implied by a hierarchical clustering (HRP's belief)
#'
#' Builds the Tree structure whose implied correlation is EXACTLY the
#' cophenetic correlation matrix 1 - 2 d^2 of the clustering: each merge
#' at cophenetic distance h contributes lam^2 = rho - rho_parent with
#' rho = 1 - 2 h^2; unit total variance per runner. A non-monotonic
#' linkage (an inversion, as centroid and median linkage produce) has no
#' tree representation and is refused with an error.
#'
#' @param Z scipy-style linkage matrix (n-1 rows; columns: merged node
#'   ids 0-based, distance, size)
#' @return a \code{\link{Tree}} structure
#' @export
tree_from_linkage <- function(Z) {
  Z <- as.matrix(Z)
  n <- nrow(Z) + 1L
  nT <- 2L * n - 1L
  parent <- integer(nT)                    # 0 = root
  # d[t] = 1 - rho_t = 2 h^2, kept directly: 1 - (1 - 2 h^2) cancels to
  # a few ulps for near-duplicate leaves (python Tree.from_linkage, #430)
  d <- rep(1, nT)
  for (k in seq_len(nrow(Z))) {
    a <- as.integer(Z[k, 1]) + 1L          # 0-based ids -> 1-based
    b <- as.integer(Z[k, 2]) + 1L
    t <- n + k
    parent[a] <- t; parent[b] <- t
    # floor rho at zero: the tree race cannot represent negative
    # dependence, so merges above the h = 1/sqrt(2) horizon leave
    # branches independent
    d[t] <- min(2 * Z[k, 3]^2, 1)
  }
  lam <- numeric(nT)
  # the nonnegative increments are a PREMISE, checked as in Python's
  # Tree.from_linkage: centroid and median linkage invert, and clipping
  # the negative increment returns a covariance that is not the
  # cophenetic one promised (#133)
  bad <- numeric(0); bad_t <- integer(0)
  for (t in (n + 1L):nT) {
    pa <- parent[t]
    lam2 <- (if (pa > 0) d[pa] else 1) - d[t]     # rho_t - rho_pa
    if (lam2 < -1e-9) { bad <- c(bad, lam2); bad_t <- c(bad_t, t) }
    lam[t] <- sqrt(max(lam2, 0))
  }
  if (length(bad)) {
    w <- which.min(bad)
    stop(sprintf(paste0(
      "this linkage is not monotonic: node %d merges %.3g BELOW its ",
      "parent, and %d node(s) do. A tree race is a nested variance ",
      "decomposition, so an inversion has no representation in it -- ",
      "clipping the negative increment would return a different ",
      "covariance from the cophenetic one this promises. Use a monotonic ",
      "method (average, complete, ward), or build the Tree with explicit ",
      "parent/strength."), bad_t[w] - 1L, -bad[w], length(bad)),
      call. = FALSE)
  }
  # D_i = 2 h^2 at the leaf's first merge, EXACTLY: the old absolute
  # floor pmax(., 1e-10) changed every near-duplicate branch (h = 1e-6
  # priced 0.556 where the cophenetic model gives pnorm(1)), #430. A
  # merge at height 0 (coincident leaves) is refused by name.
  D <- vapply(seq_len(n), function(i)
    if (parent[i] > 0) d[parent[i]] else 1, numeric(1))
  if (any(D <= 0)) {
    stop(sprintf(paste0(
      "leaf %d merges at height 0 (%d leaf/leaves do): coincident leaves ",
      "have zero idiosyncratic variance, which a tree race cannot price. ",
      "Merge the duplicates, or perturb them deliberately."),
      which(D <= 0)[1] - 1L, sum(D <= 0)), call. = FALSE)
  }
  Tree(cluster = seq_len(n), loading = numeric(n), D = D,
       parent = parent, strength = lam)
}

#' @rdname tree_from_linkage
#' @param hc an \code{\link[stats]{hclust}} object
#' @export
tree_from_hclust <- function(hc) {
  m <- hc$merge
  n <- nrow(m) + 1L
  Z <- matrix(0, n - 1L, 4)
  for (k in seq_len(nrow(m))) {
    id <- function(x) if (x < 0) -x - 1L else n + x - 1L   # to 0-based
    Z[k, ] <- c(id(m[k, 1]), id(m[k, 2]), hc$height[k], 0)
  }
  tree_from_linkage(Z)
}
