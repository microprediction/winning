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
#' @param leaves optional Tree leaf labels in leaf-node order; NULL uses
#'   the sorted cluster labels (#372)
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
Tree <- function(cluster, loading, D, parent, strength, leaves = NULL)
  .structure("Tree", cluster = cluster, loading = loading, D = D,
             parent = parent, strength = strength, leaves = leaves)

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
                                   points = points, qa = qa,
                                   leaves = s$leaves))
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
  if (cls == "Nested" || cls == "Tree") {
    out <- .abilities_from_structure(
      p, s, function(m) .dispatch_probabilities(m, s, points = points,
                                                qa = qa, qf = qf),
      points = points, n_iter = ctl(n_iter, 400), tol = ctl(tol, 1e-9))
    return(out$mu)
  }
  stop("unknown structure ", cls)
}

# Total per-runner variance of a structure (shared effects plus
# idiosyncratic): the marginal the generic inverter's independent
# surrogate matches. Python structures.structure_variances.
.structure_variances <- function(s) {
  cls <- class(s)[1]
  n <- length(s$cluster)
  D <- .as_idio(s$D, n, positive = TRUE)
  if (cls == "Blocks")
    return(D + rowSums(.as_loadings(s$loading, n, name = "loading")^2))
  if (cls == "Nested") {
    tot <- D + rowSums(.as_loadings(s$loading, n, name = "loading")^2)
    if (!is.null(s$coupling) && s$gamma != 0)
      tot <- tot + s$gamma^2 *
        rowSums(.as_loadings(s$coupling, n, name = "coupling")^2)
    return(tot)
  }
  if (cls == "Tree") {
    inv <- .tree_leaf_index(s$cluster, s$leaves)         # (#372)
    nC <- max(inv)
    lam <- as.numeric(s$strength)
    parent <- .validate_tree(s$parent, lam, nC)
    # a common ancestor's shock cancels and is no part of the race (#489)
    lam <- .without_common_shock(parent, lam, nC)
    anc <- numeric(nC)
    for (cc in seq_len(nC)) {
      u <- cc
      while (parent[u] > 0) { anc[cc] <- anc[cc] + lam[parent[u]]^2; u <- parent[u] }
    }
    return(D + .scalar_loading(s$loading)^2 + anc[inv])
  }
  stop("unknown structure ", cls)
}

# Generic grammar inversion, python races._abilities_from_structure: the
# exact forward through the dispatch, a damped log-residual fixed point
# preconditioned by the own-slope of the independent race at matched
# total variances, in the field's own units. It replaced a port of
# polish._invert_generic whose start and steps were unit-free, so the
# same Nested/Tree race missed by 0.41 of a share when written at
# c = 1e-3 and 0.18 at c = 1e3 (#100). Returns list(mu, converged,
# resid, iters).
.abilities_from_structure <- function(p, s, forward, points = 257,
                                      n_iter = 120, tol = 1e-8) {
  target <- .rescaled_target(as.numeric(p))
  logt <- log(target)
  totvar <- .structure_variances(s)
  scale <- sqrt(stats::median(totvar))
  mu <- -(logt - mean(logt)) / 2 * scale
  alpha <- 0.7; last <- Inf
  err <- Inf; iters <- 0L
  for (it in seq_len(n_iter)) {
    iters <- it
    phat <- forward(mu)
    resid <- log(pmax(phat, 1e-300)) - logt
    err <- max(abs(resid))
    if (err < tol) break
    if (err > last) alpha <- max(alpha * 0.5, 0.05)
    last <- err
    ps <- race_probabilities(mu, D = totvar, points = points,
                             return_slopes = TRUE)
    dlogp <- pmin(ps$slopes / pmax(ps$p, 1e-300), -1e-6 / scale)
    lim <- pmin(2, 10 * abs(resid)) * scale
    mu <- mu - pmin(pmax(alpha * resid / dlogp, -lim), lim)
    mu <- mu - mean(mu)
  }
  list(mu = mu, converged = err < tol, resid = err, iters = iters)
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
# A scipy-style linkage checked at the constructor's own boundary (python
# structures._as_linkage): row k merges two DISTINCT clusters that already
# exist and are still active, at a finite non-negative height. A
# self-reference built a cyclic parent (#328), ids were truncated, and a
# negative height was squared into the tree of its absolute value (#459).
.check_linkage <- function(Z) {
  if (length(Z) == 0L) return(matrix(numeric(0), 0, 4))
  if (!is.numeric(Z) || length(dim(Z)) != 2L || ncol(Z) < 3L)
    stop("a linkage must be a numeric (n-1) x 4 matrix", call. = FALSE)
  n <- nrow(Z) + 1L
  used <- logical(2L * n - 1L)
  for (k in seq_len(nrow(Z))) {
    r <- k - 1L                                  # 0-based row, as scipy
    h <- Z[k, 3]
    if (!is.finite(h) || h < 0)
      stop(sprintf("linkage row %d merges at height %s: merge heights are finite non-negative distances (#459)",
                   r, format(h)), call. = FALSE)
    for (cc in Z[k, 1:2]) {
      if (!is.finite(cc) || cc != floor(cc))
        stop(sprintf("linkage row %d names cluster %s: cluster ids are whole numbers (#328)",
                     r, format(cc)), call. = FALSE)
      if (cc < 0 || cc >= n + r)
        stop(sprintf("linkage row %d uses cluster %d before it is formed: row %d may merge only ids below %d (#328)",
                     r, as.integer(cc), r, n + r), call. = FALSE)
      if (used[cc + 1L])
        stop(sprintf("linkage row %d reuses cluster %d, already merged by an earlier row (#328)",
                     r, as.integer(cc)), call. = FALSE)
    }
    if (Z[k, 1] == Z[k, 2])
      stop(sprintf("linkage row %d merges cluster %d with itself (#328)",
                   r, as.integer(Z[k, 1])), call. = FALSE)
    used[Z[k, 1:2] + 1L] <- TRUE
  }
  Z
}

tree_from_linkage <- function(Z) {
  Z <- .check_linkage(as.matrix(Z))
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
  # seq_len, not (n + 1L):nT -- at one leaf that is 2:1 = c(2, 1) (#382)
  for (t in n + seq_len(n - 1L)) {
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
