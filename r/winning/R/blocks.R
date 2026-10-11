# Block, nested and tree races -- base-R port of winning/factor/blocks.py.
# Min-wins abilities, mean-zero gauge, Gaussian base. Rank r >= 3 cluster
# effects need scrambled Sobol nodes and are not provided in pure R
# (rank 1 and the special rank 2 are; pass nodes explicitly otherwise).

.TINY <- 1e-300

# Winner-bulk window covering every RETAINED conditional race (max-wins):
# per-runner conditional locations range over [mu - amp, mu + amp] with amp
# the largest shared-effect shift the quadrature nodes can produce, and the
# IDIOSYNCRATIC sd sets the local scale. Replaces an independent-marginal
# proxy whose lower crossing drifted upward with n under near-common shocks
# while the winner sat O(1) lower (28 percent of mass lost on a 400-runner
# cluster at correlation 0.99; fourteenth review).
.window_nodes <- function(mu, sd, amp, delta = 1e-12, pad_sds = 2) {
  if (length(amp) == 1) amp <- rep(amp, length(mu))
  m_lo <- mu - amp
  m_hi <- mu + amp
  smax <- max(max(sd), 1e-12)
  Fx <- function(x, m) exp(sum(log(pmax(stats::pnorm((x - m) / sd), .TINY))))
  lo <- min(m_lo) - 9 * smax
  step <- 9 * smax
  for (i in 1:60) {
    if (Fx(lo, m_lo) <= delta) break
    lo <- lo - step; step <- 2 * step
  }
  hi <- max(m_hi) + 9 * smax
  step <- 9 * smax
  for (i in 1:60) {
    if (Fx(hi, m_hi) >= 1 - delta) break
    hi <- hi + step; step <- 2 * step
  }
  a <- lo; b <- hi
  for (i in 1:70) {
    m <- 0.5 * (a + b)
    if (Fx(m, m_lo) < delta) a <- m else b <- m
  }
  xlo <- a
  a <- xlo; b <- hi
  for (i in 1:70) {
    m <- 0.5 * (a + b)
    if (Fx(m, m_hi) < 1 - delta) a <- m else b <- m
  }
  c(xlo - pad_sds * smax, b + pad_sds * smax)
}

# Tree races take SCALAR (rank-one) leaf-cluster loadings; only the block
# forward kernel prices rank r. as.numeric() on a matrix silently flattens
# it, which priced garbage -- refuse instead.
.scalar_loading <- function(loading) {
  if (is.matrix(loading)) {
    if (ncol(loading) > 1) {
      stop(paste0("tree races take scalar (rank-one) leaf-cluster ",
                  "loadings; rank-r leaf effects are supported by the ",
                  "block grammar only."))
    }
    return(as.numeric(loading))
  }
  as.numeric(loading)
}

# Raw lattice mass is a diagnostic, not a nuisance: a material defect means
# the window missed the winner's region, and normalizing it away returns
# confident wrong shares. Stop instead.
.checked_mass <- function(raw, kind, mass_tol = 5e-3) {
  t_ <- sum(raw)
  # non-finite mass must fail explicitly, not fall into the comparison
  if (!is.finite(t_) || abs(t_ - 1) > mass_tol) {
    stop(sprintf(paste0(
      "%s lattice captured total mass %.4f (defect %.2e > %g): the window ",
      "missed part of the winner distribution and the shares would be ",
      "silently wrong if normalized. Raise points=, or report this field: ",
      "the node-aware window should have covered it."),
      kind, t_, abs(t_ - 1), mass_tol))
  }
  raw / t_
}

.cluster_index <- function(cluster) {
  lv <- sort(unique(cluster))
  match(cluster, lv)                       # np.unique return_inverse
}

.cluster_nodes <- function(r, qa) {
  if (r == 0) {
    # the EMPTY PRODUCT: one node of weight 1 with no columns. Rank 1
    # and rank 2 were handled and everything else fell to the rank >= 3
    # refusal below, so an (n, 0) matrix -- the documented spelling of
    # "no factors" -- was turned away with a message about Sobol nodes
    # for high rank (#309).
    return(list(nodes = matrix(numeric(0), 1L, 0L), w = 1))
  }
  if (r == 1) {
    h <- .hermite1(qa)
    return(list(nodes = matrix(h$nodes, ncol = 1), w = h$weights))
  }
  if (r == 2) {
    # python: nodes [[a, b] for a in an for b in an], weights u*v, same order
    h <- .hermite1(qa)
    nodes <- cbind(rep(h$nodes, each = qa), rep(h$nodes, times = qa))
    w <- rep(h$weights, each = qa) * rep(h$weights, times = qa)
    return(list(nodes = nodes, w = w / sum(w)))
  }
  stop("rank >= 3 cluster effects need Sobol nodes; supply nodes= or rank <= 2")
}

# Max-wins rank-1 field kernel (public functions negate).
.block_max <- function(mu, sd, cluster, v, points, qa) {
  # rank zero is the independent race; as.numeric() made it numeric(0)
  # and the kernel indexed past it (#508)
  if (is.matrix(v) && ncol(v) == 0L) v <- rep(0, length(mu))
  if (is.matrix(v) && ncol(v) > 1) {
    return(.block_max_r(mu, sd, cluster, v, points, qa))
  }
  v <- as.numeric(v)
  n <- length(mu)
  inv <- .cluster_index(cluster)
  ord <- order(inv)                        # stable
  mu_o <- mu[ord]; sd_o <- sd[ord]; v_o <- v[ord]; c_o <- inv[ord]
  h <- .hermite1(qa)
  an <- h$nodes; aw <- h$weights
  amp <- abs(v_o) * max(abs(an))
  lh <- .window_nodes(mu_o, sd_o, amp)
  x <- seq(lh[1], lh[2], length.out = points)
  dx <- x[2] - x[1]
  nC <- max(c_o)
  xm <- matrix(x, n, points, byrow = TRUE)
  S <- array(0, c(nC, qa, points))
  logF <- array(0, c(n, qa, points))
  pdf <- array(0, c(n, qa, points))
  for (q in seq_len(qa)) {
    z <- (xm - mu_o - v_o * an[q]) / sd_o
    lf <- log(pmax(stats::pnorm(z), .TINY))
    logF[, q, ] <- lf
    pdf[, q, ] <- exp(-0.5 * z * z) / (sd_o * sqrt(2 * pi))
    S[, q, ] <- rowsum(lf, c_o)
  }
  G <- matrix(0, nC, points)
  for (q in seq_len(qa)) G <- G + aw[q] * exp(pmin(matrix(S[, q, ], dim(S)[1], dim(S)[3]), 0))
  logG <- log(pmax(G, .TINY))
  rest <- exp(pmin(matrix(colSums(logG), nC, points, byrow = TRUE) - logG, 0))
  hmat <- matrix(0, n, points)
  for (q in seq_len(qa)) {
    hmat <- hmat + aw[q] * pdf[, q, ] * exp(pmin(S[c_o, q, ] - logF[, q, ], 0))
  }
  p_o <- rowSums(hmat * rest[c_o, , drop = FALSE]) * dx
  p <- numeric(n); p[ord] <- p_o
  pmax(p, 0)
}

# Max-wins rank-r blocks: loading matrix V (n, r), per-cluster r-dim effect.
.block_max_r <- function(mu, sd, cluster, V, points, qa, nodes = NULL) {
  V <- as.matrix(V)
  n <- length(mu)
  r <- ncol(V)
  inv <- .cluster_index(cluster)
  ord <- order(inv)
  mu_o <- mu[ord]; sd_o <- sd[ord]; V_o <- V[ord, , drop = FALSE]
  c_o <- inv[ord]
  if (is.null(nodes)) nodes <- .cluster_nodes(r, qa)
  Fm <- nodes$nodes; w <- nodes$w
  Q <- nrow(Fm)
  amp <- sqrt(rowSums(V_o^2)) * max(sqrt(rowSums(Fm^2)))
  lh <- .window_nodes(mu_o, sd_o, amp)
  x <- seq(lh[1], lh[2], length.out = points)
  dx <- x[2] - x[1]
  nC <- max(c_o)
  xm <- matrix(x, n, points, byrow = TRUE)
  shift <- V_o %*% t(Fm)                   # (n, Q)
  S <- array(0, c(nC, Q, points))
  logF <- array(0, c(n, Q, points))
  pdf <- array(0, c(n, Q, points))
  for (q in seq_len(Q)) {
    z <- (xm - mu_o - shift[, q]) / sd_o
    lf <- log(pmax(stats::pnorm(z), .TINY))
    logF[, q, ] <- lf
    pdf[, q, ] <- exp(-0.5 * z * z) / (sd_o * sqrt(2 * pi))
    S[, q, ] <- rowsum(lf, c_o)
  }
  G <- matrix(0, nC, points)
  for (q in seq_len(Q)) G <- G + w[q] * exp(pmin(matrix(S[, q, ], dim(S)[1], dim(S)[3]), 0))
  logG <- log(pmax(G, .TINY))
  rest <- exp(pmin(matrix(colSums(logG), nC, points, byrow = TRUE) - logG, 0))
  hmat <- matrix(0, n, points)
  for (q in seq_len(Q)) {
    hmat <- hmat + w[q] * pdf[, q, ] * exp(pmin(S[c_o, q, ] - logF[, q, ], 0))
  }
  p_o <- rowSums(hmat * rest[c_o, , drop = FALSE]) * dx
  p <- numeric(n); p[ord] <- p_o
  pmax(p, 0)
}

#' Block race: independent rank-1 (or rank-r) cluster effects
#'
#' @param mu abilities (min-wins), length n
#' @param cluster cluster labels, length n (any comparable values)
#' @param loading numeric length n (rank 1) or n x r matrix
#' @param D idiosyncratic variances, length n
#' @param points lattice size
#' @param qa cluster-effect quadrature order
#' @return win probabilities summing to one
#' @export
block_race_probabilities <- function(mu, cluster, loading, D,
                                     points = 257, qa = 9) {
  mu <- as.numeric(mu)
  # a scalar D is one variance per contestant, as at every boundary (#510)
  sd <- sqrt(.as_idio(D, length(mu), positive = TRUE))
  p <- .block_max(-mu, sd, cluster, loading, points, qa)
  .checked_mass(p, "block race")
}

#' Nested race: block race plus one global factor
#'
#' @inheritParams block_race_probabilities
#' @param coupling per-runner loading on the global factor (vector, or
#'   n x k matrix for k global factors with k <= 2)
#' @param gamma interpolates from independent blocks (0) to fully
#'   coupled (1)
#' @param qf global-factor quadrature order
#' @return win probabilities summing to one
#' @export
nested_race_probabilities <- function(mu, cluster, loading, D,
                                      coupling = NULL, gamma = 1.0,
                                      points = 257, qa = 9, qf = 15) {
  if (is.null(coupling) || gamma == 0) {
    return(block_race_probabilities(mu, cluster, loading, D,
                                    points = points, qa = qa))
  }
  mu <- as.numeric(mu)
  # the shared loading contract: (n, k), (k, n) or a length-n vector.
  # Transposing once and trusting the result let a 2x2 coupling at n = 4
  # recycle g %*% f across the field as a different loading (#469)
  g <- .as_loadings(coupling, length(mu), name = "coupling")
  if (ncol(g) == 1) {
    h <- .hermite1(qf)
    fn <- matrix(h$nodes, ncol = 1)
    fw <- h$weights
  } else {
    cn <- .cluster_nodes(ncol(g), qf)
    fn <- cn$nodes; fw <- cn$w
  }
  sd <- sqrt(.as_idio(D, length(mu), positive = TRUE))
  p <- numeric(length(mu))
  for (q in seq_len(nrow(fn))) {
    # average the RAW conditional masses (each near one) and normalize
    # once: normalizing each conditional separately hides a window defect
    p <- p + fw[q] * .block_max(-(mu + gamma * as.numeric(g %*% fn[q, ])),
                                sd, cluster, loading, points, qa)
  }
  .checked_mass(p, "nested race")
}

# Strengths with every COMMON ancestor's set to zero (python's
# blocks._without_common_shock). A node above every cluster adds the same
# shock to everyone and cancels from every difference; kept, it widened
# the lattice window until an irrelevant root strength of 50 raised a
# mass defect at 257 points (#489). `parent` is 1-based with 0 = root.
.without_common_shock <- function(parent, lam, nC) {
  hits <- integer(length(parent))
  for (cc in seq_len(nC)) {
    u <- cc
    while (parent[u] > 0) { u <- parent[u]; hits[u] <- hits[u] + 1L }
  }
  lam[hits == nC] <- 0
  lam
}

# A tree is a tree: one root, every parent an existing node, no cycles,
# the cluster nodes are leaves -- the contract of python's
# _validate_tree. Each parent == 0 node kept its own unit cavity, so a
# forest (parent = c(0, 0)) priced each component without the others
# and reached .checked_mass with total mass 2, misreported as a lattice
# defect (#517); a cycle spun the `while (parent[u] > 0)` walks forever.
.validate_tree <- function(parent, strength, n_clusters) {
  if (!is.numeric(parent) && !is.logical(parent))
    stop("parent must be a numeric vector of 1-based node indices",
         call. = FALSE)
  parent <- as.numeric(parent)
  parent[is.na(parent)] <- 0                      # 0/NA = root
  nT <- length(parent)
  if (length(strength) != nT)
    stop(sprintf(paste("strength must have one entry per tree node;",
                       "got %d for %d nodes"), length(strength), nT),
         call. = FALSE)
  if (any(!is.finite(strength)))
    stop("strength has a non-finite entry", call. = FALSE)
  bad <- which(!is.finite(parent) | parent != round(parent) | parent < 0 |
               parent > nT)
  if (length(bad))
    stop(sprintf("parent[%d] = %s is not 0/NA or a node index in 1..%d",
                 bad[1], format(parent[bad[1]]), nT), call. = FALSE)
  parent <- as.integer(parent)
  selfp <- which(parent == seq_len(nT))
  if (length(selfp))
    stop(sprintf("parent contains a cycle: %d -> %d", selfp[1], selfp[1]),
         call. = FALSE)
  roots <- which(parent == 0L)
  if (length(roots) != 1L)
    stop(sprintf("a tree has exactly one root (parent 0/NA); got %d%s",
                 length(roots),
                 if (length(roots)) paste0(" (", paste(roots, collapse = ", "),
                                           ")") else ""),
         call. = FALSE)
  state <- integer(nT)                    # 0 unseen, 1 on path, 2 done
  for (t in seq_len(nT)) {
    path <- integer(0); u <- t
    while (u > 0L && state[u] == 0L) {
      state[u] <- 1L; path <- c(path, u); u <- parent[u]
    }
    if (u > 0L && state[u] == 1L) {
      k <- match(u, path)
      stop(paste("parent contains a cycle:",
                 paste(c(path[k:length(path)], u), collapse = " -> ")),
           call. = FALSE)
    }
    state[path] <- 2L
  }
  if (n_clusters > nT)
    stop(sprintf("%d clusters but only %d tree nodes", n_clusters, nT),
         call. = FALSE)
  kids <- which(parent > 0L & parent <= n_clusters)
  if (length(kids))
    stop(sprintf(paste("node %d has parent %d, a cluster node; the first %d",
                       "nodes are the clusters and must be leaves"),
                 kids[1], parent[kids[1]], n_clusters), call. = FALSE)
  parent
}

#' Tree race: hierarchy of uniform shared effects
#'
#' Leaf clusters keep per-member loadings; each internal node t applies
#' strength[t] uniformly to every leaf beneath it. Node indices continue
#' past the leaf clusters; parent[t] gives the tree with the root's
#' parent 0 (or NA). Two message passes on the lattice.
#'
#' @inheritParams block_race_probabilities
#' @param parent integer vector over all nodes, 1-based; root has 0/NA
#' @param strength per-node shared-effect strength
#' @return win probabilities summing to one
#' @export
tree_race_probabilities <- function(mu, cluster, loading, D, parent,
                                    strength, points = 257, qa = 9) {
  mu <- as.numeric(mu)
  m <- -mu
  sd <- sqrt(.as_idio(D, length(mu), positive = TRUE))
  v <- .scalar_loading(loading)
  lam <- as.numeric(strength)
  n <- length(m)
  inv <- .cluster_index(cluster)
  nC <- max(inv)
  parent <- .validate_tree(parent, lam, nC)        # 0 = root (#517)
  nT <- length(parent)
  lam <- .without_common_shock(parent, lam, nC)          # #489
  ord <- order(inv)
  mu_o <- m[ord]; sd_o <- sd[ord]; v_o <- v[ord]; c_o <- inv[ord]
  h <- .hermite1(qa)
  an <- h$nodes; aw <- h$weights
  depth_shift <- numeric(nT)
  # traverse by TREE depth, not by accumulated |strength|: zero strengths
  # (from_linkage's floored merges) tie the |lambda| path sums and a tied
  # sort visits children before their parents, reading cavities still at
  # their initial value.
  depth_hops <- integer(nT)
  for (t in seq_len(nT)) {
    s_ <- 0; d_ <- 0L; u <- t
    while (parent[u] > 0) {
      s_ <- s_ + abs(lam[parent[u]]); d_ <- d_ + 1L; u <- parent[u]
    }
    depth_shift[t] <- s_
    depth_hops[t] <- d_
  }
  path_var <- numeric(nC)
  for (cc in seq_len(nC)) {
    s_ <- 0; u <- cc
    while (parent[u] > 0) { s_ <- s_ + lam[parent[u]]^2; u <- parent[u] }
    path_var[cc] <- s_
  }
  amp <- (abs(v_o) + depth_shift[c_o]) * max(abs(an))
  lh <- .window_nodes(mu_o, sd_o, amp)
  x <- seq(lh[1], lh[2], length.out = points)
  dx <- x[2] - x[1]
  xm <- matrix(x, n, points, byrow = TRUE)
  S <- array(0, c(nC, qa, points))
  logF <- array(0, c(n, qa, points))
  pdf <- array(0, c(n, qa, points))
  for (q in seq_len(qa)) {
    z <- (xm - mu_o - v_o * an[q]) / sd_o
    lf <- log(pmax(stats::pnorm(z), .TINY))
    logF[, q, ] <- lf
    pdf[, q, ] <- exp(-0.5 * z * z) / (sd_o * sqrt(2 * pi))
    S[, q, ] <- rowsum(lf, c_o)
  }
  G <- matrix(0, nT, points)
  for (q in seq_len(qa)) {
    G[1:nC, ] <- G[1:nC, ] + aw[q] * exp(pmin(matrix(S[, q, ], dim(S)[1], dim(S)[3]), 0))
  }
  children <- vector("list", nT)
  for (t in seq_len(nT)) {
    if (parent[t] > 0) children[[parent[t]]] <- c(children[[parent[t]]], t)
  }
  shift_eval <- function(g, delta) {
    stats::approx(x - delta, g, xout = x, rule = 2, ties = "ordered")$y
  }
  up <- setdiff(seq_len(nT), seq_len(nC))
  up <- up[order(-depth_hops[up])]
  for (t in up) {
    acc <- numeric(points)
    for (q in seq_len(qa)) {
      prod <- rep(1, points)
      for (cc in children[[t]]) prod <- prod * shift_eval(G[cc, ], lam[t] * an[q])
      acc <- acc + aw[q] * prod
    }
    G[t, ] <- pmax(acc, 0)
  }
  R <- matrix(1, nT, points)
  down <- order(depth_hops)
  for (t in down) {
    pa <- parent[t]
    if (pa == 0) next
    sm <- numeric(points)
    for (q in seq_len(qa)) sm <- sm + aw[q] * shift_eval(R[pa, ], -lam[pa] * an[q])
    prod <- rep(1, points)
    for (s_ in children[[pa]]) if (s_ != t) prod <- prod * G[s_, ]
    R[t, ] <- pmax(sm * prod, 0)
  }
  hmat <- matrix(0, n, points)
  for (q in seq_len(qa)) {
    hmat <- hmat + aw[q] * pdf[, q, ] * exp(pmin(S[c_o, q, ] - logF[, q, ], 0))
  }
  p_o <- rowSums(hmat * R[c_o, , drop = FALSE]) * dx
  p <- numeric(n); p[ord] <- p_o
  .checked_mass(pmax(p, 0), "tree race")
}

#' Exact Jacobian d p / d mu of the block race (min-wins)
#'
#' Rows sum to zero; off-diagonals are the symmetric tie densities.
#' @inheritParams block_race_probabilities
#' @return n x n matrix
#' @export
block_race_jacobian <- function(mu, cluster, loading, D,
                                points = 257, qa = 9) {
  if (is.matrix(loading) && ncol(loading) > 1) {
    stop(paste0("block_race_jacobian is implemented for rank-one cluster ",
                "loadings only; rank-r tie densities are tracked work."))
  }
  mu <- as.numeric(mu)
  m <- -mu
  sd <- sqrt(.as_idio(D, length(mu), positive = TRUE))
  # rank zero: no shared effect (#508)
  v <- if (is.matrix(loading) && ncol(loading) == 0L) rep(0, length(mu))
       else as.numeric(loading)
  n <- length(mu)
  inv <- .cluster_index(cluster)
  ord <- order(inv)
  mu_o <- m[ord]; sd_o <- sd[ord]; v_o <- v[ord]; c_o <- inv[ord]
  n_c <- max(c_o)
  h <- .hermite1(qa)
  an <- h$nodes; aw <- h$weights
  amp <- abs(v_o) * max(abs(an))
  lh <- .window_nodes(mu_o, sd_o, amp)
  x <- seq(lh[1], lh[2], length.out = points)
  dx <- x[2] - x[1]
  xm <- matrix(x, n, points, byrow = TRUE)
  S <- array(0, c(n_c, qa, points))
  logF <- array(0, c(n, qa, points))
  pdf <- array(0, c(n, qa, points))
  for (q in seq_len(qa)) {
    z <- (xm - mu_o - v_o * an[q]) / sd_o
    lf <- log(pmax(stats::pnorm(z), .TINY))
    logF[, q, ] <- lf
    pdf[, q, ] <- exp(-0.5 * z * z) / (sd_o * sqrt(2 * pi))
    S[, q, ] <- rowsum(lf, c_o)
  }
  G <- matrix(0, n_c, points)
  for (q in seq_len(qa)) G <- G + aw[q] * exp(pmin(matrix(S[, q, ], dim(S)[1], dim(S)[3]), 0))
  logG <- log(pmax(G, .TINY))
  logG_all <- colSums(logG)
  Rc <- exp(pmin(matrix(logG_all, n_c, points, byrow = TRUE) - logG, 0))
  hmat <- matrix(0, n, points)
  for (q in seq_len(qa)) {
    hmat <- hmat + aw[q] * pdf[, q, ] * exp(pmin(S[c_o, q, ] - logF[, q, ], 0))
  }
  Gall <- exp(pmin(logG_all, 0))
  U <- hmat * Rc[c_o, , drop = FALSE] /
    matrix(sqrt(pmax(Gall, .TINY)), n, points, byrow = TRUE) * sqrt(dx)
  J <- -(U %*% t(U))
  for (ci in seq_len(n_c)) {
    idx <- which(c_o == ci)
    k <- length(idx)
    if (k == 1) next
    term <- matrix(0, k, k)
    for (q in seq_len(qa)) {
      Rcq <- Rc[ci, ]
      Fk <- matrix(logF[idx, q, ], k, points)     # (k, points)
      Pk <- matrix(pdf[idx, q, ], k, points)
      for (ii in seq_len(k)) {
        base <- matrix(S[ci, q, ] - Fk[ii, ], k, points, byrow = TRUE)
        lo2 <- exp(pmin(base - Fk, 0))
        wrow <- Pk[ii, ] * Rcq
        term[ii, ] <- term[ii, ] + aw[q] *
          rowSums(Pk * lo2 * matrix(wrow, k, points, byrow = TRUE)) * dx
      }
    }
    J[idx, idx] <- -term
  }
  diag(J) <- 0
  diag(J) <- -rowSums(J)
  Jf <- matrix(0, n, n)
  Jf[ord, ord] <- J
  -Jf                                       # chain rule: p_min(mu) = p_max(-mu)
}

#' Exact Jacobian of the nested race (mixture of block Jacobians)
#' @inheritParams nested_race_probabilities
#' @return n x n matrix
#' @export
nested_race_jacobian <- function(mu, cluster, loading, D, coupling = NULL,
                                 gamma = 1.0, points = 257, qa = 9, qf = 15) {
  if (is.null(coupling) || gamma == 0) {
    return(block_race_jacobian(mu, cluster, loading, D,
                               points = points, qa = qa))
  }
  mu <- as.numeric(mu)
  # the shared loading contract: (n, k), (k, n) or a length-n vector.
  # Transposing once and trusting the result let a 2x2 coupling at n = 4
  # recycle g %*% f across the field as a different loading (#469)
  g <- .as_loadings(coupling, length(mu), name = "coupling")
  if (ncol(g) == 1) {
    h <- .hermite1(qf)
    fn <- matrix(h$nodes, ncol = 1); fw <- h$weights
  } else {
    cn <- .cluster_nodes(ncol(g), qf)
    fn <- cn$nodes; fw <- cn$w
  }
  n <- length(mu)
  J <- matrix(0, n, n)
  for (q in seq_len(nrow(fn))) {
    J <- J + fw[q] * block_race_jacobian(
      mu + gamma * as.numeric(g %*% fn[q, ]), cluster, loading, D,
      points = points, qa = qa)
  }
  J
}

#' Invert the block race: centred min-wins mu reproducing p
#'
#' Sub-resolution targets are bounds, not measurements: they are floored
#' at max(1e-14, min positive / 1000).
#'
#' @inheritParams block_race_probabilities
#' @param p positive target probabilities
#' @param tol convergence tolerance
#' @param max_iter Newton iterations after the fixed-point globalizer
#' @return list(mu, residual, iterations)
#' @export
abilities_from_block_race <- function(p, cluster, loading, D,
                                      points = 257, qa = 9,
                                      tol = 1e-10, max_iter = 25) {
  p_t <- .rescaled_target(as.numeric(p))       # scale-safe (#463)
  n <- length(p_t)
  floor_ <- max(1e-14, min(p_t[p_t > 0]) * 1e-3)
  p_t <- pmax(p_t, floor_); p_t <- p_t / sum(p_t)
  lt <- log(p_t)
  ones <- matrix(1 / n, n, n)
  forward <- function(m) block_race_probabilities(m, cluster, loading, D,
                                                  points = points, qa = qa)
  mu <- -(lt - mean(lt))
  eta <- 1.0
  lp <- log(pmax(forward(mu), .TINY))
  err <- max(abs(lp - lt))
  for (i in 1:200) {
    if (err < 0.2) break
    mu_n <- mu - eta * (lt - lp); mu_n <- mu_n - mean(mu_n)
    lp_n <- log(pmax(forward(mu_n), .TINY))
    e_n <- max(abs(lp_n - lt))
    if (e_n < err) {
      mu <- mu_n; lp <- lp_n; err <- e_n
      eta <- min(eta * 1.2, 1.5)
    } else {
      eta <- eta * 0.5
      if (eta < 1e-4) break
    }
  }
  for (it in seq_len(max_iter)) {
    pv <- pmax(forward(mu), .TINY); pv <- pv / sum(pv)
    r <- log(pv) - lt
    cur <- max(abs(r))
    if (cur < tol) return(list(mu = mu - mean(mu), residual = cur,
                               iterations = it))
    J <- block_race_jacobian(mu, cluster, loading, D,
                             points = points, qa = qa)
    Jl <- J / pv
    step <- tryCatch(qr.solve(Jl + ones, -r, tol = 1e-12),
                     error = function(e) qr.coef(qr(Jl + ones), -r))
    step[is.na(step)] <- 0
    nn <- sqrt(sum(step^2))
    if (nn > 5) step <- step * 5 / nn
    for (k in 1:8) {
      mu_n <- mu + step; mu_n <- mu_n - mean(mu_n)
      p_n <- pmax(forward(mu_n), .TINY); p_n <- p_n / sum(p_n)
      if (max(abs(log(p_n) - lt)) < cur) { mu <- mu_n; break }
      step <- step * 0.5
    }
  }
  pv <- pmax(forward(mu), .TINY); pv <- pv / sum(pv)
  list(mu = mu - mean(mu), residual = max(abs(log(pv) - lt)),
       iterations = max_iter)
}

#' Jacobian of the tree race (min-wins)
#'
#' Same-cluster term exact under the downward message; cross-cluster by
#' a Gram approximation, exact when clusters share no ancestor effects.
#' Feasibility-critical callers (polish_race) verify on the exact
#' forward map and fall back to finite differences.
#'
#' @inheritParams tree_race_probabilities
#' @return n x n matrix with zero row sums
#' @export
tree_race_jacobian <- function(mu, cluster, loading, D, parent, strength,
                               points = 257, qa = 9) {
  mu <- as.numeric(mu)
  m <- -mu
  sd <- sqrt(.as_idio(D, length(mu), positive = TRUE))
  v <- .scalar_loading(loading)
  lam <- as.numeric(strength)
  n <- length(m)
  inv <- .cluster_index(cluster)
  nC <- max(inv)
  parent <- .validate_tree(parent, lam, nC)        # 0 = root (#517)
  nT <- length(parent)
  # a common shock cancels: without it a bare root takes the exact path
  lam <- .without_common_shock(parent, lam, nC)          # #489
  strength <- lam
  if (nT > nC && any(lam[(nC + 1):nT] != 0)) {
    # with any ancestor effect the cross-cluster Gram product is not the
    # derivative (two leaves under a common root: forward invariant in
    # the root strength, J[1,2] 0.2467 -> 0.1792 at strength 0.8); use
    # the central difference of the exact forward, as python (#350)
    h <- 1e-5 * sqrt(stats::median(sd^2 + v^2))
    J <- matrix(0, n, n)
    for (j in seq_len(n)) {
      e <- numeric(n); e[j] <- h
      J[, j] <- (tree_race_probabilities(mu + e, cluster, loading, D, parent,
                                         strength, points = points, qa = qa) -
                 tree_race_probabilities(mu - e, cluster, loading, D, parent,
                                         strength, points = points, qa = qa)) /
        (2 * h)
    }
    return(J)
  }
  ord <- order(inv)
  mu_o <- m[ord]; sd_o <- sd[ord]; v_o <- v[ord]; c_o <- inv[ord]
  h1 <- .hermite1(qa)
  an <- h1$nodes; aw <- h1$weights
  depth_shift <- numeric(nT)
  # traverse by TREE depth, not by accumulated |strength|: zero strengths
  # (from_linkage's floored merges) tie the |lambda| path sums and a tied
  # sort visits children before their parents, reading cavities still at
  # their initial value.
  depth_hops <- integer(nT)
  for (t in seq_len(nT)) {
    s_ <- 0; d_ <- 0L; u <- t
    while (parent[u] > 0) {
      s_ <- s_ + abs(lam[parent[u]]); d_ <- d_ + 1L; u <- parent[u]
    }
    depth_shift[t] <- s_
    depth_hops[t] <- d_
  }
  path_var <- numeric(nC)
  for (cc in seq_len(nC)) {
    s_ <- 0; u <- cc
    while (parent[u] > 0) { s_ <- s_ + lam[parent[u]]^2; u <- parent[u] }
    path_var[cc] <- s_
  }
  amp <- (abs(v_o) + depth_shift[c_o]) * max(abs(an))
  lh <- .window_nodes(mu_o, sd_o, amp)
  x <- seq(lh[1], lh[2], length.out = points)
  dx <- x[2] - x[1]
  xm <- matrix(x, n, points, byrow = TRUE)
  S <- array(0, c(nC, qa, points))
  logF <- array(0, c(n, qa, points))
  pdf <- array(0, c(n, qa, points))
  for (q in seq_len(qa)) {
    z <- (xm - mu_o - v_o * an[q]) / sd_o
    lf <- log(pmax(stats::pnorm(z), .TINY))
    logF[, q, ] <- lf
    pdf[, q, ] <- exp(-0.5 * z * z) / (sd_o * sqrt(2 * pi))
    S[, q, ] <- rowsum(lf, c_o)
  }
  G <- matrix(0, nT, points)
  for (q in seq_len(qa)) {
    G[1:nC, ] <- G[1:nC, ] + aw[q] * exp(pmin(matrix(S[, q, ], nC, points), 0))
  }
  children <- vector("list", nT)
  root <- 0L
  for (t in seq_len(nT)) {
    if (parent[t] > 0) children[[parent[t]]] <- c(children[[parent[t]]], t)
    else root <- t
  }
  shift_eval <- function(g, delta) stats::approx(x - delta, g, xout = x,
                                                 rule = 2,
                                                 ties = "ordered")$y
  up <- setdiff(seq_len(nT), seq_len(nC))
  up <- up[order(-depth_hops[up])]
  for (t in up) {
    acc <- numeric(points)
    for (q in seq_len(qa)) {
      prod <- rep(1, points)
      for (cc in children[[t]]) prod <- prod * shift_eval(G[cc, ], lam[t] * an[q])
      acc <- acc + aw[q] * prod
    }
    G[t, ] <- pmax(acc, 0)
  }
  R <- matrix(1, nT, points)
  for (t in order(depth_hops)) {
    pa <- parent[t]
    if (pa == 0) next
    sm <- numeric(points)
    for (q in seq_len(qa)) sm <- sm + aw[q] * shift_eval(R[pa, ], -lam[pa] * an[q])
    prod <- rep(1, points)
    for (s_ in children[[pa]]) if (s_ != t) prod <- prod * G[s_, ]
    R[t, ] <- pmax(sm * prod, 0)
  }
  hmat <- matrix(0, n, points)
  for (q in seq_len(qa)) {
    hmat <- hmat + aw[q] * pdf[, q, ] * exp(pmin(S[c_o, q, ] - logF[, q, ], 0))
  }
  Gr <- pmax(G[root, ], .TINY)
  U <- hmat * R[c_o, , drop = FALSE] /
    matrix(sqrt(Gr), n, points, byrow = TRUE) * sqrt(dx)
  J <- -(U %*% t(U))
  for (ci in seq_len(nC)) {
    idx <- which(c_o == ci)
    k <- length(idx)
    if (k == 1) next
    term <- matrix(0, k, k)
    for (q in seq_len(qa)) {
      Rcq <- R[ci, ]
      Fk <- matrix(logF[idx, q, ], k, points)
      Pk <- matrix(pdf[idx, q, ], k, points)
      for (ii in seq_len(k)) {
        base <- matrix(S[ci, q, ] - Fk[ii, ], k, points, byrow = TRUE)
        lo2 <- exp(pmin(base - Fk, 0))
        wrow <- Pk[ii, ] * Rcq
        term[ii, ] <- term[ii, ] + aw[q] *
          rowSums(Pk * lo2 * matrix(wrow, k, points, byrow = TRUE)) * dx
      }
    }
    J[idx, idx] <- -term
  }
  diag(J) <- 0
  diag(J) <- -rowSums(J)
  Jf <- matrix(0, n, n)
  Jf[ord, ord] <- J
  -Jf
}
