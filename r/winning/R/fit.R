# Dense-covariance intake: the python package's fit_covariance pipeline,
# algorithm-faithful (certified quotient factor fit, blocks and residual
# promotion on the PROJECTED residual, closing (P.P) d = diag(P R P)
# solve). Nodes come from the port's Halton rule, so cross-language
# parity is at the fitted-model-covariance level (V V' + diag(D)), not
# the node level.

.nnls_active_set <- function(G, c0, max_iter = 200L) {
  # min 1/2 d' G d - c0' d  s.t. d >= 0, G SPD (Lawson-Hanson on the
  # normal equations; n-dimensional, G = P.P is tiny)
  n <- length(c0)
  d <- rep(0, n)
  passive <- rep(FALSE, n)
  for (it in seq_len(max_iter)) {
    w <- c0 - G %*% d
    w[passive] <- -Inf
    j <- which.max(w)
    if (w[j] <= 1e-12) break
    passive[j] <- TRUE
    repeat {
      s <- rep(0, n)
      idx <- which(passive)
      s[idx] <- solve(G[idx, idx, drop = FALSE], c0[idx])
      if (all(s[idx] > 0)) { d <- s; break }
      neg <- idx[s[idx] <= 0]
      alpha <- min(d[neg] / (d[neg] - s[neg]))
      d <- d + alpha * (s - d)
      passive[which(passive)[abs(d[which(passive)]) < 1e-14]] <- FALSE
      d[!passive] <- 0
    }
  }
  d
}

.factor_model_projected <- function(C, k, n_outer = 60L) {
  n <- nrow(C)
  P <- diag(n) - 1 / n
  B <- qr.Q(qr(P))[, seq_len(n - 1), drop = FALSE]
  S <- t(B) %*% C %*% B
  D <- rep(0.5 * mean(diag(C)), n)
  G <- P * P
  W <- matrix(0, n - 1, k)
  for (it in seq_len(n_outer)) {
    R <- S - t(B * D) %*% B
    e <- eigen(R, symmetric = TRUE)
    lam <- pmax(e$values[seq_len(k)], 0)
    W <- e$vectors[, seq_len(k), drop = FALSE] * rep(sqrt(lam), each = n - 1)
    A <- S - W %*% t(W)
    c0 <- rowSums((B %*% A) * B)
    D_new <- .nnls_active_set(G, c0)
    if (max(abs(D_new - D)) < 1e-12) { D <- D_new; break }
    D <- D_new
  }
  list(V = B %*% W, D = pmax(D, 1e-8))
}

#' Fit a dense covariance to the race grammar
#'
#' One-call intake for \code{race_probabilities(mu, cov = )}: k global
#' factors by the quotient-space fit (only \code{P Sigma P} is
#' choice-relevant), average-linkage blocks and residual promotion on
#' the projected residual, and a closing diagonal solve of the
#' identified problem's normal equations.
#'
#' @param C covariance or correlation matrix
#' @param k global factor rank
#' @param m residual eigencolumns to promote
#' @param blocks cluster count for the block stage (default
#'   \code{max(2, min(n/5, 20))})
#' @param nodes count of Halton nodes for the returned rule
#' @return list with V, D, F, W ready for \code{race_probabilities}
#' @export
fit_covariance <- function(C, k = 3L, m = 5L, blocks = NULL,
                           nodes = 2048L) {
  C <- as.matrix(C)
  n <- nrow(C)
  s <- sqrt(pmax(diag(C), 1e-12))
  corr <- C / outer(s, s)
  fit <- .factor_model_projected(C, min(k, n - 1L))
  V <- fit$V
  if (is.null(blocks)) blocks <- max(2L, min(n %/% 5L, 20L))
  P <- diag(n) - 1 / n
  R <- P %*% (C - V %*% t(V) - diag(fit$D)) %*% P
  cluster <- rep(0L, n)
  v <- rep(0, n)
  if (n >= 3L && blocks >= 2L) {
    dm <- stats::as.dist(sqrt(pmin(pmax(0.5 * (1 - corr), 0), 1)))
    cluster <- stats::cutree(stats::hclust(dm, method = "average"),
                             k = blocks) - 1L
    for (cc in unique(cluster)) {
      idx <- which(cluster == cc)
      if (length(idx) < 2L) next
      Rb <- R[idx, idx, drop = FALSE]
      diag(Rb) <- 0
      e <- eigen(Rb, symmetric = TRUE)
      if (e$values[1] > 0)
        v[idx] <- e$vectors[, 1] * sqrt(e$values[1])
    }
  }
  uc <- unique(cluster)
  BD <- matrix(0, n, length(uc))
  for (j in seq_along(uc)) {
    idx <- which(cluster == uc[j])
    BD[idx, j] <- v[idx]
  }
  E <- R - BD %*% t(BD)
  diag(E) <- 0
  e <- eigen(E, symmetric = TRUE)
  m_eff <- min(m, n)
  Vres <- e$vectors[, seq_len(m_eff), drop = FALSE] *
    rep(sqrt(pmax(e$values[seq_len(m_eff)], 0)), each = n)
  Vall <- cbind(V, Vres, BD)
  keep <- colSums(Vall ^ 2) > 1e-10 * sum(diag(C)) / n
  if (!any(keep)) keep[1] <- TRUE
  Vall <- Vall[, keep, drop = FALSE]
  close_fit <- function(Vc) {
    rhs <- diag(P %*% (C - Vc %*% t(Vc)) %*% P)
    Dc <- pmax(solve(P * P, rhs), 1e-3 * mean(diag(C)))
    Rm <- P %*% (C - Vc %*% t(Vc) - diag(Dc)) %*% P
    list(D = Dc, res = max(abs(Rm)))
  }
  a1 <- close_fit(Vall)
  # second arm: pure eigen fit at the same total rank (greedy
  # factor+blocks allocation is the wrong shape for globally smooth
  # covariance); smaller choice-relevant residual wins, pipeline on ties
  rank <- ncol(Vall)
  eC <- eigen(C, symmetric = TRUE)
  Veig <- eC$vectors[, seq_len(rank), drop = FALSE] *
    rep(sqrt(pmax(eC$values[seq_len(rank)], 0)), each = n)
  a2 <- close_fit(Veig)
  if (a2$res < a1$res) { Vall <- Veig; D <- a2$D } else D <- a1$D
  hw <- .halton_normal_nodes(ncol(Vall), nodes)
  list(V = Vall, D = D, F = hw$F, W = hw$W)
}
