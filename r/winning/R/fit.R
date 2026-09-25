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
  # The same contract python states, in the same order and with the
  # same messages. R had NONE of it: a negative variance was clamped to
  # the floor, NA and Inf propagated into the fit, and a non-square
  # matrix was read by nrow() alone. That last one is why this block
  # sits ABOVE the n == 1 return rather than below it -- a 1 x 2 matrix
  # has nrow 1, so the one-runner branch would otherwise answer for it
  # from C[1, 1] and drop the second column (#277).
  if (nrow(C) != ncol(C))
    stop(sprintf("cov= must be square; got %d x %d", nrow(C), ncol(C)))
  n <- nrow(C)
  if (!all(is.finite(C))) stop("cov= contains NaN or inf")
  asym <- max(abs(C - t(C)))
  if (asym > 1e-8 * max(max(abs(C)), 1e-300))
    stop(sprintf(paste("cov= is not symmetric (max asymmetry %.2e); pass",
                       "(C + t(C))/2 if the asymmetry is numerical noise"),
                 asym))
  C <- 0.5 * (C + t(C))
  lam_min <- min(eigen(C, symmetric = TRUE, only.values = TRUE)$values)
  if (lam_min < -1e-8 * max(sum(diag(C)) / n, 1e-300))
    stop(sprintf(paste("cov= is not positive semidefinite (min eigenvalue",
                       "%.2e); this is not a covariance matrix. Project to",
                       "the PSD cone first if it came from noisy",
                       "estimation."),
                 lam_min))
  if (n == 1L) {
    # A one-runner field has no covariance STRUCTURE: nothing for a
    # factor to correlate, the whole variance idiosyncratic, and the
    # race is 1 whatever it is. .factor_model_projected is asked for
    # min(k, n - 1) = 0 factors and hands back a 0 x 0 matrix, which
    # then fails to multiply; python divided by zero at the same point
    # (#273). Report it as what it is: an exact fit of rank zero.
    return(list(V = matrix(0, 1L, 1L), D = pmax(C[1L, 1L], 1e-12),
                F = matrix(0, 1L, 1L), W = 1,
                rank = 0L, n = 1L, bound = 0, clamp = 0,
                residual = 0, contrast_residual = 0, degraded = FALSE))
  }
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
  # the clamp is reported, not re-derived: the degradation test used to
  # compare D against 1e-6 * diag while close_fit clamped at 1e-3 * mean,
  # three orders of magnitude apart, so a clamp-bound fit counted zero
  # bound entries and was priced instead of routed (#189)
  d_clamp <- 1e-3 * mean(diag(C))
  close_fit <- function(Vc) {
    rhs <- diag(P %*% (C - Vc %*% t(Vc)) %*% P)
    # The closing solve is against P o P = a I + b 11' with a = 1 - 2/n,
    # b = 1/n^2. At n = 2 that a is exactly ZERO, so the matrix is rank
    # one and solve() threw for EVERY 2x2 covariance, public cov= calls
    # included (#181). The python reference has had the two-runner branch
    # all along: one contrast, so the total is spread evenly.
    if (n <= 2L) {
      a <- 1 - 2 / n
      b <- 1 / (n * n)
      Dc <- pmax(rep(max(sum(rhs), 0) / (a + n * b) / n, n), d_clamp)
    } else {
      Dc <- pmax(solve(P * P, rhs), d_clamp)
    }
    Rm <- P %*% (C - Vc %*% t(Vc) - diag(Dc)) %*% P
    list(D = Dc, res = max(abs(Rm)))
  }
  a1 <- close_fit(Vall)
  # second arm: pure eigen fit at the same total rank (greedy
  # factor+blocks allocation is the wrong shape for globally smooth
  # covariance); smaller choice-relevant residual wins, pipeline on ties
  # clamp to n: the greedy allocation (k global + m eigendirections +
  # one per block) can ask for more columns than there are eigenvectors
  # -- k=3, m=5 and 2 blocks is 10 at n=8 -- and the python reference's
  # _top_eigen truncates silently where seq_len() here ran off the end
  # of eC$vectors ("subscript out of bounds", an n=8 exact-rank-3
  # correlation, the #118 fixture)
  rank <- min(ncol(Vall), n)
  eC <- eigen(C, symmetric = TRUE)
  Veig <- eC$vectors[, seq_len(rank), drop = FALSE] *
    rep(sqrt(pmax(eC$values[seq_len(rank)], 0)), each = n)
  a2 <- close_fit(Veig)
  if (a2$res < a1$res) { Vall <- Veig; D <- a2$D } else D <- a1$D
  hw <- .halton_normal_nodes(ncol(Vall), nodes)
  # The two failure classes the python reference keys on, reported so a
  # caller can see them: the fit degenerates to full rank or drives D onto
  # its floor (the conditional race is then a near-step the factor nodes
  # cannot resolve, and the residual checks are silent about it), or it
  # reproduces cov badly. Python routes either case to GHK; this package
  # has no GHK, so it prices the fit and says so -- see the warning in
  # race_probabilities(). A gap that does not announce itself is the one
  # failure mode that looks like an answer.
  bound <- sum(D <= 2 * d_clamp)
  # the pairwise-contrast residual python also keys on: a near-singular
  # difference variance the fit did not hold makes head-to-head
  # probabilities badly wrong while the global residual stays small
  Sig <- Vall %*% t(Vall) + diag(D)
  cv_fit <- outer(diag(Sig), diag(Sig), "+") - 2 * Sig
  cv_true <- outer(diag(C), diag(C), "+") - 2 * C
  scale_cv <- pmax(cv_true, 1e-12 * mean(diag(C)))
  contrast_res <- max(abs(cv_fit - cv_true) / scale_cv)
  residual <- min(a1$res, a2$res)
  list(V = Vall, D = D, F = hw$F, W = hw$W,
       rank = ncol(Vall), n = n,
       bound = bound,
       clamp = d_clamp,
       residual = residual,
       contrast_residual = contrast_res,
       degraded = (bound > 0 || ncol(Vall) >= n ||
                   contrast_res > 0.05 ||
                   residual > 0.05 * mean(diag(C))))
}
