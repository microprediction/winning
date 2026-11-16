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

# ||P (C - V V' - diag D) P||_F^2, the identified objective
.projected_sq <- function(C, V, D) {
  n <- nrow(C)
  P <- diag(n) - 1 / n
  sum((P %*% (C - V %*% t(V) - diag(D, n)) %*% P)^2)
}

# Exact argmin_{d >= 0} d'Gd/2 - c'd for G = P o P = a I + b 11'
# (a = 1 - 2/n, b = 1/n^2), by water-filling: port of python's
# _nnls_centered_gram.
.nnls_centered_gram <- function(c0, n, n_pass = 100L) {
  a <- 1 - 2 / n
  b <- 1 / (n * n)
  s <- max(sum(c0), 0) / (a + b * n)
  if (n <= 2L) return(rep(max(s, 0) / n, n))
  for (it in seq_len(n_pass)) {
    mask <- c0 > b * s
    s_new <- sum(c0[mask]) / (a + b * sum(mask))
    if (abs(s_new - s) <= 1e-15 * max(1, abs(s))) { s <- s_new; break }
    s <- s_new
  }
  pmax((c0 - b * s) / a, 0)
}

.factor_model_projected <- function(C, k, n_outer = 60L, D0 = NULL) {
  n <- nrow(C)
  P <- diag(n) - 1 / n
  B <- qr.Q(qr(P))[, seq_len(n - 1), drop = FALSE]
  S <- t(B) %*% C %*% B
  # D0 exists for multistart: the alternation is nonconvex, and its
  # default start can stall at a nonglobal stationary point (#427)
  D <- if (is.null(D0)) rep(0.5 * mean(diag(C)), n) else as.numeric(D0)
  G <- P * P
  W <- matrix(0, n - 1, k)
  best <- list(W = W, D = D, obj = Inf)
  hit_cap <- TRUE
  for (it in seq_len(n_outer)) {
    R <- S - t(B * D) %*% B
    e <- eigen(R, symmetric = TRUE)
    lam <- pmax(e$values[seq_len(k)], 0)
    W <- e$vectors[, seq_len(k), drop = FALSE] * rep(sqrt(lam), each = n - 1)
    A <- S - W %*% t(W)
    c0 <- rowSums((B %*% A) * B)
    D_new <- .nnls_active_set(G, c0)
    # descent enforced, as python: a sweep that does not improve the
    # objective is discarded and the alternation stops at the best iterate
    obj <- .projected_sq(C, B %*% W, D_new)
    if (obj > best$obj * (1 + 1e-12)) {
      W <- best$W; D <- best$D; hit_cap <- FALSE; break
    }
    best <- list(W = W, D = D_new, obj = obj)
    if (max(abs(D_new - D)) < 1e-12) { D <- D_new; hit_cap <- FALSE; break }
    D <- D_new
  }
  # hit_cap: the alternation ran out of sweeps rather than stopping on
  # its own rule (#1051), so fit_covariance may finish it
  list(V = B %*% W, D = .projected_floor(C, D), hit_cap = hit_cap)
}

# RELATIVE floor, as python's factor_model_projected: an absolute 1e-8
# inflates a tiny variance (C = 1e-12 I) by orders of magnitude
.projected_floor <- function(C, D) {
  dc <- diag(C)
  floor <- 1e-8 * pmax(dc, 1e-6 * max(mean(dc), 1e-300))
  pmax(D, floor)
}

# Finish a projected factor fit the alternation left unconverged: port of
# python's core._polish_projected (#1018, #1051). The alternation is
# block coordinate descent and converges linearly; on an EXACT
# rank-1-plus-diagonal C at n = 5 its 60 sweeps stop with projected
# residuals up to 1.5e-3. This is Gauss-Newton on the variable-projection
# objective f(D) = min_V ||P (C - V V' - diag D) P||_F^2: with N spanning
# the eigenvectors of S - B' diag(D) B the top-k PSD fit does not keep
# (eigenvalues l) and X = B N, the step solves (Pi o Pi) delta =
# (X o X) l, Pi = X X' -- the alternation's own D-step with P o P
# replaced by the projector that also removes the factor directions.
# Quadratic at zero residual. D stays nonnegative (bound variables with
# an outward step are frozen, then the step is clipped), a step is taken
# only if it lowers f (halving otherwise), and at most max_eval
# eigendecompositions are spent. Returns list(V, D, f) with D before the
# variance floor; the caller keeps it only if it lowers the objective.
.polish_projected <- function(C, k, D, max_eval = 60L) {
  n <- nrow(C)
  # orthonormal Helmert basis of the mean-zero subspace, as python
  B <- matrix(0, n, n - 1L)
  for (j in seq_len(n - 1L)) {
    B[seq_len(j), j] <- 1
    B[j + 1L, j] <- -j
    B[, j] <- B[, j] / sqrt(j * (j + 1))
  }
  S <- t(B) %*% C %*% B
  budget <- max_eval
  evaluate <- function(D) {
    budget <<- budget - 1L
    e <- eigen(S - t(B * D) %*% B, symmetric = TRUE)
    lam <- e$values
    U <- e$vectors
    keep <- rep(FALSE, length(lam))
    kk <- min(k, length(lam))
    if (kk > 0) keep[seq_len(kk)] <- lam[seq_len(kk)] > 0
    lc <- lam[!keep]
    V <- B %*% (U[, seq_len(kk), drop = FALSE] *
                  rep(sqrt(pmax(lam[seq_len(kk)], 0)), each = n - 1L))
    if (ncol(V) < k) V <- cbind(V, matrix(0, n, k - ncol(V)))
    list(f = sum(lc * lc), lc = lc, X = B %*% U[, !keep, drop = FALSE],
         V = V)
  }
  D <- pmax(as.numeric(D), 0)
  cur <- evaluate(D)
  while (budget > 0 && cur$f > 0) {
    H <- (cur$X %*% t(cur$X))^2
    h <- as.numeric((cur$X^2) %*% cur$lc)
    free <- rep(TRUE, n)
    for (pass in 1:3) {
      d <- rep(0, n)
      if (any(free)) {
        Hf <- H[free, free, drop = FALSE]
        # minimum-norm least squares, as numpy's lstsq
        sv <- svd(Hf)
        tol <- max(dim(Hf)) * .Machine$double.eps * max(sv$d)
        dinv <- ifelse(sv$d > tol, 1 / sv$d, 0)
        d[free] <- sv$v %*% (dinv * (t(sv$u) %*% h[free]))
      }
      blocked <- free & (D <= 0) & (d < 0)
      if (!any(blocked)) break
      free <- free & !blocked
    }
    t_ <- 1
    accepted <- FALSE
    while (budget > 0) {
      Dn <- pmax(D + t_ * d, 0)
      nxt <- evaluate(Dn)
      if (nxt$f < cur$f) { accepted <- TRUE; break }
      t_ <- t_ * 0.5
    }
    if (!accepted) break
    gain <- (cur$f - nxt$f) / cur$f
    D <- Dn
    cur <- nxt
    if (gain < 1e-10) break
  }
  list(V = cur$V, D = D, f = cur$f)
}

# Warn when the rank cuts through a degenerate eigenvalue of the
# centred covariance (port of python's _warn_if_rank_splits_a_tie). Any
# k directions from a tied group fit equally well and imply DIFFERENT
# races: six equal blocks at k = 3 moved a pair price 3.6 points under a
# relabelling, with no warning in this port (#383). The 5% filter is
# python's: every centred spectrum ends in a tied bulk.
.warn_if_rank_splits_a_tie <- function(C, k, tol = 1e-6) {
  n <- nrow(C)
  if (k >= n - 1L || k < 1L) return(invisible(NULL))
  P <- diag(n) - 1 / n
  lam <- eigen(P %*% C %*% P, symmetric = TRUE, only.values = TRUE)$values
  probe <- min(2L * k + 4L, n - 1L)
  lam <- lam[seq_len(probe)]
  if (length(lam) <= k || lam[1] <= 0) return(invisible(NULL))
  scale <- lam[1]
  if (lam[k] - lam[k + 1L] > tol * scale) return(invisible(NULL))
  if (lam[k] < 0.05 * scale) return(invisible(NULL))
  lo <- k                       # 1-based index of the first tied value
  while (lo > 1L && lam[lo - 1L] - lam[lo] <= tol * scale) lo <- lo - 1L
  hi <- k + 1L
  while (hi + 1L <= length(lam) && lam[hi] - lam[hi + 1L] <= tol * scale)
    hi <- hi + 1L
  tied <- hi - lo + 1L
  warning(sprintf(paste0(
    "rank %d splits a tied eigenvalue of the centered covariance: ",
    "eigenvalues %d to %d are equal to within %g relative%s, so the fit ",
    "picks an arbitrary %d of %d equally optimal directions. Every choice ",
    "gives the same residual and a DIFFERENT race. This is not ",
    "approximation error and no optimizer fixes it; pass k=%d or more to ",
    "take the whole tied group."),
    k, lo, hi, tol,
    if (hi >= probe) " (and the tie may extend past the probe)" else "",
    k - lo + 1L, tied, hi), call. = FALSE)
  invisible(NULL)
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
  # halve BEFORE adding, or C + t(C) overflows to Inf for finite
  # entries near the double ceiling (#279)
  C <- 0.5 * C + 0.5 * t(C)
  # positive semidefiniteness in CORRELATION units (#666, as python
  # core._validate_covariance): a tolerance of 1e-8 times the mean raw
  # variance let an unrelated variance of 1e12 hide a pair with
  # correlation 2, and the absolute 1e-300 floor passed [[0, c], [c, 0]]
  # at c = 1e-309. No negative variance; a zero variance has a zero row;
  # the standardised positive-variance block has min eigenvalue >= -1e-8.
  psd_msg <- paste("cov= is not positive semidefinite; this is not a",
                   "covariance matrix. Project to the PSD cone first if",
                   "it came from noisy estimation.")
  d <- diag(C)
  if (any(d < 0))
    stop(sprintf("%s (variance [%d] = %.2e < 0)", psd_msg,
                 which(d < 0)[1], d[which(d < 0)[1]]), call. = FALSE)
  pos <- d > 0
  if (!all(pos)) {
    for (i in which(!pos))
      if (any(C[i, -i] != 0))
        stop(sprintf("%s (variance [%d] is zero but its row is not)",
                     psd_msg, i), call. = FALSE)
  }
  if (any(pos)) {
    sd_ <- sqrt(d[pos])
    R <- C[pos, pos, drop = FALSE] / outer(sd_, sd_)
    lam_min <- min(eigen(R, symmetric = TRUE, only.values = TRUE)$values)
    if (lam_min < -1e-8)
      stop(sprintf("%s (min eigenvalue of the correlation matrix %.2e)",
                   psd_msg, lam_min), call. = FALSE)
  }
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
  # a common shift cancels from every contrast: fit the representative
  # without it, so C and C + a 11' are fitted identically (#465, python
  # core._without_common_shift)
  C <- .without_common_shift(C)
  s <- sqrt(pmax(diag(C), 1e-12))
  corr <- C / outer(s, s)
  kk <- min(k, n - 1L)
  fit <- .factor_model_projected(C, kk)
  .warn_if_rank_splits_a_tie(C, kk)          # #383, python parity
  # Multistart rescue, ported from python's fit_covariance: the default
  # start stalled on an EXACT rank-2-plus-diagonal 12x12 matrix at
  # projected objective 1.04 (contrast residual 0.159, a pair probability
  # 1.79 points off) where a diagonal-heavy start reaches 1e-14 (#427).
  # Tried only when the default objective is materially above zero.
  Pc <- diag(n) - 1 / n
  scale2 <- sum((Pc %*% C %*% Pc)^2)
  obj0 <- .projected_sq(C, fit$V, fit$D)
  if (obj0 > 1e-12 * max(scale2, 1e-300)) {
    lam_top <- sum(sort(eigen(C, symmetric = TRUE, only.values = TRUE)$values,
                        decreasing = TRUE)[seq_len(kk)]) / n
    for (D_start in list(0.9 * diag(C), pmax(diag(C) - lam_top, 1e-3))) {
      f2 <- .factor_model_projected(C, kk, D0 = D_start)
      o2 <- .projected_sq(C, f2$V, f2$D)
      if (o2 < obj0) { fit <- f2; obj0 <- o2 }
    }
  }
  # #1051, python parity (#1018): the alternation converges linearly,
  # and on an EXACT rank-1-plus-diagonal C at n = 5 every start ran out of
  # sweeps with projected residuals up to 1.5e-3. A kept start that hit
  # the sweep cap is finished by Gauss-Newton on the same objective; the
  # polished start is a candidate only if it lowers that objective by more
  # than a relative 1e-10, and replaces the fit only if the FINAL
  # choice-relevant residual is strictly lower in its largest entry and
  # no higher in Frobenius norm. A start that stopped on its own rule is
  # untouched.
  polished <- NULL
  if (isTRUE(fit$hit_cap) && n >= 3L && n <= 800L) {
    pz <- .polish_projected(C, kk, fit$D)
    D1 <- .projected_floor(C, pz$D)
    if (.projected_sq(C, pz$V, D1) <
        .projected_sq(C, fit$V, fit$D) * (1 - 1e-10))
      polished <- list(V = pz$V, D = D1)
  }
  if (is.null(blocks)) blocks <- max(2L, min(n %/% 5L, 20L))
  P <- diag(n) - 1 / n
  cluster <- rep(0L, n)
  use_blocks <- n >= 3L && blocks >= 2L
  if (use_blocks) {
    dm <- stats::as.dist(sqrt(pmin(pmax(0.5 * (1 - corr), 0), 1)))
    cluster <- stats::cutree(stats::hclust(dm, method = "average"),
                             k = blocks) - 1L
  }
  uc <- unique(cluster)
  m_eff <- min(m, n)
  # The floor is RELATIVE to each runner's own variance, as python's
  # _close. The absolute 1e-3 * mean(diag(C)) floor destroyed near-singular
  # contrasts: on diag(1e-8, 1e-8, 1) Var(X1 - X2) >= D1 + D2 >= 6.7e-4
  # against the true 2e-8, a near-certain head-to-head priced as a coin
  # flip (#407). And it is the LOWER-BOUNDED least squares (d = floor + x,
  # x >= 0), not an unconstrained solve clipped afterwards, which is not
  # the constrained minimiser because P o P couples the coordinates. The
  # degradation test still keys on the floor close_fit uses (#189).
  #
  # The MULTIPLIER stays this port's 1e-3, not python's 1e-6. What #407
  # needs is the per-runner scale and the bounded solve; on the #407
  # fixture the floor is then 1e-11 against D = 1e-8, untouched. Python's
  # 1e-6 adds nothing a caller can use -- a runner whose idiosyncratic
  # share is under 0.1% of its own variance is the near-Dirac regime, out
  # of scope -- and it costs: a degenerate full-rank fit (k + m + blocks
  # >= n) puts D on the floor, so 1e-6 makes every conditional race
  # 32x sharper than 1e-3 did (sharpness 1166 vs 37), the refinement asks
  # for 40,000 lattice points against the 8193 cap, one forward costs 11x
  # (3.95 s vs 0.35 s at n = 8) and its slopes are under-resolved noise,
  # so an inverse on such a fit ran 60 sweeps for 4 minutes without
  # converging where it had taken 4 s.
  d_clamp <- 1e-3 * pmax(diag(C), 1e-6 * mean(diag(C)))
  close_fit <- function(Vc) {
    rhs <- diag(P %*% (C - Vc %*% t(Vc)) %*% P)
    # P o P = a I + b 11' with a = 1 - 2/n, b = 1/n^2. At n = 2, a = 0
    # and the matrix is rank one (#181): one contrast, the total is
    # spread evenly.
    a <- 1 - 2 / n
    b <- 1 / (n * n)
    if (n <= 2L) {
      Dc <- pmax(rep(max(sum(rhs), 0) / (a + n * b) / n, n), d_clamp)
    } else {
      Dc <- d_clamp + .nnls_centered_gram(rhs - (a * d_clamp +
                                                   b * sum(d_clamp)), n)
    }
    Rm <- P %*% (C - Vc %*% t(Vc) - diag(Dc)) %*% P
    list(D = Dc, res = max(abs(Rm)))
  }
  # second arm: pure eigen fit at the same total rank (greedy
  # factor+blocks allocation is the wrong shape for globally smooth
  # covariance); smaller choice-relevant residual wins, pipeline on ties
  # clamp to n: the greedy allocation (k global + m eigendirections +
  # one per block) can ask for more columns than there are eigenvectors
  # -- k=3, m=5 and 2 blocks is 10 at n=8 -- and the python reference's
  # _top_eigen truncates silently where seq_len() here ran off the end
  # of eC$vectors ("subscript out of bounds", an n=8 exact-rank-3
  # correlation, the #118 fixture)
  eC <- NULL
  stages <- function(V, D0) {
    R <- P %*% (C - V %*% t(V) - diag(D0)) %*% P
    v <- rep(0, n)
    if (use_blocks) {
      for (cc in uc) {
        idx <- which(cluster == cc)
        if (length(idx) < 2L) next
        Rb <- R[idx, idx, drop = FALSE]
        diag(Rb) <- 0
        e <- eigen(Rb, symmetric = TRUE)
        if (e$values[1] > 0)
          v[idx] <- e$vectors[, 1] * sqrt(e$values[1])
      }
    }
    BD <- matrix(0, n, length(uc))
    for (j in seq_along(uc)) {
      idx <- which(cluster == uc[j])
      BD[idx, j] <- v[idx]
    }
    E <- R - BD %*% t(BD)
    diag(E) <- 0
    e <- eigen(E, symmetric = TRUE)
    Vres <- e$vectors[, seq_len(m_eff), drop = FALSE] *
      rep(sqrt(pmax(e$values[seq_len(m_eff)], 0)), each = n)
    Vall <- cbind(V, Vres, BD)
    keep <- colSums(Vall ^ 2) > 1e-10 * sum(diag(C)) / n
    if (!any(keep)) keep[1] <- TRUE
    Vall <- Vall[, keep, drop = FALSE]
    a1 <- close_fit(Vall)
    rank <- min(ncol(Vall), n)
    if (is.null(eC)) eC <<- eigen(C, symmetric = TRUE)
    Veig <- eC$vectors[, seq_len(rank), drop = FALSE] *
      rep(sqrt(pmax(eC$values[seq_len(rank)], 0)), each = n)
    a2 <- close_fit(Veig)
    if (a2$res < a1$res) list(V = Veig, D = a2$D, res = a2$res)
    else list(V = Vall, D = a1$D, res = a1$res)
  }
  fro <- function(st) sqrt(sum((P %*% (C - st$V %*% t(st$V) -
                                         diag(st$D, n)) %*% P)^2))
  st <- stages(fit$V, fit$D)
  if (!is.null(polished)) {
    st1 <- stages(polished$V, polished$D)
    if (st1$res < st$res && fro(st1) <= fro(st)) st <- st1
  }
  Vall <- st$V
  D <- st$D
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
  residual <- st$res
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


# A race-equivalent representative of C with its removable common shift
# taken out (#465). When every off-diagonal is positive, m = min C_ij is
# removed elementwise (exact near the shift); if that over-removes, the
# largest removable part t = -min_{1'x=1} x'(C - m)x is restored from one
# KKT solve. A field with any non-positive covariance is unchanged.
.without_common_shift <- function(C) {
  n <- nrow(C)
  if (n < 2L) return(C)
  m <- min(C[row(C) != col(C)])
  if (!(m > 0)) return(C)
  B <- C - m
  K <- rbind(cbind(B, 1), c(rep(1, n), 0))
  rhs <- c(rep(0, n), 1)
  sol <- tryCatch(solve(K, rhs),
                  error = function(e) qr.solve(K, rhs, tol = 1e-12))
  x <- sol[seq_len(n)]
  v <- sum(x * (B %*% x))
  if (is.finite(v) && v < 0) B <- B - v
  B
}
