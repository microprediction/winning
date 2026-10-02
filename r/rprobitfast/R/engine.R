# Exact multinomial probit behind the mlogit interface.
#
# Model: U_ij = x_ij' beta + eps_ij, choice = argmax_j U_ij, with
# eps = V f + sqrt(D) z: rank-2 factor loadings V (reference row zero,
# lower-triangular free block) and unit idiosyncratic D. The differenced
# covariance is then W W' + 11' + I with W the free block, which covers
# every positive-definite differenced covariance up to scale -- the same
# identified space, and the same degree-of-freedom count, as mlogit's
# differenced-Cholesky parameterization (J=4: five covariance
# parameters either way). Scale is fixed by D = 1 rather than by
# L[1,1] = 1, so coefficient VECTORS differ from mlogit's by one common
# scalar while the maximized log-likelihood is directly comparable.
#
# Probability: conditional on the factor f AND the chosen alternative's
# own noise z_k, rivals are independent, so
#   P(k | f, z) = prod_{j != k} Phi( (dmu_kj + (v_k - v_j)'f + z) / 1 )
# and p_k integrates (f, z) over a Gauss-Hermite grid. Every
# observation shares the node set, so the whole likelihood vectorizes
# into a handful of pnorm calls: no simulation, no per-observation
# loop, deterministic to quadrature accuracy.

.gh1 <- function(Q) {
  J <- diag(0, Q)
  off <- sqrt(seq_len(Q - 1))
  J[cbind(seq_len(Q - 1), 2:Q)] <- off
  J[cbind(2:Q, seq_len(Q - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  list(x = e$values, w = e$vectors[1, ]^2)
}

# log of phi(a)/Phi(a), the inverse Mills ratio. pnorm(log.p = TRUE),
# never log(pmax(pnorm(a), 1e-300)): pnorm underflows to exactly 0
# below about -37, so the floor turned every deep tail into the SAME
# number, log(1e-300) = -690.78, the objective went flat and this ratio
# underflowed to a score of exactly zero (#270).
.log_mills <- function(a) dnorm(a, log = TRUE) - pnorm(a, log.p = TRUE)

# Laplace-tilt the chosen alternative's own-noise quadrature. The fixed
# Gauss-Hermite rule samples z where the PRIOR has its mass, and for a
# large observed contrast the integrand's mass is far outside it: at a
# gap of -20 the mode is at z = 10 and a 7-node rule reaches |z| < 3.8.
# With g(z) = -z^2/2 + sum_j logPhi(b_j + z) (unit variances here),
#   g'(z)  = -z + sum_j lam(A_j)
#   g''(z) = -1 - sum_j lam(A_j)(A_j + lam(A_j))
# and lam(a)(a + lam(a)) = -lam'(a) > 0 for every a, so g'' <= -1: g is
# strictly concave, its mode is unique, and Newton converges from
# anywhere. Nodes go at z* + sigma*x and the change of measure is undone
# exactly by log sigma - z^2/2 + x^2/2, so the shift moves the NODES and
# not the integrand.
.nodes3 <- function(Qf = 11, Qz = 11, r = 2) {
  g <- .gh1(Qf); gz <- .gh1(Qz)
  grids <- do.call(expand.grid, c(rep(list(g$x), r), list(gz$x)))
  wg <- do.call(expand.grid, c(rep(list(g$w), r), list(gz$w)))
  W <- apply(as.matrix(wg), 1, prod)
  keep <- W > 1e-10 * max(W)
  list(F = as.matrix(grids)[keep, , drop = FALSE], W = W[keep] / sum(W[keep]))
}

# Equal-weight nodes over (factor^r, own noise) for the sharp regime:
# scrambled Sobol in r + 1 dimensions (R/sobol.R), as the python
# reference's nodes_for_likelihood (2^10 points, seed 0). It was a
# four-prime Halton table, so r = 4 -- the full triangular rank at five
# alternatives -- indexed an NA fifth base and every rank-4 fit died in
# setup, even on the Gauss-Hermite branch (#388).
.sobol_nodes3 <- function(r, m = 10L) {
  .sobol_normal(r + 1L, 2L^m, seed = 0L)
}

# The node-family dispatch statistic, pairwise-safe and gauge-fixed, as
# python's winning.likelihood.sharpness_bound and julia's: only loading
# DIFFERENCES decide a choice, so centre V first, then sqrt(2) max_i
# ||(PV)_i|| bounds max_ij ||V_i - V_j|| / sqrt(2) (unit idiosyncratic
# variance here, so no D divisor). The raw max ||V_i|| was the pre-#213
# statistic: anchored rows 1 and -2.7 have norm <= 2.7 but differ by 3.7,
# so a fit was priced on the 7-point Hermite tensor 1.37 points high where
# python and julia escalate (#419).
.mnp_sharpness <- function(V) {
  V <- as.matrix(V)
  Vc <- sweep(V, 2L, colMeans(V), "-")
  sqrt(2) * max(sqrt(rowSums(Vc^2)))
}

# Lazily built sharp-regime rule: constructed on first use only.
.lazy_sharp_nodes <- function(r, m = 10L) {
  cache <- NULL
  function() {
    if (is.null(cache)) cache <<- .sobol_nodes3(r, m)
    cache
  }
}

# One chosen row per observation, keyed by observation. The wrappers
# used to flatten every truthy row with which() and hand the result to
# the core positionally, so an observation with two choices cancelled
# one with none: the total was still T, the length check passed, and
# observation 1's second choice became observation 2's (#324, #435).
# `ids` are integer codes 1..T and the rows are already sorted by
# (ids, alt); `labels` maps codes back to the caller's ids for messages.
.choices_by_id <- function(ids, alt, chosen, labels = NULL) {
  y <- chosen
  if (is.factor(y)) y <- as.character(y)
  if (is.logical(y)) {
    v <- y
  } else if (is.numeric(y)) {
    if (any(!is.na(y) & !(y %in% c(0, 1))))
      stop(sprintf(paste("the chosen indicator must be logical or 0/1;",
                         "got %s"), format(y[which(!is.na(y) &
                                                   !(y %in% c(0, 1)))[1]])),
           call. = FALSE)
    v <- y == 1
  } else {
    v <- as.logical(y)
    if (any(is.na(v) & !is.na(y)))
      stop("the chosen indicator must be logical or 0/1", call. = FALSE)
  }
  if (anyNA(v)) {
    i <- which(is.na(v))[1]
    stop(sprintf(paste("the chosen indicator is missing for observation",
                       "'%s'; drop that observation deliberately"),
                 if (is.null(labels)) ids[i] else labels[ids[i]]),
         call. = FALSE)
  }
  Tn <- max(ids)
  cnt <- tabulate(ids[v], nbins = Tn)
  bad <- which(cnt != 1L)
  if (length(bad))
    stop(sprintf(paste("each observation must choose exactly one",
                       "alternative; observation '%s' has %d chosen rows",
                       "(%d of %d observations are malformed)"),
                 if (is.null(labels)) bad[1] else labels[bad[1]],
                 cnt[bad[1]], length(bad), Tn), call. = FALSE)
  choice <- integer(Tn)
  choice[ids[v]] <- alt[v]
  choice
}

# Negative log-likelihood AND analytic score, fully vectorized.
# Sharpness-aware: when the loadings make the factor integrand a
# near-step, Gauss-Hermite under-integrates at any order and the
# OPTIMIZER EXPLOITS THE HOLES (observed: a runaway to ||w|| ~ 300 with
# a fake 20-nat likelihood gain that collapses under denser rules), so
# past sharpness 3 the evaluation switches to scrambled Sobol nodes --
# the same family-escalation rule as winning::race_probabilities.
# `nodes_sharp` may be a node list or a zero-argument function returning
# one, so a caller can build the sharp rule only when it is selected.
#
# Score: with a_ijq = dmu_ij + s_jq and posterior node weights
# omega_iq = w_q exp(sum_j log Phi) / p_i, the derivative of log p_i in
# a_ijq is omega_iq * lambda(a_ijq), lambda = phi/Phi (the Mills ratio),
# and beta / loading gradients follow by the chain rule. One extra pass
# over arrays the likelihood already computes; replaces 2*npar numeric
# evaluations per gradient.
.nll_core <- function(theta, Xb, choice, J, r, nodes, nodes_sharp,
                      want_grad = TRUE) {
  X <- Xb[[1]]
  # Every observation must be accounted for. The loop below walks the
  # LEGAL labels and gathers the rows matching each, so a row whose
  # choice is outside 1..J is never visited: logp stays 0 there, which
  # adds zero negative log-likelihood and zero gradient, and the fit
  # silently optimised a SUBSET while reporting it as the whole. NA rows
  # did the same (#194).
  # X is stacked long, J rows per observation, so the observation count
  # is nrow(X) / J -- the same Tn the loop below uses.
  .Tn <- nrow(X) / J
  if (length(choice) != .Tn)
    stop(sprintf(paste0("choice must have one entry per observation: ",
                        "got %d for %d observations"),
                 length(choice), .Tn), call. = FALSE)
  bad <- which(is.na(choice) | choice < 1L | choice > J)
  if (length(bad))
    stop(sprintf(paste0("choice[%d] = %s is not an alternative in 1..%d; ",
                        "%d of %d observations are not. They would be ",
                        "dropped in silence, which raises the ",
                        "log-likelihood because there is less of it"),
                 bad[1], as.character(choice[bad[1]]), J, length(bad),
                 length(choice)), call. = FALSE)
  nb <- ncol(X)
  beta <- theta[seq_len(nb)]
  wfree <- theta[-seq_len(nb)]
  V <- matrix(0, J, r)
  k <- 1L
  fill <- list()
  # seq_len, not `(col + 1L):J`: at J = 2 and the default r = 2L the
  # col = 2 pass evaluates 3:2, which in R is the two-element vector
  # c(3, 2) rather than empty, so the first assignment is V[3, 2] on a
  # 2x2 matrix and every binary-choice fit died out of bounds (#183).
  # nw counts only one free loading there, so wfree[2] was out of range
  # too. seq_len(J - col) is empty exactly when it should be.
  for (col in seq_len(r)) for (row in col + seq_len(J - col)) {
    V[row, col] <- wfree[k]; fill[[k]] <- c(row, col); k <- k + 1L
  }
  Tn <- nrow(X) / J
  mu <- matrix(X %*% beta, nrow = Tn, ncol = J, byrow = TRUE)
  nd <- if (.mnp_sharpness(V) > 3.0) {
    if (is.function(nodes_sharp)) nodes_sharp() else nodes_sharp
  } else nodes
  Fq <- nd$F[, seq_len(r), drop = FALSE]
  zq <- nd$F[, r + 1L]
  Wq <- nd$W
  Q <- length(Wq)
  Vf <- Fq %*% t(V)
  logp <- numeric(Tn)
  gbeta <- numeric(nb)
  gV <- matrix(0, J, r)
  for (k_alt in seq_len(J)) {
    idx <- which(choice == k_alt)
    if (!length(idx)) next
    Ti <- length(idx)
    dmu <- mu[idx, k_alt] - mu[idx, , drop = FALSE]
    rivals <- setdiff(seq_len(J), k_alt)
    logPhi <- vector("list", J)
    A <- vector("list", J)
    acc <- matrix(0, Ti, Q)
    for (j in rivals) {
      A[[j]] <- outer(dmu[, j], Vf[, k_alt] - Vf[, j] + zq, "+")
      logPhi[[j]] <- pnorm(A[[j]], log.p = TRUE)
      acc <- acc + logPhi[[j]]
    }
    m <- apply(acc, 1, max)
    pw <- exp(acc - m) * rep(Wq, each = Ti)
    rs <- rowSums(pw)
    logp[idx] <- m + log(pmax(rs, 1e-300))
    if (!want_grad) next
    omega <- pw / rs                       # (Ti, Q) posterior node weights
    rowsK <- (idx - 1L) * J + k_alt
    for (j in rivals) {
      lam <- exp(dnorm(A[[j]], log = TRUE) - logPhi[[j]])
      wl <- omega * lam                    # (Ti, Q)
      g_i <- rowSums(wl)                   # d logp_i / d dmu_ij
      rowsJ <- (idx - 1L) * J + j
      gbeta <- gbeta + colSums((X[rowsK, , drop = FALSE]
                                - X[rowsJ, , drop = FALSE]) * g_i)
      H <- wl %*% Fq                       # (Ti, r)
      hc <- colSums(H)
      gV[k_alt, ] <- gV[k_alt, ] + hc
      gV[j, ] <- gV[j, ] - hc
    }
  }
  val <- -sum(logp)
  if (!want_grad) return(list(value = val, grad = NULL))
  gw <- vapply(fill, function(rc) gV[rc[1], rc[2]], 0)
  list(value = val, grad = -c(gbeta, gw))
}

# value-only wrapper (kept for tests and diagnostics)
.nll <- function(theta, Xb, choice, J, r, nodes, nodes_sharp) {
  .nll_core(theta, Xb, choice, J, r, nodes, nodes_sharp,
            want_grad = FALSE)$value
}

#' Exact multinomial probit, mlogit-style interface
#'
#' @param formula choice ~ alternative-specific covariates,
#'   e.g. mode ~ price + catch (intercepts added per non-reference
#'   alternative automatically).
#' @param data a dfidx object as used by mlogit (long format).
#' @param r number of factor columns (default 2 covers the full
#'   identified covariance at J = 4).
#' @param Qf,Qz Gauss-Hermite orders for factor and own-noise nodes.
