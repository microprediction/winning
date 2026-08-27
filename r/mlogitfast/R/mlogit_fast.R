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

.nodes3 <- function(Qf = 11, Qz = 11, r = 2) {
  g <- .gh1(Qf); gz <- .gh1(Qz)
  grids <- do.call(expand.grid, c(rep(list(g$x), r), list(gz$x)))
  wg <- do.call(expand.grid, c(rep(list(g$w), r), list(gz$w)))
  W <- apply(as.matrix(wg), 1, prod)
  keep <- W > 1e-10 * max(W)
  list(F = as.matrix(grids)[keep, , drop = FALSE], W = W[keep] / sum(W[keep]))
}

.halton_nodes3 <- function(r, m = 10L) {
  primes <- c(2, 3, 5, 7)[seq_len(r + 1L)]
  n <- 2L^m
  H <- vapply(primes, function(b) {
    idx <- seq_len(n) + 20L
    h <- numeric(n); f <- 1 / b; i <- idx
    while (any(i > 0)) { h <- h + f * (i %% b); i <- i %/% b; f <- f / b }
    h
  }, numeric(n))
  list(F = qnorm(pmin(pmax(H, 1e-12), 1 - 1e-12)),
       W = rep(1 / n, n))
}

# negative log-likelihood, fully vectorized across observations.
# Sharpness-aware: when the loadings make the factor integrand a
# near-step, Gauss-Hermite under-integrates at any order and the
# OPTIMIZER EXPLOITS THE HOLES (observed: a runaway to ||w|| ~ 300 with
# a fake 20-nat likelihood gain that collapses under denser rules), so
# past sharpness 3 the evaluation switches to Halton nodes -- the same
# family-escalation rule as winning::race_probabilities.
.nll <- function(theta, Xb, choice, J, r, nodes, nodes_sharp) {
  nb <- ncol(Xb[[1]])
  beta <- theta[seq_len(nb)]
  wfree <- theta[-seq_len(nb)]
  V <- matrix(0, J, r)
  # reference row stays zero; free block is strictly-lower-triangular
  # by column: col 1 fills rows 2..J, col 2 rows 3..J, and so on
  k <- 1L
  for (col in seq_len(r)) for (row in (col + 1L):J) {
    V[row, col] <- wfree[k]; k <- k + 1L
  }
  Tn <- nrow(Xb[[1]]) / J
  mu <- matrix(Xb[[1]] %*% beta, nrow = Tn, ncol = J, byrow = TRUE)
  sharp <- max(sqrt(rowSums(V^2)))
  nd <- if (sharp > 3.0) nodes_sharp else nodes
  Fq <- nd$F[, seq_len(r), drop = FALSE]
  zq <- nd$F[, r + 1L]
  Wq <- nd$W
  Q <- length(Wq)
  # conditional means shift per alternative: (Q, J)
  Vf <- Fq %*% t(V)
  ll <- 0
  logp <- numeric(Tn)
  # loop over the CHOSEN alternative only (J small); vectorize obs x nodes
  for (k_alt in seq_len(J)) {
    idx <- which(choice == k_alt)
    if (!length(idx)) next
    dmu <- mu[idx, k_alt] - mu[idx, , drop = FALSE]      # (Ti, J)
    acc <- matrix(0, length(idx), Q)
    for (j in seq_len(J)) {
      if (j == k_alt) next
      shift <- Vf[, k_alt] - Vf[, j] + zq                # (Q,)
      acc <- acc + log(pmax(pnorm(outer(dmu[, j], shift, "+")), 1e-300))
    }
    m <- apply(acc, 1, max)
    logp[idx] <- m + log(pmax(exp(acc - m) %*% Wq, 1e-300))
  }
  -sum(logp)
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
mlogit_fast <- function(formula, data, r = 2L, Qf = 7L, Qz = 7L,
                        maxit = 400L) {
  t0 <- Sys.time()
  idx <- if (!is.null(data$idx)) data$idx else attr(data, "idx")
  alt_f <- idx[[2]]
  alt <- as.integer(alt_f)
  J <- max(alt)
  # dfidx wraps columns in xseries, under which %in% misbehaves; coerce
  chosen_rows <- which(as.logical(data[[as.character(formula[[2]])]]))
  Tn <- length(chosen_rows)
  choice <- alt[chosen_rows]
  rhs <- attr(terms(formula), "term.labels")
  Xcov <- as.matrix(as.data.frame(data)[, rhs, drop = FALSE])
  Xint <- matrix(0, nrow(Xcov), J - 1L)
  for (j in 2:J) Xint[alt == j, j - 1L] <- 1
  X <- cbind(Xint, Xcov)
  colnames(X) <- c(paste0("(Intercept):", levels(alt_f)[2:J]), rhs)
  nb <- ncol(X)
  nw <- sum(vapply(seq_len(r), function(cl) J - cl, 0L))
  nodes <- .nodes3(Qf, Qz, r)
  nodes_sharp <- .halton_nodes3(r, 10L)
  theta0 <- c(rep(0, nb), rep(0.1, nw))
  fit <- optim(theta0, .nll, Xb = list(X), choice = choice, J = J, r = r,
               nodes = nodes, nodes_sharp = nodes_sharp, method = "BFGS",
               control = list(maxit = maxit, reltol = 1e-8))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  beta <- fit$par[seq_len(nb)]
  names(beta) <- colnames(X)
  structure(list(coefficients = beta,
                 covariance_par = fit$par[-seq_len(nb)],
                 logLik = -fit$value, time = secs,
                 convergence = fit$convergence, J = J, r = r),
            class = "mlogit_fast")
}

print.mlogit_fast <- function(x, ...) {
  cat(sprintf("exact multinomial probit  logLik %.3f  (%.1f s, %s)\n",
              x$logLik, x$time,
              if (x$convergence == 0) "converged" else "NOT converged"))
  print(round(x$coefficients, 4))
  invisible(x)
}
