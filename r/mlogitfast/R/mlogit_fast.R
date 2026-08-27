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

# Negative log-likelihood AND analytic score, fully vectorized.
# Sharpness-aware: when the loadings make the factor integrand a
# near-step, Gauss-Hermite under-integrates at any order and the
# OPTIMIZER EXPLOITS THE HOLES (observed: a runaway to ||w|| ~ 300 with
# a fake 20-nat likelihood gain that collapses under denser rules), so
# past sharpness 3 the evaluation switches to Halton nodes -- the same
# family-escalation rule as winning::race_probabilities.
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
  nb <- ncol(X)
  beta <- theta[seq_len(nb)]
  wfree <- theta[-seq_len(nb)]
  V <- matrix(0, J, r)
  k <- 1L
  fill <- list()
  for (col in seq_len(r)) for (row in (col + 1L):J) {
    V[row, col] <- wfree[k]; fill[[k]] <- c(row, col); k <- k + 1L
  }
  Tn <- nrow(X) / J
  mu <- matrix(X %*% beta, nrow = Tn, ncol = J, byrow = TRUE)
  sharp <- max(sqrt(rowSums(V^2)))
  nd <- if (sharp > 3.0) nodes_sharp else nodes
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
      logPhi[[j]] <- log(pmax(pnorm(A[[j]]), 1e-300))
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
  memo <- new.env()
  fn <- function(th) {
    if (!is.null(memo$th) && identical(th, memo$th)) return(memo$out$value)
    memo$th <- th
    memo$out <- .nll_core(th, list(X), choice, J, r, nodes, nodes_sharp)
    memo$out$value
  }
  gr <- function(th) {
    if (!is.null(memo$th) && identical(th, memo$th)) return(memo$out$grad)
    memo$th <- th
    memo$out <- .nll_core(th, list(X), choice, J, r, nodes, nodes_sharp)
    memo$out$grad
  }
  fit <- optim(theta0, fn, gr, method = "BFGS",
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
