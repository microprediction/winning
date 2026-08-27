# rprobit_fast: exact MNP estimation for long-format choice data, the
# model class of the Rprobit package (Bauer, MACML-based), priced
# exactly instead of by Mendell-Elston / Solow-Joe approximation.
#
# The engine (R/engine.R, sync-guarded copy of r/mlogitfast's) is the
# factor-conditional product likelihood with analytic score and the
# sharpness-escalation node rule; see r/mlogitfast/README.md for the
# measured behavior including the quadrature-hole war story and the
# boundary-seeking caveat.

#' Exact multinomial probit from long-format choice data
#'
#' @param df data.frame in long format: one row per (observation,
#'   alternative), with columns id (observation), alt (alternative,
#'   integer or factor), chosen (0/1 or logical), and covariates.
#' @param covariates character vector of alternative-specific covariate
#'   column names.
#' @param r factor rank (r = 2 spans the full identified covariance at
#'   four alternatives).
rprobit_fast <- function(df, covariates, r = 2L, Qf = 7L, Qz = 7L,
                         maxit = 400L) {
  t0 <- Sys.time()
  alt <- as.integer(as.factor(df$alt))
  J <- max(alt)
  ids <- as.integer(as.factor(df$id))
  ord <- order(ids, alt)
  df <- df[ord, , drop = FALSE]
  alt <- alt[ord]; ids <- ids[ord]
  Tn <- length(unique(ids))
  stopifnot(nrow(df) == Tn * J)
  chosen <- which(as.logical(df$chosen))
  choice <- alt[chosen]
  Xcov <- as.matrix(df[, covariates, drop = FALSE])
  Xint <- matrix(0, nrow(Xcov), J - 1L)
  for (j in 2:J) Xint[alt == j, j - 1L] <- 1
  X <- cbind(Xint, Xcov)
  colnames(X) <- c(paste0("asc_", 2:J), covariates)
  nb <- ncol(X)
  nw <- sum(vapply(seq_len(r), function(cl) J - cl, 0L))
  nodes <- .nodes3(Qf, Qz, r)
  nodes_sharp <- .halton_nodes3(r, 10L)
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
  theta0 <- c(rep(0, nb), rep(0.1, nw))
  fit <- optim(theta0, fn, gr, method = "BFGS",
               control = list(maxit = maxit, reltol = 1e-8))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  beta <- fit$par[seq_len(nb)]
  names(beta) <- colnames(X)
  wnorm <- 0
  k <- nb + 1L
  V <- matrix(0, J, r)
  ki <- 1L
  for (col in seq_len(r)) for (row in (col + 1L):J) {
    V[row, col] <- fit$par[nb + ki]; ki <- ki + 1L
  }
  structure(list(coefficients = beta,
                 covariance_par = fit$par[-seq_len(nb)],
                 boundary = max(sqrt(rowSums(V^2))) > 50,
                 logLik = -fit$value, time = secs,
                 convergence = fit$convergence, J = J, r = r),
            class = "rprobit_fast")
}

print.rprobit_fast <- function(x, ...) {
  cat(sprintf("exact MNP  logLik %.3f  (%.1f s, %s%s)\n",
              x$logLik, x$time,
              if (x$convergence == 0) "converged" else "NOT converged",
              if (x$boundary) ", BOUNDARY-SEEKING covariance" else ""))
  print(round(x$coefficients, 4))
  invisible(x)
}
