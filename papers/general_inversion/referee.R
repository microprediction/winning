# Replay cases.json through the two field-standard R implementations:
#   mvtnorm::pmvnorm  (Genz-Bretz, the canonical central referee)
#   TruncatedNormal::pmvnorm  (Botev minimax tilting, bounded RELATIVE
#                              error, the deep-tail referee)
# p_i = P(X_j - X_i >= 0 for all j) with d ~ N(m_d, S_d), evaluated as
# P(d - m_d >= -m_d): lower = -m_d, upper = Inf, centered sigma = S_d.
# Output: referee_out.json.

suppressMessages({library(mvtnorm); library(TruncatedNormal)})
have_jsonlite <- requireNamespace("jsonlite", quietly = TRUE)
stopifnot(have_jsonlite)
cases <- jsonlite::fromJSON("cases.json", simplifyVector = FALSE)

out <- list()
for (case in cases) {
  mu <- unlist(case$mu); n <- length(mu)
  V  <- do.call(rbind, lapply(case$V, unlist))
  D  <- unlist(case$D)
  Sig <- V %*% t(V) + diag(D)
  genz <- genz_err <- botev <- botev_relerr <- numeric(n)
  for (i in 1:n) {
    A <- diag(n)[-i, , drop = FALSE]; A[, i] <- -1
    m <- as.vector(A %*% mu); S <- A %*% Sig %*% t(A)
    # NB: TruncatedNormal also exports pmvnorm and masks mvtnorm's;
    # the namespace must be explicit or the bare call silently returns 1.
    g <- mvtnorm::pmvnorm(lower = -m, upper = rep(Inf, n - 1), sigma = S,
                          algorithm = GenzBretz(maxpts = 2e6,
                                                abseps = 1e-10))
    genz[i] <- as.numeric(g)
    e <- attr(g, "error"); genz_err[i] <- if (is.null(e)) NA else e
    b <- TruncatedNormal::pmvnorm(mu = rep(0, n - 1), sigma = S,
                                  lb = -m, ub = rep(Inf, n - 1), B = 2e4)
    botev[i] <- as.numeric(b)
    re <- attr(b, "relerr")
    botev_relerr[i] <- if (is.null(re)) NA else as.numeric(re)
  }
  out[[case$name]] <- list(genz = genz, genz_err = genz_err,
                           botev = botev, botev_relerr = botev_relerr)
  cat(sprintf("%s done\n", case$name))
}
jsonlite::write_json(out, "referee_out.json", digits = 16, auto_unbox = TRUE)
