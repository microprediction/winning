# Independent adjudication: Botev minimax tilting (TruncatedNormal) with
# relative-accuracy estimates, per selected runner.
suppressMessages({library(TruncatedNormal); library(jsonlite)})
cs <- fromJSON("referee_cases.json")
mu <- cs$mu; C <- as.matrix(cs$C); idx <- cs$idx + 1  # 1-based
n <- length(mu)
cat(sprintf("%4s %12s %12s %10s %9s\n", "i", "ours", "botev", "rel.err", "botev.acc"))
set.seed(1)
for (j in seq_along(idx)) {
  i <- idx[j]
  o <- setdiff(seq_len(n), i)
  m <- mu[o] - mu[i]
  S <- C[o, o] - outer(C[o, i], rep(1, n - 1)) -
       outer(rep(1, n - 1), C[i, o]) + C[i, i]
  est <- tryCatch(
    TruncatedNormal::pmvnorm(mu = m, sigma = S,
                             lb = rep(0, n - 1), ub = rep(Inf, n - 1),
                             B = 10000),
    error = function(e) NA)
  ours <- cs$p_ours[j]
  if (is.na(est[1])) { cat(sprintf("%4d %12.3e %12s\n", i-1, ours, "ERR")); next }
  relerr <- ours / as.numeric(est) - 1
  acc <- attr(est, "relerr"); if (is.null(acc)) acc <- NA
  cat(sprintf("%4d %12.3e %12.3e %+9.1f%% %9.5f\n",
              i - 1, ours, as.numeric(est), 100 * relerr, acc))
}
