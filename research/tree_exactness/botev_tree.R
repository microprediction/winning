suppressMessages({library(TruncatedNormal); library(jsonlite)})
cs <- fromJSON("botev_case.json")
mu <- cs$mu; S <- as.matrix(cs$Sig); n <- length(mu)
set.seed(1)
for (j in seq_along(cs$idx)) {
  i <- cs$idx[j] + 1
  o <- setdiff(seq_len(n), i)
  m <- mu[o] - mu[i]
  SS <- S[o, o] - outer(S[o, i], rep(1, n - 1)) -
        outer(rep(1, n - 1), S[i, o]) + S[i, i]
  est <- TruncatedNormal::pmvnorm(mu = m, sigma = SS,
                                  lb = rep(0, n - 1), ub = rep(Inf, n - 1), B = 20000)
  cat(sprintf("entry %3d  kernel %.4e  botev %.4e  ratio %.3f\n",
              i - 1, cs$p_kernel[j], as.numeric(est),
              cs$p_kernel[j] / as.numeric(est)))
}
