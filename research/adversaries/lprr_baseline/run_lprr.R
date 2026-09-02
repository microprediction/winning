# Per-winner reduced-rank Genz (mvtnorm::lpRR) as the strongest
# factor-aware baseline: winner i's share is the rectangle
# P(Y <= 0), Y_j = U_j - U_i (max-wins), with covariance
# B_i B_i' + diag(D_{-i}), B_i = [v_j - v_i, -sqrt(d_i)] of k+1
# columns. One lpRR call per winner, common scrambled-Sobol factor
# draws across winners, complete-vector timing.
suppressMessages(library(mvtnorm))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
inst <- fromJSON(args[1])
R_draws <- as.integer(args[2])

mu <- inst$mu; V <- matrix(unlist(inst$V), nrow = length(mu),
                           byrow = TRUE)
d <- inst$d
N <- length(mu); k <- ncol(V)

# common scrambled Sobol over the k+1 latent dimensions (qrng if
# installed, else pseudo-random common draws, flagged in the output)
set.seed(11)
qrng_available <- requireNamespace("qrng", quietly = TRUE)
if (qrng_available) {
  U <- qrng::sobol(R_draws, d = k + 1, randomize = "digital.shift")
} else {
  U <- matrix(runif(R_draws * (k + 1)), ncol = k + 1)
}
Z <- qnorm(t(U))                     # (k+1) x R, common across winners

t0 <- proc.time()[3]
logp <- numeric(N)
for (i in 1:N) {
  vi <- V[i, ]; di <- d[i]
  B <- cbind(sweep(V[-i, , drop = FALSE], 2, vi), -sqrt(di))
  Dm <- d[-i]
  up <- mu[i] - mu[-i]
  logp[i] <- lpRR(lower = rep(-Inf, N - 1), upper = up,
                  mean = 0, B = B, D = Dm, Z = Z, log.p = TRUE)
}
secs <- proc.time()[3] - t0
cat(toJSON(list(logp = logp, seconds = secs,
                sobol = qrng_available), digits = 12))
