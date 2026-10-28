# GHK keeps its tail slope (#270): the truncated draw was floored at
# u * Phi(b) = 1e-300, pinning draws behind a deep bound at
# qnorm(1e-300). References: 1-d conditional-normal quadrature.
test_that("GHK log probability tracks a quadrature far below 1e-300", {
  v <- c(0.3, -0.2, 0.1)
  Sigma <- outer(v, v) + diag(3)
  set.seed(1)
  u <- matrix(runif(8000), 4000, 2)
  refs <- list(c(-10, -37.51799189156518), c(-60, -1131.5948698859718),
               c(-80, -2001.2502944154676))
  for (r in refs) {
    lp <- winning:::.ghk_one(c(r[1], 0, 0.5), Sigma, 1L, u)$logP
    expect_lt(abs(lp - r[2]), 0.05)
  }
})
