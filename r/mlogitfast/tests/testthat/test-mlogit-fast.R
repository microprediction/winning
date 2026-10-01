# Run by R CMD check through tests/testthat.R, so the package is
# already attached and its internals are in scope. These files used
# to source(file.path("..", "..", "R", ...)) instead, which only
# works from the repo and fails inside a check -- which is how they
# went unrun (#142).
test_that("nll is finite and decreases from start on a synthetic problem", {
  set.seed(3)
  J <- 3; Tn <- 400; r <- 2
  V <- rbind(0, matrix(c(1.2, 0, 0.7, 0.9), 2, 2))
  X <- cbind(matrix(rep(diag(1, J)[, -1], Tn), ncol = J - 1, byrow = TRUE),
             rnorm(Tn * J))
  beta_true <- c(0.4, -0.3, 0.8)
  mu <- matrix(X %*% beta_true, Tn, J, byrow = TRUE)
  eps <- t(V %*% matrix(rnorm(2 * Tn), 2)) + matrix(rnorm(Tn * J), Tn, J)
  choice <- max.col(mu + eps)
  nodes <- .nodes3(7, 7, r); ns <- .halton_nodes3(r, 9L)
  th0 <- c(rep(0, 3), rep(0.1, 3))
  v0 <- .nll(th0, list(X), choice, J, r, nodes, ns)
  expect_true(is.finite(v0))
  th1 <- th0; th1[1:3] <- beta_true
  expect_lt(.nll(th1, list(X), choice, J, r, nodes, ns), v0)
})

test_that("sharpness escalation switches node families", {
  J <- 3; r <- 2
  nodes <- .nodes3(7, 7, r); ns <- .halton_nodes3(r, 9L)
  X <- cbind(matrix(0, 3 * J, J - 1), rnorm(3 * J))
  # sharp covariance parameters must route to the Halton set: the nll
  # values under the two calls differ only if the branch is taken, so
  # probe via node-count sensitivity at sharp vs mild theta
  th_mild <- c(0, 0, 0, 0.1, 0.1, 0.1)
  th_sharp <- c(0, 0, 0, 50, 50, 50)
  v_m <- .nll(th_mild, list(X), c(1, 2, 3), J, r, nodes, ns)
  v_s <- .nll(th_sharp, list(X), c(1, 2, 3), J, r, nodes, ns)
  expect_true(is.finite(v_m) && is.finite(v_s))
})

test_that("the triangular loading loop is empty at J = 2, not backwards", {
  # `(col + 1L):J` is c(3, 2) at J = 2, not empty, so the loop wrote
  # V[3, 2] on a 2x2 matrix and every binary-choice fit died out of
  # bounds (#183). The same loop is in rprobitfast, which carries the
  # end-to-end fit test; this pins the sequence itself.
  visited <- function(J, r) {
    out <- character(0)
    for (col in seq_len(r)) for (row in col + seq_len(J - col))
      out <- c(out, sprintf("%d,%d", row, col))
    out
  }
  expect_equal(visited(2L, 2L), "2,1")
  expect_equal(visited(1L, 1L), character(0))
  expect_true(all(vapply(strsplit(visited(4L, 2L), ","),
                         function(p) as.integer(p[1]) <= 4L, TRUE)))
})

# --- the tail likelihood is finite and its score is not zero (#270) ---
#
# logPhi was log(pmax(pnorm(a), 1e-300)). pnorm underflows to exactly
# zero below about -37, so every deep tail collapsed to the SAME
# -690.78 and the objective went FLAT; the Mills ratio then underflowed
# to a score of exactly ZERO. An optimizer declares convergence
# precisely where an observation is most badly contradicted.
#
# pnorm(log.p = TRUE) removes both. It does NOT remove the quadrature
# error underneath -- the fixed rule samples the own noise where the
# prior has mass, not the integrand -- so the values are still wrong
# in the tail, and the last test here pins that rather than pretending
# otherwise.

.binary_ll <- function(gap, Qf = 7L, Qz = 7L) {
  nd <- .nodes3(Qf = Qf, Qz = Qz, r = 1L)
  ndS <- .halton_nodes3(1L, m = 10L)
  X <- matrix(c(gap, 0), nrow = 2L, ncol = 1L)
  -.nll_core(c(1, 0), list(X), 1L, 2L, 1L, nd, ndS,
             want_grad = FALSE)$value
}

test_that("the objective does not go flat in the tail", {
  lls <- vapply(c(-40, -60, -100, -200), .binary_ll, 0)
  expect_true(all(is.finite(lls)))
  expect_true(all(diff(lls) < -1))          # strictly falling, by a lot
  expect_lt(min(lls), -1e4)                 # far past the old -690.78
})

test_that("the score is not zero and points the right way", {
  nd <- .nodes3(Qf = 7L, Qz = 7L, r = 1L)
  ndS <- .halton_nodes3(1L, m = 10L)
  prev <- NULL
  for (gap in c(-40, -60, -100, -200)) {
    X <- matrix(c(gap, 0), nrow = 2L, ncol = 1L)
    g <- .nll_core(c(1, 0), list(X), 1L, 2L, 1L, nd, ndS,
                   want_grad = TRUE)$grad
    expect_true(all(is.finite(g)))
    expect_true(any(abs(g) > 1e-6))         # not the zero score
    s <- abs(g[1])
    if (!is.null(prev)) expect_gt(s, prev)  # grows with the surprise
    prev <- s
  }
})

test_that("moderate gaps are accurate, so the tail failure is the rule", {
  for (gap in c(-1, -2, -3)) {
    ex <- pnorm(gap / sqrt(2), log.p = TRUE)
    expect_lt(abs(.binary_ll(gap) - ex) / abs(ex), 1e-3)
  }
})

test_that("the remaining quadrature error is recorded, not claimed fixed", {
  # a measurement so a later change closing #270's third defect has
  # something to move: at a gap of -20 this is 38% away
  ex <- pnorm(-20 / sqrt(2), log.p = TRUE)
  got <- .binary_ll(-20)
  expect_gt(abs(got - ex) / abs(ex), 0.3)
  expect_lt(got, ex)                        # understates the probability
})

test_that("an ordinary panel still fits", {
  set.seed(5)
  n <- 80L; J <- 3L
  df <- data.frame(id = rep(seq_len(n), each = J),
                   alt = rep(1:J, times = n), x = rnorm(n * J))
  u <- 0.7 * df$x + rnorm(n * J)
  df$chosen <- unlist(lapply(split(u, df$id),
                             function(v) as.integer(seq_along(v) == which.max(v))))
  nd <- .nodes3(Qf = 5L, Qz = 5L, r = 1L)
  ndS <- .halton_nodes3(1L, m = 8L)
  X <- matrix(df$x, ncol = 1L)
  ch <- apply(matrix(df$chosen, nrow = n, ncol = J, byrow = TRUE), 1,
              which.max)
  r <- .nll_core(c(0.7, 0, 0), list(X), as.integer(ch), J, 1L, nd, ndS,
                 want_grad = TRUE)
  expect_true(is.finite(r$value))
  expect_true(all(is.finite(r$grad)))
})
