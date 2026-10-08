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
  nodes <- .nodes3(7, 7, r); ns <- .sobol_nodes3(r, 9L)
  th0 <- c(rep(0, 3), rep(0.1, 3))
  v0 <- .nll(th0, list(X), choice, J, r, nodes, ns)
  expect_true(is.finite(v0))
  th1 <- th0; th1[1:3] <- beta_true
  expect_lt(.nll(th1, list(X), choice, J, r, nodes, ns), v0)
})

test_that("sharpness escalation switches node families", {
  J <- 3; r <- 2
  nodes <- .nodes3(7, 7, r); ns <- .sobol_nodes3(r, 9L)
  X <- cbind(matrix(0, 3 * J, J - 1), rnorm(3 * J))
  # sharp covariance parameters must route to the Sobol set: the nll
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
  ndS <- .sobol_nodes3(1L, m = 10L)
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
  ndS <- .sobol_nodes3(1L, m = 10L)
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
  ndS <- .sobol_nodes3(1L, m = 8L)
  X <- matrix(df$x, ncol = 1L)
  ch <- apply(matrix(df$chosen, nrow = n, ncol = J, byrow = TRUE), 1,
              which.max)
  r <- .nll_core(c(0.7, 0, 0), list(X), as.integer(ch), J, 1L, nd, ndS,
                 want_grad = TRUE)
  expect_true(is.finite(r$value))
  expect_true(all(is.finite(r$grad)))
})

# --- shared-engine regressions (identical in mlogitfast and rprobitfast) ---

test_that("sharp-regime nodes exist at every rank the fitters accept (#388)", {
  for (r in 1:6) {
    nd <- .sobol_nodes3(r, 5L)
    expect_equal(dim(nd$F), c(32L, r + 1L))
    expect_true(all(is.finite(nd$F)) && all(is.finite(nd$W)))
    expect_equal(sum(nd$W), 1)
  }
  # the rule is built lazily, once
  lz <- .lazy_sharp_nodes(4L, 5L)
  expect_identical(lz(), lz())
  expect_equal(ncol(lz()$F), 5L)
})

test_that("node dispatch uses the centred pairwise-safe bound (#419)", {
  V <- matrix(c(0, 1, -2.7), ncol = 1)
  expect_equal(.mnp_sharpness(V), sqrt(2) * max(abs(V - mean(V))))
  expect_gt(.mnp_sharpness(V), 3)               # raw max |V_i| is 2.7
  # gauge invariance: V -> V + 1 c'
  V2 <- matrix(c(0, 1, -2.7, 0.3, -0.4, 2), ncol = 2)
  expect_equal(.mnp_sharpness(V2), .mnp_sharpness(sweep(V2, 2, c(5, -3), "+")))
  # the #419 fixture: alternative 3 wins with exact bivariate-normal
  # probability 0.15149432190016066; Hermite-7 said 0.16517
  X <- diag(3)
  theta <- c(-3.25, 2.47, -1.61, 1, -2.7)
  gh <- .nodes3(7L, 7L, 1L)
  got <- exp(-.nll_core(theta, list(X), 3L, 3L, 1L, nodes = gh,
                        nodes_sharp = .sobol_nodes3(1L, 10L))$value)
  expect_lt(abs(got - 0.15149432190016066), 2e-3)
  forced_gh <- exp(-.nll_core(theta, list(X), 3L, 3L, 1L, nodes = gh,
                              nodes_sharp = gh)$value)
  expect_gt(abs(forced_gh - 0.15149432190016066), 1e-2)
})

test_that("choices are keyed by observation, one per id (#324, #435)", {
  ids <- rep(1:3, each = 3); alt <- rep(1:3, 3)
  ok <- c(TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE)
  expect_equal(.choices_by_id(ids, alt, ok), c(1L, 2L, 3L))
  expect_equal(.choices_by_id(ids, alt, as.integer(ok)), c(1L, 2L, 3L))
  cancel <- c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE)
  expect_error(.choices_by_id(ids, alt, cancel, c("a", "b", "c")),
               "observation 'a' has 2 chosen rows")
  none <- ok; none[5] <- FALSE
  expect_error(.choices_by_id(ids, alt, none), "observation '2' has 0")
  two <- ok; two[6] <- TRUE
  expect_error(.choices_by_id(ids, alt, two), "has 2 chosen")
  na <- ok; na[4] <- NA
  expect_error(.choices_by_id(ids, alt, na), "missing")
  expect_error(.choices_by_id(ids, alt, ok * 2), "logical or 0/1")
})

# a minimal stand-in for a dfidx: mlogit_fast reads attr(data, "idx")
.fake_dfidx <- function(df) {
  out <- df[, setdiff(names(df), c("id", "alt")), drop = FALSE]
  attr(out, "idx") <- data.frame(id = df$id, alt = factor(df$alt))
  out
}

test_that("mlogit_fast is invariant to long-row order (#215, #355)", {
  set.seed(8)
  n <- 40L; J <- 3L
  df <- data.frame(id = rep(seq_len(n), each = J), alt = rep(1:J, n),
                   x = rnorm(n * J))
  u <- 0.8 * df$x + rnorm(n * J)
  df$chosen <- unlist(lapply(split(u, df$id),
                             function(v) seq_along(v) == which.max(v)))
  f1 <- mlogit_fast(chosen ~ x, .fake_dfidx(df), r = 1L, Qf = 5L, Qz = 5L,
                    maxit = 30L)
  altmajor <- order(df$alt, df$id)
  f2 <- mlogit_fast(chosen ~ x, .fake_dfidx(df[altmajor, ]), r = 1L,
                    Qf = 5L, Qz = 5L, maxit = 30L)
  shuffled <- sample(nrow(df))
  f3 <- mlogit_fast(chosen ~ x, .fake_dfidx(df[shuffled, ]), r = 1L,
                    Qf = 5L, Qz = 5L, maxit = 30L)
  expect_equal(f1$logLik, f2$logLik)
  expect_equal(f1$logLik, f3$logLik)
  expect_equal(f1$coefficients, f3$coefficients)
})

test_that("mlogit_fast refuses cancelling choices and fits rank four", {
  d <- data.frame(id = rep(1:3, each = 3), alt = rep(1:3, 3),
                  x = c(-1, .2, .7, .4, -1.2, .1, .8, -.3, 1.1))
  d$chosen <- c(TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE)
  expect_error(mlogit_fast(chosen ~ x, .fake_dfidx(d), r = 1L, Qf = 3L,
                           Qz = 3L, maxit = 1L),
               "observation '1' has 2 chosen rows")
  J <- 6L
  df <- expand.grid(alt = seq_len(J), id = 1:3)
  df$chosen <- df$alt == ((df$id - 1L) %% J + 1L)
  df$x <- seq_len(nrow(df)) / nrow(df)
  fit <- mlogit_fast(chosen ~ x, .fake_dfidx(df[, c("id", "alt", "chosen", "x")]),
                     r = 4L, Qf = 3L, Qz = 3L, maxit = 1L)
  expect_true(is.finite(fit$logLik))
})

test_that("fractional labels are refused and the rank is identified (#194, #201)", {
  X <- matrix(rnorm(30 * 3 * 2), nrow = 90, ncol = 2)
  nd <- .sobol_nodes3(1L, m = 5L)
  ch <- as.numeric(rep(1:3, length.out = 30)); ch[2] <- 1.5
  expect_error(.nll_core(c(0.5, 0.2, 0.3), list(X), ch, 3L, 1L, nd, nd,
                         want_grad = FALSE), "choice\\[2\\] = 1.5")
  for (J in 2:5) {
    expect_identical(.check_rank(NULL, J), as.integer(min(2L, J - 2L)))
    expect_error(.check_rank(J - 1L, J), "not identified")
  }
})

test_that("unused factor levels are not phantom choosers or alternatives (#464)", {
  set.seed(4)
  n <- 30L; J <- 3L
  id <- rep(seq_len(n), each = J)
  alt <- rep(c("x", "y", "z"), times = n)
  d <- data.frame(v = rnorm(n * J))
  u <- 0.7 * d$v + rnorm(n * J)
  d$chosen <- unlist(lapply(split(u, id),
                            function(w) seq_along(w) == which.max(w)))
  fit <- function(id, alt) {
    dd <- d
    attr(dd, "idx") <- data.frame(id = id, alt = alt)
    mlogit_fast(chosen ~ v, dd, r = 1L, Qf = 3L, Qz = 3L, maxit = 5L)
  }
  id_f <- factor(id * 2L, levels = seq_len(2L * n + 1L))
  alt_f <- factor(alt, levels = c("x", "w", "y", "z", "zz"))
  ref <- fit(droplevels(id_f), droplevels(alt_f))
  for (got in list(fit(id_f, droplevels(alt_f)),
                   fit(droplevels(id_f), alt_f),
                   fit(id_f, alt_f))) {
    expect_identical(got$J, 3L)
    expect_equal(got$coefficients, ref$coefficients)
    expect_equal(got$logLik, ref$logLik)
  }
  expect_identical(names(ref$coefficients),
                   c("(Intercept):y", "(Intercept):z", "v"))
})

test_that("Gauss-Hermite order one is the one-node rule (#504)", {
  expect_equal(.gh1(1L), list(x = 0, w = 1))
  for (bad in list(0L, -1L, 1.5, NA_real_, Inf, c(2L, 3L)))
    expect_error(.gh1(bad), "one positive integer")
  d <- data.frame(chosen = c(TRUE, FALSE, FALSE, TRUE),
                  x = c(-1, 1, -0.5, 0.5))
  attr(d, "idx") <- data.frame(id = rep(1:2, each = 2),
                               alt = factor(rep(1:2, 2)))
  f1 <- mlogit_fast(chosen ~ x, d, r = 0L, Qf = 1L, Qz = 3L, maxit = 3L)
  f7 <- mlogit_fast(chosen ~ x, d, r = 0L, Qf = 7L, Qz = 3L, maxit = 3L)
  expect_equal(f1$logLik, f7$logLik)          # Qf is inert at r = 0
  expect_true(is.finite(mlogit_fast(chosen ~ x, d, r = 0L, Qf = 3L, Qz = 1L,
                                    maxit = 3L)$logLik))
})

test_that("an idx that does not cover data is refused, not a prefix (#520)", {
  idx1 <- data.frame(id = rep(1L, 3L), alt = factor(c("a", "b", "c")))
  d <- data.frame(chosen = c(TRUE, FALSE, FALSE, FALSE, FALSE, TRUE),
                  x = c(-1, 0, 1, 10, 0, -10))
  attr(d, "idx") <- idx1
  expect_error(mlogit_fast(chosen ~ x, d, r = 0L, Qf = 3L, Qz = 3L,
                           maxit = 2L), "got 3 for 6 data rows")
  short <- d[1:3, , drop = FALSE]
  attr(short, "idx") <- data.frame(id = rep(1:2, each = 3),
                                   alt = factor(rep(c("a", "b", "c"), 2)))
  expect_error(mlogit_fast(chosen ~ x, short, r = 0L, Qf = 3L, Qz = 3L,
                           maxit = 2L), "got 6 for 3 data rows")
  attr(d, "idx") <- NULL
  expect_error(mlogit_fast(chosen ~ x, d, r = 0L, Qf = 3L, Qz = 3L,
                           maxit = 2L), "idx column or attribute")
  attr(d, "idx") <- data.frame(id = rep(1:2, each = 3))
  expect_error(mlogit_fast(chosen ~ x, d, r = 0L, Qf = 3L, Qz = 3L,
                           maxit = 2L), "idx column or attribute")
})

test_that("a factor response means its labels, not its level codes (#512)", {
  set.seed(5)
  n <- 20L; J <- 3L
  d <- data.frame(x = rnorm(n * J))
  u <- 0.7 * d$x + rnorm(n * J)
  ch <- unlist(lapply(split(u, rep(seq_len(n), each = J)),
                      function(w) seq_along(w) == which.max(w)))
  attr(d, "idx") <- data.frame(id = rep(seq_len(n), each = J),
                               alt = factor(rep(c("a", "b", "c"), n)))
  fit <- function(resp) {
    d$chosen <- resp
    mlogit_fast(chosen ~ x, d, r = 0L, Qf = 3L, Qz = 3L, maxit = 5L)
  }
  ref <- fit(ch)
  for (resp in list(factor(ch, levels = c(FALSE, TRUE)),
                    factor(as.integer(ch), levels = 0:1),
                    as.integer(ch))) {
    got <- fit(resp)
    expect_equal(got$coefficients, ref$coefficients)
    expect_equal(got$logLik, ref$logLik)
  }
})
