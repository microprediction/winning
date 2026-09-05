# Top-k memberships q_i = P(X_i among the k smallest), their mu- and
# sigma-Jacobians, the rank marginals, and the inversions: locations
# from one membership curve, (loc, scale) jointly from two. Port of
# winning/factor/topk.py -- the cavity count distribution with
# stable-direction deconvolution; see the python module docstring for
# derivations and the two-branch refusal of exact-rank targets.

.clip01 <- function(v) pmin(pmax(v, 0), 1)

.count_window <- function(mu, sd, k, fn, delta = 1e-12, pad_sds = 2.0) {
  n <- length(mu)
  smax <- max(max(sd), 1e-12)
  mean_count <- function(x) sum(1 - fn((x - mu) / sd)$S)
  lo <- min(mu) - 9 * smax
  step <- 9 * smax
  for (it in 1:60) {
    if (mean_count(lo) <= delta) break
    lo <- lo - step
    step <- step * 2
  }
  target_hi <- min(k + 2 * log(1 / delta) +
                     sqrt(2 * (k + 1) * log(1 / delta)), n - 1e-4)
  hi <- max(mu) + 9 * smax
  step <- 9 * smax
  for (it in 1:60) {
    if (mean_count(hi) >= target_hi) break
    hi <- hi + step
    step <- step * 2
  }
  a <- lo; b <- hi
  for (it in 1:70) {
    m <- 0.5 * (a + b)
    if (mean_count(m) < delta) a <- m else b <- m
  }
  xlo <- a
  a <- xlo; b <- hi
  for (it in 1:70) {
    m <- 0.5 * (a + b)
    if (mean_count(m) < target_hi) a <- m else b <- m
  }
  c(xlo - pad_sds * smax, b + pad_sds * smax)
}

.topk_grid <- function(mu, sd, k, fn, points, delta = 1e-12) {
  w <- .count_window(mu, sd, k, fn, delta)
  x <- seq(w[1], w[2], length.out = points)
  z <- outer(x, mu, "-") / matrix(sd, points, length(mu), byrow = TRUE)
  b <- fn(z)
  list(x = x, dx = x[2] - x[1], z = z,
       S = b$S, f = b$f, fp = b$fp, F = .clip01(1 - b$S))
}

.count_distribution <- function(F) {
  L <- nrow(F); n <- ncol(F)
  C <- matrix(0, L, n + 1)
  C[, 1] <- 1
  for (j in 1:n) {
    f <- F[, j]
    idx <- 2:(j + 1)
    C[, idx] <- C[, idx, drop = FALSE] * (1 - f) +
      C[, idx - 1, drop = FALSE] * f
    C[, 1] <- C[, 1] * (1 - f)
  }
  C
}

.loo_cdf <- function(C, F, k) {
  L <- nrow(F); n <- ncol(F)
  out <- matrix(0, n, L)
  for (i in 1:n) {
    Fi <- F[, i]; Si <- 1 - Fi
    fwd <- Si >= Fi
    s <- pmax(Si, 1e-300)
    Q <- .clip01(C[, 1] / s)
    acc_f <- Q
    if (k >= 2) for (m in 2:k) {
      Q <- .clip01((C[, m] - Fi * Q) / s)
      acc_f <- acc_f + Q
    }
    f <- pmax(Fi, 1e-300)
    Qb <- .clip01(C[, n + 1] / f)
    acc_b <- Qb
    if (n - 1 >= k + 1) for (mm in seq(n - 1, k + 1, by = -1)) {
      Qb <- .clip01((C[, mm + 1] - Si * Qb) / f)
      acc_b <- acc_b + Qb
    }
    out[i, ] <- .clip01(ifelse(fwd, acc_f, 1 - acc_b))
  }
  out
}

.loo_pmf <- function(C, F, i) {
  L <- nrow(F); n <- ncol(F)
  Fi <- F[, i]; Si <- 1 - Fi
  fwd <- Si >= Fi
  s <- pmax(Si, 1e-300)
  Qf <- matrix(0, L, n)
  Qf[, 1] <- .clip01(C[, 1] / s)
  if (n >= 2) for (m in 2:n)
    Qf[, m] <- .clip01((C[, m] - Fi * Qf[, m - 1]) / s)
  f <- pmax(Fi, 1e-300)
  Qb <- matrix(0, L, n)
  Qb[, n] <- .clip01(C[, n + 1] / f)
  if (n >= 2) for (m in seq(n - 1, 1, by = -1))
    Qb[, m] <- .clip01((C[, m + 1] - Si * Qb[, m + 1]) / f)
  Q <- Qb
  Q[fwd, ] <- Qf[fwd, , drop = FALSE]
  Q
}

.pair_pmf_at <- function(Qi, F, i, k) {
  L <- nrow(F); n <- ncol(F)
  out <- matrix(0, n, L)
  for (j in 1:n) {
    if (j == i) next
    Fj <- F[, j]; Sj <- 1 - Fj
    fwd <- Sj >= Fj
    s <- pmax(Sj, 1e-300)
    Q <- .clip01(Qi[, 1] / s)
    if (k >= 2) for (m in 2:k) Q <- .clip01((Qi[, m] - Fj * Q) / s)
    f <- pmax(Fj, 1e-300)
    Qb <- .clip01(Qi[, n] / f)
    if (n - 1 >= k + 1) for (mm in seq(n - 1, k + 1, by = -1))
      Qb <- .clip01((Qi[, mm] - Sj * Qb) / f)
    out[j, ] <- ifelse(fwd, Q, Qb)
  }
  out
}

.topk_with_slopes <- function(mu, sd, k, fn, points) {
  g <- .topk_grid(mu, sd, k, fn, points)
  C <- .count_distribution(g$F)
  cdf <- .loo_cdf(C, g$F, k)
  n <- length(mu)
  dens <- t(g$f) / sd                       # (n, L)
  dmu <- -t(g$fp) / (sd * sd)
  list(q = rowSums(dens * cdf) * g$dx,
       slopes = rowSums(dmu * cdf) * g$dx)
}

.checked_topk <- function(raw, k, kind, mass_tol = 5e-3) {
  t <- sum(raw)
  if (!is.finite(t) || abs(t - k) > mass_tol * k)
    stop(sprintf(paste0(
      "%s captured total membership %.4f where exactly %d slots exist: ",
      "the window or the deconvolution missed part of the field. Raise ",
      "points=, or report this field."), kind, t, k))
  .clip01(raw * (k / t))
}

.topk_factor_nodes <- function(V, n, qa) {
  Vm <- if (is.matrix(V)) V else matrix(V, ncol = 1)
  r <- ncol(Vm)
  if (r > 2)
    stop("top_k_probabilities is implemented for factor rank <= 2")
  Vm <- sweep(Vm, 2, colMeans(Vm))
  cn <- .cluster_nodes(r, qa)
  list(Vm = Vm, nodes = cn$nodes, w = cn$w / sum(cn$w))
}

top_k_probabilities <- function(mu, k, V = NULL, D = NULL,
                                base = "normal", points = 513, qa = 15) {
  n <- length(mu)
  k <- as.integer(k)
  if (k < 1 || k > n - 1)
    stop(sprintf("k must be in [1, n-1]; got k=%d, n=%d", k, n))
  sd <- sqrt(if (is.null(D)) rep(1, n) else D)
  fn <- if (is.function(base)) base else .BASES[[base]]
  if (is.null(V))
    return(.checked_topk(.topk_with_slopes(mu, sd, k, fn, points)$q,
                         k, "top-k race"))
  fac <- .topk_factor_nodes(V, n, qa)
  raw <- rep(0, n)
  for (q in seq_len(nrow(fac$nodes))) {
    shift <- as.vector(fac$Vm %*% fac$nodes[q, ])
    raw <- raw + fac$w[q] *
      .topk_with_slopes(mu + shift, sd, k, fn, points)$q
  }
  .checked_topk(raw, k, "top-k race")
}

bottom_k_probabilities <- function(mu, k, V = NULL, D = NULL,
                                   base = "normal", points = 513, qa = 15) {
  n <- length(mu)
  k <- as.integer(k)
  if (k < 1 || k > n - 1)
    stop(sprintf("k must be in [1, n-1]; got k=%d, n=%d", k, n))
  1 - top_k_probabilities(mu, n - k, V = V, D = D, base = base,
                          points = points, qa = qa)
}

top_k_jacobians <- function(mu, k, D = NULL, base = "normal",
                            points = 513) {
  n <- length(mu)
  k <- as.integer(k)
  if (k < 1 || k > n - 1)
    stop(sprintf("k must be in [1, n-1]; got k=%d, n=%d", k, n))
  Dv <- if (is.null(D)) rep(1, n) else D
  sd <- sqrt(Dv)
  fn <- if (is.function(base)) base else .BASES[[base]]
  g <- .topk_grid(mu, sd, k, fn, points)
  dens <- t(g$f) / sd                       # (n, L)
  zdens <- t(g$z * g$f) / sd
  C <- .count_distribution(g$F)
  Jm <- matrix(0, n, n)
  Js <- matrix(0, n, n)
  for (i in 1:n) {
    Qi <- .loo_pmf(C, g$F, i)
    pair <- .pair_pmf_at(Qi, g$F, i, k)
    kern <- pair * matrix(dens[i, ], n, ncol(pair), byrow = TRUE)
    row_mu <- rowSums(kern * dens) * g$dx
    row_sd <- rowSums(kern * zdens) * g$dx
    row_mu[i] <- 0
    row_mu[i] <- -sum(row_mu)
    cdf_i <- rowSums(Qi[, seq_len(k), drop = FALSE])
    dfdsd <- -(g$z[, i] * g$fp[, i] + g$f[, i]) / Dv[i]
    row_sd[i] <- sum(dfdsd * cdf_i) * g$dx
    Jm[i, ] <- row_mu
    Js[i, ] <- row_sd
  }
  list(Jmu = Jm, Jsigma = Js)
}

rank_probabilities <- function(mu, D = NULL, base = "normal",
                               points = 513) {
  n <- length(mu)
  sd <- sqrt(if (is.null(D)) rep(1, n) else D)
  fn <- if (is.function(base)) base else .BASES[[base]]
  g <- .topk_grid(mu, sd, n - 1, fn, points)
  C <- .count_distribution(g$F)
  P <- matrix(0, n, n)
  for (i in 1:n) {
    Qi <- .loo_pmf(C, g$F, i)
    P[i, ] <- colSums(Qi * (g$f[, i] / sd[i])) * g$dx
  }
  rows <- rowSums(P)
  cols <- colSums(P)
  if (any(!is.finite(P)) || max(abs(rows - 1)) > 5e-3 ||
      max(abs(cols - 1)) > 5e-3)
    stop("rank marginals defective; raise points=")
  .clip01(P / rows)
}

.validated_topk_target <- function(q, k, n, target_floor) {
  target <- as.numeric(q)
  if (length(target) != n)
    stop(sprintf("target has %d entries for %d runners", length(target), n))
  floored <- rep(FALSE, n)
  if (!is.null(target_floor)) {
    if (!(target_floor > 0)) stop("target_floor must be positive")
    floored <- target < target_floor
    target <- pmax(target, target_floor)
  } else if (any(target <= 0)) {
    stop(paste0(
      "all target memberships must be positive: a zero top-k probability ",
      "has no finite inverse. Pass target_floor= to floor small entries ",
      "deliberately."))
  }
  target <- target * (k / sum(target))
  if (any(target >= 1))
    stop(paste0(
      "after renormalizing to k slots, a target membership is >= 1: ",
      "certain membership has no finite inverse."))
  list(target = target, floored = floored)
}

abilities_from_topk <- function(q, k, V = NULL, D = NULL, base = "normal",
                                points = 513, qa = 15, n_iter = 80,
                                tol = 1e-8, target_floor = NULL,
                                return_info = FALSE) {
  n <- length(q)
  k <- as.integer(k)
  if (k < 1 || k > n - 1)
    stop(sprintf("k must be in [1, n-1]; got k=%d, n=%d", k, n))
  vt <- .validated_topk_target(q, k, n, target_floor)
  target <- vt$target
  sd <- sqrt(if (is.null(D)) rep(1, n) else D)
  fn <- if (is.function(base)) base else .BASES[[base]]
  fac <- if (is.null(V)) NULL else .topk_factor_nodes(V, n, qa)

  logit_t <- log(target) - log1p(-target)
  logt <- log(target)
  mu <- -(logt - mean(logt)) / 2
  alpha <- if (n > 2) 1.0 else 0.7
  resid_max <- Inf
  iters <- 0
  for (it in 1:n_iter) {
    iters <- it
    if (is.null(fac)) {
      ws <- .topk_with_slopes(mu, sd, k, fn, points)
      qraw <- ws$q
      sl <- ws$slopes
    } else {
      qraw <- rep(0, n)
      sl <- rep(0, n)
      for (j in seq_len(nrow(fac$nodes))) {
        shift <- as.vector(fac$Vm %*% fac$nodes[j, ])
        ws <- .topk_with_slopes(mu + shift, sd, k, fn, points)
        qraw <- qraw + fac$w[j] * ws$q
        sl <- sl + fac$w[j] * ws$slopes
      }
    }
    qhat <- .checked_topk(qraw, k, "top-k inversion")
    resid <- (log(pmax(qhat, 1e-300)) - log(pmax(1 - qhat, 1e-300))) -
      logit_t
    resid_max <- max(abs(resid))
    if (resid_max < tol) break
    dlogit <- pmin(sl / pmax(qhat * (1 - qhat), 1e-300), -1e-6)
    lim <- pmin(2, 10 * abs(resid))
    mu <- mu - pmin(pmax(alpha * resid / dlogit, -lim), lim)
    mu <- mu - mean(mu)
  }
  converged <- resid_max < tol
  if (!converged && !return_info)
    warning(sprintf(paste0(
      "abilities_from_topk did not converge: max |logit residual| %.2e ",
      "after %d iterations (tol %.0e)"), resid_max, iters, tol))
  if (return_info)
    return(list(mu = mu, converged = converged,
                max_logit_residual = resid_max, iterations = iters,
                floored = vt$floored))
  mu
}

loc_scale_from_topk_pair <- function(q1, k1, q2, k2, D0 = NULL,
                                     base = "normal", points = 513,
                                     n_iter = 60, tol = 1e-8, ridge = 0,
                                     mu0 = NULL, return_info = FALSE) {
  n <- length(q1)
  k1 <- as.integer(k1); k2 <- as.integer(k2)
  if (k1 == k2)
    stop("k1 == k2 gives one curve twice: scale is unidentified")
  for (kk in c(k1, k2)) if (kk < 1 || kk > n - 1)
    stop(sprintf("k must be in [1, n-1]; got k=%d, n=%d", kk, n))
  t1 <- .validated_topk_target(q1, k1, n, NULL)$target
  t2 <- .validated_topk_target(q2, k2, n, NULL)$target
  lt1 <- log(t1) - log1p(-t1)
  lt2 <- log(t2) - log1p(-t2)

  sd <- if (is.null(D0)) rep(1, n) else sqrt(D0)
  if (!is.null(mu0)) {
    mu <- as.numeric(mu0) - mean(mu0)
  } else {
    if (k1 < k2) { ka <- k1; ta <- t1 } else { ka <- k2; ta <- t2 }
    # warm start only: the LM loop refines, loose tolerance by design
    mu <- abilities_from_topk(ta, ka, D = sd^2, base = base,
                              points = points, n_iter = 20, tol = 1e-3,
                              return_info = TRUE)$mu
  }
  sqr <- sqrt(max(ridge, 0))

  logits <- function(m, s) {
    qh1 <- pmin(pmax(top_k_probabilities(m, k1, D = s^2, base = base,
                                         points = points), 1e-300),
                1 - 1e-15)
    qh2 <- pmin(pmax(top_k_probabilities(m, k2, D = s^2, base = base,
                                         points = points), 1e-300),
                1 - 1e-15)
    list(r = c(log(qh1) - log1p(-qh1) - lt1,
               log(qh2) - log1p(-qh2) - lt2,
               sqr * log(s)),
         qh1 = qh1, qh2 = qh2)
  }

  lg <- logits(mu, sd)
  r <- lg$r; qh1 <- lg$qh1; qh2 <- lg$qh2
  cost <- sum(r * r)
  resid_max <- max(abs(r[1:(2 * n)]))
  lam <- 1e-6
  iters <- 0
  last_accepted <- TRUE
  for (it in 1:n_iter) {
    iters <- it
    if (resid_max < tol) break
    blocks <- list()
    pairs <- list(list(k = k1, qh = qh1), list(k = k2, qh = qh2))
    for (bi in 1:2) {
      Jp <- top_k_jacobians(mu, pairs[[bi]]$k, D = sd^2, base = base,
                            points = points)
      g <- 1 / pmax(pairs[[bi]]$qh * (1 - pairs[[bi]]$qh), 1e-300)
      blocks[[bi]] <- cbind(Jp$Jmu * g,
                            sweep(Jp$Jsigma, 2, sd, "*") * g)
    }
    J <- rbind(blocks[[1]], blocks[[2]],
               cbind(matrix(0, n, n), sqr * diag(n)))
    JtJ <- crossprod(J)
    Jtr <- as.vector(crossprod(J, r))
    accepted <- FALSE
    for (attempt in 1:8) {
      step <- tryCatch(solve(JtJ + lam * diag(2 * n), -Jtr),
                       error = function(e) NULL)
      if (is.null(step)) { lam <- lam * 8; next }
      mu_n <- mu + step[1:n]
      ls_n <- pmin(pmax(log(sd) + step[(n + 1):(2 * n)], -3), 3)
      cc <- exp(mean(ls_n))
      sd_n <- exp(ls_n - mean(ls_n))
      mu_n <- (mu_n - mean(mu_n)) / cc
      lg_n <- tryCatch(logits(mu_n, sd_n), error = function(e) NULL)
      if (is.null(lg_n)) { lam <- lam * 8; next }
      cost_n <- sum(lg_n$r * lg_n$r)
      if (cost_n < cost) {
        mu <- mu_n; sd <- sd_n; r <- lg_n$r
        qh1 <- lg_n$qh1; qh2 <- lg_n$qh2
        cost <- cost_n
        resid_max <- max(abs(r[1:(2 * n)]))
        lam <- max(lam / 3, 1e-10)
        accepted <- TRUE
        break
      }
      lam <- lam * 8
    }
    last_accepted <- accepted
    if (!accepted) break
  }
  # with a ridge the penalized optimum keeps a nonzero fit residual by
  # design: an LM stall there is the answer, not a failure
  converged <- resid_max < tol || (sqr > 0 && !last_accepted)
  if (!converged && !return_info)
    warning(sprintf(paste0(
      "loc_scale_from_topk_pair did not converge: max |logit residual| ",
      "%.2e after %d iterations (tol %.0e)"), resid_max, iters, tol))
  if (return_info)
    return(list(mu = mu, sd = sd, converged = converged,
                max_logit_residual = resid_max, iterations = iters))
  list(mu = mu, sd = sd)
}

loc_scale_from_win_and_second <- function(p_win, p_second, D0 = NULL,
                                          base = "normal", points = 513,
                                          n_iter = 60, tol = 1e-8,
                                          ridge = 0, mu0 = NULL,
                                          return_info = FALSE) {
  # win plus EXACTLY-second marginals: P(2nd) + P(win) = P(top-2), the
  # well-posed pair. Each marginal renormalized to unit mass first.
  n <- length(p_win)
  if (length(p_second) != n)
    stop("p_win and p_second must have equal length")
  if (any(p_win <= 0) || any(p_second <= 0))
    stop("all win and second probabilities must be positive")
  p1 <- p_win / sum(p_win)
  top2 <- p1 + p_second / sum(p_second)
  loc_scale_from_topk_pair(p1, 1, top2, 2, D0 = D0, base = base,
                           points = points, n_iter = n_iter, tol = tol,
                           ridge = ridge, mu0 = mu0,
                           return_info = return_info)
}

.rank_marginal_with_jacobian <- function(mu, sd, r, fn, points) {
  n <- length(mu)
  g <- .topk_grid(mu, sd, n - 1, fn, points)
  dens <- t(g$f) / sd                       # (n, L)
  C <- .count_distribution(g$F)
  p <- numeric(n)
  J <- matrix(0, n, n)
  for (i in 1:n) {
    Qi <- .loo_pmf(C, g$F, i)
    p[i] <- sum(Qi[, r] * dens[i, ]) * g$dx
    pair <- .pair_pmf_at(Qi, g$F, i, r)
    if (r >= 2) pair <- pair - .pair_pmf_at(Qi, g$F, i, r - 1)
    kern <- pair * matrix(dens[i, ], n, ncol(pair), byrow = TRUE)
    row <- rowSums(kern * dens) * g$dx
    row[i] <- 0
    row[i] <- -sum(row)
    J[i, ] <- row
  }
  list(p = p, J = J)
}

abilities_from_rank_marginal <- function(p, r, mu0 = NULL, D = NULL,
                                         base = "normal", points = 513,
                                         n_iter = 60, tol = 1e-8,
                                         return_info = FALSE) {
  # invert one EXACT-rank marginal at frozen scales: two-branched for
  # r >= 2, mu0 selects the branch. See the python docstring.
  n <- length(p)
  r <- as.integer(r)
  if (r < 1 || r > n)
    stop(sprintf("rank must be in [1, n]; got r=%d, n=%d", r, n))
  if (any(p <= 0))
    stop("all rank probabilities must be positive")
  logt <- log(p / sum(p))
  sd <- sqrt(if (is.null(D)) rep(1, n) else D)
  fn <- if (is.function(base)) base else .BASES[[base]]
  mu <- if (is.null(mu0)) rep(0, n) else as.numeric(mu0) - mean(mu0)

  st <- .rank_marginal_with_jacobian(mu, sd, r, fn, points)
  resid <- log(pmax(st$p, 1e-300)) - logt
  cost <- sum(resid * resid)
  resid_max <- max(abs(resid))
  lam <- 1e-6
  iters <- 0
  for (it in 1:n_iter) {
    iters <- it
    if (resid_max < tol) break
    Jlog <- st$J / pmax(st$p, 1e-300)
    A <- crossprod(Jlog)
    gvec <- as.vector(crossprod(Jlog, resid))
    accepted <- FALSE
    for (attempt in 1:8) {
      step <- tryCatch(solve(A + lam * diag(n), -gvec),
                       error = function(e) NULL)
      if (is.null(step)) { lam <- lam * 8; next }
      mu_n <- mu + step
      mu_n <- mu_n - mean(mu_n)
      st_n <- .rank_marginal_with_jacobian(mu_n, sd, r, fn, points)
      r_n <- log(pmax(st_n$p, 1e-300)) - logt
      cost_n <- sum(r_n * r_n)
      if (cost_n < cost) {
        mu <- mu_n; st <- st_n; resid <- r_n; cost <- cost_n
        resid_max <- max(abs(resid))
        lam <- max(lam / 3, 1e-10)
        accepted <- TRUE
        break
      }
      lam <- lam * 8
    }
    if (!accepted) break
  }
  converged <- resid_max < tol
  if (!converged && !return_info)
    warning(sprintf(paste0(
      "abilities_from_rank_marginal did not converge: max |log ",
      "residual| %.2e after %d iterations (tol %.0e)"),
      resid_max, iters, tol))
  if (return_info)
    return(list(mu = mu, converged = converged,
                max_log_residual = resid_max, iterations = iters))
  mu
}
