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
  if (length(Q) != 1L || !is.numeric(Q) || !is.finite(Q) || Q < 1 ||
      Q != round(Q))
    stop(sprintf("a Gauss-Hermite order must be one positive integer; got %s",
                 paste(format(Q), collapse = ", ")), call. = FALSE)
  Q <- as.integer(Q)
  J <- diag(0, Q)
  # i and i + 1, not 2:Q: at Q = 1, 2:Q is c(2, 1), not empty, and the
  # one-node rule (x = 0, w = 1) died on an out-of-bounds index (#504)
  i <- seq_len(Q - 1L)
  off <- sqrt(i)
  J[cbind(i, i + 1L)] <- off
  J[cbind(i + 1L, i)] <- off
  e <- eigen(J, symmetric = TRUE)
  list(x = e$values, w = e$vectors[1, ]^2)
}

# log of phi(a)/Phi(a), the inverse Mills ratio. pnorm(log.p = TRUE),
# never log(pmax(pnorm(a), 1e-300)): pnorm underflows to exactly 0
# below about -37, so the floor turned every deep tail into the SAME
# number, log(1e-300) = -690.78, the objective went flat and this ratio
# underflowed to a score of exactly zero (#270).
.log_mills <- function(a) dnorm(a, log = TRUE) - pnorm(a, log.p = TRUE)

# Own-noise order when the caller gives none: 7 below J = 6, 11 from
# J = 6, where the product of J - 1 rival CDFs is sharper than seven nodes
# resolve (python: winning.likelihood.default_qz, #173).
.default_qz <- function(J) if (J >= 6L) 11L else 7L

# The chosen alternative's own-noise nodes are tilted per observation
# (.tilt_mode, below): the fixed Gauss-Hermite rule sampled z where the
# PRIOR has its mass, and for a large observed contrast the integrand's
# mass is far outside it -- at a gap of -20 the binary likelihood was
# -143.2 against the exact -103.6 (#270).
.nodes3 <- function(Qf = 11, Qz = 11, r = 2) {
  g <- .gh1(Qf); gz <- .gh1(Qz)
  grids <- do.call(expand.grid, c(rep(list(g$x), r), list(gz$x)))
  wg <- do.call(expand.grid, c(rep(list(g$w), r), list(gz$w)))
  W <- apply(as.matrix(wg), 1, prod)
  keep <- W > 1e-10 * max(W)
  list(F = as.matrix(grids)[keep, , drop = FALSE], W = W[keep] / sum(W[keep]))
}

# All permutations of 1..r, lexicographic.
.permutations <- function(r) {
  if (r == 0L) return(list(integer(0)))
  out <- list()
  for (i in seq_len(r)) for (p in .permutations(r - 1L)) {
    rest <- setdiff(seq_len(r), i)
    out[[length(out) + 1L]] <- c(i, rest[p])
  }
  out
}

# Factor orientation is arbitrary: V and V Q price the same law for any
# orthogonal Q, in particular a column sign flip or permutation. The
# Gauss-Hermite tensor is invariant under both; a Sobol cloud is not, so
# the sharp-branch probability moved when one loading column changed
# sign (#605). The cloud is the orbit of base points under the signed
# permutations of the factor coordinates (all 2^r r! while that is at
# most 384, i.e. r <= 4; the 2^r sign flips beyond), at about the same
# node count. General rotations are not covered.
.factor_symmetries <- function(r, n) {
  if (r == 0L) return(list(perms = list(integer(0)), signs = list(numeric(0))))
  signs <- lapply(0:(2^r - 1), function(s)
    ifelse(bitwAnd(s, 2^(seq_len(r) - 1L)) == 0, 1, -1))
  full <- 2^r * factorial(r)
  if (full <= min(384, n))
    return(list(perms = .permutations(r), signs = signs))
  if (2^r <= n) return(list(perms = list(seq_len(r)), signs = signs))
  list(perms = list(seq_len(r)), signs = list(rep(1, r)))
}

# Equal-weight nodes over (factor^r, own noise) for the sharp regime:
# scrambled Sobol in r + 1 dimensions (R/sobol.R), symmetrised over the
# factor signs and permutations (#605). It was a four-prime Halton
# table, so r = 4 -- the full triangular rank at five alternatives --
# indexed an NA fifth base and every rank-4 fit died in setup, even on
# the Gauss-Hermite branch (#388).
.sobol_nodes3 <- function(r, m = 10L) {
  n <- 2L^m
  sym <- .factor_symmetries(r, n)
  ord <- length(sym$perms) * length(sym$signs)
  nb <- as.integer(ceiling(n / ord))
  base <- .sobol_normal(r + 1L, nb, seed = 0L)$F
  blocks <- list()
  for (pm in sym$perms) for (sg in sym$signs) {
    blk <- base
    if (r > 0L)
      blk[, seq_len(r)] <- base[, pm, drop = FALSE] *
        matrix(sg, nb, r, byrow = TRUE)
    blocks[[length(blocks) + 1L]] <- blk
  }
  F <- do.call(rbind, blocks)
  list(F = F, W = rep(1 / nrow(F), nrow(F)))
}

# The node-family dispatch statistic, pairwise-safe and gauge-fixed, as
# python's winning.likelihood.sharpness_bound and julia's: only loading
# DIFFERENCES decide a choice, so centre V first, then sqrt(2) max_i
# ||(PV)_i|| bounds max_ij ||V_i - V_j|| / sqrt(2) (unit idiosyncratic
# variance here, so no D divisor). The raw max ||V_i|| was the pre-#213
# statistic: anchored rows 1 and -2.7 have norm <= 2.7 but differ by 3.7,
# so a fit was priced on the 7-point Hermite tensor 1.37 points high where
# python and julia escalate (#419).
.mnp_sharpness <- function(V) {
  V <- as.matrix(V)
  Vc <- sweep(V, 2L, colMeans(V), "-")
  sqrt(2) * max(sqrt(rowSums(Vc^2)))
}

# The blend's statistic must be differentiable and symmetric in the
# alternatives: the max over row norms has a cusp at ties. A p-norm of
# the centred row norms (p = 32) is smooth, permutation symmetric, at
# least the max and at most J^(1/p) above it -- python's
# likelihood._smooth_sharpness. Returns the value and its V-gradient.
.mnp_smooth_sharpness <- function(V) {
  V <- as.matrix(V)
  if (ncol(V) == 0L) return(list(value = 0, grad = V))
  C <- sweep(V, 2L, colMeans(V), "-")
  n2 <- rowSums(C^2)
  M <- max(n2)
  if (!(M > 0)) return(list(value = 0, grad = V * 0))
  p <- 32
  u <- n2 / M
  Su <- sum(u^(p / 2))
  sn <- sqrt(M) * Su^(1 / p)
  dn2 <- 0.5 * sn * u^(p / 2 - 1) / (M * Su)
  G <- 2 * dn2 * C
  G <- sweep(G, 2L, colMeans(G), "-")
  list(value = sqrt(2) * sn, grad = sqrt(2) * G)
}

# Across the switch from the Hermite tensor to the Sobol cloud the
# likelihood is the probability-weighted blend (1 - w) P_GH + w P_Sobol
# with a C1 smoothstep w over sharpness [2.9, 3.1]; the hard switch at 3
# moved the binary #420 fixture's log-likelihood by 0.065.
.blend_weight <- function(sharp) {
  lo <- 2.9; hi <- 3.1
  t <- (sharp - lo) / (hi - lo)
  if (t <= 0) return(c(w = 0, dw = 0))
  if (t >= 1) return(c(w = 1, dw = 0))
  c(w = t * t * (3 - 2 * t), dw = 6 * t * (1 - t) / (hi - lo))
}

# Lazily built sharp-regime rule: constructed on first use only.
.lazy_sharp_nodes <- function(r, m = 10L) {
  cache <- NULL
  function() {
    if (is.null(cache)) cache <<- .sobol_nodes3(r, m)
    cache
  }
}

# One chosen row per observation, keyed by observation. The wrappers
# used to flatten every truthy row with which() and hand the result to
# the core positionally, so an observation with two choices cancelled
# one with none: the total was still T, the length check passed, and
# observation 1's second choice became observation 2's (#324, #435).
# `ids` are integer codes 1..T and the rows are already sorted by
# (ids, alt); `labels` maps codes back to the caller's ids for messages.
.choices_by_id <- function(ids, alt, chosen, labels = NULL) {
  y <- chosen
  if (is.factor(y)) y <- as.character(y)
  if (is.logical(y)) {
    v <- y
  } else if (is.numeric(y)) {
    if (any(!is.na(y) & !(y %in% c(0, 1))))
      stop(sprintf(paste("the chosen indicator must be logical or 0/1;",
                         "got %s"), format(y[which(!is.na(y) &
                                                   !(y %in% c(0, 1)))[1]])),
           call. = FALSE)
    v <- y == 1
  } else {
    # labels: "TRUE"/"FALSE" (and spellings as.logical knows) or "0"/"1",
    # which is how a factor response arrives (#512)
    v <- as.logical(y)
    v[!is.na(y) & y == "1"] <- TRUE
    v[!is.na(y) & y == "0"] <- FALSE
    if (any(is.na(v) & !is.na(y)))
      stop("the chosen indicator must be logical or 0/1", call. = FALSE)
  }
  if (anyNA(v)) {
    i <- which(is.na(v))[1]
    stop(sprintf(paste("the chosen indicator is missing for observation",
                       "'%s'; drop that observation deliberately"),
                 if (is.null(labels)) ids[i] else labels[ids[i]]),
         call. = FALSE)
  }
  Tn <- max(ids)
  cnt <- tabulate(ids[v], nbins = Tn)
  bad <- which(cnt != 1L)
  if (length(bad))
    stop(sprintf(paste("each observation must choose exactly one",
                       "alternative; observation '%s' has %d chosen rows",
                       "(%d of %d observations are malformed)"),
                 if (is.null(labels)) bad[1] else labels[bad[1]],
                 cnt[bad[1]], length(bad), Tn), call. = FALSE)
  choice <- integer(Tn)
  choice[ids[v]] <- alt[v]
  choice
}

# Largest factor rank the unit-idiosyncratic parameterization
# identifies. With D fixed to one and a zero reference row the free
# loadings number r*J - r*(r+1)/2 while the differenced covariance has
# J*(J-1)/2 - 1 shape degrees of freedom; at r >= J - 1 the loading
# block is full rank, D = 1 no longer fixes the utility scale, and
# (2 beta, V') prices every choice like (beta, V). The old default
# r = 2 was nonidentified at J = 3 (#201).
.check_rank <- function(r, J) {
  rmax <- max(J - 2L, 0L)
  if (is.null(r)) return(as.integer(min(2L, rmax)))
  r <- .check_count(r, "factor rank r")
  if (r > rmax)
    stop(sprintf(paste0("factor rank r = %s is not identified at J = %d ",
                        "alternatives: with unit idiosyncratic variances ",
                        "the utility scale is fixed only for ",
                        "r <= J - 2 = %d (#201)"),
                 paste(r, collapse = ","), J, rmax), call. = FALSE)
  r
}


# The own-noise tilt (#128, #270): each observation's own-noise nodes move
# to the mode m of log h(z), h(z) = phi(z) prod_j Phi(c_j + z), and scale
# with its curvature, z = m + sigma x; the change of measure rides in the
# weights (log sigma - z^2/2 + x^2/2), so the rule integrates the same h.
# The mode is taken with the factors integrated out in the Gaussian
# approximation (rival noise widened to t_j^2 = 1 + |v_k - v_j|^2). m is
# a FIXED smooth map -- four Newton steps on a Mills-ratio surrogate
# (lambda ~ (sqrt(a^2 + 4) - a) / 2), then two exact ones -- with its
# derivatives carried along, so the analytic score is the exact gradient
# of the returned value. Port of winning/likelihood.py::_tilt_mode (unit
# own-noise variance here). c0: R x n contrasts mu_k - mu_j; tv: the R
# effective rival sds.
.tilt_mode <- function(c0, tv) {
  R <- nrow(c0); n <- ncol(c0)
  m <- numeric(n)
  dm_c <- matrix(0, R, n); dm_t <- matrix(0, R, n)
  rho <- rep(1 / tv, n)                     # s_k / t_j, R x n layout
  for (it in 1:6) {
    a <- (c0 + rep(m, each = R)) * rho
    if (it <= 4L) {
      q <- sqrt(a * a + 4)
      lam <- 0.5 * (q - a)
      kap <- 0.5 * (1 - a / q)              # -dlam/da
      kp <- -2 / q^3                        # dkap/da
    } else {
      lam <- exp(dnorm(a, log = TRUE) - pnorm(a, log.p = TRUE))
      kap <- lam * (a + lam)
      kp <- lam * (1 - kap) - kap * (a + lam)
    }
    g1 <- -m + colSums(rho * lam)
    h <- 1 + colSums(rho^2 * kap)
    S3 <- rep(colSums(rho^3 * kp), each = R)
    hR <- rep(h, each = R)
    dg1_c <- -(rho * kap) * rho - hR * dm_c
    dh_c <- rho^2 * kp * rho + S3 * dm_c
    dg1_t <- rho * (kap * a - lam) * rho - hR * dm_t
    dh_t <- -rho^2 * (2 * kap + kp * a) * rho + S3 * dm_t
    sc <- rep(g1 / h^2, each = R)
    dm_c <- dm_c + dg1_c / hR - sc * dh_c
    dm_t <- dm_t + dg1_t / hR - sc * dh_t
    m <- m + g1 / h
  }
  list(m = m, h = h, dm_c = dm_c, dh_c = dh_c, dm_t = dm_t, dh_t = dh_t)
}

# Per-observation log P(choice) with per-observation scores on one node
# set: logp (Tn), dmu (Tn x J), GV (Tn x J x r). `tilt` moves the
# own-noise nodes (the Gauss-Hermite branch only: on a paired Sobol cloud
# the change-of-measure weight is unbounded and QMC error grew tenfold).
.mnp_terms <- function(mu, V, choice, nd, tilt = TRUE, want_grad = TRUE) {
  Tn <- nrow(mu); J <- ncol(mu); r <- ncol(V)
  Fq <- nd$F[, seq_len(r), drop = FALSE]
  xq <- nd$F[, r + 1L]
  lW <- log(nd$W)
  Q <- length(lW)
  Vf <- Fq %*% t(V)
  logp <- numeric(Tn)
  dmu <- matrix(0, Tn, J)
  GV <- array(0, c(Tn, J, r))
  for (k_alt in seq_len(J)) {
    idx <- which(choice == k_alt)
    if (!length(idx)) next
    Ti <- length(idx)
    rivals <- setdiff(seq_len(J), k_alt)
    R <- length(rivals)
    c0 <- t(mu[idx, k_alt] - mu[idx, rivals, drop = FALSE])     # R x Ti
    dk <- matrix(V[k_alt, ], R, r, byrow = TRUE) -
      V[rivals, , drop = FALSE]
    if (tilt) {
      tv <- sqrt(1 + rowSums(dk^2))
      tm <- .tilt_mode(c0, tv)
      sig <- 1 / sqrt(tm$h)
      ds_c <- -0.5 * rep(sig / tm$h, each = R) * tm$dh_c
      ds_t <- -0.5 * rep(sig / tm$h, each = R) * tm$dh_t
      z <- outer(tm$m, rep(1, Q)) + outer(sig, xq)              # Ti x Q
      acc <- matrix(lW + 0.5 * xq^2, Ti, Q, byrow = TRUE) - 0.5 * z^2
    } else {
      sig <- rep(1, Ti)
      z <- matrix(xq, Ti, Q, byrow = TRUE)
      acc <- matrix(lW, Ti, Q, byrow = TRUE)
    }
    lam <- vector("list", R)
    for (i in seq_len(R)) {
      j <- rivals[i]
      A <- outer(c0[i, ], Vf[, k_alt] - Vf[, j], "+") + z
      lP <- pnorm(A, log.p = TRUE)
      acc <- acc + lP
      if (want_grad) lam[[i]] <- exp(dnorm(A, log = TRUE) - lP)
    }
    mx <- acc[cbind(seq_len(Ti), max.col(acc, ties.method = "first"))]
    pw <- exp(acc - mx)
    rs <- rowSums(pw)
    logp[idx] <- log(sig) + mx + log(pmax(rs, 1e-300))
    if (!want_grad) next
    omega <- pw / rs
    if (tilt) {
      gp <- -z
      for (i in seq_len(R)) gp <- gp + lam[[i]]
      Eg <- rowSums(omega * gp)
      Egx <- rowSums(omega * gp * rep(xq, each = Ti))
    }
    for (i in seq_len(R)) {
      j <- rivals[i]
      wl <- omega * lam[[i]]
      dc <- rowSums(wl)
      H <- wl %*% Fq                                             # Ti x r
      if (tilt) {
        dc <- dc + ds_c[i, ] / sig + Eg * tm$dm_c[i, ] + Egx * ds_c[i, ]
        dt <- ds_t[i, ] / sig + Eg * tm$dm_t[i, ] + Egx * ds_t[i, ]
        if (r > 0L) H <- H + outer(dt / tv[i], dk[i, ])
      }
      dmu[idx, k_alt] <- dmu[idx, k_alt] + dc
      dmu[idx, j] <- dmu[idx, j] - dc
      if (r > 0L) {
        GV[idx, k_alt, ] <- GV[idx, k_alt, ] + H
        GV[idx, j, ] <- GV[idx, j, ] - H
      }
    }
  }
  list(logp = logp, dmu = dmu, GV = GV)
}

# J = 2 in closed form: P(k) = Phi(c / s), c = mu_k - mu_j,
# s^2 = 2 + |v_k - v_j|^2, with its analytic score (python's
# likelihood._choice_pair at unit D).
.mnp_pair <- function(mu, V, choice) {
  Tn <- nrow(mu); r <- ncol(V)
  rows <- seq_len(Tn)
  k <- choice; j <- 3L - choice
  cc <- mu[cbind(rows, k)] - mu[cbind(rows, j)]
  d <- V[k, , drop = FALSE] - V[j, , drop = FALSE]
  s <- sqrt(2 + rowSums(d^2))
  zz <- cc / s
  logp <- pnorm(zz, log.p = TRUE)
  lam <- exp(dnorm(zz, log = TRUE) - logp)
  dmu <- matrix(0, Tn, 2L)
  dmu[cbind(rows, k)] <- lam / s
  dmu[cbind(rows, j)] <- -lam / s
  GV <- array(0, c(Tn, 2L, r))
  if (r > 0L) {
    gd <- (-lam * cc / s^3) * d
    for (cl in seq_len(r)) {
      GV[cbind(rows, k, cl)] <- gd[, cl]
      GV[cbind(rows, j, cl)] <- -gd[, cl]
    }
  }
  list(logp = logp, dmu = dmu, GV = GV)
}

# Negative log-likelihood AND analytic score. J = 2 is the closed form;
# J >= 3 integrates the factors and the chosen alternative's own noise on
# the Gauss-Hermite tensor with tilted own-noise nodes, or past sharpness
# 3 a symmetrised Sobol cloud, blended continuously across the switch
# (#420). `nodes_sharp` may be a node list or a zero-argument function
# returning one, so a caller can build the sharp rule only when it is
# selected. `offset` is a fixed, coefficient-one addition to each row's
# utility (long format, one entry per row of X, #641). The result also
# carries the per-observation log-probabilities `logp`.
.nll_core <- function(theta, Xb, choice, J, r, nodes, nodes_sharp,
                      want_grad = TRUE, offset = NULL) {
  X <- Xb[[1]]
  # Every observation must be accounted for. The loop below walks the
  # LEGAL labels and gathers the rows matching each, so a row whose
  # choice is outside 1..J is never visited: logp stays 0 there, which
  # adds zero negative log-likelihood and zero gradient, and the fit
  # silently optimised a SUBSET while reporting it as the whole. NA rows
  # did the same (#194).
  # X is stacked long, J rows per observation, so the observation count
  # is nrow(X) / J -- the same Tn the loop below uses.
  .Tn <- nrow(X) / J
  if (length(choice) != .Tn)
    stop(sprintf(paste0("choice must have one entry per observation: ",
                        "got %d for %d observations"),
                 length(choice), .Tn), call. = FALSE)
  # A fractional label such as 1.5 is in range but matches no
  # `choice == k_alt` bucket, so it was dropped exactly like an
  # out-of-range one; integral doubles such as 1.0 are fine (#194).
  bad <- which(is.na(choice) | !is.finite(choice) | choice < 1L |
                 choice > J | choice != round(choice))
  if (length(bad))
    stop(sprintf(paste0("choice[%d] = %s is not an alternative in 1..%d; ",
                        "%d of %d observations are not. They would be ",
                        "dropped in silence, which raises the ",
                        "log-likelihood because there is less of it"),
                 bad[1], as.character(choice[bad[1]]), J, length(bad),
                 length(choice)), call. = FALSE)
  nb <- ncol(X)
  beta <- theta[seq_len(nb)]
  wfree <- theta[-seq_len(nb)]
  V <- matrix(0, J, r)
  k <- 1L
  fill <- list()
  # seq_len, not `(col + 1L):J`: at J = 2 and the default r = 2L the
  # col = 2 pass evaluates 3:2, which in R is the two-element vector
  # c(3, 2) rather than empty, so the first assignment is V[3, 2] on a
  # 2x2 matrix and every binary-choice fit died out of bounds (#183).
  # nw counts only one free loading there, so wfree[2] was out of range
  # too. seq_len(J - col) is empty exactly when it should be.
  for (col in seq_len(r)) for (row in col + seq_len(J - col)) {
    V[row, col] <- wfree[k]; fill[[k]] <- c(row, col); k <- k + 1L
  }
  Tn <- nrow(X) / J
  eta <- X %*% beta
  if (!is.null(offset)) eta <- eta + offset
  mu <- matrix(eta, nrow = Tn, ncol = J, byrow = TRUE)
  choice <- as.integer(choice)
  sgrad <- NULL
  if (J == 2L) {
    tr <- .mnp_pair(mu, V, choice)
  } else {
    sm <- .mnp_smooth_sharpness(V)
    bw <- .blend_weight(sm$value)
    sharp_nd <- function()
      if (is.function(nodes_sharp)) nodes_sharp() else nodes_sharp
    if (bw[["w"]] == 0) {
      tr <- .mnp_terms(mu, V, choice, nodes, TRUE, want_grad)
    } else if (bw[["w"]] == 1) {
      tr <- .mnp_terms(mu, V, choice, sharp_nd(), FALSE, want_grad)
    } else {
      w <- bw[["w"]]
      tg <- .mnp_terms(mu, V, choice, nodes, TRUE, want_grad)
      ts <- .mnp_terms(mu, V, choice, sharp_nd(), FALSE, want_grad)
      x1 <- log1p(-w) + tg$logp
      x2 <- log(w) + ts$logp
      mx <- pmax(x1, x2)
      lp <- mx + log(exp(x1 - mx) + exp(x2 - mx))
      a <- exp(x1 - lp); b <- exp(x2 - lp)
      tr <- list(logp = lp, dmu = a * tg$dmu + b * ts$dmu,
                 GV = a * tg$GV + b * ts$GV)
      # d logp / dV through w: (P_S - P_GH) / P * dw/dsharp * dsharp/dV
      gw <- (exp(ts$logp - lp) - exp(tg$logp - lp)) * bw[["dw"]]
      sgrad <- list(gw = gw, G = sm$grad)
    }
  }
  logp <- tr$logp
  val <- -sum(logp)
  if (!want_grad) return(list(value = val, grad = NULL, logp = logp))
  gbeta <- colSums(X * as.vector(t(tr$dmu)))
  gV <- if (r > 0L) apply(tr$GV, c(2L, 3L), sum) else matrix(0, J, 0)
  if (!is.null(sgrad)) gV <- gV + sum(sgrad$gw) * sgrad$G
  gw <- vapply(fill, function(rc) gV[rc[1], rc[2]], 0)
  list(value = val, grad = -c(gbeta, gw), logp = logp)
}

# value-only wrapper (kept for tests and diagnostics)
.nll <- function(theta, Xb, choice, J, r, nodes, nodes_sharp) {
  .nll_core(theta, Xb, choice, J, r, nodes, nodes_sharp,
            want_grad = FALSE)$value
}

# Raw per-alternative quadrature masses at theta, Tn x J: the fit
# maximizes the raw mass of each observed choice, and these do not sum to
# one exactly; `mass_defect` in a fit is max |1 - rowSums| (#376).
.raw_probabilities <- function(theta, X, J, r, nodes, nodes_sharp,
                               offset = NULL) {
  Tn <- nrow(X) / J
  P <- matrix(0, Tn, J)
  for (k in seq_len(J))
    P[, k] <- exp(.nll_core(theta, list(X), rep(k, Tn), J, r, nodes,
                            nodes_sharp, want_grad = FALSE,
                            offset = offset)$logp)
  P
}

# max c'x s.t. A x <= b, x >= 0 (b >= 0) by the condensed-tableau simplex
# with Bland's rule, which terminates on the degenerate b = 0 rows the
# separation problem is made of. Returns list(x, value, ok).
.lp_max <- function(cvec, A, b, tol = 1e-12) {
  m <- nrow(A); n <- ncol(A)
  Tb <- rbind(cbind(A, b), c(-cvec, 0))
  basis <- n + seq_len(m); nonbasis <- seq_len(n)
  for (iter in seq_len(50L * (m + n))) {
    neg <- which(Tb[m + 1L, seq_len(n)] < -tol)
    if (!length(neg)) {
      x <- numeric(n + m)
      x[basis] <- Tb[seq_len(m), n + 1L]
      return(list(x = x[seq_len(n)], value = Tb[m + 1L, n + 1L], ok = TRUE))
    }
    e <- neg[which.min(nonbasis[neg])]
    cand <- which(Tb[seq_len(m), e] > tol)
    if (!length(cand)) return(list(x = numeric(n), value = Inf, ok = FALSE))
    ratio <- Tb[cand, n + 1L] / Tb[cand, e]
    best <- min(ratio)
    tie <- cand[abs(ratio - best) <= 1e-15]
    rr <- tie[which.min(basis[tie])]
    p <- Tb[rr, e]
    col <- Tb[, e]
    rowr <- Tb[rr, ]
    Tb <- Tb - outer(col / p, rowr)
    Tb[rr, ] <- rowr / p
    Tb[, e] <- -col / p
    Tb[rr, e] <- 1 / p
    tmp <- basis[rr]; basis[rr] <- nonbasis[e]; nonbasis[e] <- tmp
  }
  list(x = numeric(n), value = 0, ok = FALSE)
}

# A recession direction of the mean design, or NULL. Every choice
# probability rises with each of its own contrasts, so if some d gives
# every observed contrast (x_chosen - x_j)' d >= 0, with at least one
# strict, the likelihood rises along beta + s d for ever and has no
# finite maximizer (complete or quasi-complete separation); optim
# stopped where the score fell under its tolerance and reported
# convergence 0, with logLik 0 at a coefficient of 112.9 (#642). The LP
# max 1'(A d) s.t. A d >= 0, |d|_inf <= 1 on unit-scaled columns; python's
# mnprobit._separation_direction.
.separation_direction <- function(X, choice, J) {
  p <- ncol(X)
  Tn <- nrow(X) / J
  if (p == 0L || J < 2L) return(NULL)
  base <- (seq_len(Tn) - 1L) * J
  rows <- list()
  for (j in seq_len(J)) {
    sel <- which(choice != j)
    if (length(sel))
      rows[[length(rows) + 1L]] <- X[base[sel] + choice[sel], , drop = FALSE] -
        X[base[sel] + j, , drop = FALSE]
  }
  A <- do.call(rbind, rows)
  cs <- apply(abs(A), 2L, max)
  cs[!(cs > 0)] <- 1
  As <- sweep(A, 2L, cs, "/")
  g <- colSums(As)
  nr <- nrow(As)
  I <- diag(p)
  Alp <- rbind(cbind(-As, As), cbind(I, matrix(0, p, p)),
               cbind(matrix(0, p, p), I))
  sol <- .lp_max(c(g, -g), Alp, c(rep(0, nr), rep(1, 2L * p)))
  if (!sol$ok || !(sol$value > 1e-9)) return(NULL)
  d <- sol$x[seq_len(p)] - sol$x[p + seq_len(p)]
  marg <- drop(As %*% d)
  if (min(marg) < -1e-9 || max(marg) <= 1e-7) return(NULL)
  d <- d / cs
  d <- d / sqrt(sum(d^2))
  names(d) <- colnames(X)
  d
}

.warn_separated <- function(d) {
  warning(sprintf(paste("the mean design is separated: moving the",
                        "coefficients along (%s) never lowers any observed",
                        "choice probability and raises at least one, so the",
                        "likelihood has no finite maximizer; convergence is",
                        "reported as 2 and `separation` holds the",
                        "direction (#642)"),
                  paste(sprintf("%s = %.4g", names(d), d), collapse = ", ")),
          call. = FALSE)
}

# A choice needs at least two alternatives: with one, every probability
# is one, every coefficient unidentified, and rprobit_fast died on
# `for (j in 2:J)` with "subscript out of bounds" (#581).
.check_two_alternatives <- function(J) {
  if (!is.finite(J) || J < 2L)
    stop(sprintf(paste("a choice model needs at least two alternatives;",
                       "the data have %d. With one, every coefficient is",
                       "unidentified (#581)"), as.integer(J)), call. = FALSE)
  invisible(NULL)
}

# The packed lower-triangular loadings, named by alternative label and
# factor column, and the J x r loading matrix with the labels as row
# names: without them a releveled or dropped alternative silently
# changed what each parameter meant (#643).
.loading_labels <- function(par, J, r, labels) {
  V <- matrix(0, J, r, dimnames = list(labels, if (r) paste0("f", seq_len(r))))
  nm <- character(0)
  k <- 1L
  for (col in seq_len(r)) for (row in col + seq_len(J - col)) {
    V[row, col] <- par[k]
    nm <- c(nm, sprintf("v[%s,%d]", labels[row], col))
    k <- k + 1L
  }
  names(par) <- nm
  list(par = par, V = V)
}

# A finite, non-negative whole number. as.integer() floored r = 1.9 to
# rank 1 and fitted a different model than the one asked for (#479);
# optim took any maxit it was handed (#560).
.check_count <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x != round(x) || x < 0)
    stop(sprintf("%s must be a finite non-negative whole number; got %s",
                 name, paste(format(x), collapse = ",")), call. = FALSE)
  as.integer(x)
}

# Mean-design columns that within-choice-set contrasts do not identify:
# a covariate common to every alternative, or one duplicating a
# generated intercept, lies on an exact likelihood ridge, and BFGS
# reported an arbitrary point on it as converged (#581). X rows are in
# (observation, alternative) order; a column is aliased when its
# contrast residual on the earlier kept columns is below tol of its own
# contrast norm (Gram scale).
.check_identified <- function(X, Tn, J, tol = 1e-10) {
  p <- ncol(X)
  if (p == 0L || J < 2L) return(invisible(NULL))
  base <- (seq_len(Tn) - 1L) * J + 1L
  Dc <- X[-base, , drop = FALSE] -
    X[rep(base, each = J - 1L), , drop = FALSE]
  G <- crossprod(Dc)
  kept <- integer(0); aliased <- integer(0)
  for (k in seq_len(p)) {
    gkk <- G[k, k]
    resid <- gkk
    if (gkk > 0 && length(kept)) {
      g <- G[kept, k]
      coef <- qr.solve(G[kept, kept, drop = FALSE], g, tol = 1e-14)
      resid <- gkk - sum(g * coef)
    }
    if (gkk > 0 && resid > tol * gkk) kept <- c(kept, k)
    else aliased <- c(aliased, k)
  }
  if (length(aliased))
    stop(sprintf(paste("mean-design column(s) %s are not identified from",
                       "within-choice-set contrasts: each is constant",
                       "across the alternatives of every observation or a",
                       "linear combination of earlier columns (the",
                       "generated intercepts come first); drop them",
                       "(#581)"),
                 paste(colnames(X)[aliased], collapse = ", ")),
         call. = FALSE)
  invisible(NULL)
}

# With alternative intercepts, an alternative nobody chose has no
# finite MLE: its intercept runs to -Inf, and BFGS stopped on its
# tolerance at a tolerance-dependent point reported as converged
# (#496). Returns the never-chosen labels (empty when all are chosen).
.never_chosen <- function(choice, J, labels) {
  miss <- setdiff(seq_len(J), choice)
  if (length(miss))
    warning(sprintf(paste("alternative(s) %s are never chosen, so their",
                          "intercepts have no finite MLE (complete",
                          "separation); convergence is reported as 2.",
                          "Drop those alternatives."),
                    paste(labels[miss], collapse = ", ")), call. = FALSE)
  labels[miss]
}

#' Exact multinomial probit, mlogit-style interface
#'
#' @param formula choice ~ alternative-specific covariates,
#'   e.g. mode ~ price + catch, an ordinary model formula (log(x), I(),
#'   interactions, contrasts); intercepts per non-reference alternative
#'   are added unless the formula has 0 + or - 1.
#' @param data long format, one row per (chooser, alternative), with an
#'   idx column or attribute (chooser, alternative); a plain data frame
#'   works, and so does a dfidx object (dfidx is not a dependency).
#' @param r number of factor columns (default 2 covers the full
#'   identified covariance at J = 4).
#' @param Qf,Qz Gauss-Hermite orders for factor and own-noise nodes;
#'   Qz = NULL takes 7 below J = 6 and 11 from J = 6 (#173).
.check_choice_sets <- function(ids, alt, J) {
  # The reshape downstream is POSITIONAL --
  #   mu <- matrix(X %*% beta, nrow = Tn, ncol = J, byrow = TRUE)
  # -- so row order alone decides which alternative a row is priced as.
  # A count check cannot see that: an observation that duplicates one
  # alternative and omits another still has J rows, so `nrow(df) == Tn *
  # J` passed and the duplicate row was priced as the missing
  # alternative, returning an ordinary-looking fit (#230). The contract
  # is one row per (observation, alternative), and that is a table, not
  # a total.
  tab <- table(ids, alt)
  bad <- which(tab != 1L, arr.ind = TRUE)
  if (nrow(bad) == 0L) return(invisible(NULL))
  ids_lab <- rownames(tab); alt_lab <- colnames(tab)
  first <- bad[order(bad[, 1], bad[, 2]), , drop = FALSE][1, ]
  count <- tab[first[1], first[2]]
  stop(sprintf(paste("each observation must contain each of the %d",
                     "alternatives exactly once; observation '%s' has %d",
                     "row(s) for alternative '%s' (%d observation-alternative",
                     "cell(s) are wrong)"),
               J, ids_lab[first[1]], count, alt_lab[first[2]], nrow(bad)),
       call. = FALSE)
}

# The right-hand side through model.frame/model.matrix, so log(x),
# I(x^2), x:z and factor contrasts mean what they mean in R. Term labels
# used to be looked up as literal column names, and the intercept flag
# was ignored (#472). The global intercept is unidentified in a choice
# model and is always dropped; the formula's intercept flag instead
# switches the alternative-specific intercepts.
.covariate_matrix <- function(formula, data) {
  df <- as.data.frame(data)
  df$idx <- NULL
  # dfidx wraps columns in xseries; model.frame wants plain vectors
  df[] <- lapply(df, function(col) {
    if (inherits(col, "xseries")) class(col) <- setdiff(class(col), "xseries")
    col
  })
  tt <- terms(formula, data = df)
  if (any(grepl("|", attr(tt, "term.labels"), fixed = TRUE)))
    stop(paste("multi-part formulas (`|`) are not supported; give",
               "alternative-specific covariates only, and `0 +` to drop",
               "the alternative intercepts"), call. = FALSE)
  tt <- delete.response(tt)
  attr(tt, "intercept") <- 1L
  mf <- model.frame(tt, df, na.action = na.pass)
  mm <- model.matrix(tt, mf)
  # offsets live in the model frame, not the model matrix: they were
  # accepted and then dropped, so chosen ~ offset(off) fitted the
  # zero-offset model (#641). model.offset sums several offset() terms.
  off <- model.offset(mf)
  if (!is.null(off)) {
    off <- as.numeric(off)
    bad <- which(!is.finite(off))
    if (length(bad))
      stop(sprintf(paste("offset row %d is %s; offsets must be finite",
                         "(%d of %d rows are not)"),
                   bad[1], format(off[bad[1]]), length(bad), length(off)),
           call. = FALSE)
  }
  list(X = mm[, colnames(mm) != "(Intercept)", drop = FALSE], offset = off)
}

mlogit_fast <- function(formula, data, r = NULL, Qf = 7L, Qz = NULL,
                        maxit = 400L) {
  t0 <- Sys.time()
  maxit <- .check_count(maxit, "maxit")
  idx <- if (!is.null(data$idx)) data$idx else attr(data, "idx")
  # a stale or short idx used to select a prefix of data in silence: ord
  # was built from idx alone, so rows past it never entered the fit (#520)
  # (dfidx may append nesting columns after these two)
  if (is.null(idx) || length(dim(idx)) != 2L || ncol(idx) < 2L)
    stop(paste("data needs an idx column or attribute whose first two",
               "columns are (chooser, alternative)"), call. = FALSE)
  if (NROW(idx) != nrow(data))
    stop(sprintf("idx must have exactly nrow(data) rows (got %d for %d data rows)",
                 NROW(idx), nrow(data)), call. = FALSE)
  idx <- as.data.frame(idx)
  # droplevels: a filtered panel keeps its unused factor levels, and as
  # integer codes those levels left gaps that were read as phantom
  # choosers or alternatives (#464). Code the OBSERVED labels only.
  alt_f <- droplevels(as.factor(idx[[2]]))
  alt <- as.integer(alt_f)
  J <- max(alt)
  .check_two_alternatives(J)
  id_f <- droplevels(as.factor(idx[[1]]))
  ids <- as.integer(id_f)
  # Canonical (observation, alternative) order BEFORE anything is built.
  # The core reshape is positional, so a dfidx in alternative-major or
  # shuffled order priced each block's first row as alternative 1 whatever
  # its label: the same labelled data gave nll 0.0342 or 23.94 by row
  # order alone (#215, #355). rprobit_fast already sorted; this did not.
  ord <- order(ids, alt)
  alt <- alt[ord]; ids <- ids[ord]
  # the same positional reshape lives here (#230)
  .check_choice_sets(ids, alt, J)
  # dfidx wraps columns in xseries, under which %in% misbehaves; coerce.
  # But unclass on a factor exposes its level CODES (FALSE/TRUE -> 1/2),
  # so a factor response goes through its labels instead (#512)
  resp_raw <- data[[as.character(formula[[2]])]]
  resp <- if (is.factor(resp_raw)) as.character(resp_raw) else
    as.vector(unclass(resp_raw))
  resp <- resp[ord]
  # exactly one chosen row per observation, keyed by id (#324, #435)
  choice <- .choices_by_id(ids, alt, resp, levels(id_f))
  Tn <- length(choice)
  r <- .check_rank(r, J)
  cm <- .covariate_matrix(formula, data)
  Xcov <- cm$X[ord, , drop = FALSE]
  off <- if (is.null(cm$offset)) NULL else cm$offset[ord]
  if (attr(terms(formula), "intercept") == 1L) {
    Xint <- matrix(0, nrow(Xcov), J - 1L)
    for (j in 2:J) Xint[alt == j, j - 1L] <- 1
    X <- cbind(Xint, Xcov)
    colnames(X) <- c(paste0("(Intercept):", levels(alt_f)[2:J]),
                     colnames(Xcov))
  } else {
    # `0 +` / `- 1` drops the alternative-specific intercepts (#472)
    if (!ncol(Xcov))
      stop("the formula has no covariates and no intercepts to fit",
           call. = FALSE)
    X <- Xcov
  }
  .check_identified(X, Tn, J)
  # without intercepts an unchosen alternative is not a separation
  unchosen <- if (attr(terms(formula), "intercept") == 1L)
    .never_chosen(choice, J, levels(alt_f)) else character(0)
  nb <- ncol(X)
  nw <- sum(vapply(seq_len(r), function(cl) J - cl, 0L))
  if (is.null(Qz)) Qz <- .default_qz(J)
  nodes <- .nodes3(Qf, Qz, r)
  # built only if the sharpness rule ever selects it (#388)
  nodes_sharp <- .lazy_sharp_nodes(r, 10L)
  theta0 <- c(rep(0, nb), rep(0.1, nw))
  memo <- new.env()
  fn <- function(th) {
    if (!is.null(memo$th) && identical(th, memo$th)) return(memo$out$value)
    memo$th <- th
    memo$out <- .nll_core(th, list(X), choice, J, r, nodes, nodes_sharp,
                          offset = off)
    memo$out$value
  }
  gr <- function(th) {
    if (!is.null(memo$th) && identical(th, memo$th)) return(memo$out$grad)
    memo$th <- th
    memo$out <- .nll_core(th, list(X), choice, J, r, nodes, nodes_sharp,
                          offset = off)
    memo$out$grad
  }
  fit <- optim(theta0, fn, gr, method = "BFGS",
               control = list(maxit = maxit, reltol = 1e-8))
  sep <- .separation_direction(X, choice, J)
  if (!is.null(sep) && !length(unchosen)) .warn_separated(sep)
  P <- .raw_probabilities(fit$par, X, J, r, nodes, nodes_sharp, off)
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  beta <- fit$par[seq_len(nb)]
  names(beta) <- colnames(X)
  out <- .loading_labels(fit$par[-seq_len(nb)], J, r, levels(alt_f))
  structure(list(coefficients = beta,
                 covariance_par = out$par,
                 loadings = out$V,
                 alternatives = levels(alt_f),
                 logLik = -fit$value, time = secs,
                 convergence = if (length(unchosen) || !is.null(sep)) 2L
                               else fit$convergence,
                 separation = sep,
                 # the fit maximizes raw per-choice quadrature masses;
                 # this is how far their row sums are from one (#376)
                 mass_defect = max(abs(1 - rowSums(P))),
                 J = J, r = r),
            class = "mlogit_fast")
}

print.mlogit_fast <- function(x, ...) {
  cat(sprintf("exact multinomial probit  logLik %.3f  (%.1f s, %s)\n",
              x$logLik, x$time,
              if (x$convergence == 0) "converged" else "NOT converged"))
  print(round(x$coefficients, 4))
  invisible(x)
}
