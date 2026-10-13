# The general race: one API, distributions and correlation as parameters.
# Base-R port of winning/factor/races.py (the python reference is the
# spec; parity/vectors.json pins the two together).
#
# Min-wins convention throughout. A base is a function z -> list(S, f, fp)
# giving survival, density and density derivative of a mean-zero,
# unit-variance law.

.EULER <- 0.5772156649015329

.base_normal <- function(z) {
  # the upper tail directly: 1 - pnorm(z) cancels to exactly 0 past
  # about 8.3 sd, which made a 20-sd longshot ~95x too unlikely (#96)
  S <- pmax(stats::pnorm(z, lower.tail = FALSE), 1e-300)
  f <- exp(-0.5 * z^2) / sqrt(2 * pi)
  list(S = S, f = f, fp = -z * f)
}

.base_gumbel <- function(z) {
  cc <- pi / sqrt(6)
  u <- pmin(z * cc - .EULER, 30)
  eu <- exp(u)
  S0 <- exp(-eu)
  # density and slope from the unfloored survival (#502)
  f <- cc * eu * S0
  list(S = pmax(S0, 1e-300), f = f, fp = cc * cc * eu * S0 * (1 - eu))
}

.base_logistic <- function(z) {
  cc <- pi / sqrt(3)
  u <- cc * z
  # reflected, from e = exp(-|u|) <= 1, as python: f = cc S (1 - S)
  # rounded 1 - S to zero once S hit 1, zeroing the left-tail density
  # (a runner 25 sd behind priced 7.3e-19 for 9.0e-19, #467)
  e <- exp(-abs(u))
  S <- ifelse(u > 0, e / (1 + e), 1 / (1 + e))
  f <- cc * e / (1 + e)^2
  list(S = pmax(S, 1e-300), f = f, fp = -cc * f * (1 - 2 * S))
}

.base_laplace <- function(z) {
  b <- 1 / sqrt(2)
  f <- exp(-abs(z) / b) / (2 * b)
  S <- ifelse(z < 0, 1 - 0.5 * exp(z / b), 0.5 * exp(-z / b))
  list(S = pmax(S, 1e-300), f = f, fp = -sign(z) * f / b)
}
# the density is not analytic at z = 0: the inverses take their slopes on
# the split lattice there (.kink_lattice, #267); the forward is unchanged
attr(.base_laplace, "kinks") <- 0

# Gauss-Legendre nodes and weights on [-1, 1] (Golub-Welsch), cached
.LEGENDRE <- new.env(parent = emptyenv())
.legendre <- function(m) {
  key <- as.character(m)
  if (!is.null(.LEGENDRE[[key]])) return(.LEGENDRE[[key]])
  k <- seq_len(m - 1)
  off <- k / sqrt(4 * k * k - 1)
  J <- matrix(0, m, m)
  if (m > 1) {
    J[cbind(k, k + 1)] <- off
    J[cbind(k + 1, k)] <- off
  }
  e <- eigen(J, symmetric = TRUE)
  idx <- order(e$values)
  out <- list(t = e$values[idx], w = 2 * e$vectors[1, idx]^2)
  assign(key, out, envir = .LEGENDRE)
  out
}

# Nodes and weights over the window of the uniform lattice x, split at
# every runner's kink and Gauss-Legendre on each piece: python's
# races.kink_lattice, same budget rule (outer pieces at a quarter weight,
# at least four nodes a piece, at most length(x) %/% 4 kinks kept). A
# kinked base makes the slope integrand -f'(z) JUMP at every runner's
# location, so a uniform rectangle sum of a slope or Jacobian is only
# first order there; on these pieces it is spectral (#267). Used for the
# inverses' derivatives only: the forward maps keep the uniform lattice.
.kink_lattice <- function(x, M, sd, kinks) {
  a <- x[1]; b <- x[length(x)]; L <- length(x)
  K <- as.vector(outer(M, rep(1, length(kinks))) + outer(sd, kinks))
  K <- sort(unique(K[K > a & K < b]))
  cap <- max(1L, L %/% 4L)
  if (length(K) > cap) {
    width <- (b - a) / cap
    K <- K[!duplicated(floor((K - a) / width))]
  }
  cuts <- c(a, K, b)
  P <- length(cuts) - 1
  lo <- cuts[-length(cuts)]; hi <- cuts[-1]
  eff <- hi - lo
  eff[1] <- eff[1] * 0.25
  eff[P] <- eff[P] * 0.25
  cnt <- pmax(4L, as.integer(round(L * eff / sum(eff))))
  xs <- numeric(0); ws <- numeric(0)
  for (j in seq_len(P)) {
    gl <- .legendre(cnt[j])
    mid <- 0.5 * (hi[j] + lo[j]); half <- 0.5 * (hi[j] - lo[j])
    xs <- c(xs, mid + half * gl$t); ws <- c(ws, half * gl$w)
  }
  list(x = xs, w = ws)
}

.kinks_of <- function(fn) {
  k <- attr(fn, "kinks")
  if (is.numeric(k) && length(k)) k else NULL
}

# Exact log survivals of the named bases, used by the integrator only
# where S was floored at 1e-300 (#502): summed as mass, the floor
# plateaued a rare winner at ~1e-300 (normal mu = c(0, 53, 106): 1.0e-300
# for 1.105e-307). Results above that range are unchanged bit for bit.
attr(.base_normal, "log_sf") <- function(z)
  stats::pnorm(z, lower.tail = FALSE, log.p = TRUE)
attr(.base_gumbel, "log_sf") <- function(z)
  -exp(pmin(z * pi / sqrt(6) - .EULER, 30))
# the min-Gumbel is asymmetric, so its lower-tail CDF cannot be the
# reflected survival; 1 - S rounded a representable last place to zero
# (mu = c(-35, 0, 0): 0 for 2.05e-39, #710)
attr(.base_gumbel, "cdf") <- function(z)
  -expm1(-exp(pmin(z * pi / sqrt(6) - .EULER, 30)))
attr(.base_logistic, "log_sf") <- function(z) {
  u <- pi / sqrt(3) * z
  -pmax(u, 0) - log1p(exp(-abs(u)))
}
attr(.base_laplace, "log_sf") <- function(z) {
  b <- 1 / sqrt(2)
  ifelse(z < 0, log1p(-0.5 * exp(-abs(z) / b)), log(0.5) - abs(z) / b)
}

.log_survival <- function(fn, z, S) {
  logS <- log(S)
  lsf <- attr(fn, "log_sf")
  if (!is.null(lsf)) {
    low <- S <= 1e-300
    # bounded below: an overflowing log (mu = 1e308) gave -Inf and
    # -Inf - -Inf a NaN race; the density is 0 wherever this binds
    if (any(low)) logS[low] <- pmax(lsf(z)[low], -1e300, na.rm = TRUE)
  }
  logS
}

.BASES <- list(normal = .base_normal, gumbel = .base_gumbel,
               logistic = .base_logistic, laplace = .base_laplace)
.SPANS <- list(normal = c(8, 8), gumbel = c(22, 8),
               logistic = c(16, 16), laplace = c(18, 18))

.hermite1 <- function(order) {
  off <- sqrt(seq_len(order - 1))
  J <- matrix(0, order, order)
  J[cbind(seq_len(order - 1), seq_len(order - 1) + 1)] <- off
  J[cbind(seq_len(order - 1) + 1, seq_len(order - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  idx <- order(e$values)
  w <- e$vectors[1, idx]^2
  list(nodes = e$values[idx], weights = w / sum(w))
}

# Dependency-free Halton sequence mapped through qnorm: equal-weight
# nodes for E over N(0, I_r). Used when the sharpness escalation calls
# for a low-discrepancy family (see .race_setup); adequate for r <= 4.
# The bases are the first r primes, generated: a fixed 32-prime table
# made r = 33 index past its end, and the NA base then failed inside the
# radical-inverse loop, on a full-contrast-rank race the API accepts and
# the Python reference prices (#402). fit_covariance calls this too.
.halton_normal_nodes <- function(r, n) {
  primes <- .first_primes(r)
  H <- vapply(primes, function(b) {
    idx <- seq_len(n) + 20L          # drop the first few, standard hygiene
    h <- numeric(n)
    f <- 1 / b
    i <- idx
    while (any(i > 0)) {
      h <- h + f * (i %% b)
      i <- i %/% b
      f <- f / b
    }
    h
  }, numeric(n))
  F <- qnorm(pmin(pmax(H, 1e-12), 1 - 1e-12))
  list(F = matrix(F, ncol = r), W = rep(1 / n, n))
}

.race_setup <- function(mu, V, D, F, W, base) {
  mu <- as.numeric(mu)
  n <- length(mu)
  # One idiosyncratic variance per contestant, or a scalar broadcast on
  # purpose -- the contract winning.shapes.as_idio states. R recycles a
  # short vector in silence whenever its length divides n, so
  # D = c(2, 9) at n = 4 priced exactly the race D = c(2, 9, 2, 9)
  # prices: same numbers, no warning. Found by the cross-port
  # divergence scan, where python, the browser and julia all refuse it.
  # positive, as python's as_idio(D, n, positive=TRUE): a zero D is a
  # point mass the lattice divides by, which crashed in .bulk_window with
  # "missing value where TRUE/FALSE needed" instead of naming D.
  D <- .as_idio(D, n, positive = TRUE)
  if (is.null(V)) {
    V <- matrix(0, n, 1)
    F <- matrix(0, 1, 1)
    W <- 1
  } else {
    # a SCALAR V is the same loading for everyone -- the spelling
    # winning.shapes.as_loadings documents, which python and the browser
    # accept and R refused (cross-port divergence scan). as.matrix() on
    # a length-1 vector gives a 1x1, which is one contestant with one
    # factor, not n contestants sharing a loading.
    if (length(V) == 1L && is.null(dim(V))) V <- matrix(as.numeric(V), n, 1L)
    V <- as.matrix(V)
    # (rank, n) is the SAME race as (n, rank), which as_loadings
    # documents and abilities_from_race already did for itself a few
    # hundred lines down -- .race_setup did not, so R was inconsistent
    # with the contract AND with itself. The ambiguity is real only at
    # rank == n, where the contract wins: n rows is n contestants.
    if (nrow(V) != n && ncol(V) == n) V <- t(V)
    if (nrow(V) != n)
      stop(sprintf("V must have one row per contestant; got %d for %d",
                   nrow(V), n), call. = FALSE)
    # gauge-fix matching the python reference: a common loading column
    # shifts every conditional mean equally and cannot move an argmin,
    # so center V and make node selection and the lattice window
    # invariant under V -> V + 1 c'
    V <- .gauge_center(V)                                    # (#471)
    if (is.null(F) || is.null(W)) {
      # adaptive order matching the python reference: sharp conditional
      # races (small D relative to loadings) need more factor nodes.
      # Dispatch on the pairwise-safe bound sqrt(2) max |(PV)_i|/sqrt(D_i),
      # which bounds the pairwise contrast sharpness from above
      sharp <- sqrt(2) * max(sqrt(rowSums(V^2)) / sqrt(pmax(D, 1e-300)))
      r <- ncol(V)
      # per-rank (Gauss-Hermite order cap, sharpness past which even that
      # order loses to the low-discrepancy family); see GH_RULE in the
      # python reference for the measurements behind each number
      cap <- if (r == 1) 201 else if (r == 2) 41 else if (r == 3) 31 else 15
      sharp_max <- if (r == 2) 3.75 else if (r == 3) 4.75 else 3.0
      if (r >= 2 && sharp > sharp_max) {
        # matching the python reference: past this sharpness the factor
        # integrand is a near-step and Gauss-Hermite converges slowly at
        # any order; escalate the FAMILY to a low-discrepancy rule.
        # Python uses scrambled Sobol; here dependency-free Halton.
        hw <- .halton_normal_nodes(r, 2^13)
        F <- hw$F
        W <- hw$W
      } else if (r == 1 && ceiling(8 * sharp) > 80) {
        # rank-1 extreme sharpness (matching the python reference):
        # Gauss-Hermite is the wrong family for a near-step integrand;
        # an equal-weight midpoint-quantile grid scaled with sharpness
        # replaces it (TV 0.65 -> 6e-3 at the same node count)
        Q <- as.integer(min(ceiling(8 * sharp), 4001))
        F <- matrix(qnorm((seq_len(Q) - 0.5) / Q), ncol = 1)
        W <- rep(1 / Q, Q)
      } else if (cap^r > 1e5) {
        # high-rank tensor footgun (matching python): past a 1e5-node
        # tensor budget escalate to low-discrepancy nodes
        hw <- .halton_normal_nodes(r, 2^13)
        F <- hw$F
        W <- hw$W
      } else {
        Q <- as.integer(min(max(ceiling(8 * sharp), 15), cap))
        hw <- hermite_nodes(ncol(V), order = Q)
        F <- hw$F
        W <- hw$W
      }
    }
  }
  fn <- if (is.function(base)) base else .BASES[[base]]
  # a callable base may declare its own span (attr(base, "span")), as
  # python's getattr(base, "span", (12, 12)); the fixed +/-12 clipped a
  # caller's polynomial tail under window = "span" too (#106)
  span <- if (is.function(base)) {
    sp <- attr(base, "span")
    if (is.null(sp)) c(12, 12) else rep_len(as.numeric(sp), 2L)
  } else {
    s <- .SPANS[[base]]
    if (is.null(s)) c(12, 12) else s
  }
  # Every node carries exactly the loadings' rank, and there is one
  # weight per node. F with the wrong number of ROWS was accepted and
  # priced a different quadrature outright -- 0.64616 0.12271 0.23064
  # 0.00049 where the right answer is 0.38173 0.30061 0.13687 0.18079 --
  # too few weights returned all NA, and extra weights were ignored. A
  # short F did fail, but with "non-conformable arguments", which names
  # nothing the caller passed. Same contract as the browser's
  # asFactorNodes (#290) and julia's.
  Fm <- as.matrix(F)
  rk <- ncol(V)
  if (ncol(Fm) != rk)
    stop(sprintf(paste("F must have one column per loading; got %d for",
                       "rank %d -- a short F prices a lower-rank model"),
                 ncol(Fm), rk), call. = FALSE)
  if (nrow(Fm) < 1L)
    stop("F is empty; there are no quadrature nodes", call. = FALSE)
  if (any(!is.finite(Fm)))
    stop("F has a non-finite node", call. = FALSE)
  if (length(W) != nrow(Fm))
    stop(sprintf("W must have one weight per factor node; got %d for %d nodes",
                 length(W), nrow(Fm)), call. = FALSE)
  F <- Fm

  # W and c*W describe the SAME factor law, so normalise here, as
  # python's _setup does through winning.shapes.as_weights and as
  # abilities_from_race already does for itself further down. The
  # forward divides its accumulated shares by their total and was
  # invariant either way; race_jacobian does not, so J(W) and J(10W)
  # differed by 1.473 -- the same defect as #281 in the standalone
  # javascript module, and #290 in the browser tree.
  #
  # Normalise through the LARGEST weight, never the raw total: finite
  # weights such as c(1e308, 1e308) overflow only in sum(W), and that
  # was refused as a non-finite total although it is the same law as
  # c(1, 1) (#415).
  W <- as.numeric(W)
  if (any(!is.finite(W)) || any(W < 0))
    stop("W must be finite and non-negative; a factor law has no negative mass",
         call. = FALSE)
  Wmax <- if (length(W)) max(W) else 0
  if (!(Wmax > 0))
    stop(sprintf("W must have a positive total; got %s", format(sum(W))),
         call. = FALSE)
  W <- W / Wmax
  W <- W / sum(W)
  list(mu = mu, V = V, D = D, F = as.matrix(F), W = as.numeric(W),
       fn = fn, left = span[1], right = span[2])
}

# Lattice over the WINNER distribution's bulk, not the ability span --
# port of races._bulk_window: bisection on a conservative winner-cdf
# envelope built from the CALLER'S base survival `fn`, both edges
# bracketed before they are bisected, delta relaxed (by factors of 100,
# up to 1e-4, with a warning) when the requested quantiles will not fit
# the point budget, and the pad widened by a base's declared span.
# It used pnorm for every base and a fixed +/-9 sd start: a Student-t
# custom base was truncated (TV 4.8e-3 against a wide span at 2001
# points, where python's base-aware window is 1.0e-5) and no point count
# could repair the wrong interval (#106).
# LOG-survival alone for the shipped bases: the window evaluates it ~160
# times, and calling the full base also built the density and its slope
# only to discard them -- ~0.6 ms of a 1.4 ms independent n = 10 race
# (python's #112 removed the same cost). Floored at log(1e-300) as the
# bases floor S.
.LOG_TINY <- log(1e-300)
.LOG_SURVIVALS <- list(
  normal = function(z) pmax(stats::pnorm(z, lower.tail = FALSE, log.p = TRUE),
                            .LOG_TINY),
  gumbel = function(z) pmax(-exp(pmin(z * (pi / sqrt(6)) - .EULER, 30)),
                            .LOG_TINY))

.bulk_window <- function(M_all, sd, points, delta, fn = NULL) {
  logS_of <- if (is.null(fn) || identical(fn, .base_normal))
    .LOG_SURVIVALS$normal
  else if (identical(fn, .base_gumbel)) .LOG_SURVIVALS$gumbel
  else function(z) log(pmax(fn(z)$S, 1e-300))
  mu_lo <- apply(M_all, 2, min)
  mu_hi <- apply(M_all, 2, max)
  s <- sd
  G <- function(x) 1 - exp(sum(logS_of((x - mu_lo) / s)))
  H <- function(x) 1 - exp(sum(logS_of((x - mu_hi) / s)))
  bracket <- function(x0, step0, ok, sgn) {
    step <- step0
    for (i in seq_len(60)) {
      if (ok(x0)) return(x0)
      x0 <- x0 + sgn * step
      step <- step * 2
    }
    warning(paste("bulk window could not bracket the requested quantile",
                  "after 60 doublings; this base's tail is heavier than the",
                  "lattice can span, and the window is truncated rather than",
                  "quantile-exact."), call. = FALSE)
    x0
  }
  pad <- 2 * max(s)
  if (!is.null(fn)) {
    sp <- attr(fn, "span")
    if (!is.null(sp)) pad <- max(pad, 0.25 * max(sp) * max(s))
  }
  window_at <- function(d) {
    step0 <- max(9 * max(s), 1e-12)
    lo0 <- bracket(min(mu_lo) - 9 * max(s), step0, function(x) G(x) <= d, -1)
    hi0 <- bracket(max(mu_hi) + 9 * max(s), step0,
                   function(x) H(x) >= 1 - d, +1)
    # up to 80 halvings, stopping once the midpoint is one of the ends
    # (adjacent doubles): every later step would leave a and b unchanged,
    # so the window is bit-identical and ~25 evaluations cheaper
    a <- lo0; b <- hi0
    for (i in 1:80) {
      m <- 0.5 * (a + b)
      if (m == a || m == b) break
      if (G(m) < d) a <- m else b <- m
    }
    xlo <- a
    a <- xlo; b <- hi0
    for (i in 1:80) {
      m <- 0.5 * (a + b)
      if (m == a || m == b) break
      if (H(m) < 1 - d) a <- m else b <- m
    }
    c(xlo - pad, b + pad)
  }
  # times the base's central scale, as python and the browser (#600)
  budget <- 0.5 * min(s) * .base_resolution(fn) * max(points - 1, 1)
  d <- .as_delta(delta)
  w <- window_at(d)
  while (w[2] - w[1] > budget && d < 1e-4) {
    d <- min(d * 100, 1e-4)
    w <- window_at(d)
  }
  if (d > delta)
    warning(sprintf(paste("bulk window relaxed delta from %.0e to %.0e: this",
                          "base's tail puts the requested quantile further",
                          "out than %d points can resolve, so the window is",
                          "%.3g units wide at the relaxed delta and exact",
                          "there. Raise points= to tighten it."),
                    delta, d, as.integer(points), w[2] - w[1]), call. = FALSE)
  seq(w[1], w[2], length.out = points)
}

#' Win probabilities of the general race, all N in one field pass
#'
#' @param mu numeric abilities (min-wins: lower is better)
#' @param V optional N x k loading matrix
#' @param D idiosyncratic variances (default 1)
#' @param F,W optional factor nodes and weights
#' @param base "normal", "gumbel", or a function z -> list(S, f, fp)
#' @param points lattice size (default 257)
#' @param return_slopes also return d p_raw_i / d mu_i (inversion
#'   preconditioner), normalized as p is
#' @param structure optional covariance grammar (see
#'   \code{\link{Independent}}); overrides V/D
#' @param window "bulk" (winner-bulk lattice, default) or "span"
#' @param delta omitted winner mass bound for the bulk window
#' @param qa,qf quadrature orders for structure dispatch
#' @param nodes deprecated alias for list(F, W)
#' @return probabilities summing to one, or list(p, slopes)
#' @export
.warn_degraded_cov <- function(fit, where) {
  # matching winning.factor.races._fit_cov: the same two failure classes,
  # and the same advice, except that python can route around it here
  if (!isTRUE(fit$degraded)) return(invisible(NULL))
  warning(sprintf(
    "%s: the cov= grammar fit is degraded (rank %d of %d%s). The %s race is then a near-step the factor nodes cannot resolve, and the covariance residual checks do not see it -- expect percent-level error (4.6e-3 measured against the python reference on an exactly 3-factor correlation at n=8). The python package routes this case to scrambled-Sobol GHK and this port has none, so the two DISAGREE here by design; use winning (python) race_probabilities(cov=) if that matters.",
    where, fit$rank, fit$n,
    if (fit$bound > 0) sprintf(", idiosyncratic floor bound on %d of %d entries", fit$bound, fit$n) else "",
    "conditional"), call. = FALSE)
  invisible(NULL)
}

.jacobi_sweeps <- function(mu, forward, scale, alpha, n_iter, tol) {
  backoffs <- 0L
  # Own-slope-preconditioned coordinate sweeps on the mean-zero quotient,
  # matching python's races._jacobi_sweeps. `forward(mu)` returns
  # list(resid, dres): the residual in whatever space the caller inverts
  # in, model minus target, and its own-slopes (negative).
  #
  # alpha_base is the Richardson value, a persistent fact about the
  # Jacobian; penalty is caution after a sweep that failed to contract, a
  # transient, restored on the next good sweep. One number for both can
  # only ratchet down (#178). A monotone iteration riding one mode is
  # summed rather than waited out (Aitken), above 1e3 tolerances only.
  alpha_base <- alpha
  penalty <- 1
  prev <- NULL
  prev_step <- NULL
  rmax <- Inf
  cached <- NULL  # forward of an accepted Aitken trial
  iters <- 0L
  for (it in seq_len(n_iter)) {
    iters <- it
    fw <- cached
    if (is.null(fw)) {
      fw <- tryCatch(forward(mu), error = function(e) e)
      if (inherits(fw, "error")) {
        # the forward refused this iterate (a lattice defect, or a
        # membership above one): back off halfway toward the last priced
        # iterate, as python's _jacobi_sweeps (#267). The first forward
        # still errors; a persistent refusal returns the last priced
        # iterate as not converged.
        if (is.null(prev)) stop(fw)
        if (backoffs >= 20L)
          return(list(mu = prev$mu, converged = prev$rmax < tol,
                      resid = prev$rmax, iters = iters))
        backoffs <- backoffs + 1L
        mu <- prev$mu + 0.5 * (mu - prev$mu)
        penalty <- max(0.5 * penalty, 0.1)
        prev_step <- NULL
        next
      }
    }
    cached <- NULL
    resid <- fw$resid
    dlogp <- fw$dres
    rmax <- max(abs(resid))
    rrms <- sqrt(mean(resid^2))
    if (rmax < tol) break
    if (!is.null(prev) && rmax >= prev$rmax && rrms >= prev$rrms) {
      if (penalty > 0.1) {
        penalty <- max(0.5 * penalty, 0.1)
        mu <- prev$mu; resid <- prev$resid; dlogp <- prev$dlogp
        rmax <- prev$rmax; rrms <- prev$rrms
        prev_step <- NULL
      }
    } else if (!is.null(prev) && penalty < 1) {
      penalty <- min(1, penalty / 0.75)
    }
    prev <- list(mu = mu, resid = resid, dlogp = dlogp, rmax = rmax, rrms = rrms)
    alpha <- alpha_base * penalty
    # residual-proportional step cap in the field's scale: a near-certain
    # winner's residual and own-slope both vanish and their noisy ratio
    # destabilizes the recentered fixed point
    lim <- pmin(2, 10 * abs(resid)) * scale
    step <- pmin(pmax(alpha * resid / dlogp, -lim), lim)
    step <- step - mean(step)
    extrapolated <- FALSE
    if (!is.null(prev_step)) {
      na <- sqrt(sum(prev_step^2))
      nb <- sqrt(sum(step^2))
      if (na > 0) {
        dot <- sum(step * prev_step)
        rho <- dot / (na * na)
        cosn <- if (nb > 0) dot / (na * nb) else 0
        ratio <- nb / na
        if (rho < 0) {
          lam <- 1 - (1 - rho) / alpha
          alpha_base <- min(max(2 / (2 - lam), 0.1), 1)
        } else if (cosn > 0.999 && ratio > 0.5 && ratio < 0.999 &&
                   rmax > 1e3 * tol) {
          # collinear steps decaying geometrically: sum the tail (Aitken),
          # as a capped, repriced trial (#583, matching python)
          trial <- .aitken_trial(mu, step, ratio, scale, forward, rmax)
          if (!is.null(trial)) {
            mu <- trial$mu
            cached <- trial$fwd
            prev_step <- NULL
            extrapolated <- TRUE
          }
        }
      }
    }
    if (!extrapolated) {
      prev_step <- step
      mu <- mu - step
    }
  }
  if (!(rmax < tol)) {
    # exhausted after a step no sweep priced: price the returned iterate
    # (one forward, only here) and keep the better of the two, so the
    # warning describes the mu returned (#497, matching python)
    fw <- if (is.null(cached)) forward(mu) else cached
    r_end <- max(abs(fw$resid))
    if (!is.null(prev) && !isTRUE(r_end < prev$rmax)) mu <- prev$mu
    else rmax <- r_end
  }
  list(mu = mu, converged = rmax < tol, resid = rmax, iters = iters)
}

# The Aitken-extrapolated iterate as a bounded, checked trial (#583),
# matching python's races._aitken_trial: the summed tail capped in the max
# norm at twice the field scale (never below the ordinary step) and
# repriced; NULL -- take the ordinary step -- when the forward errors, is
# non-finite, or its max residual grows tenfold.
.aitken_trial <- function(mu, step, ratio, scale, forward, rmax) {
  jump <- step / (1 - ratio)
  big <- max(abs(jump))
  cap <- max(2 * max(scale), max(abs(step)))
  if (big > cap) jump <- jump * (cap / big)
  cand <- mu - jump
  fw <- tryCatch(forward(cand), error = function(e) NULL)
  if (is.null(fw) || any(!is.finite(fw$resid)) || any(!is.finite(fw$dres)))
    return(NULL)
  if (max(abs(fw$resid)) > 10 * rmax) return(NULL)
  list(mu = cand, fwd = fw)
}

# A lattice size: a whole number >= 2 (python shapes.as_points, #444).
# dx = x[2] - x[1] was NA at points = 1 and the race returned NaN.
.as_points <- function(points) {
  v <- suppressWarnings(as.numeric(points))
  if (length(v) != 1L || is.logical(points) || !is.finite(v) || v != trunc(v))
    stop(sprintf("points must be a whole number of lattice points; got %s",
                 format(points)), call. = FALSE)
  if (v < 2)
    stop(sprintf(paste("points must be at least 2: a lattice of %d point(s)",
                       "has no spacing to integrate on"), as.integer(v)),
         call. = FALSE)
  as.integer(v)
}

# structure= describes the covariance; V/D/F/W would describe it again,
# and the forward silently dropped them while the inverse kept some (#89)
.refuse_second_covariance <- function(V, D, F, W) {
  given <- c(V = !is.null(V), D = !is.null(D), F = !is.null(F), W = !is.null(W))
  if (any(given))
    stop(sprintf(paste("structure= already describes the covariance; %s=",
                       "would describe it again. Pass one or the other."),
                 paste(names(given)[given], collapse = ", ")), call. = FALSE)
}

# The gauge, scale-safe (python core._gauge_center, #471): an exactly
# common column centres to exactly zero, and a column whose mean or
# differences overflow ([1e308] * 9, or c(1, 1, -1) * 1e308) is centred in
# units of its largest entry. Ordinary columns take the plain mean.
.gauge_center <- function(V) {
  V <- as.matrix(V)
  if (!nrow(V) || !ncol(V)) return(V)
  Vc <- suppressWarnings(sweep(V, 2, colMeans(V)))
  for (j in seq_len(ncol(V))) {
    if (all(V[, j] == V[1, j])) {
      Vc[, j] <- 0
    } else if (!all(is.finite(Vc[, j]))) {
      s <- max(abs(V[, j]))
      u <- V[, j] / s
      Vc[, j] <- (u - mean(u)) * s
    }
  }
  Vc
}

# The bulk window's tail mass: a number in (0, 0.5). delta = -1 hung the
# relaxation loop and 1 or 2 priced a different race (#445, matching
# python's races._as_delta and the browser's asDelta).
.as_delta <- function(delta) {
  if (!is.numeric(delta) || length(delta) != 1L || !is.finite(delta) ||
      !(delta > 0 && delta < 0.5))
    stop(sprintf("delta must be a number in (0, 0.5); got %s",
                 paste(deparse(delta), collapse = " ")), call. = FALSE)
  as.numeric(delta)
}

race_probabilities <- function(mu, V = NULL, D = NULL, F = NULL, W = NULL,
                               base = "normal", points = 257,
                               return_slopes = FALSE, structure = NULL,
                               window = "bulk", delta = 1e-12,
                               qa = 9, qf = 15, nodes = NULL, cov = NULL) {
  points <- .as_points(points)
  # one of two names: every value but "bulk" used to mean span (#582)
  if (!(is.character(window) && length(window) == 1L && !is.na(window) &&
        window %in% c("bulk", "span")))
    stop(sprintf("window must be \"bulk\" or \"span\"; got %s",
                 paste(deparse(window), collapse = " ")), call. = FALSE)
  delta <- .as_delta(delta)                                    # (#445)
  if (!is.null(cov)) {
    if (!is.null(structure) || !is.null(V) || !is.null(D))
      stop("cov= replaces structure=/V=/D=; pass one only")
    fit <- fit_covariance(cov)
    # the forward normal race with no slopes is the one case answerable
    # without the fit at all; everything else needs the factor form and
    # keeps the fit with its warning (matching python)
    routable <- identical(base, "normal") && !return_slopes
    if (isTRUE(fit$degraded) && routable)
      return(.ghk_race(mu, cov)$p)
    .warn_degraded_cov(fit, "race_probabilities")
    V <- fit$V; D <- fit$D; F <- fit$F; W <- fit$W
  }
  if (!is.null(structure)) {
    if (is.null(cov)) .refuse_second_covariance(V, D, F, W)
    return(.dispatch_probabilities(mu, structure, base = base,
                                   points = points, qa = qa, qf = qf,
                                   return_slopes = return_slopes,
                                   window = window, delta = delta))
  }
  if (!is.null(nodes)) { F <- nodes$F; W <- nodes$W }
  st <- .race_setup(mu, V, D, F, W, base)
  n <- length(st$mu)
  sd <- sqrt(st$D)
  Q <- nrow(st$F)
  M_all <- matrix(st$mu, Q, n, byrow = TRUE) + st$F %*% t(st$V)
  lat <- .race_lattice(st, M_all, sd, points, window, delta)
  x <- lat$x; dx <- lat$dx; points <- lat$points
  p <- numeric(n)
  slope <- numeric(n)
  xm <- matrix(x, n, points, byrow = TRUE)
  for (q in seq_len(Q)) {
    z <- (xm - M_all[q, ]) / sd
    b <- st$fn(z)
    f <- b$f / sd
    logS <- .log_survival(st$fn, z, b$S)
    L <- colSums(logS)
    rest <- exp(pmin(pmax(matrix(L, n, points, byrow = TRUE) - logS,
                          -745), 0))
    p <- p + st$W[q] * rowSums(f * rest) * dx
    slope <- slope + st$W[q] * rowSums(-b$fp / sd^2 * rest) * dx
  }
  total <- sum(p)
  if (return_slopes) return(list(p = p / total, slopes = slope / total))
  p / total
}

# The race lattice: window plus the sharpness refinement, in one place
# (race_probabilities and .split_race_slopes integrate over the same one).
.race_lattice <- function(st, M_all, sd, points, window = "bulk",
                          delta = 1e-12) {
  x <- if (identical(window, "bulk")) {
    .bulk_window(M_all, sd, points, delta, st$fn)
  } else {
    seq(min(M_all) - st$left * max(sd), max(M_all) + st$right * max(sd),
        length.out = points)
  }
  dx <- x[2] - x[1]
  smin <- min(sd)
  sharp_here <- max(sqrt(rowSums(st$V^2))) / max(smin, 1e-300)
  if (sharp_here > 25 && dx > 0.5 * smin) {
    # extreme-sharpness lattice refinement (matching python): refine to
    # ~2 points per conditional sd, capped, warn when the cap binds
    need <- ceiling((x[length(x)] - x[1]) / (0.5 * smin)) + 1
    pts2 <- min(need, 8193)
    if (pts2 > points) {
      x <- seq(x[1], x[length(x)], length.out = pts2)
      dx <- x[2] - x[1]
      points <- pts2
    }
    if (need > 8193)
      warning("conditional races sharper than the lattice can resolve ",
              "even at 8193 points; results may carry percent-level error")
  }
  list(x = x, dx = dx, points = points)
}

# Own-slopes of the race on the kink-split lattice, for a base that
# declares kinks: the uniform rectangle sum of the slope integrand
# -f'(z), which jumps at each runner's kink, is first order -- a laplace
# own-slope came out 6x too large and abilities_from_race missed an exact
# self-generated target by 0.099 (#267). Normalised by the split total,
# as python's race_probabilities(..., kink_split=True).
.split_race_slopes <- function(mu, V, D, F, W, base, points) {
  st <- .race_setup(mu, V, D, F, W, base)
  n <- length(st$mu)
  sd <- sqrt(st$D)
  Q <- nrow(st$F)
  M_all <- matrix(st$mu, Q, n, byrow = TRUE) + st$F %*% t(st$V)
  x <- suppressWarnings(.race_lattice(st, M_all, sd, points))$x
  kinks <- .kinks_of(st$fn)
  p <- numeric(n)
  slope <- numeric(n)
  for (q in seq_len(Q)) {
    kl <- .kink_lattice(x, M_all[q, ], sd, kinks)
    T <- length(kl$x)
    z <- (matrix(kl$x, n, T, byrow = TRUE) - M_all[q, ]) / sd
    b <- st$fn(z)
    f <- b$f / sd
    logS <- .log_survival(st$fn, z, b$S)
    L <- colSums(logS)
    rest <- exp(pmin(pmax(matrix(L, n, T, byrow = TRUE) - logS, -745), 0))
    p <- p + st$W[q] * as.vector((f * rest) %*% kl$w)
    slope <- slope + st$W[q] * as.vector((-b$fp / sd^2 * rest) %*% kl$w)
  }
  slope / sum(p)
}

#' Invert the general race: mean-zero mu reproducing probabilities p
#'
#' @param p positive target probabilities (normalized internally)
#' @param V,D,F,W,base,points,structure,qa,qf as in
#'   \code{\link{race_probabilities}}
#' @param n_iter maximum damped-Newton iterations
#' @param tol convergence tolerance on max |log p - log target|
#' @return mean-zero ability vector (min-wins)
#' @export
abilities_from_race <- function(p, V = NULL, D = NULL, F = NULL, W = NULL,
                                base = "normal", points = 257,
                                n_iter = 60, tol = 1e-8,
                                structure = NULL, qa = 9, qf = 15, cov = NULL) {
  points <- .as_points(points)
  tol <- .as_tolerance(tol)                                   # (#551)
  dense <- NULL
  if (!is.null(cov)) {
    if (!is.null(structure) || !is.null(V) || !is.null(D))
      stop("cov= replaces structure=/V=/D=; pass one only")
    fit <- fit_covariance(cov)
    if (isTRUE(fit$degraded) && identical(base, "normal")) {
      dense <- as.matrix(cov)
    } else {
      .warn_degraded_cov(fit, "abilities_from_race")
    }
    V <- fit$V; D <- fit$D; F <- fit$F; W <- fit$W
  }
  target <- as.numeric(p)
  # finite BEFORE the sign test: NaN slips past `<= 0` (#110). And
  # before the structure= dispatch, so every grammar shares one target
  # contract: Blocks floored a zero share and Nested/Tree logged it as
  # 1e-300, returning finite abilities for an impossible target that
  # Independent and Factor refused (#659)
  if (length(target) == 0L || any(!is.finite(target)))
    stop("target probabilities must be finite", call. = FALSE)
  if (any(target <= 0)) stop("all target probabilities must be positive")
  if (!is.null(structure)) {
    if (is.null(cov)) .refuse_second_covariance(V, D, F, W)
    # an explicit n_iter/tol reaches every grammar; left missing, each
    # branch keeps its own default budget
    return(.dispatch_abilities(p, structure, base = base, points = points,
                               qa = qa, qf = qf,
                               n_iter = if (missing(n_iter)) NULL else n_iter,
                               tol = if (missing(tol)) NULL else tol))
  }
  target <- .rescaled_target(target)
  logt <- log(target)
  n <- length(target)
  # the field's contrast scale (matching the python reference): median
  # idiosyncratic variance plus the mean factor variance under the nodes
  # actually represented, so (V, F) -> (V / c, c F) is invariant
  # the target and D must describe the SAME field: a length-2 p with a
  # length-4 D answered a two-runner race and returned two numbers,
  # silently, where python, julia and the browser all refuse (cross-port
  # divergence scan)
  # through the forward map's own validators: a length-only check let
  # D = c(-1, 2) through to the pair closed form (contrast variance 1,
  # finite answer for a covariance the forward refuses), and as.matrix
  # on a scalar V gave a 1x1 that the pair branch indexed at [2, 2] (#482)
  Dn <- .as_idio(D, n, positive = TRUE)
  Vn <- if (is.null(V)) matrix(0, n, 1) else .as_loadings(V, n)
  Vc <- .gauge_center(Vn)                                    # (#471)
  if (!is.null(V) && !is.null(F)) {
    Fq <- as.matrix(F)
    # through max(W) first: sum(W) can overflow for finite weights (#415)
    Wq <- if (is.null(W)) rep(1 / nrow(Fq), nrow(Fq)) else {
      w0 <- as.numeric(W) / max(as.numeric(W))
      w0 / sum(w0)
    }
    Fc <- sweep(Fq, 2, colSums(Fq * Wq))
    CovF <- t(Fc) %*% (Fc * Wq)
  } else {
    CovF <- diag(ncol(Vc))
  }
  SigV <- Vc %*% CovF %*% t(Vc)
  scale <- sqrt(median(Dn) + mean(diag(SigV)))
  if (n == 2 && identical(base, "normal")) {
    # a pair is one Gaussian contrast: closed form (matching python)
    # The contrast sd, formed in units of the largest entry so it neither
    # overflows (D = c(9e307, 9e307) summed to Inf and returned -Inf/Inf)
    # nor hits an absolute floor (max(var, 1e-300) changed the model below
    # contrast sd 1e-150, #100)
    cs <- sqrt(max(Dn[1], Dn[2], abs(SigV[1, 1]), abs(SigV[2, 2])))
    sd_d <- cs * sqrt(max((SigV[1, 1] / cs) / cs + (SigV[2, 2] / cs) / cs -
                          2 * (SigV[1, 2] / cs) / cs +
                          ((min(Dn[1:2]) / cs) / cs + (max(Dn[1:2]) / cs) / cs), 0))
    # invert the SMALLER share: for c(1, 1e-16) the normalized first
    # share rounds to exactly 1 and qnorm(1) = Inf, while the
    # permutation was finite (#412)
    # the shift is the caller rule's weighted MEAN, (v2 - v1) . E[F]
    shift <- if (!is.null(V) && !is.null(F))
      sum((Vc[2, ] - Vc[1, ]) * colSums(Fq * Wq)) else 0
    gap <- (if (target[1] <= target[2]) sd_d * qnorm(target[1]) else
      -sd_d * qnorm(target[2])) - shift
    mu0 <- c(-0.5 * gap, 0.5 * gap)
    if (is.null(V) || is.null(F) || !all(is.finite(mu0))) return(mu0)
    # A caller's rule need not be Gaussian (a centred two-point law has
    # the right mean and variance and a different pair map): the closed
    # form is a START, certified by the forward it claims to invert
    # (#374, matching python and the browser).
    ph <- race_probabilities(mu0, V = V, D = D, F = F, W = W, base = base,
                             points = points)
    if (max(abs(log(pmax(ph, 1e-300)) - logt)) < tol) return(mu0)
  } else {
    mu0 <- NULL
  }
  mu <- if (is.null(mu0)) -(logt - mean(logt)) / 2 * scale else mu0
  top2 <- if (n > 2) sum(sort(target, decreasing = TRUE)[1:2]) else 1
  alpha <- if (n == 2 || top2 > 0.8) 0.7 else 1.0
  if (!is.null(dense)) {
    # Invert the routed map itself, not the fit: GHK yields d log p / d mu
    # in its conditioning pass, so the same sweeps apply. Inverting the fit
    # while the forward returns GHK would make the two front doors describe
    # different races, which is python's #164.
    scale <- sqrt(mean(diag(dense)))
    mu <- -(logt - mean(logt)) / 2 * scale
    fwd <- function(m) {
      g <- .ghk_race(m, dense, want_slopes = TRUE)
      list(resid = g$logp - logt, dres = pmin(g$dlogp, -1e-6 / scale))
    }
    out <- .jacobi_sweeps(mu, fwd, scale, alpha, max(n_iter, 120), tol)
    if (!out$converged)
      warning(sprintf("abilities_from_race did not converge: max |log residual| %.2e (tol %.0e)", out$resid, tol), call. = FALSE)
    return(out$mu)
  }
  fn_b <- if (is.function(base)) base else .BASES[[base]]
  kinked <- !is.null(.kinks_of(fn_b))
  fwd <- function(m) {
    ps <- race_probabilities(m, V = V, D = D, F = F, W = W, base = base,
                             points = points, return_slopes = TRUE)
    # a kinked base: the slope from the split lattice, the value from the
    # map being inverted (#267)
    if (kinked)
      ps$slopes <- .split_race_slopes(m, V, D, F, W, base, points)
    phat <- pmax(ps$p, 1e-300)
    # the slope floor in the field's units: d log p / d mu scales as
    # 1 / scale, so an absolute -1e-6 took over past scale 1e6 and the
    # same race at D = 1e20 stopped 0.062 of a share short (#100)
    list(resid = log(phat) - logt, dres = pmin(ps$slopes / phat, -1e-6 / scale))
  }
  out <- .jacobi_sweeps(mu, fwd, scale, alpha, n_iter, tol)
  if (!out$converged)
    warning(sprintf("abilities_from_race did not converge: max |log residual| %.2e (tol %.0e)", out$resid, tol), call. = FALSE)
  out$mu
}

#' @rdname abilities_from_race
#' @export
calibrate_abilities <- function(p, V = NULL, D = NULL, F = NULL, W = NULL,
                                base = "normal", points = 257,
                                n_iter = 60, tol = 1e-8,
                                structure = NULL, qa = 9, qf = 15,
                                cov = NULL) {
  abilities_from_race(p, V = V, D = D, F = F, W = W, base = base,
                      points = points, n_iter = n_iter, tol = tol,
                      structure = structure, qa = qa, qf = qf, cov = cov)
}
