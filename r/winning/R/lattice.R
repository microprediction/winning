# Lattice operations for the horse race problem.
#
# Pure base-R port of the reference implementation in the python `winning`
# package (winning/lattice.py). A density is a numeric vector of length
# 2L+1 interpreted on the integer lattice -L..L; performances are TIMES,
# so the LOWEST draw wins. Dead heats are handled exactly through the
# multiplicity recursion of the paper. The python implementation is the
# spec; tests/testthat/test-parity.R pins this port to golden values it
# produced.

# The one boundary for a classic atom vector (#339), as python's
# as_classic_density: finite, nonnegative, odd length 2L+1 with L >= 3,
# positive total; returned normalised so raw counts are the same law as
# their frequencies. Entries down to -1e-12 of the total are round-off
# and are clipped. L <= 2 is refused because low_high pins every offset
# to [-L+2, L-2], a single point.
MIN_CLASSIC_L <- 3L

as_classic_density <- function(density, where = "density") {
  d <- as.numeric(density)
  n <- length(d)
  if (n == 0) stop(where, " must be a nonempty vector of lattice atoms")
  if (n %% 2 != 1)
    stop(where, " must have odd length 2L+1 on the symmetric lattice; got length ", n)
  if ((n - 1) %/% 2 < MIN_CLASSIC_L)
    stop(where, " has L = ", (n - 1) %/% 2, "; the classic lattice needs L >= ",
         MIN_CLASSIC_L, " (length >= ", 2 * MIN_CLASSIC_L + 1,
         ") to represent distinct offsets")
  if (!all(is.finite(d))) stop(where, " has a non-finite atom")
  total <- sum(d)
  if (!(total > 0)) stop(where, " has no positive mass")
  if (min(d) < -1e-12 * total) stop(where, " has a negative atom (", min(d), ")")
  pmax(d, 0) / total
}

# Target prices carry only relative mass (#377): finite, nonnegative,
# positive total; normalised.
as_classic_prices <- function(prices, where = "prices") {
  p <- as.numeric(prices)
  if (length(p) == 0) stop(where, " must be a nonempty vector")
  if (!all(is.finite(p))) stop(where, " has a non-finite entry")
  if (min(p) < 0) stop(where, " has a negative entry (", min(p), ")")
  total <- sum(p)
  if (!(total > 0)) stop(where, " has no positive mass")
  p / total
}

pdf_to_cdf <- function(density) cumsum(density)

cdf_to_pdf <- function(cdf) diff(c(0, cdf))

implied_L <- function(density) (length(density) - 1L) %/% 2L

integer_shift <- function(cdf, k) {
  m <- length(cdf)
  k <- max(min(k, m - 1L), -(m - 1L))
  if (k < 0) {
    a <- -k
    c(cdf[(a + 1):m], rep(cdf[m], a))
  } else if (k == 0) {
    cdf
  } else {
    # mass shifted past the top atom lumps on it, mirroring the negative
    # branch; truncating left a sub-probability CDF (#373)
    out <- c(rep(0, k), cdf[1:(m - k)])
    out[m] <- cdf[m]
    out
  }
}

low_high <- function(offset, L) {
  if (offset > -L + 2 && offset < L - 2) {
    lo <- floor(offset)
    up <- ceiling(offset)
    r <- offset - lo
    list(lo = lo, lo_coef = 1 - r, up = up, up_coef = r)
  } else if (offset >= L - 2) {
    list(lo = L - 2, lo_coef = 1, up = L - 1, up_coef = 0)
  } else {
    list(lo = -L + 1, lo_coef = 0, up = -L + 2, up_coef = 1)
  }
}

shifted_cdf <- function(cdf, offset, L) {
  lh <- low_high(offset, L)
  lh$lo_coef * integer_shift(cdf, lh$lo) +
    lh$up_coef * integer_shift(cdf, lh$up)
}

#' Density and dead-heat multiplicity of the winner (minimum) of a field
#'
#' @param densities list of numeric densities, all length 2L+1
#' @return list with elements `density`, `multiplicity`
#' @export
winner_of_many <- function(densities) {
  cdfs <- lapply(densities, pdf_to_cdf)
  m <- length(cdfs[[1]])
  cdf_min <- cdfs[[1]]
  mult <- rep(1, m)
  for (cb in cdfs[-1]) {
    fa <- cdf_to_pdf(cdf_min)
    fb <- cdf_to_pdf(cb)
    win <- fa * (1 - cb)
    draw <- fa * fb
    lose <- fb * (1 - cdf_min)
    mult <- (win * mult + draw * (mult + 1) + lose + 1e-18) /
      (win + draw + lose + 1e-18)
    cdf_min <- 1 - (1 - cdf_min) * (1 - cb)
  }
  list(density = cdf_to_pdf(cdf_min), multiplicity = mult)
}

# ---- exact dead-heat pricing (#418, #362, #348, #373) -------------------
# P_i = sum_t f_i(t) int_0^1 prod_{j != i} (S_j(t) + u f_j(t)) du: the
# equal-split winner claim, since 1/(1+M) = int_0^1 u^M du. The integrand
# is a polynomial of degree n-1 in u, so n %/% 2 + 1 Gauss-Legendre nodes
# are exact. The field is G_q(t) = prod_j (S_j + u_q f_j) and a runner's
# opponents are G_q / (S_i + u_q f_i), capped at one and nonincreasing in
# t. The lattice is padded by L-1 atoms each side so no shift loses mass.
# Mirrors winning/classic/lattice.py (_exact_payoff and friends).

gauss_legendre01 <- function(n) {
  x <- numeric(n)
  w <- numeric(n)
  for (i in seq_len(n)) {
    z <- cos(pi * (i - 1 + 0.75) / (n + 0.5))
    dp <- 1
    for (it in 1:100) {
      p0 <- 1
      p1 <- z
      if (n >= 2) for (k in 2:n) {
        p2 <- ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
        p0 <- p1
        p1 <- p2
      }
      dp <- n * (z * p1 - p0) / (z * z - 1)
      dz <- p1 / dp
      z <- z - dz
      if (abs(dz) < 1e-16) break
    }
    x[i] <- 0.5 * (z + 1)
    w[i] <- 1 / ((1 - z * z) * dp * dp)
  }
  list(nodes = rev(x), weights = rev(w))
}

exact_n_nodes <- function(n_runners) n_runners %/% 2L + 1L

padded_base_cdf <- function(density) {
  L <- implied_L(density)
  pad <- max(L - 1L, 0L)
  pdf_to_cdf(c(rep(0, pad), density, rep(0, pad)))
}

exact_field <- function(cdfs, nodes) {
  m <- length(cdfs[[1]])
  G <- matrix(1, nrow = length(nodes), ncol = m)
  for (cc in cdfs) {
    S <- pmax(1 - cc, 0)
    f <- cdf_to_pdf(cc)
    for (q in seq_along(nodes)) G[q, ] <- G[q, ] * (S + nodes[q] * f)
  }
  G
}

exact_payoff <- function(cdf, G, gl) {
  S <- pmax(1 - cdf, 0)
  f <- cdf_to_pdf(cdf)
  total <- 0
  for (q in seq_along(gl$nodes)) {
    den <- S + gl$nodes[q] * f
    ratio <- ifelse(den > 0, pmin(G[q, ] / ifelse(den > 0, den, 1), 1), 1)
    total <- total + gl$weights[q] * sum(cummin(ratio) * f)
  }
  total
}

# node doubling for big fields, as python's Q_START: start at 8 nodes and
# double (capped at the exact count) until two rules agree to 1e-14
EXACT_Q_START <- 8L
EXACT_TABLE_NODES <- 16L
EXACT_TOL <- 1e-14

exact_state_prices_from_cdfs <- function(cdfs) {
  exact <- exact_n_nodes(length(cdfs))
  q <- min(EXACT_Q_START, exact)
  prev <- NULL
  repeat {
    gl <- gauss_legendre01(q)
    G <- exact_field(cdfs, gl$nodes)
    p <- vapply(cdfs, function(cc) exact_payoff(cc, G, gl), numeric(1))
    if (q >= exact || (!is.null(prev) && max(abs(p - prev)) <= EXACT_TOL)) return(p)
    prev <- p
    q <- min(2L * q, exact)
  }
}

# The paper's interpolation table, priced against the exact field.
exact_implicit_prices <- function(padded_cdf, field_cdfs, offsets, L) {
  gl <- gauss_legendre01(min(exact_n_nodes(length(field_cdfs)), EXACT_TABLE_NODES))
  G <- exact_field(field_cdfs, gl$nodes)
  vapply(offsets, function(k) exact_payoff(shifted_cdf(padded_cdf, k, L), G, gl),
         numeric(1))
}

# np.interp semantics: ascending xp, end-clamped, largest j with xp[j] <= x
np_interp <- function(x, xp, fp) {
  vapply(x, function(v) {
    if (v <= xp[1]) return(fp[1])
    n <- length(xp)
    if (v >= xp[n]) return(fp[n])
    j <- findInterval(v, xp)
    if (j >= n) return(fp[n])
    d <- xp[j + 1] - xp[j]
    if (d <= 0) return(fp[j])
    fp[j] + (v - xp[j]) / d * (fp[j + 1] - fp[j])
  }, numeric(1))
}

#' State prices for a race of translated copies of one density
#'
#' All contestants share the performance density up to translation by
#' `offsets` (in lattice units; lower is better). Returns the expected
#' payoff of each contestant against the field, dead heats split exactly.
#'
#' @param density numeric density on the symmetric lattice (length 2L+1)
#' @param offsets numeric vector of translations, lattice units
#' @return numeric vector of state prices (not renormalized)
#' @export
state_prices_from_offsets <- function(density, offsets) {
  density <- as_classic_density(density)
  L <- implied_L(density)
  base <- padded_base_cdf(density)
  cdfs <- lapply(offsets, function(o) shifted_cdf(base, o, L))
  exact_state_prices_from_cdfs(cdfs)
}
