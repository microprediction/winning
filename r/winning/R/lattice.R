# Lattice operations for the horse race problem.
#
# Pure base-R port of the reference implementation in the python `winning`
# package (winning/lattice.py). A density is a numeric vector of length
# 2L+1 interpreted on the integer lattice -L..L; performances are TIMES,
# so the LOWEST draw wins. Dead heats are handled exactly through the
# multiplicity recursion of the paper. The python implementation is the
# spec; tests/testthat/test-parity.R pins this port to golden values it
# produced.

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
    c(rep(0, k), cdf[1:(m - k)])
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

# Expected payoff of a contestant with cdf `cdf` against the field
# (cdf_all, mult_all): 1 if strictly best, 1/(1+multiplicity) on a tie.
# Left-tail multiplicity by inversion, right tail by the stable asymptotic
# form, switching at the first mode of the contestant density; the rest's
# cdf is forced monotone. Epsilons follow the reference exactly.
expected_payoff_sum <- function(cdf, cdf_all, mult_all) {
  f1 <- cdf_to_pdf(cdf)
  S <- 1 - cdf_all
  S1 <- 1 - cdf
  Srest <- (S + 1e-18) / (S1 + 1e-6)
  cdf_rest <- 1 - Srest
  f_rest <- cdf_to_pdf(cdf_rest)

  numer <- mult_all * f1 * Srest + mult_all * (f1 + S1) * f_rest -
    f1 * (Srest + f_rest)
  denom <- f_rest * (f1 + S1)
  mult_left <- (1e-18 + numer) / (1e-18 + denom)
  T1 <- (S1 + 1e-18) / (f1 + 1e-6)
  Trest <- (Srest + 1e-18) / (f_rest + 1e-6)
  mult_right <- mult_all * Trest / (1 + T1) + mult_all - (1 + Trest) / (1 + T1)
  k <- which.max(f1)
  mult_rest <- mult_left
  mult_rest[k:length(f1)] <- mult_right[k:length(f1)]

  run <- cummax(cdf_rest)
  fr <- diff(c(0, run))
  sum(f1 * (1 - run) + f1 * fr / (1 + mult_rest))
}

# Expected payoff of the base density shifted to each offset (float
# offsets blend the two integer shifts), against a fixed field.
implicit_state_prices <- function(base_cdf, cdf_all, mult_all, offsets, L) {
  vapply(offsets, function(k) {
    if (k == trunc(k)) {
      expected_payoff_sum(integer_shift(base_cdf, as.integer(k)),
                          cdf_all, mult_all)
    } else {
      lh <- low_high(k, L)
      lh$lo_coef * expected_payoff_sum(integer_shift(base_cdf, lh$lo),
                                       cdf_all, mult_all) +
        lh$up_coef * expected_payoff_sum(integer_shift(base_cdf, lh$up),
                                         cdf_all, mult_all)
    }
  }, numeric(1))
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
#' payoff of each contestant against the field, dead heats included.
#'
#' @param density numeric density on the symmetric lattice (length 2L+1)
#' @param offsets numeric vector of translations, lattice units
#' @return numeric vector of state prices (not renormalized)
#' @export
state_prices_from_offsets <- function(density, offsets) {
  L <- implied_L(density)
  base_cdf <- pdf_to_cdf(density)
  cdfs <- lapply(offsets, function(o) shifted_cdf(base_cdf, o, L))
  fold <- winner_of_many_cdfs(cdfs)
  implicit_state_prices(base_cdf, fold$cdf, fold$multiplicity, offsets, L)
}

winner_of_many_cdfs <- function(cdfs) {
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
  list(cdf = cdf_min, multiplicity = mult)
}
