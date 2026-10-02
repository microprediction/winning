# The horse race problem: infer relative ability from win probabilities.
#
# Reference: Cotton, "Inferring relative ability from winning probability
# in multi-entrant contests" (SIAM), and the python `winning` package,
# of which this is a base-R port (winning/lattice_calibration.py).

NAN_DIVIDEND <- 2000  # longshots with no bid

normalize <- function(p) p / sum(p)

#' Risk-neutral probabilities from Australian-style dividends
#' @param dividends numeric decimal prices, NA allowed
#' @param nan_value dividend assigned to NA entries
#' @return numeric probabilities summing to one
#' @export
prices_from_dividends <- function(dividends, nan_value = 2000) {
  # Only a MISSING quote -- NA or NaN -- becomes nan_value. A
  # non-positive dividend, and -Inf with it, is worth nothing and prices
  # at 0; +Inf prices at 1/Inf = 0 on its own. This divided by the
  # dividend unconditionally, so a dividend of 0 gave Inf and then NaN
  # after normalising, and a negative dividend came back as a NEGATIVE
  # probability. Normalising only when the total is positive is python's
  # rule too: an all-infinite book is all zeros, not 0/0 (#242).
  d <- ifelse(is.na(dividends), nan_value, dividends)
  p <- ifelse(d <= 0, 0, 1 / d)
  s <- sum(p)
  if (s > 0) p / s else p
}

#' Australian-style dividends from probabilities
#' @param prices numeric win probabilities
#' @param multiplicity dead-heat multiplicity divisor (default 1)
#' @return numeric dividends
#' @export
dividends_from_prices <- function(prices, multiplicity = 1.0) {
  p <- normalize(prices)
  ifelse(!is.na(p) & p > 0, 1 / (multiplicity * p), NA_real_)
}

#' Solve the horse race problem: offsets matching given state prices
#'
#' The fixed-point iteration of the paper: tabulate offset -> implied
#' price against the current field, correct each offset by the table's
#' reading of target minus current price, rebuild the field.
#'
#' @param prices numeric state prices (positive, ideally summing to one)
#' @param density performance density on the symmetric lattice
#' @param offset_samples descending offsets for the interpolation table
#'   (default: the reference's half-lattice grid)
#' @param implied_offsets_guess starting offsets, one per price (default zeros)
#' @param n_iter fixed-point iterations (default 3)
#' @return numeric offsets in lattice units (lower is better)
#' @export
solve_for_implied_offsets <- function(prices, density,
                                      offset_samples = NULL,
                                      implied_offsets_guess = NULL,
                                      n_iter = 3) {
  density <- as_classic_density(density)
  prices <- as_classic_prices(prices)
  L <- implied_L(density)
  if (is.null(offset_samples)) {
    offset_samples <- rev(seq.int(-(L %/% 2), (L %/% 2) - 1L))
  } else if (length(offset_samples) == 0) {
    stop("offset_samples is empty; there is nothing to interpolate against")
  } else if (!all(is.finite(offset_samples))) {
    stop("offset_samples has a non-finite offset")
  } else if (any(diff(offset_samples) > 0)) {
    stop("offset_samples must be descending")
  }
  # one starting offset per target price; the default was L %/% 3
  # offsets -- lattice width as contestant count (#369)
  if (is.null(implied_offsets_guess)) {
    implied_offsets_guess <- rep(0, length(prices))
  } else if (length(implied_offsets_guess) != length(prices)) {
    stop("implied_offsets_guess must have one starting offset per price: got ",
         length(implied_offsets_guess), " for ", length(prices), " prices")
  }
  # defect correction a <- a + T^{-1}(p) - T^{-1}(P(a)) against the exact
  # engine, so the fixed point is the exact forward map
  # (see winning/classic/lattice_calibration.py)
  base <- padded_base_cdf(density)
  implied <- as.numeric(implied_offsets_guess)
  cdfs <- lapply(implied, function(o) shifted_cdf(base, o, L))
  for (i in seq_len(n_iter)) {
    tab <- exact_implicit_prices(base, cdfs, offset_samples, L)
    current <- exact_state_prices_from_cdfs(cdfs)
    implied <- implied + np_interp(prices, tab, offset_samples) -
      np_interp(current, tab, offset_samples)
    cdfs <- lapply(implied, function(o) shifted_cdf(base, o, L))
  }
  implied
}

#' Ability implied by state prices
#' @param prices numeric win probabilities (positive)
#' @param density performance density on the symmetric lattice
#' @param unit lattice spacing used when the density was constructed
#' @return numeric abilities (lower is better), in units of `unit`
#' @export
state_price_implied_ability <- function(prices, density, unit = 1.0) {
  guess <- rep(0, length(prices))
  solve_for_implied_offsets(prices, density,
                            implied_offsets_guess = guess) * unit
}

#' Ability implied by dividends (decimal odds)
#' @param dividends numeric decimal prices, NA allowed
#' @param density performance density on the symmetric lattice
#' @param nan_value dividend assigned to NA entries
#' @param unit lattice spacing used when the density was constructed
#' @return numeric abilities (lower is better)
#' @export
dividend_implied_ability <- function(dividends, density,
                                     nan_value = 2000, unit = 1.0) {
  p <- prices_from_dividends(dividends, nan_value = nan_value)
  state_price_implied_ability(p, density, unit = unit)
}

#' State prices implied by ability (the forward direction)
#' @param ability numeric abilities (lower is better)
#' @param density performance density on the symmetric lattice
#' @param unit lattice spacing used when the density was constructed
#' @return numeric state prices
#' @export
ability_implied_state_prices <- function(ability, density, unit = 1.0) {
  offsets <- ability / unit
  L <- implied_L(density)
  # center, as the reference's extended handling does before pricing
  offsets <- offsets - round(mean(range(offsets)))
  state_prices_from_offsets(density, offsets)
}

#' Dividends implied by ability
#' @param ability numeric abilities (lower is better)
#' @param density performance density on the symmetric lattice
#' @param unit lattice spacing used when the density was constructed
#' @return numeric dividends (inverse state prices)
#' @export
ability_implied_dividends <- function(ability, density, unit = 1.0) {
  1 / ability_implied_state_prices(ability, density, unit = unit)
}
