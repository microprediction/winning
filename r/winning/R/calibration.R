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
  d <- ifelse(is.na(dividends), nan_value, dividends)
  normalize(1 / d)
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
#' The fixed-point iteration of the paper: build the winner-of-field
#' density, tabulate offset -> implied price, interpolate the target
#' prices to offsets, rebuild the field; three iterations suffice.
#'
#' @param prices numeric state prices (positive, ideally summing to one)
#' @param density performance density on the symmetric lattice
#' @param offset_samples descending offsets for the interpolation table
#'   (default: the reference's half-lattice grid)
#' @param implied_offsets_guess starting offsets (default: the reference's)
#' @param n_iter fixed-point iterations (default 3)
#' @return numeric offsets in lattice units (lower is better)
#' @export
solve_for_implied_offsets <- function(prices, density,
                                      offset_samples = NULL,
                                      implied_offsets_guess = NULL,
                                      n_iter = 3) {
  L <- implied_L(density)
  if (is.null(offset_samples)) {
    offset_samples <- rev(seq.int(-(L %/% 2), (L %/% 2) - 1L))
  } else if (any(diff(offset_samples) > 0)) {
    stop("offset_samples must be descending")
  }
  if (is.null(implied_offsets_guess)) {
    implied_offsets_guess <- seq.int(0L, (L %/% 3) - 1L)
  }
  base_cdf <- pdf_to_cdf(density)
  cdfs <- lapply(implied_offsets_guess,
                 function(o) shifted_cdf(base_cdf, o, L))
  implied <- prices
  for (i in seq_len(n_iter)) {
    fold <- winner_of_many_cdfs(cdfs)
    tab <- implicit_state_prices(base_cdf, fold$cdf, fold$multiplicity,
                                 offset_samples, L)
    implied <- np_interp(prices, tab, offset_samples)
    cdfs <- lapply(implied, function(o) shifted_cdf(base_cdf, o, L))
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
