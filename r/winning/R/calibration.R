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
ability_implied_state_prices <- function(ability, density, unit = 1.0,
                                         max_depth = 3L) {
  # the reference's full extended handling, not just centring: a runner
  # hundreds of support widths behind pushed the viable ones off the
  # lattice edge, where the shift clamp made them identical -- a 75/25
  # pair priced 50/50 once a no-chance third runner was added (#393)
  .state_prices_from_extended_offsets(density, as.numeric(ability) / unit,
                                     max_depth = max_depth)
}

# --- port of winning/classic/lattice.py's extended-offset handling ---

.mean_ignoring_inf <- function(o) mean(o[is.finite(o)])

# python's int(): truncation toward zero
.int_centered <- function(o) o - trunc(.mean_ignoring_inf(o))

.approximate_support_width <- function(density, tol = 1e-12) {
  supp <- which(density > tol)
  max(supp) - min(supp)
}

.dilate_density <- function(density, unit_ratio = 2) {
  L <- implied_L(density)
  out <- numeric(2L * L + 1L)
  for (k in seq_along(density)) {
    lh <- low_high((k - 1L - L) / unit_ratio, L)
    for (pr in list(c(lh$lo, lh$lo_coef), c(lh$up, lh$up_coef))) {
      rel <- min(2L * L, max(pr[1] + L, 0)) + 1L
      out[rel] <- out[rel] + density[k] * pr[2]
    }
  }
  out / sum(out)
}

.divide_offsets <- function(centered) {
  n <- length(centered)
  if (n == 2L) return(mean(centered))
  srt <- sort(centered)
  max_best <- min(20L, as.integer(n / 6 + 2))
  gaps <- abs(diff(c(srt[1], srt)))[seq_len(min(n, max_best + 1L))]
  i0 <- max(1L, which.max(gaps) - 1L)        # python's 0-based index
  (srt[i0] + srt[i0 + 1L]) / 2
}

.clustered_state_prices <- function(density, offsets, fast, unit_ratio,
                                    max_depth) {
  dd <- .dilate_density(density, unit_ratio)
  doff <- offsets / unit_ratio
  dsp <- .state_prices_from_extended_offsets(dd, doff, max_depth - 1L)
  n <- length(offsets)
  if (length(fast) == n || length(fast) == 0L || max_depth <= 0L)
    return(dsp)
  slow <- setdiff(seq_len(n), fast)
  slow_rel <- if (length(slow) == 1L) 1 else
    .state_prices_from_extended_offsets(dd, .int_centered(doff[slow]),
                                       max_depth - 2L)
  fast_rel <- .state_prices_from_extended_offsets(
    density, .int_centered(offsets[fast]), max_depth - 1L)
  slow_share <- sum(dsp[slow])
  sp <- numeric(n)
  sp[slow] <- slow_rel * slow_share
  sp[fast] <- fast_rel * (1 - slow_share)
  sp / sum(sp)
}

# State prices from offsets that may be infinite or far off the lattice
#
# Port of the python reference's state_prices_from_extended_offsets:
# Inf offsets (no chance) are removed, -Inf split the pot,
# runners beyond the density's support from the best are set to
# Inf, a lone standout is a walkover, and fields too wide for the
# lattice are clustered and dilated recursively.
# @param density performance density on the symmetric lattice
# @param offsets numeric offsets in lattice units (lower is better)
# @param max_depth recursion budget for the clustering
# @param unit_ratio dilation ratio for the clustered approximation
# @return numeric state prices
.state_prices_from_extended_offsets <- function(density, offsets,
                                               max_depth = 3L,
                                               unit_ratio = 3) {
  offsets <- as.numeric(offsets)
  n <- length(offsets)
  if (anyNA(offsets)) stop("offsets must not be NA", call. = FALSE)
  if (n == 1L) return(1)
  pinf <- offsets == Inf
  if (any(pinf)) {
    if (all(pinf)) return(rep(1 / n, n))
    sp <- numeric(n)
    sp[!pinf] <- .state_prices_from_extended_offsets(
      density, .int_centered(offsets[!pinf]), max_depth)
    return(sp)
  }
  ninf <- offsets == -Inf
  if (any(ninf)) {
    sp <- numeric(n)
    sp[ninf] <- 1 / sum(ninf)
    return(sp)
  }
  L <- implied_L(density)
  W <- as.integer(.approximate_support_width(density))
  bad <- offsets > min(offsets) + W          # no chance of winning
  if (any(bad)) {
    aug <- offsets
    aug[bad] <- Inf
    return(.state_prices_from_extended_offsets(density, .int_centered(aug),
                                              max_depth))
  }
  # python's walkover test compares min(diff_to_best), which is always 0,
  # so it never fires; kept for fidelity
  diff_best <- offsets - min(offsets)
  if (min(diff_best) > W * sqrt(n)) {
    sp <- numeric(n)
    sp[which.min(offsets)] <- 1
    return(sp)
  }
  co <- .int_centered(offsets)
  # Hanging = the shifted density would lose mass off the lattice (or the
  # offset is past the exact range of low_high). Python tests the cruder
  # |o| <= L - W, which misfires whenever the support width W exceeds L:
  # then even an all-zero field "hangs" and is clustered and dilated --
  # python's own round trip on skew_normal_density(L=500, unit=0.01)
  # is 0.015 off because of it, against 2e-6 for the direct pricing.
  # Measuring the actual support keeps every in-window field on the exact
  # primitive and agrees with python wherever its bound is meaningful.
  supp <- which(density > 1e-12) - 1L - L
  left <- which(floor(co) + min(supp) < -L | co <= -L + 2)
  right <- which(ceiling(co) + max(supp) > L | co >= L - 2)
  if (!length(left) && !length(right))
    return(state_prices_from_offsets(density, co))
  if (max_depth == 0L) {
    co[right] <- Inf
    co[left] <- -Inf
    return(.state_prices_from_extended_offsets(density, co, 0L))
  }
  divider <- .divide_offsets(co)
  fast <- which(co < divider)
  .clustered_state_prices(density, co, fast, unit_ratio, max_depth - 1L)
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
