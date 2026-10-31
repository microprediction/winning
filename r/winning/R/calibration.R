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
#'   (default: every representable offset, L-2 down to -(L-2))
#' @param implied_offsets_guess starting offsets, one per price (default zeros)
#' @param n_iter iteration budget (default 100); the iteration stops early
#'   once the exact reprice of the current field is within `tol` of every
#'   target, or when two successive steps fail to reduce the error, and
#'   returns the best iterate it assessed
#' @param tol stopping tolerance on the maximum price error (default 1e-6),
#'   finite and positive
#' @return numeric offsets in lattice units (lower is better)
#' @export
solve_for_implied_offsets <- function(prices, density,
                                      offset_samples = NULL,
                                      implied_offsets_guess = NULL,
                                      n_iter = CALIBRATION_MAX_ITER,
                                      tol = CALIBRATION_TOL) {
  density <- as_classic_density(density)
  prices <- as_classic_prices(prices)
  tol <- .as_tolerance(tol)
  # a loop bound: zero or negative skipped every step and returned the
  # warm start as a calibration (#543); the browser refuses these too
  if (length(n_iter) != 1L || !is.numeric(n_iter) || !is.finite(n_iter) ||
      n_iter < 1 || n_iter != round(n_iter))
    stop("n_iter must be a positive integer number of iterations; got ",
         deparse(n_iter), ". A zero or negative budget skips the solve and ",
         "returns the warm start.", call. = FALSE)
  L <- implied_L(density)
  core <- NULL
  if (is.null(offset_samples)) {
    offset_samples <- .default_offset_samples(L)
    core <- rev(seq.int(-(L %/% 2), (L %/% 2) - 1L))   # the old half table
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
  # after each step the field is re-centred by the integer part of its
  # mean (an exact lattice translation): the table is absolute, so a
  # drifting field wasted half of it and a 97/3 book stalled at 88/12
  # however many iterations ran (#498)
  # each iterate is assessed by its exact reprice before the next step:
  # stop once the best miss is within tol, or two successive iterates fail
  # to improve on it (a plateau), and return the best iterate; n_iter
  # bounds the steps (#904)
  best <- NULL; best_miss <- Inf; stalls <- 0L
  assess <- function(field, current) {
    miss <- max(abs(current - prices))
    if (miss < best_miss) {
      best <<- field; best_miss <<- miss; stalls <<- 0L
    } else {
      stalls <<- stalls + 1L
    }
    best_miss <= tol || stalls >= 2L
  }
  exhausted <- TRUE
  for (i in seq_len(n_iter)) {
    current <- exact_state_prices_from_cdfs(cdfs)
    if (assess(implied, current)) { exhausted <- FALSE; break }
    st <- .step_table(base, cdfs, offset_samples, core, L, prices, current)
    implied <- implied + np_interp(prices, st$tab, st$samples) -
      np_interp(current, st$tab, st$samples)
    implied <- implied - trunc(mean(implied))
    cdfs <- lapply(implied, function(o) shifted_cdf(base, o, L))
  }
  if (exhausted) assess(implied, exact_state_prices_from_cdfs(cdfs))
  implied <- best
  cdfs <- lapply(implied, function(o) shifted_cdf(base, o, L))
  .warn_if_unconverged(exact_state_prices_from_cdfs(cdfs), prices)
  implied
}

# Default interpolation table for the classic inverse: every integer
# offset the lattice represents, L-2 down to -(L-2) (low_high pins
# anything beyond). The half-lattice default endpoint-clamped targets the
# forward reaches: a 97/3 book repriced at 88/12 (#498). python's
# default_offset_samples.
.default_offset_samples <- function(L) rev(seq.int(-(L - 2L), L - 2L))

# With the default table, price the central slice first and extend to the
# full range only when a lookup reaches the slice's ends: inside it the
# interpolation is identical (same integer samples, same field), so the
# answer is the full table's at the half table's cost.
.step_table <- function(base, cdfs, offset_samples, core, L, prices, current) {
  if (is.null(core))
    return(list(samples = offset_samples,
                tab = exact_implicit_prices(base, cdfs, offset_samples, L)))
  t <- exact_implicit_prices(base, cdfs, core, L)
  lo <- t[1]; hi <- t[length(t)]
  x <- c(prices, current)
  if (all(x > lo & x < hi)) return(list(samples = core, tab = t))
  top <- offset_samples[offset_samples > core[1]]
  bottom <- offset_samples[offset_samples < core[length(core)]]
  ext <- function(k) if (length(k)) exact_implicit_prices(base, cdfs, k, L) else numeric(0)
  list(samples = offset_samples, tab = c(ext(top), t, ext(bottom)))
}

# The inverse iterates until its own exact reprice is within
# CALIBRATION_TOL of every target, or CALIBRATION_MAX_ITER steps have run,
# or two successive steps fail to reduce the error, and returns the best
# iterate it assessed. The budget was a fixed three sweeps: ordinary books
# reach 1e-4 to 1e-7 in three, but an atomic law converges slowly and a
# three-runner two-atom market generated by the forward itself came back
# 1.27 points off (#904). The stall rule is for a target the lattice
# cannot resolve, such as a straggler priced at exactly zero, where the
# offset drifted along a plateau for the whole budget. python's
# CALIBRATION_MAX_ITER / CALIBRATION_TOL and the same rule.
CALIBRATION_MAX_ITER <- 100L
CALIBRATION_TOL <- 1e-6

# a calibration whose own reprice misses its target by more than this is
# reported: the target is outside what the lattice represents or n_iter
# is too small (#498); python's CALIBRATION_WARN_TOL
CALIBRATION_WARN_TOL <- 1e-3

.warn_if_unconverged <- function(repriced, prices) {
  miss <- max(abs(repriced - prices))
  if (miss > CALIBRATION_WARN_TOL)
    warning(sprintf(paste0(
      "solve_for_implied_offsets did not reach the target: max |price error| = %.3g ",
      "after calibration. The target may lie outside what this lattice represents ",
      "(an exact zero, or a longshot past the lattice edge: use a wider L or finer ",
      "unit), or n_iter is too small."), miss), call. = FALSE)
  invisible(miss)
}

.check_unit <- function(unit, where) {
  # the lattice spacing is finite and positive: a negative one reversed
  # every ability, zero tied the field, NaN/Inf gave non-finite (#649)
  if (length(unit) != 1L || !is.numeric(unit) || !is.finite(unit) || unit <= 0)
    stop(where, ": unit must be a finite positive number; got ",
         deparse(unit), call. = FALSE)
  invisible(unit)
}

.check_max_depth <- function(max_depth, where) {
  # a recursion budget, a whole number >= 0 as the browser requires: the
  # recursion stops only on max_depth == 0, so -1 and TRUE ran as depth 1
  # and 1.5 and Inf as depth 2 (#775)
  if (length(max_depth) != 1L || !is.numeric(max_depth) || !is.finite(max_depth) ||
      max_depth != round(max_depth) || max_depth < 0)
    stop(where, ": max_depth must be a whole number >= 0; got ",
         deparse(max_depth), call. = FALSE)
  invisible(max_depth)
}

#' Ability implied by state prices
#' @param prices numeric win probabilities (nonnegative; a zero price
#'   returns ability Inf)
#' @param density performance density on the symmetric lattice
#' @param unit lattice spacing used when the density was constructed
#' @param n_iter,tol iteration budget and stopping tolerance of
#'   [solve_for_implied_offsets()]
#' @return numeric abilities (lower is better), in units of `unit`
#' @export
state_price_implied_ability <- function(prices, density, unit = 1.0,
                                        n_iter = CALIBRATION_MAX_ITER,
                                        tol = CALIBRATION_TOL) {
  .check_unit(unit, "state_price_implied_ability")
  # A zero price -- a scratched runner, or the zero, negative or infinite
  # dividend prices_from_dividends maps to 0 -- is not calibrated: its
  # ability is Inf (lower is better, so it never wins;
  # ability_implied_state_prices prices it at 0) and the rest are
  # calibrated among themselves. The finite lattice inverse returned an
  # ordinary offset that repriced it at 4.5% (16% at L=3) (#589).
  p <- as_classic_prices(prices)
  live <- which(p > 0)
  ability <- rep(Inf, length(p))
  if (length(live) == 1L) {
    ability[live] <- 0
  } else {
    ability[live] <- solve_for_implied_offsets(p[live], density,
                                               n_iter = n_iter, tol = tol) * unit
  }
  ability
}

#' Ability implied by dividends (decimal odds)
#' @param dividends numeric decimal prices, NA allowed
#' @param density performance density on the symmetric lattice
#' @param nan_value dividend assigned to NA entries
#' @param unit lattice spacing used when the density was constructed
#' @param n_iter,tol iteration budget and stopping tolerance of
#'   [solve_for_implied_offsets()]
#' @return numeric abilities (lower is better)
#' @export
dividend_implied_ability <- function(dividends, density,
                                     nan_value = 2000, unit = 1.0,
                                     n_iter = CALIBRATION_MAX_ITER,
                                     tol = CALIBRATION_TOL) {
  p <- prices_from_dividends(dividends, nan_value = nan_value)
  state_price_implied_ability(p, density, unit = unit, n_iter = n_iter, tol = tol)
}

#' State prices implied by ability (the forward direction)
#' @param ability numeric abilities (lower is better)
#' @param density performance density on the symmetric lattice
#' @param unit lattice spacing used when the density was constructed
#' @return numeric state prices
#' @export
ability_implied_state_prices <- function(ability, density, unit = 1.0,
                                         max_depth = 3L) {
  .check_unit(unit, "ability_implied_state_prices")
  .check_max_depth(max_depth, "ability_implied_state_prices")
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
  # ... except that low_high mixes a fractional offset's floor and ceil
  # shifts, so a runner just past W can still dead-heat the best at the
  # support endpoint: [0, 1.0001] at W = 1 was [1, 0] against the exact
  # 0.125 claim (#794). Only a runner whose lowest cell lies past the
  # best's highest is certainly out. A near miss is priced exactly when
  # the field fits the lattice, and dropped as before when it would hang.
  kept <- which(!(floor(offsets) > ceiling(min(offsets)) + W))
  # python's walkover test compares min(diff_to_best), which is always 0,
  # so it never fires; kept for fidelity
  diff_best <- offsets - min(offsets)
  if (min(diff_best) > W * sqrt(n)) {
    sp <- numeric(n)
    sp[which.min(offsets)] <- 1
    return(sp)
  }
  co <- .int_centered(offsets[kept])
  # Hanging = the shifted density would lose mass off the lattice (or the
  # offset is past the exact range of low_high). Python tests the cruder
  # |o| <= L - W, which misfires whenever the support width W exceeds L:
  # then even an all-zero field "hangs" and is clustered and dilated --
  # python's own round trip on skew_normal_density(L=500, unit=0.01)
  # is 0.015 off because of it, against 2e-6 for the direct pricing.
  # Measuring the actual support keeps every in-window field on the exact
  # primitive and agrees with python wherever its bound is meaningful.
  #
  # Once W reaches L the support fills the lattice (the standard L=500,
  # unit=0.01 normal has W = 1000) and every nonzero offset "leaves" it,
  # so this test sent python's own 8:4:2:1 round trip to the dilating
  # cluster path as well, 0.017 off (#725). The exact primitive piles
  # mass leaving the left edge at -L and drops it past +L, which keeps
  # who wins up to edge dead heats; for such a density only an offset
  # past the representable range hangs -- python's rule there too.
  if (W >= L) {
    left <- which(co < -(L - 2))
    right <- which(co > L - 2)
  } else {
    supp <- which(density > 1e-12) - 1L - L
    # low_high represents an offset of exactly +-(L - 2); only one past it
    # is clamped, so the range test is strict (#773)
    left <- which(floor(co) + min(supp) < -L | co < -L + 2)
    right <- which(ceiling(co) + max(supp) > L | co > L - 2)
  }
  if (!length(left) && !length(right)) {
    if (length(kept) == n) return(state_prices_from_offsets(density, co))
    sp <- numeric(n)
    sp[kept] <- if (length(kept) == 1L) 1 else state_prices_from_offsets(density, co)
    return(sp)
  }
  if (any(bad)) {
    aug <- offsets
    aug[bad] <- Inf
    return(.state_prices_from_extended_offsets(density, .int_centered(aug),
                                              max_depth))
  }
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
