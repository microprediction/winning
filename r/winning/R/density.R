# Skew-normal lattice density, matching the python reference.

symmetric_lattice <- function(L, unit) unit * seq(-L, L)

mean_of_density <- function(density, unit = 1.0) {
  L <- implied_L(density)
  sum(density * symmetric_lattice(L, unit))
}

fractional_shift <- function(cdf, x) {
  L <- implied_L(cdf)
  shifted_cdf(cdf, x, L)
}

fractional_shift_density <- function(density, x) {
  cdf_to_pdf(fractional_shift(pdf_to_cdf(density), x))
}

center_density <- function(density) {
  fractional_shift_density(density, -mean_of_density(density, unit = 1.0))
}

#' Skew-normal performance density on a symmetric lattice
#'
#' @param L half-width: the density has length 2L+1 on lattice -L..L
#' @param unit lattice spacing (performance units per lattice step)
#' @param loc location shift, in performance units
#' @param scale scale, in performance units
#' @param a skew parameter (a > 0 puts the fat tail on the right,
#'   i.e. slow stragglers, the usual racing shape)
#' @return numeric density of length 2L+1, centered then shifted by loc
#' @export
skew_normal_density <- function(L, unit, loc = 0, scale = 1.0, a = 2.0) {
  x <- symmetric_lattice(L, unit)
  t <- (x - loc) / scale
  density <- 2 / scale * stats::dnorm(t) * stats::pnorm(a * t)
  density <- density / sum(density)
  density <- center_density(density)
  # the reference applies the cdf-shift machinery to the density vector
  # directly for the final loc shift; replicated verbatim (loc = 0 is
  # unaffected)
  fractional_shift(density, loc / unit)
}
