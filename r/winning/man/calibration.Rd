\name{dividend_implied_ability}
\alias{dividend_implied_ability}
\alias{state_price_implied_ability}
\alias{ability_implied_state_prices}
\alias{ability_implied_dividends}
\alias{solve_for_implied_offsets}
\title{The horse race problem: ability from win probabilities, and back}
\usage{
dividend_implied_ability(dividends, density, nan_value = 2000, unit = 1.0)

state_price_implied_ability(prices, density, unit = 1.0)

ability_implied_state_prices(ability, density, unit = 1.0)

ability_implied_dividends(ability, density, unit = 1.0)

solve_for_implied_offsets(prices, density, offset_samples = NULL,
  implied_offsets_guess = NULL, n_iter = 3)
}
\arguments{
\item{dividends}{numeric decimal prices (Australian style), \code{NA} allowed}
\item{prices}{numeric state prices (win probabilities), positive}
\item{ability}{numeric relative abilities; performances are times, so
lower is better}
\item{density}{numeric performance density on the symmetric lattice
\code{-L..L} (length \code{2L+1}), e.g. from
\code{\link{skew_normal_density}}}
\item{nan_value}{dividend assigned to \code{NA} entries}
\item{unit}{lattice spacing assumed when the density was constructed}
\item{offset_samples}{descending offsets for the interpolation table
(default: the reference's half-lattice grid)}
\item{implied_offsets_guess}{starting offsets}
\item{n_iter}{fixed-point iterations (default 3)}
}
\value{
The \code{*_implied_ability} functions return numeric abilities (lower is
better) in units of \code{unit}; \code{ability_implied_state_prices}
returns state prices; \code{ability_implied_dividends} their inverses;
\code{solve_for_implied_offsets} raw lattice offsets.
}
\description{
Given win probabilities for a multi-entrant contest in which every
contestant's performance is a translated copy of one density, infer the
translations (relative ability), or price the contest in the forward
direction, with dead heats handled exactly. Implements the lattice
fixed-point algorithm of Cotton (2021), \doi{10.1137/19M1276261}; this
is a base-R port of the reference python package \code{winning}.
}
\examples{
d <- skew_normal_density(L = 200, unit = 0.02, a = 1.5)
ability <- dividend_implied_ability(c(2, 6, 6, 20), d)
p <- ability_implied_state_prices(ability, d)
}
