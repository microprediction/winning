\name{prices_from_dividends}
\alias{prices_from_dividends}
\alias{dividends_from_prices}
\title{Convert between dividends and probabilities}
\usage{
prices_from_dividends(dividends, nan_value = 2000)

dividends_from_prices(prices, multiplicity = 1.0)
}
\arguments{
\item{dividends}{numeric decimal prices, \code{NA} allowed}
\item{prices}{numeric win probabilities}
\item{nan_value}{dividend assigned to \code{NA} entries}
\item{multiplicity}{dead-heat multiplicity divisor}
}
\value{Numeric vector: normalized probabilities, or dividends.}
\description{
Naive renormalization between Australian-style dividends (decimal
prices) and risk-neutral win probabilities.
}
