\name{state_prices_from_offsets}
\alias{state_prices_from_offsets}
\alias{winner_of_many}
\title{Lattice race primitives}
\usage{
state_prices_from_offsets(density, offsets)

winner_of_many(densities)
}
\arguments{
\item{density}{numeric density on the symmetric lattice (length 2L+1)}
\item{offsets}{numeric translations in lattice units (lower is better)}
\item{densities}{list of numeric densities, all the same length}
}
\value{
\code{state_prices_from_offsets} returns the expected payoff of each
contestant against the field (1 if strictly first, split on dead heats).
\code{winner_of_many} returns a list with the \code{density} of the
minimum of the field and the dead-heat \code{multiplicity} lattice.
}
\description{
The two primitives of the lattice race: the distribution of the winning
(minimum) time of a field, with exact dead-heat multiplicity tracking,
and the state prices of translated copies of a common density.
}
