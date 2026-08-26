\name{skew_normal_density}
\alias{skew_normal_density}
\title{Skew-normal performance density on a symmetric lattice}
\usage{
skew_normal_density(L, unit, loc = 0, scale = 1.0, a = 2.0)
}
\arguments{
\item{L}{half-width: the density has length 2L+1 on lattice -L..L}
\item{unit}{lattice spacing (performance units per lattice step)}
\item{loc}{location shift, in performance units}
\item{scale}{scale, in performance units}
\item{a}{skew (a > 0 puts the fat tail on the right: slow stragglers,
the usual racing shape)}
}
\value{Numeric density of length 2L+1.}
\description{
The reference performance distribution of the horse race problem:
proportional to \code{2/scale * dnorm(t) * pnorm(a t)}, normalized and
centered on the lattice.
}
