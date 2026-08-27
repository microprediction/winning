## R CMD check results

0 errors | 0 warnings | 1 note

* "New submission": this is the package's first CRAN submission.

A third note about HTML Tidy appears only on the development machine
(old local tidy binary) and does not concern the package.

## Package background

Base-R implementation (no compiled code, no dependencies beyond stats)
of the lattice contest-calibration algorithms of Cotton (2021),
doi:10.1137/19M1276261, and their structured-covariance extensions. The
package is parity-locked against the reference Python implementation:
an embedded vector file of 22 scenarios is replayed by both, most at
machine precision (see tests/).
