## Resubmission

This is a resubmission of a new package. The previous submission
(0.1.0) was returned with two requests, both addressed:

* "Please write function names with parentheses as in
  `mvtnorm::pmvnorm()`" — done throughout the DESCRIPTION, the Rd file,
  the roxygen comments and the README.

* "Is there some reference about the method you can add in the
  Description field in the form Authors (year) <doi:10.....>?" — the
  Description now cites Cotton (2026) <doi:10.2139/ssrn.7307363>, which
  describes the method this package implements, and Butler and Moffitt
  (1982) <doi:10.2307/1912613>, the factor-conditioning quadrature it
  extends. Both DOIs pass the incoming checks.

## R CMD check results

0 errors, 0 warnings, 2 NOTEs.

* "New submission" — expected for a first-time package.
* An HTML-tidy / V8 tooling note from the local check environment
  (HTML validation and math rendering skipped); not a package issue.

Checked with R CMD check --as-cran on R release (macOS). No reverse
dependencies (new package).
