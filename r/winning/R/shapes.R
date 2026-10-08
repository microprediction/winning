# Shared shape contracts for the public R entry points, parallel to
# winning/shapes.py. Base R recycles a short vector in silence whenever
# its length divides n, so every boundary that adds or divides by a
# per-runner vector must check the length itself; otherwise D = c(1, 4)
# at n = 4 prices the race D = c(1, 4, 1, 4) (#319) and a 2-row V at
# n = 4 prices a repeated-loading model (#343), with no warning.

# Idiosyncratic variances: scalar (broadcast) or exactly n finite
# entries; negative is never a variance; positive = TRUE (the lattice
# kernels, which divide by the sd) also refuses an exact zero.
.as_idio <- function(D, n, positive = TRUE, name = "D") {
  if (is.null(D)) return(rep(1, n))
  A <- as.numeric(D)
  if (length(A) == 1L) A <- rep(A, n)
  if (length(A) != n)
    stop(sprintf(paste("%s must be a scalar or one idiosyncratic variance",
                       "per contestant; got %d for %d contestants"),
                 name, length(A), n), call. = FALSE)
  if (any(!is.finite(A)))
    stop(sprintf("%s has a non-finite entry: it is a variance and must be finite",
                 name), call. = FALSE)
  if (any(A < 0)) {
    i <- which(A < 0)[1]
    stop(sprintf("%s[%d] = %g is a negative variance", name, i, A[i]),
         call. = FALSE)
  }
  if (positive && any(A <= 0)) {
    i <- which(A <= 0)[1]
    stop(sprintf(paste("%s must be strictly positive here: %s[%d] is zero.",
                       "A zero variance is a point mass, and the lattice",
                       "divides by its standard deviation."), name, name, i),
         call. = FALSE)
  }
  A
}

# Factor loadings as an (n, rank) matrix: scalar (common loading),
# length-n vector (rank one), (n, rank), or (rank, n) transposed. Any
# other shape is refused before quadrature.
.as_loadings <- function(V, n, name = "V") {
  if (is.null(dim(V))) {
    v <- as.numeric(V)
    if (any(!is.finite(v)))
      stop(sprintf("%s has a non-finite entry", name), call. = FALSE)
    if (length(v) == 1L) return(matrix(v, n, 1L))
    if (length(v) == n) return(matrix(v, n, 1L))
  } else {
    A <- as.matrix(V)
    storage.mode(A) <- "double"
    if (any(!is.finite(A)))
      stop(sprintf("%s has a non-finite entry", name), call. = FALSE)
    if (length(dim(A)) == 2L) {
      if (nrow(A) == n) return(A)
      if (ncol(A) == n) return(t(A))
    }
  }
  stop(sprintf(paste("%s must carry one row per contestant: expected a",
                     "scalar, a vector of length %d, or an (%d, rank)",
                     "matrix; got %s"), name, n, n,
               if (is.null(dim(V))) sprintf("length %d", length(V))
               else paste0("dim ", paste(dim(V), collapse = "x"))),
       call. = FALSE)
}

# An iteration budget is a COUNT: a single finite non-negative whole
# number. `for (it in 1:n_iter)` with n_iter = 0 runs it = 1, 0 -- two
# updates reported as zero (#441) -- so loops use seq_len() and callers
# validate here first.
.as_iter_budget <- function(n_iter, name = "n_iter") {
  v <- suppressWarnings(as.numeric(n_iter))
  if (length(v) != 1L || is.na(v) || !is.finite(v) || v < 0 || v != trunc(v))
    stop(sprintf("%s must be a single finite non-negative whole number; got %s",
                 name, paste(format(n_iter), collapse = " ")), call. = FALSE)
  as.integer(v)
}

# A tolerance is one finite positive number: `resid < Inf` certified the
# first finite residual of every inverse as converged, returning warm
# starts 15-24 points off (#551). The browser's asTolerance refuses it.
.as_tolerance <- function(tol, name = "tol") {
  if (!is.numeric(tol) || length(tol) != 1L || is.na(tol) ||
      !is.finite(tol) || tol <= 0)
    stop(sprintf("%s must be a single finite positive number; got %s",
                 name, paste(format(tol), collapse = " ")), call. = FALSE)
  as.numeric(tol)
}

# A penalty weight: one finite non-negative number. sqrt(max(ridge, 0))
# read every negative ridge as 0 and a vector as its max (#558).
.as_nonnegative <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
      x < 0)
    stop(sprintf("%s must be a single finite non-negative number; got %s",
                 name, paste(format(x), collapse = " ")), call. = FALSE)
  as.numeric(x)
}

# A target is a law up to a positive factor. When its SUM overflows --
# every entry finite, e.g. c(4e307, 2e307, 1e307, 1e307) -- rescale by
# the max first, as python does since #300; the ratios, which are all a
# target carries, survive. Conditional, so every ordinary input stays
# bit-identical: the branch is taken only where the plain division
# already returned NaN (#326).
.rescaled_target <- function(target, mass = NULL) {
  tot <- sum(target)
  if (!is.finite(tot)) {
    target <- target / max(target)
    tot <- sum(target)
  }
  # `mass` keeps the caller's own arithmetic, so the ordinary path is
  # bit-identical to the expression it replaced
  if (is.null(mass)) target / tot else target * (mass / tot)
}

# A warm start selects a branch of an inverse; it must not redefine the
# field. R recycled a 2n mu0 against n targets and returned 2n abilities
# (#554). Finite, exactly n entries, then centred.
.as_warm_start <- function(mu0, n, name = "mu0") {
  m <- suppressWarnings(as.numeric(mu0))
  if (length(m) != n)
    stop(sprintf("%s has %d entries for %d runners", name, length(m), n),
         call. = FALSE)
  if (any(!is.finite(m)))
    stop(sprintf("%s has a non-finite entry", name), call. = FALSE)
  m - mean(m)
}
