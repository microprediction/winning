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
