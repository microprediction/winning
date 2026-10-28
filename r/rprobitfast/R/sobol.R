# Scrambled Sobol nodes, dependency-free base R.
#
# GENERATED FILE: the same text is shipped in r/mvtnormfast, r/mlogitfast
# and r/rprobitfast (each package is self-contained), and a test in each
# package checks the unscrambled points against scipy's.
#
# Why not Halton: the old fixed-prime Halton tables failed outright past
# their length (rank 7 in mvtnormfast, #143; r + 1 = 5 in the MNP
# likelihoods, #388) and, below that, plain Halton's low-dimensional
# projections degrade past three dimensions -- a rank-4 rectangle was
# 1.06% high at 8192 points (#143). The python reference uses scrambled
# Sobol (scipy.stats.qmc.Sobol); this is the same construction: Joe-Kuo
# new-joe-kuo-6.21201 direction numbers (the table scipy ships, first
# 64 dimensions), a random linear matrix scramble plus a digital shift.
# The scramble randomness comes from a private Park-Miller generator, so
# the user's RNG stream is never touched and the nodes are deterministic.

.SOBOL_MAXDIM <- 64L
.SOBOL_BITS <- 30L
.SOBOL_POLY <- c(
  1, 3, 7, 11, 13, 19, 25, 37, 41, 47, 55, 59, 61, 67, 91, 97, 103, 109,
  115, 131, 137, 143, 145, 157, 167, 171, 185, 191, 193, 203, 211, 213,
  229, 239, 241, 247, 253, 285, 299, 301, 333, 351, 355, 357, 361, 369,
  391, 397, 425, 451, 463, 487, 501, 529, 539, 545, 557, 563, 601, 607,
  617, 623, 631, 637
)
.SOBOL_VINIT <- list(
  c(1L), c(1L), c(1L, 3L), c(1L, 3L, 1L), c(1L, 1L, 1L), c(1L, 1L, 3L,
  3L), c(1L, 3L, 5L, 13L), c(1L, 1L, 5L, 5L, 17L), c(1L, 1L, 5L, 5L,
  5L), c(1L, 1L, 7L, 11L, 19L), c(1L, 1L, 5L, 1L, 1L), c(1L, 1L, 1L, 3L,
  11L), c(1L, 3L, 5L, 5L, 31L), c(1L, 3L, 3L, 9L, 7L, 49L), c(1L, 1L,
  1L, 15L, 21L, 21L), c(1L, 3L, 1L, 13L, 27L, 49L), c(1L, 1L, 1L, 15L,
  7L, 5L), c(1L, 3L, 1L, 15L, 13L, 25L), c(1L, 1L, 5L, 5L, 19L, 61L),
  c(1L, 3L, 7L, 11L, 23L, 15L, 103L), c(1L, 3L, 7L, 13L, 13L, 15L, 69L),
  c(1L, 1L, 3L, 13L, 7L, 35L, 63L), c(1L, 3L, 5L, 9L, 1L, 25L, 53L),
  c(1L, 3L, 1L, 13L, 9L, 35L, 107L), c(1L, 3L, 1L, 5L, 27L, 61L, 31L),
  c(1L, 1L, 5L, 11L, 19L, 41L, 61L), c(1L, 3L, 5L, 3L, 3L, 13L, 69L),
  c(1L, 1L, 7L, 13L, 1L, 19L, 1L), c(1L, 3L, 7L, 5L, 13L, 19L, 59L),
  c(1L, 1L, 3L, 9L, 25L, 29L, 41L), c(1L, 3L, 5L, 13L, 23L, 1L, 55L),
  c(1L, 3L, 7L, 3L, 13L, 59L, 17L), c(1L, 3L, 1L, 3L, 5L, 53L, 69L),
  c(1L, 1L, 5L, 5L, 23L, 33L, 13L), c(1L, 1L, 7L, 7L, 1L, 61L, 123L),
  c(1L, 1L, 7L, 9L, 13L, 61L, 49L), c(1L, 3L, 3L, 5L, 3L, 55L, 33L),
  c(1L, 3L, 1L, 15L, 31L, 13L, 49L, 245L), c(1L, 3L, 5L, 15L, 31L, 59L,
  63L, 97L), c(1L, 3L, 1L, 11L, 11L, 11L, 77L, 249L), c(1L, 3L, 1L, 11L,
  27L, 43L, 71L, 9L), c(1L, 1L, 7L, 15L, 21L, 11L, 81L, 45L), c(1L, 3L,
  7L, 3L, 25L, 31L, 65L, 79L), c(1L, 3L, 1L, 1L, 19L, 11L, 3L, 205L),
  c(1L, 1L, 5L, 9L, 19L, 21L, 29L, 157L), c(1L, 3L, 7L, 11L, 1L, 33L,
  89L, 185L), c(1L, 3L, 3L, 3L, 15L, 9L, 79L, 71L), c(1L, 3L, 7L, 11L,
  15L, 39L, 119L, 27L), c(1L, 1L, 3L, 1L, 11L, 31L, 97L, 225L), c(1L,
  1L, 1L, 3L, 23L, 43L, 57L, 177L), c(1L, 3L, 7L, 7L, 17L, 17L, 37L,
  71L), c(1L, 3L, 1L, 5L, 27L, 63L, 123L, 213L), c(1L, 1L, 3L, 5L, 11L,
  43L, 53L, 133L), c(1L, 3L, 5L, 5L, 29L, 17L, 47L, 173L, 479L), c(1L,
  3L, 3L, 11L, 3L, 1L, 109L, 9L, 69L), c(1L, 1L, 1L, 5L, 17L, 39L, 23L,
  5L, 343L), c(1L, 3L, 1L, 5L, 25L, 15L, 31L, 103L, 499L), c(1L, 1L, 1L,
  11L, 11L, 17L, 63L, 105L, 183L), c(1L, 1L, 5L, 11L, 9L, 29L, 97L,
  231L, 363L), c(1L, 1L, 5L, 15L, 19L, 45L, 41L, 7L, 383L), c(1L, 3L,
  7L, 7L, 31L, 19L, 83L, 137L, 221L), c(1L, 1L, 1L, 3L, 23L, 15L, 111L,
  223L, 83L), c(1L, 1L, 5L, 13L, 31L, 15L, 55L, 25L, 161L), c(1L, 1L,
  3L, 13L, 25L, 47L, 39L, 87L, 257L)
)

# direction numbers v_1..v_bits of dimension d, scaled to 30-bit integers
# (same recurrence as scipy.stats._sobol._initialize_v)
.sobol_direction <- function(d) {
  bits <- .SOBOL_BITS
  m <- integer(bits)
  if (d == 1L) {
    m[] <- 1L
  } else {
    p <- .SOBOL_POLY[d]
    deg <- as.integer(floor(log2(p) + 1e-9))
    m[seq_len(deg)] <- .SOBOL_VINIT[[d]][seq_len(deg)]
    if (deg < bits) for (j in (deg + 1L):bits) {
      newv <- m[j - deg]
      pow2 <- 1
      for (k in seq_len(deg)) {
        pow2 <- pow2 * 2
        if (bitwAnd(bitwShiftR(as.integer(p), deg - k), 1L) == 1L)
          newv <- bitwXor(newv, as.integer(pow2 * m[j - k]))
      }
      m[j] <- newv
    }
  }
  m * 2^(bits - seq_len(bits))
}

# unit-cube Sobol points: n x r, rows in (0, 1). seed = NULL gives the
# unscrambled sequence (point 0 is the origin); an integer seed gives a
# deterministic linear-matrix-scrambled, digitally shifted copy.
.sobol_unit <- function(r, n, seed = 0L) {
  r <- as.integer(r); n <- as.integer(n)
  if (r < 1L || r > .SOBOL_MAXDIM)
    stop(sprintf(paste("Sobol nodes are tabulated for 1..%d dimensions;",
                       "got %d"), .SOBOL_MAXDIM, r), call. = FALSE)
  bits <- .SOBOL_BITS
  nb <- max(1L, as.integer(ceiling(log2(max(n, 2L)))))
  if (nb > bits) stop("too many Sobol points requested", call. = FALSE)
  state <- if (is.null(seed)) NULL else
    (as.numeric(seed) %% 2147483646) + 1
  draw <- function(k) {               # k Park-Miller uniforms in [0, 1)
    out <- numeric(k)
    for (i in seq_len(k)) {
      state <<- (16807 * state) %% 2147483647
      out[i] <- (state - 1) / 2147483646
    }
    out
  }
  pow <- 2^(bits - seq_len(bits))     # bit i (from the top) -> value
  idx <- seq_len(n) - 1L
  idx <- bitwXor(idx, bitwShiftR(idx, 1L))   # Gray-code order, as scipy
  U <- matrix(0, n, r)
  for (d in seq_len(r)) {
    v <- .sobol_direction(d)[seq_len(nb)]
    shift <- 0
    if (!is.null(state)) {
      # lower-triangular unit-diagonal binary L applied to the digits of
      # every direction number; since points are XORs of direction
      # numbers this scrambles every point by the same linear map
      L <- diag(bits)
      lo <- lower.tri(L)
      L[lo] <- as.numeric(draw(sum(lo)) < 0.5)
      B <- vapply(v, function(x) as.numeric(bitwAnd(as.integer(x),
                                                    as.integer(pow)) > 0),
                  numeric(bits))
      B <- matrix(B, nrow = bits)
      v <- colSums(((L %*% B) %% 2) * pow)
      shift <- sum(as.numeric(draw(bits) < 0.5) * pow)
    }
    x <- rep(as.integer(shift), n)
    for (j in seq_len(nb)) {
      sel <- bitwAnd(idx, as.integer(2^(j - 1L))) != 0L
      x[sel] <- bitwXor(x[sel], as.integer(v[j]))
    }
    U[, d] <- x / 2^bits
  }
  U
}

# equal-weight standard normal nodes over N(0, I_r)
.sobol_normal <- function(r, n, seed = 0L) {
  U <- .sobol_unit(r, n, seed)
  list(F = matrix(qnorm(pmin(pmax(U, 1e-12), 1 - 1e-12)), ncol = r),
       W = rep(1 / n, n))
}
