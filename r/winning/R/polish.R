# Polish a race onto linear constraints -- port of winning/factor/polish.py.
# polish_race solves: minimise ||mu - mu0||^2 subject to A p(mu) <= b in
# the mean-zero gauge, via an augmented Lagrangian on the exact race
# Jacobian (the python reference uses SLSQP; the two agree on the
# constrained optimum to optimizer tolerance).

#' Exact Jacobian d p_i / d mu_j of the general race
#'
#' One field pass; rows sum to zero; off-diagonals are positive tie
#' densities, factored stably as [f_i prod_k S_k / S_i] x [f_j / S_j].
#'
#' @inheritParams race_probabilities
#' @return n x n matrix
#' @export
race_jacobian <- function(mu, V = NULL, D = NULL, F = NULL, W = NULL,
                          base = "normal", points = 501, structure = NULL,
                          qa = 9, qf = 15) {
  if (!is.null(structure)) {
    cls <- class(structure)[1]
    if (cls == "Independent")
      return(race_jacobian(mu, V = NULL, D = structure$D, base = base,
                           points = points))
    if (cls == "Factor")
      return(race_jacobian(mu, V = structure$V, D = structure$D,
                           base = base, points = points))
    if (cls == "Blocks")
      return(block_race_jacobian(mu, structure$cluster, structure$loading,
                                 structure$D, points = points, qa = qa))
    if (cls == "Nested")
      return(nested_race_jacobian(mu, structure$cluster, structure$loading,
                                  structure$D, coupling = structure$coupling,
                                  gamma = structure$gamma, points = points,
                                  qa = qa, qf = qf))
    if (cls == "Tree")
      return(tree_race_jacobian(mu, structure$cluster, structure$loading,
                                structure$D, structure$parent,
                                structure$strength, points = points, qa = qa))
    stop("race_jacobian: structure ", cls, " not yet supported")
  }
  st <- .race_setup(mu, V, D, F, W, base)
  n <- length(st$mu)
  sd <- sqrt(st$D)
  Q <- nrow(st$F)
  M_all <- matrix(st$mu, Q, n, byrow = TRUE) + st$F %*% t(st$V)
  x <- seq(min(M_all) - st$left * max(sd), max(M_all) + st$right * max(sd),
           length.out = points)
  dx <- x[2] - x[1]
  xm <- matrix(x, n, points, byrow = TRUE)
  J <- matrix(0, n, n)
  for (q in seq_len(Q)) {
    z <- (xm - M_all[q, ]) / sd
    b <- st$fn(z)
    f <- b$f / sd
    logS <- log(b$S)
    logf <- log(pmax(f, 1e-300))
    L <- colSums(logS)
    Lm <- matrix(L, n, points, byrow = TRUE)
    P1 <- exp(pmin(pmax(logf + Lm - logS, -745), 40))
    P2 <- exp(pmin(pmax(logf - logS, -745), 40))
    J <- J + st$W[q] * (P1 %*% t(P2)) * dx
  }
  diag(J) <- 0
  diag(J) <- -rowSums(J)
  J
}

#' Assemble (A, b) rows for concentration caps: A p <= b
#'
#' @param n number of contestants
#' @param name_caps scalar or length-n per-name caps (NA entries skipped)
#' @param groups list of list(indices, cap) group caps
#' @return list(A, b)
#' @export
concentration_matrix <- function(n, name_caps = NULL, groups = NULL) {
  rows <- list(); bs <- numeric(0)
  if (!is.null(name_caps)) {
    caps <- rep_len(as.numeric(name_caps), n)
    for (i in seq_len(n)) {
      if (is.finite(caps[i])) {
        r <- numeric(n); r[i] <- 1
        rows[[length(rows) + 1]] <- r
        bs <- c(bs, caps[i])
      }
    }
  }
  if (!is.null(groups)) {
    for (g in groups) {
      r <- numeric(n); r[g[[1]]] <- 1
      rows[[length(rows) + 1]] <- r
      bs <- c(bs, g[[2]])
    }
  }
  if (!length(rows)) return(list(A = matrix(0, 0, n), b = numeric(0)))
  list(A = do.call(rbind, rows), b = bs)
}

#' Nearest race satisfying concentration constraints
#'
#' Weights ARE race probabilities; clipping breaks model-consistency.
#' Returns the nearest race (in ability space, mean-zero gauge) whose
#' probabilities satisfy the caps.
#'
#' @param p0 current weights (inverted to abilities internally), or give
#'   mu0 directly
#' @param mu0 abilities (min-wins)
#' @inheritParams race_probabilities
#' @param name_caps,groups see \code{\link{concentration_matrix}}
#' @param A,b explicit constraint rows, A p <= b
#' @param tol optimizer tolerance
#' @param max_iter outer iterations
#' @return list(p, mu, info)
#' @export
polish_race <- function(p0 = NULL, mu0 = NULL, V = NULL, D = NULL,
                        F = NULL, W = NULL, base = "normal", points = 257,
                        name_caps = NULL, groups = NULL, A = NULL, b = NULL,
                        tol = 1e-9, max_iter = 60, structure = NULL) {
  if (!is.null(structure)) {
    forward <- function(m) race_probabilities(m, structure = structure,
                                              points = points)
    jac <- function(m) race_jacobian(m, structure = structure,
                                     points = points)
    invert <- function(p) abilities_from_race(p, structure = structure,
                                              points = points)
  } else {
    forward <- function(m) race_probabilities(m, V = V, D = D, F = F, W = W,
                                              base = base, points = points)
    jac <- function(m) race_jacobian(m, V = V, D = D, F = F, W = W,
                                     base = base, points = points)
    invert <- function(p) abilities_from_race(p, V = V, D = D, F = F, W = W,
                                              base = base, points = points)
  }
  if (is.null(mu0)) {
    if (is.null(p0)) stop("give p0 or mu0")
    mu0 <- invert(as.numeric(p0))
  }
  mu0 <- as.numeric(mu0) - mean(mu0)
  n <- length(mu0)
  cm <- concentration_matrix(n, name_caps = name_caps, groups = groups)
  A0 <- cm$A; b0 <- cm$b
  if (!is.null(A)) {
    A0 <- rbind(A0, matrix(A, ncol = n))
    b0 <- c(b0, b)
  }
  if (!length(b0)) return(list(p = forward(mu0), mu = mu0,
                               info = list(active = integer(0), nit = 0)))
  # augmented Lagrangian on c(m) = b0 - A0 p(m) >= 0, mean-zero by
  # projection inside the objective and gradient
  lam <- numeric(length(b0))
  rho <- 10
  m <- mu0
  nit <- 0
  for (outer in seq_len(20)) {
    obj <- function(mm) {
      mm <- mm - mean(mm)
      cvec <- b0 - as.numeric(A0 %*% forward(mm))
      psi <- pmax(0, lam - rho * cvec)
      0.5 * sum((mm - mu0)^2) + sum(psi^2 - lam^2) / (2 * rho)
    }
    grd <- function(mm) {
      mm <- mm - mean(mm)
      cvec <- b0 - as.numeric(A0 %*% forward(mm))
      psi <- pmax(0, lam - rho * cvec)
      gc <- -A0 %*% jac(mm)                 # d c / d m
      g <- (mm - mu0) - as.numeric(t(gc) %*% psi)
      g - mean(g)
    }
    res <- stats::optim(m, obj, grd, method = "BFGS",
                        control = list(maxit = 200, reltol = 1e-12))
    m <- res$par - mean(res$par)
    nit <- nit + res$counts[1]
    cvec <- b0 - as.numeric(A0 %*% forward(m))
    lam <- pmax(0, lam - rho * cvec)
    viol <- max(0, -min(cvec))
    if (viol < 1e-8 && outer > 1) break
    rho <- min(rho * 3, 1e6)
  }
  p <- forward(m)
  slack <- b0 - as.numeric(A0 %*% p)
  if (-min(slack) > 1e-6) {
    # the analytic Jacobian may be approximate (tree: cross-cluster Gram);
    # restore feasibility with exact finite-difference constraint
    # gradients -- the forward map is always exact
    jac_fd <- function(mm, h = 1e-6) {
      Jn <- matrix(0, n, n)
      for (j in seq_len(n)) {
        e <- numeric(n); e[j] <- h
        Jn[, j] <- (forward(mm + e) - forward(mm - e)) / (2 * h)
      }
      Jn
    }
    lam <- numeric(length(b0)); rho <- 10
    for (outer in seq_len(20)) {
      obj <- function(mm) {
        mm <- mm - mean(mm)
        cvec <- b0 - as.numeric(A0 %*% forward(mm))
        psi <- pmax(0, lam - rho * cvec)
        0.5 * sum((mm - mu0)^2) + sum(psi^2 - lam^2) / (2 * rho)
      }
      grd <- function(mm) {
        mm <- mm - mean(mm)
        cvec <- b0 - as.numeric(A0 %*% forward(mm))
        psi <- pmax(0, lam - rho * cvec)
        gc <- -A0 %*% jac_fd(mm)
        g <- (mm - mu0) - as.numeric(t(gc) %*% psi)
        g - mean(g)
      }
      res <- stats::optim(m, obj, grd, method = "BFGS",
                          control = list(maxit = 200, reltol = 1e-12))
      m <- res$par - mean(res$par)
      nit <- nit + res$counts[1]
      cvec <- b0 - as.numeric(A0 %*% forward(m))
      lam <- pmax(0, lam - rho * cvec)
      if (max(0, -min(cvec)) < 1e-8 && outer > 1) break
      rho <- min(rho * 3, 1e6)
    }
    p <- forward(m)
    slack <- b0 - as.numeric(A0 %*% p)
  }
  list(p = p, mu = m,
       info = list(active = which(slack < 1e-6), nit = as.integer(nit),
                   max_violation = max(0, -min(slack)),
                   mu_distance = sqrt(sum((m - mu0)^2))))
}
