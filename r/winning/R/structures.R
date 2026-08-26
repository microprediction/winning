# One race, five covariance grammars -- port of winning/factor/structures.py.
# Constructors return classed lists accepted by race_probabilities,
# abilities_from_race / calibrate_abilities, race_jacobian and polish_race
# via their structure= argument. D is always the idiosyncratic VARIANCE.

.structure <- function(cls, ...) structure(list(...), class = c(cls, "winning_structure"))

#' Covariance grammars for the general race
#'
#' \code{Independent(D)}: Sigma = diag(D). \code{Factor(V, D)}:
#' Sigma = V V' + diag(D). \code{Blocks(cluster, loading, D)}:
#' block-diagonal rank-1 (or rank-r) plus diagonal.
#' \code{Nested(cluster, loading, D, coupling, gamma)}: blocks plus one
#' global factor, gamma dialing the coupling from 0 (independent blocks)
#' to 1 (fully coupled). \code{Tree(cluster, loading, D, parent,
#' strength)}: a hierarchy of uniform shared effects (parent is 1-based,
#' root marked 0 or NA).
#'
#' @param D idiosyncratic variances
#' @param V loading matrix
#' @param cluster cluster labels
#' @param loading within-cluster loadings
#' @param coupling loadings on the global factor
#' @param gamma coupling strength in [0, 1]
#' @param parent 1-based parent index per node (root: 0/NA)
#' @param strength per-node shared-effect strength
#' @return a structure object for the structure= argument
#' @name structures
NULL

#' @rdname structures
#' @export
Independent <- function(D) .structure("Independent", D = D)

#' @rdname structures
#' @export
Factor <- function(V, D) .structure("Factor", V = V, D = D)

#' @rdname structures
#' @export
Blocks <- function(cluster, loading, D)
  .structure("Blocks", cluster = cluster, loading = loading, D = D)

#' @rdname structures
#' @export
Nested <- function(cluster, loading, D, coupling, gamma = 1.0)
  .structure("Nested", cluster = cluster, loading = loading, D = D,
             coupling = coupling, gamma = gamma)

#' @rdname structures
#' @export
Tree <- function(cluster, loading, D, parent, strength)
  .structure("Tree", cluster = cluster, loading = loading, D = D,
             parent = parent, strength = strength)

.dispatch_probabilities <- function(mu, s, base = "normal", points = 257,
                                    qa = 9, qf = 15, return_slopes = FALSE) {
  cls <- class(s)[1]
  if (cls == "Independent") {
    return(race_probabilities(mu, V = NULL, D = s$D, base = base,
                              points = points, return_slopes = return_slopes))
  }
  if (cls == "Factor") {
    return(race_probabilities(mu, V = s$V, D = s$D, base = base,
                              points = points, return_slopes = return_slopes))
  }
  if (return_slopes) stop("return_slopes is available for Independent/Factor only")
  if (cls == "Blocks") {
    return(block_race_probabilities(mu, s$cluster, s$loading, s$D,
                                    points = points, qa = qa))
  }
  if (cls == "Nested") {
    return(nested_race_probabilities(mu, s$cluster, s$loading, s$D,
                                     coupling = s$coupling, gamma = s$gamma,
                                     points = points, qa = qa, qf = qf))
  }
  if (cls == "Tree") {
    return(tree_race_probabilities(mu, s$cluster, s$loading, s$D,
                                   s$parent, s$strength,
                                   points = points, qa = qa))
  }
  stop("unknown structure ", cls)
}

.dispatch_abilities <- function(p, s, base = "normal", points = 257,
                                qa = 9, qf = 15) {
  cls <- class(s)[1]
  if (cls == "Independent") {
    return(abilities_from_race(p, V = NULL, D = s$D, base = base,
                               points = points))
  }
  if (cls == "Factor") {
    return(abilities_from_race(p, V = s$V, D = s$D, base = base,
                               points = points))
  }
  if (cls == "Blocks") {
    return(abilities_from_block_race(p, s$cluster, s$loading, s$D,
                                     points = points, qa = qa)$mu)
  }
  if (cls == "Nested") {
    return(.invert_generic(p, function(m)
      nested_race_probabilities(m, s$cluster, s$loading, s$D,
                                coupling = s$coupling, gamma = s$gamma,
                                points = points, qa = qa, qf = qf)))
  }
  if (cls == "Tree") {
    return(.invert_generic(p, function(m)
      tree_race_probabilities(m, s$cluster, s$loading, s$D,
                              s$parent, s$strength,
                              points = points, qa = qa)))
  }
  stop("unknown structure ", cls)
}

# Adaptive fixed-point inversion for structures without a Jacobian --
# port of polish._invert_generic.
.invert_generic <- function(p, forward, tol = 1e-9, max_iter = 400) {
  p <- as.numeric(p); p <- p / sum(p)
  lt <- log(pmax(p, 1e-300))
  mu <- -(lt - mean(lt))
  eta <- 1.0
  lp <- log(pmax(forward(mu), 1e-300))
  err <- max(abs(lp - lt))
  for (i in seq_len(max_iter)) {
    if (err < tol) break
    mu_n <- mu - eta * (lt - lp); mu_n <- mu_n - mean(mu_n)
    lp_n <- log(pmax(forward(mu_n), 1e-300))
    e <- max(abs(lp_n - lt))
    if (e < err) {
      mu <- mu_n; lp <- lp_n; err <- e
      eta <- min(eta * 1.2, 1.5)
    } else {
      eta <- eta * 0.5
      if (eta < 1e-4) break
    }
  }
  mu
}
