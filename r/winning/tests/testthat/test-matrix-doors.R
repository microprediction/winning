# Every exported door must refuse a matrix in any argument that holds one
# value per contestant, rather than read it column-major (#903). A 2 x 2 has
# the right number of cells for a four-runner field, so a length check alone
# passes it; as.numeric then prices c(m[1,1], m[2,1], m[1,2], m[2,2]), a
# different field.
#
# Discovery: every export with a formal named in PER_COORD must have a valid
# all-vector call in CALLS, or the test fails. Each PER_COORD argument the
# call supplies is then replaced, one at a time, by a 2 x 2 matrix holding the
# same four cells. The unmodified call must run, so a refusal is not a
# blanket failure.

PER_COORD <- c("mu", "mu0", "p", "q", "q1", "q2", "p0", "ability", "prices",
               "dividends", "p_win", "p_second", "D", "D0")

dens <- skew_normal_density(L = 25, unit = 0.1, loc = 0, scale = 1, a = 0)
mu <- c(-2, -1, 2, 3)
p <- c(0.4, 0.3, 0.2, 0.1)
D4 <- c(1, 1.2, 0.9, 1.1)
V4 <- c(0.1, 0.2, 0.3, 0.4)
blk <- list(cluster = c(1, 1, 2, 2), loading = c(0.5, 0.5, 0.5, 0.5), D = D4)
q1 <- top_k_probabilities(mu, 1)
q2 <- top_k_probabilities(mu, 2)
z4 <- c(0, 0, 0, 0)
tree <- list(parent = c(3, 3, 0), strength = c(0.3, 0.3, 0))

CALLS <- list(
  race_probabilities          = list(mu = mu, D = D4),
  race_jacobian               = list(mu = mu, D = D4),
  abilities_from_race         = list(p = p, D = D4),
  calibrate_abilities         = list(p = p, D = D4),
  polish_race                 = list(p0 = p, mu0 = z4, D = D4),
  win_probabilities_factor    = list(mu = mu, V = V4, D = D4),
  abilities_from_probabilities_factor = list(p = p, V = V4, D = D4),
  top_k_probabilities         = list(mu = mu, k = 2, D = D4),
  bottom_k_probabilities      = list(mu = mu, k = 2, D = D4),
  top_k_jacobians             = list(mu = mu, k = 2, D = D4),
  abilities_from_topk         = list(q = q2, k = 2, D = D4),
  rank_probabilities          = list(mu = mu, D = D4),
  abilities_from_rank_marginal = list(p = p, r = 2, mu0 = z4, D = D4),
  loc_scale_from_topk_pair    = list(q1 = q1, k1 = 1, q2 = q2, k2 = 2,
                                     D0 = D4, mu0 = z4),
  loc_scale_from_win_and_second = list(p_win = p, p_second = c(0.3, 0.3, 0.2, 0.2)),
  block_race_probabilities    = c(list(mu = mu), blk),
  block_race_jacobian         = c(list(mu = mu), blk),
  abilities_from_block_race   = c(list(p = p), blk),
  nested_race_probabilities   = c(list(mu = mu), blk),
  nested_race_jacobian        = c(list(mu = mu), blk),
  tree_race_probabilities     = c(list(mu = mu), blk, tree),
  tree_race_jacobian          = c(list(mu = mu), blk, tree),
  ability_implied_state_prices = list(ability = mu, density = dens),
  ability_implied_dividends   = list(ability = mu, density = dens),
  state_price_implied_ability = list(prices = p, density = dens),
  dividend_implied_ability    = list(dividends = 1 / p, density = dens),
  dividends_from_prices       = list(prices = p),
  prices_from_dividends       = list(dividends = 1 / p),
  solve_for_implied_offsets   = list(prices = p, density = dens),
  Independent                 = list(D = D4),
  Factor                      = list(V = V4, D = D4),
  Blocks                      = blk,
  Nested                      = c(blk, list(coupling = 0.3)),
  Tree                        = c(blk, tree)
)

ns <- asNamespace("winning")
# A structure descriptor only records its arguments; the refusal is where a
# race reads them, so a descriptor is swept through race_probabilities.
DESCRIPTORS <- c("Independent", "Factor", "Blocks", "Nested", "Tree")
door <- function(f) {
  g <- get(f, ns)
  if (f %in% DESCRIPTORS)
    function(...) race_probabilities(mu, structure = g(...))
  else g
}
exports <- Filter(function(f) is.function(get(f, ns)),
                  getNamespaceExports("winning"))
doors <- Filter(function(f) any(names(formals(get(f, ns))) %in% PER_COORD),
                exports)

test_that("every export taking a per-contestant argument is swept (#903)", {
  missing <- setdiff(doors, names(CALLS))
  expect(length(missing) == 0,
         paste("doors not swept (add them to CALLS):",
               paste(missing, collapse = ", ")))
})

test_that("the all-vector calls run (#903)", {
  for (f in intersect(doors, names(CALLS)))
    expect_no_error(suppressWarnings(do.call(door(f), CALLS[[f]])),
                    message = f)
})

test_that("each per-contestant argument refuses a matrix (#903)", {
  for (f in intersect(doors, names(CALLS))) {
    args <- CALLS[[f]]
    for (a in intersect(names(args), PER_COORD)) {
      if (length(args[[a]]) != 4L) next
      bad <- args
      bad[[a]] <- matrix(args[[a]], 2, byrow = TRUE)
      expect_error(suppressWarnings(do.call(door(f), bad)), "plain vector",
                   info = paste(f, "accepted a 2 x 2", a))
    }
  }
})
