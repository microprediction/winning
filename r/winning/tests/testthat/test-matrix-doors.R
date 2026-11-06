# Every exported door whose first argument is one value per contestant must
# refuse a matrix rather than read it column-major (#903). A 2 x 2 matrix has
# the right number of cells for a four-runner field, so a length check alone
# passes it; as.numeric then prices c(m[1,1], m[2,1], m[1,2], m[2,2]), a
# different field. Discovery: any new export whose first formal is one of
# PER_COORD must be listed in EXTRA, or this test fails.

PER_COORD <- c("mu", "p", "q", "p0", "ability", "prices", "dividends", "p_win")

dens <- skew_normal_density(L = 25, unit = 0.1, loc = 0, scale = 1, a = 0)
blk <- list(cluster = c(1, 1, 2, 2), loading = c(0.5, 0.5, 0.5, 0.5), D = 1)

EXTRA <- list(
  race_probabilities          = list(),
  race_jacobian               = list(),
  abilities_from_race         = list(),
  calibrate_abilities         = list(),
  polish_race                 = list(mu0 = c(0, 0, 0, 0)),
  win_probabilities_factor    = list(V = c(0.1, 0.2, 0.3, 0.4)),
  abilities_from_probabilities_factor = list(V = c(0.1, 0.2, 0.3, 0.4)),
  top_k_probabilities         = list(k = 2),
  bottom_k_probabilities      = list(k = 2),
  top_k_jacobians             = list(k = 2),
  abilities_from_topk         = list(k = 2),
  rank_probabilities          = list(),
  abilities_from_rank_marginal = list(r = 2),
  block_race_probabilities    = blk,
  block_race_jacobian         = blk,
  abilities_from_block_race   = blk,
  nested_race_probabilities   = blk,
  nested_race_jacobian        = blk,
  tree_race_probabilities     = blk,
  tree_race_jacobian          = blk,
  ability_implied_state_prices = list(density = dens),
  ability_implied_dividends   = list(density = dens),
  state_price_implied_ability = list(density = dens),
  dividend_implied_ability    = list(density = dens),
  dividends_from_prices       = list(),
  prices_from_dividends       = list(),
  solve_for_implied_offsets   = list(density = dens),
  loc_scale_from_win_and_second = list(p_second = c(0.3, 0.3, 0.2, 0.2))
)

test_that("every per-contestant door refuses a matrix (#903)", {
  ns <- asNamespace("winning")
  doors <- Filter(function(f) {
    g <- get(f, ns)
    is.function(g) && length(formals(g)) && names(formals(g))[1] %in% PER_COORD
  }, getNamespaceExports("winning"))
  missing <- setdiff(doors, names(EXTRA))
  expect(length(missing) == 0,
         paste("doors not swept (add them to EXTRA):",
               paste(missing, collapse = ", ")))
  # probabilities for p-like doors, abilities for mu-like ones: both 2 x 2
  m_mu <- matrix(c(-2, 3, -1, 2), 2, byrow = TRUE)
  m_p <- matrix(c(0.4, 0.3, 0.2, 0.1), 2, byrow = TRUE)
  m_div <- matrix(c(2.5, 3.3, 5, 10), 2, byrow = TRUE)
  for (f in intersect(doors, names(EXTRA))) {
    first <- names(formals(get(f, ns)))[1]
    m <- switch(first, mu = , ability = m_mu, dividends = m_div, m_p)
    args <- c(stats::setNames(list(m), first), EXTRA[[f]])
    if (identical(f, "polish_race")) args$mu0 <- c(0, 0, 0, 0)
    expect_error(do.call(get(f, ns), args), "plain vector",
                 info = paste(f, "accepted a 2 x 2", first))
  }
})

test_that("the same doors still accept the plain vector (#903)", {
  # the refusal must not be a blanket failure: the vector form prices
  mu <- c(-2, 3, -1, 2)
  p <- race_probabilities(mu)
  expect_equal(sum(p), 1, tolerance = 1e-9)
  expect_equal(length(abilities_from_race(p)), 4)
  expect_equal(sum(top_k_probabilities(mu, k = 2)), 2, tolerance = 1e-6)
  expect_equal(sum(block_race_probabilities(mu, blk$cluster, blk$loading, D = 1)),
               1, tolerance = 1e-6)
})
