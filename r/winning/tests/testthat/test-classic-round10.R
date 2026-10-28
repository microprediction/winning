# Classic round-10 regressions. Each failed on the pre-fix tree.

# --- the extended forward's no-chance removal test (#794) -------------
#
# The raw fractional gap was compared with the integer support width W,
# but low_high mixes a fractional offset's floor and ceil shifts, so its
# support reaches one cell further: [0, 1.0001] on a two-atom law (W = 1)
# was priced [1, 0] against the exact 0.125 dead-heat claim.

two_atom_794 <- function() {
  d <- numeric(31)                                 # L = 15
  d[16] <- 0.5
  d[17] <- 0.5
  d
}

test_that("a runner just past the support width keeps its dead-heat claim (#794)", {
  d <- two_atom_794()
  expect_lt(max(abs(ability_implied_state_prices(c(0, 1.0001), d) -
                    c(0.8750125, 0.1249875))), 1e-15)
  expect_lt(max(abs(ability_implied_state_prices(c(1.0001, 0), d) -
                    c(0.1249875, 0.8750125))), 1e-15)
  for (g in c(0.9999, 1, 1.0001, 1.5, 1.9999, 2, 2.5, 3, 7.25)) {
    for (off in list(c(0, g), c(g, 0), c(0.5, 0.5 + g))) {
      expect_lt(max(abs(ability_implied_state_prices(off, d) -
                        state_prices_from_offsets(d, off))), 1e-15)
    }
  }
})

test_that("disjoint runners still take the zero-price path (#794)", {
  d <- two_atom_794()
  for (g in c(2, 2.5, 3, 7.25)) {
    expect_identical(ability_implied_state_prices(c(0, g), d), c(1, 0))
    expect_identical(ability_implied_state_prices(c(g, 0), d), c(0, 1))
  }
  expect_lt(max(abs(ability_implied_state_prices(c(0, 1), d) - c(0.875, 0.125))), 1e-15)
})
