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

# --- skew_normal_density's half-width (#868) --------------------------
#
# L went straight into seq(-L, L): 2.5 and 2 came back as length 6 and 5
# vectors that every classic entry point refuses.

test_that("skew_normal_density refuses a half-width the classic lattice cannot use (#868)", {
  for (L in list(2.5, 2, 0, -3, TRUE, "3", NaN, Inf, NA_real_, c(3, 4), numeric(0)))
    expect_error(skew_normal_density(L, 0.1), "whole number >= 3", info = deparse(L))
})

test_that("a whole half-width still builds the same density (#868)", {
  d3 <- skew_normal_density(3, 0.1, a = 0)
  expect_length(d3, 7)
  expect_lt(abs(sum(d3) - 1), 1e-12)
  expect_identical(skew_normal_density(3L, 0.1, a = 0), d3)
  expect_length(state_prices_from_offsets(d3, c(0, 1)), 2)
  expect_length(skew_normal_density(50, 0.1), 101)
})

# --- np_interp at a duplicated lower knot (#842) ----------------------
#
# v <= xp[1] returned fp[1], the FIRST of a duplicated lower knot, where
# numpy takes the last. A runner priced at exactly zero mid-calibration
# then read its offset off the wrong end of the zero-price plateau, and
# a self-generated compact five-runner book missed by 0.0152.

test_that("np_interp takes the last of a duplicated lower knot, as numpy (#842)", {
  expect_identical(np_interp(0, c(0, 0, 0.5), c(10, 5, 0)), 5)
  expect_identical(np_interp(0, c(0, 0, 0, 0.5), c(10, 5, 3, 0)), 3)
  # unchanged: below the lower knot, duplicated interior and upper knots,
  # ordinary ascending knots
  expect_identical(np_interp(-1, c(0, 0, 0.5), c(10, 5, 0)), 10)
  expect_identical(np_interp(1, c(0, 1, 1, 2), c(0, 5, 7, 9)), 7)
  expect_identical(np_interp(2, c(0, 2, 2), c(0, 5, 7)), 7)
  expect_identical(np_interp(c(0, 0.5, 1, 3), c(0, 1, 2), c(4, 5, 7)), c(4, 4.5, 5, 7))
})

test_that("the compact five-runner inverse reaches its own target (#842)", {
  d <- c(rep(0, 9), 0.0625, 0.125, 0.1875, 0.25, 0.1875, 0.125, 0.0625, rep(0, 9))
  truth <- c(-0.3870526359606288, -3.8859224668266945, -0.5618939753478802,
             -6.990429141603045, 0.6690584211927497)
  target <- state_prices_from_offsets(d, truth)
  expect_no_warning(ab <- solve_for_implied_offsets(target, d))
  expect_lt(max(abs(state_prices_from_offsets(d, ab) - target)), 1e-5)
})
