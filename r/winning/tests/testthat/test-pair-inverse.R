# The pair inverse keeps a tiny share whatever the label order (#412):
# c(1, 1e-16) normalizes to c(1, 1e-16) because 1 + 1e-16 == 1, and the
# closed form took qnorm(1) = Inf while the permutation was finite.
test_that("pair inverse is finite and mirrors under relabeling", {
  D <- c(0.5, 0.5)
  for (t in c(1e-12, 1e-16, 1e-20, 1e-50)) {
    a <- abilities_from_race(c(t, 1), D = D)
    b <- abilities_from_race(c(1, t), D = D)
    expect_true(all(is.finite(b)))
    expect_equal(b, rev(a), tolerance = 1e-12)
    z <- qnorm(t / (1 + t))           # the contrast at unit sd
    expect_equal(a[2] - a[1], z, tolerance = 1e-10)
  }
})
