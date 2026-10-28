# Without this driver R CMD check never runs tests/testthat/, so the
# files there were dead weight: three of the four packages had testthat
# tests and no way to reach them (#142).
library(testthat)
library(rprobitfast)

test_check("rprobitfast")
