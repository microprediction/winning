# R parity checker: rebuilds the same-named scenarios as gen_vectors.py
# from the embedded inputs and asserts agreement with the python
# reference. Run from the repo root:
#
#   Rscript parity/check.R
#
# Requires jsonlite (checker only; the package itself is dependency-free).

suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE))
    stop("install.packages('jsonlite') to run the parity checker")
})
pkg <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE),
                                                 value = TRUE))), "..",
                 "r", "winning")
for (f in list.files(file.path(pkg, "R"), full.names = TRUE)) source(f)

vec <- jsonlite::fromJSON(file.path(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))), "vectors.json"),
  simplifyVector = TRUE)
inp <- vec$inputs

mu <- inp$mu
V1 <- as.matrix(inp$V1)
V2 <- as.matrix(inp$V2)
D <- inp$D
cl <- inp$cluster
ld <- inp$loading
ld2 <- as.matrix(inp$loading2)
cp <- inp$coupling
pa_r <- ifelse(inp$parent < 0, 0L, inp$parent + 1L)   # to 1-based, root 0
stg <- inp$strength
pt <- inp$p_target

density <- skew_normal_density(L = inp$classic_L, unit = inp$classic_unit,
                               a = inp$classic_a)

runs <- list(
  independent_normal = function() race_probabilities(mu, D = D, points = 257),
  factor1_normal = function() race_probabilities(mu, V = V1, D = D, points = 257),
  factor2_normal = function() race_probabilities(mu, V = V2, D = D, points = 257),
  factor2_slopes = function()
    race_probabilities(mu, V = V2, D = D, points = 257,
                       return_slopes = TRUE)$slopes,
  factor2_span = function()
    race_probabilities(mu, V = V2, D = D, points = 501, window = "span"),
  gumbel_independent = function()
    race_probabilities(mu, D = rep(pi^2 / 6, length(mu)), base = "gumbel",
                       points = 1001),
  blocks_r1 = function() block_race_probabilities(mu, cl, ld, D, points = 257),
  blocks_r2 = function() block_race_probabilities(mu, cl, ld2, D, points = 257),
  nested = function()
    nested_race_probabilities(mu, cl, ld, D, coupling = cp, gamma = 0.7,
                              points = 257),
  tree = function()
    tree_race_probabilities(mu, cl, ld, D, pa_r, stg, points = 257),
  jacobian_factor = function() race_jacobian(mu, V = V1, D = D, points = 257),
  jacobian_blocks = function() block_race_jacobian(mu, cl, ld, D, points = 257),
  jacobian_nested = function()
    nested_race_jacobian(mu, cl, ld, D, coupling = cp, gamma = 0.7,
                         points = 257),
  invert_factor = function() abilities_from_race(pt, V = V1, D = D, points = 257),
  invert_blocks = function()
    abilities_from_block_race(pt, cl, ld, D, points = 257)$mu,
  classic_ability = function() dividend_implied_ability(inp$dividends, density),
  classic_state_prices = function()
    state_prices_from_offsets(density, vec$scenarios$classic_ability$value),
  polish_p = function()
    polish_race(p0 = pt, V = V1, D = D, points = 257, name_caps = 0.15)$p,
  polish_mu = function()
    polish_race(p0 = pt, V = V1, D = D, points = 257, name_caps = 0.15)$mu
)

fails <- 0
for (name in names(vec$scenarios)) {
  sc <- vec$scenarios[[name]]
  ref <- unlist(sc$value)
  got <- tryCatch(unlist(runs[[name]]()), error = function(e) e)
  if (inherits(got, "error")) {
    cat(sprintf("FAIL  %-22s error: %s\n", name, conditionMessage(got)))
    fails <- fails + 1
    next
  }
  d <- max(abs(got - ref))
  ok <- d <= sc$tol
  cat(sprintf("%s  %-22s max|diff| %.3e  (tol %.0e)\n",
              if (ok) "ok  " else "FAIL", name, d, sc$tol))
  if (!ok) fails <- fails + 1
}
if (fails > 0) stop(fails, " parity failures")
cat("all scenarios match the python reference\n")
