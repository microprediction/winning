# Laplace inverses on exact self-generated targets (#267). The uniform
# lattice's slope integrand -f'(z) jumps at every runner's kink, so its
# rectangle-sum slopes and Jacobians are first order and the inverses
# stalled; they now take derivatives from the split lattice and keep the
# residual on the map being inverted. The forward maps are unchanged.
test_that("laplace race inverse round-trips", {
  mu <- c(-1.5, 1.5, 0); D <- c(0.1, 6, 2)
  for (pts in c(257, 513)) {
    t <- suppressWarnings(race_probabilities(mu, D = D, base = "laplace", points = pts))
    a <- suppressWarnings(abilities_from_race(t, D = D, base = "laplace", points = pts))
    b <- suppressWarnings(race_probabilities(a, D = D, base = "laplace", points = pts))
    expect_lt(max(abs(b - t)), 1e-8)
  }
})

test_that("laplace top-k and loc/scale inverses round-trip", {
  mu <- c(-1.0758897613734006, 0.8221703173592687, 1.2747228138148785,
          -1.7811144122853875, 0.7601110424846411)
  sd <- c(0.5289511696798439, 1.2097101565229562, 0.9433803641440728,
          1.3242591310799288, 1.2509595951816537)
  q1 <- top_k_probabilities(mu, 1, D = sd^2, base = "laplace")
  q2 <- top_k_probabilities(mu, 2, D = sd^2, base = "laplace")
  f <- loc_scale_from_topk_pair(q1, 1, q2, 2, base = "laplace", return_info = TRUE)
  expect_true(f$converged)
  expect_lt(max(abs(top_k_probabilities(f$mu, 2, D = f$sd^2, base = "laplace") - q2)), 1e-8)
  mu <- c(-1.7151211049140525, 1.6647723741888125, 0.5038679121443425,
          -0.5829490360565223, 0.12942985463742007)
  D <- c(0.7813762250327798, 0.7548860941612756, 1.0407350057755838,
         0.5488380588062226, 2.0439199583301906)
  q <- top_k_probabilities(mu, 3, D = D, base = "laplace")
  f <- abilities_from_topk(q, 3, D = D, base = "laplace", return_info = TRUE)
  expect_true(f$converged)
  expect_lt(max(abs(top_k_probabilities(f$mu, 3, D = D, base = "laplace") - q)), 1e-8)
})

test_that("laplace exact-rank inverse fits the public rank map", {
  mu <- c(-0.32582296524196863, 0.47305250354111195, 0.6627286607399583,
          1.3682900704443455, 1.9737782655283809)
  sd <- c(1.1240951818414033, 1.6630497237201780, 0.37187049128115174,
          0.4599949211347848, 0.7765970396809280)
  mu0 <- c(-0.34953938962426034, 0.5153643666766584, 0.6403211534721777,
           1.3358619652222843, 2.0055403112666683)
  t <- rank_probabilities(mu, D = sd^2, base = "laplace")[, 2]
  f <- abilities_from_rank_marginal(t, 2, mu0 = mu0, D = sd^2,
                                    base = "laplace", return_info = TRUE)
  expect_true(f$converged)
  expect_lt(max(abs((f$mu - mean(f$mu)) - (mu - mean(mu)))), 1e-6)
})
