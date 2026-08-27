suppressMessages(library(jsonlite))
pkg <- "~/github/winning/r/winning/R"
for (f in list.files(pkg, full.names = TRUE)) source(f)
cases <- fromJSON("r_cases.json", simplifyVector = FALSE)
worst <- 0; fails <- 0
for (cs in cases) {
  mu <- unlist(cs$mu); D <- unlist(cs$D)
  p <- switch(cs$kind,
    factor = race_probabilities(mu, V = matrix(unlist(cs$V), nrow = length(mu),
                                               byrow = TRUE), D = D, points = 257),
    blocks = block_race_probabilities(mu, unlist(cs$cluster), unlist(cs$loading),
                                      D, points = 257),
    nested = nested_race_probabilities(mu, unlist(cs$cluster), unlist(cs$loading),
                                       D, coupling = unlist(cs$coupling),
                                       gamma = cs$gamma, points = 257),
    tree = {
      pa <- unlist(cs$parent); pa <- ifelse(pa < 0, 0L, pa + 1L)
      tree_race_probabilities(mu, unlist(cs$cluster), unlist(cs$loading), D,
                              pa, unlist(cs$strength), points = 257)
    })
  d <- max(abs(p - unlist(cs$p)))
  worst <- max(worst, d)
  if (d > 1e-10) {
    fails <- fails + 1
    cat(sprintf("FAIL seed=%d kind=%s max|diff|=%.3e\n", cs$seed, cs$kind, d))
  }
}
cat(sprintf("%d cases, %d failures, worst %.3e\n", length(cases), fails, worst))
