# Emit one ACCEPT/REFUSE verdict per case, for the R port.
suppressWarnings({
args <- commandArgs(trailingOnly = TRUE)
root <- dirname(dirname(normalizePath(args[1])))
e <- new.env()
for (f in list.files(file.path(root, "r", "winning", "R"), full.names = TRUE))
  sys.source(f, envir = e)
attach(e, warn.conflicts = FALSE)
txt <- paste(readLines(args[1], warn = FALSE), collapse = "")
num <- function(x) if (is.list(x)) vapply(x, function(v)
    if (is.character(v)) as.numeric(v) else as.numeric(v), 0) else
  if (is.character(x)) as.numeric(x) else x
# minimal JSON: reuse jsonlite if present, else fail loudly
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("need jsonlite")
d <- jsonlite::fromJSON(txt, simplifyVector = FALSE)
res <- list()
for (cs in d$cases) {
  id <- cs$id
  v <- tryCatch({
    mu <- if (!is.null(cs$mu)) num(cs$mu) else NULL
    D  <- if (!is.null(cs$D))  num(cs$D)  else NULL
    V  <- if (!is.null(cs$V))  num(cs$V)  else NULL
    p <- if (cs$verb == "race") race_probabilities(mu, V = V, D = D)
         else if (cs$verb == "inverse") abilities_from_race(num(cs$p), D = D)
         else top_k_probabilities(mu, num(cs$k), D = D)
    list(verdict = if (all(is.finite(p))) "ACCEPT" else "ACCEPT_NONFINITE",
         value = as.numeric(head(p, 6)))
  }, error = function(err) list(verdict = "REFUSE",
                                error = substr(conditionMessage(err), 1, 40)))
  res[[id]] <- v
}
cat(jsonlite::toJSON(res, auto_unbox = TRUE, digits = 12))
})
