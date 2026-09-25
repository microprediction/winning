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
    # a bare JSON number stays a scalar: rbind-ing it would hand the
    # library a 1x1 matrix and test the runner, not the port
    asV <- function(x) if (is.list(x)) do.call(rbind, lapply(x, num)) else num(x)
    V  <- if (!is.null(cs$Vt)) asV(cs$Vt)
          else if (!is.null(cs$V)) asV(cs$V) else NULL
    p <- if (cs$verb == "hermite") hermite_nodes(num(cs$r), order = num(cs$order))$F
         else if (cs$verb == "rank") rank_probabilities(mu, D = D)
         else if (cs$verb == "bottomk") bottom_k_probabilities(mu, num(cs$k), D = D)
         else if (cs$verb == "race") race_probabilities(mu, V = V, D = D)
         else if (cs$verb == "inverse") abilities_from_race(num(cs$p), D = D)
         else top_k_probabilities(mu, num(cs$k), D = D)
    # SHAPE for the node rule, values for everything else
    val <- if (cs$verb == "hermite") as.numeric(dim(p))
           # R fills a matrix COLUMN-major; the reference ravels ROW-major
           else as.numeric(head(as.numeric(if (is.matrix(p)) t(p) else p), 6))
    list(verdict = if (all(is.finite(p))) "ACCEPT" else "ACCEPT_NONFINITE",
         value = val)
  }, error = function(err) list(verdict = "REFUSE",
                                error = substr(conditionMessage(err), 1, 40)))
  res[[id]] <- v
}
cat(jsonlite::toJSON(res, auto_unbox = TRUE, digits = 12))
})
