.onLoad = \(...) {
  requireNamespace("Rcpp", quietly = TRUE)
}

utils::globalVariables(".data")

## usethis namespace: start
#' @useDynLib pc, .registration = TRUE
## usethis namespace: end
NULL
