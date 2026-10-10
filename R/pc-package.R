## usethis namespace: start
#' @useDynLib pc, .registration = TRUE
## usethis namespace: end
NULL

utils::globalVariables(".data")

.onLoad = \(...) {
  requireNamespace("Rcpp", quietly = TRUE)
}
