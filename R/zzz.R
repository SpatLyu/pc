.onLoad = \(...) {
  requireNamespace("Rcpp", quietly = TRUE)
}

utils::globalVariables(".data")
