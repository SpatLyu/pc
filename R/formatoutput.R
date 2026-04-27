#' @export
#' @noRd
print.pc_single = \(x,...) {
  print(x$summary)
}

#' @export
#' @noRd
print.pc_ops = \(x,...) {
  res = x$param
  cat(paste0("The suggested E, k, tau is ", res[1], ", ", res[2], " and ", res[3]), "\n")
}
