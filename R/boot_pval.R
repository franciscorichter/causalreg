boot_pval <- function(formula, family, data, B = 100, use_gam = FALSE, ...) {
  n <- nrow(data)
  pr <- numeric(B)
  for (i in seq_len(B)) {
    idx <- sample.int(n, replace = TRUE)
    h <- .fit_model(formula, family, data[idx, , drop = FALSE], use_gam, ...)
    pr[i] <- sum(residuals(h, type = "pearson")^2) / n
  }
  prob <- mean(pr >= 1)
  2 * min(prob, 1 - prob)
}
