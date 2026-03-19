boot_pval <- function(formula, family, data, B = 100, use_gam = FALSE,
                      ncores = 1L, ...) {
  n <- nrow(data)
  dots <- list(...)

  boot_one <- function(i) {
    idx <- sample.int(n, replace = TRUE)
    h <- do.call(.fit_model, c(list(formula = formula, family = family,
                                    data = data[idx, , drop = FALSE],
                                    use_gam = use_gam), dots))
    sum(residuals(h, type = "pearson")^2) / n
  }

  if (ncores > 1L) {
    pr <- unlist(parallel::mclapply(seq_len(B), boot_one,
                                   mc.cores = ncores, mc.set.seed = TRUE))
  } else {
    pr <- vapply(seq_len(B), boot_one, numeric(1))
  }

  prob <- mean(pr >= 1)
  2 * min(prob, 1 - prob)
}
