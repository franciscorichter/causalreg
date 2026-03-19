test_that("cgam finds correct causal model for Poisson (chi-square, all)", {
  n <- 1000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(sin(X1)))
  X2 <- log(Y + 1) + rnorm(n, 0, 0.5)
  data <- data.frame(X1, X2, Y)

  result <- cgam(Y ~ s(X1) + s(X2), "poisson", data, pval = "chi-square", search = "all")
  expect_equal(result$model.opt, "Y ~ s(X1)")
  expect_length(result$models, 3)
  expect_length(result$pv, 3)
  expect_length(result$bic, 3)
})

test_that("cgam finds correct causal model for Poisson (chi-square, stepwise)", {
  n <- 1000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(sin(X1)))
  X2 <- log(Y + 1) + rnorm(n, 0, 0.5)
  data <- data.frame(X1, X2, Y)

  result <- cgam(Y ~ s(X1) + s(X2), "poisson", data, pval = "chi-square", search = "stepwise")
  expect_equal(result$model.opt, "Y ~ s(X1)")
})

test_that("cgam finds correct causal model for binomial (bootstrap, all)", {
  n <- 1000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rbinom(n, 1, exp(X1) / (1 + exp(X1)))
  flip <- rbinom(n, 1, 0.1)
  X2 <- (1 - flip) * Y + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, Y)

  set.seed(1)
  result <- cgam(Y ~ s(X1) + s(X2), "binomial", data, pval = "bootstrap", search = "all")
  expect_equal(result$model.opt, "Y ~ s(X1)")
})

test_that("cgam finds correct causal model for binomial (bootstrap, stepwise)", {
  n <- 1000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rbinom(n, 1, exp(X1) / (1 + exp(X1)))
  flip <- rbinom(n, 1, 0.1)
  X2 <- (1 - flip) * Y + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, Y)

  set.seed(1)
  result <- cgam(Y ~ s(X1) + s(X2), "binomial", data, pval = "bootstrap", search = "stepwise")
  expect_equal(result$model.opt, "Y ~ s(X1)")
})

test_that("cgam match.arg works for pval and search", {
  n <- 100
  set.seed(1)
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(sin(X1)))
  data <- data.frame(X1, Y)

  expect_error(cgam(Y ~ s(X1), "poisson", data, pval = "invalid"))
  expect_error(cgam(Y ~ s(X1), "poisson", data, search = "invalid"))
})
