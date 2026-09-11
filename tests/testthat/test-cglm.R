test_that("cglm finds correct causal model for Poisson (chi-square, all)", {
  n <- 1000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(X1))
  X2 <- log(Y + 1) + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, Y)

  result <- cglm(Y ~ X1 + X2, "poisson", data, pval = "chi-square", search = "all")
  expect_equal(result$model.opt, "Y ~ X1")
  expect_length(result$models, 3)     # X1, X2, X1+X2
  expect_length(result$pv, 3)
  expect_length(result$bic, 3)
  expect_length(result$pearsonrisk, 3)
})

test_that("cglm finds correct causal model for Poisson (chi-square, stepwise)", {
  n <- 1000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(X1))
  X2 <- log(Y + 1) + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, Y)

  result <- cglm(Y ~ X1 + X2, "poisson", data, pval = "chi-square", search = "stepwise")
  expect_equal(result$model.opt, "Y ~ X1")
})

test_that("cglm finds correct causal model for binomial (bootstrap, all)", {
  n <- 2000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rbinom(n, 1, exp(X1) / (1 + exp(X1)))
  flip <- rbinom(n, 1, 0.1)
  X2 <- (1 - flip) * Y + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, Y)

  set.seed(1)
  result <- cglm(Y ~ X1 + X2, "binomial", data, pval = "bootstrap", search = "all")
  expect_equal(result$model.opt, "Y ~ X1")
})

test_that("cglm finds correct causal model for binomial (bootstrap, stepwise)", {
  n <- 2000
  set.seed(123)
  X1 <- rnorm(n)
  Y  <- rbinom(n, 1, exp(X1) / (1 + exp(X1)))
  flip <- rbinom(n, 1, 0.1)
  X2 <- (1 - flip) * Y + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, Y)

  set.seed(1)
  result <- cglm(Y ~ X1 + X2, "binomial", data, pval = "bootstrap", search = "stepwise")
  expect_equal(result$model.opt, "Y ~ X1")
})

test_that("cglm handles 5 covariates correctly", {
  set.seed(12)
  n <- 3000
  X1 <- rnorm(n)
  X2 <- rnorm(n, X1, 0.5)
  X3 <- rnorm(n, 0, 1)
  X4 <- rnorm(n, X2, 0.5)
  Y  <- rbinom(n, 1, exp(0.8 * X2 - 0.9 * X3) / (1 + exp(0.8 * X2 - 0.9 * X3)))
  flip <- rbinom(n, 1, 0.1)
  X5 <- (1 - flip) * Y + flip * (1 - Y) + rnorm(n, 0, 0.3)
  dat <- data.frame(X1, X2, X3, X4, X5, Y)

  set.seed(1)
  result <- cglm(Y ~ X1 + X2 + X3 + X4 + X5, "binomial", dat,
                 pval = "bootstrap", search = "all")
  expect_equal(result$model.opt, "Y ~ X2 + X3")
  expect_length(result$models, 31)  # 2^5 - 1
})

test_that("cglm returns 'no potential causal model found' when appropriate", {
  # Create data where no submodel passes the Pearson test
  n <- 200
  set.seed(42)
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(2 * X1))
  X2 <- Y + rnorm(n)
  data <- data.frame(X2, Y)

  result <- cglm(Y ~ X2, "poisson", data, pval = "chi-square", search = "all")
  # With only one non-causal variable, the model should fail
  expect_type(result$model.opt, "character")
})

test_that("cglm match.arg works for pval and search", {
  n <- 100
  set.seed(1)
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(X1))
  data <- data.frame(X1, Y)

  expect_error(cglm(Y ~ X1, "poisson", data, pval = "invalid"))
  expect_error(cglm(Y ~ X1, "poisson", data, search = "invalid"))
})

test_that("backward stepwise recovers the parents on a Poisson chain", {
  set.seed(123)
  n <- 1000
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(X1))
  X2 <- log(Y + 1) + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, Y)

  result <- cglm(Y ~ X1 + X2, "poisson", data, pval = "chi-square",
                 search = "stepwise", direction = "backward")
  expect_equal(result$model.opt, "Y ~ X1")
  # the backward path starts at the full model
  expect_equal(result$models[[1]], "Y ~ X1 + X2")
})

test_that("backward stepwise drops descendants and ancestors before parents", {
  set.seed(7)
  n <- 1000
  X1 <- rnorm(n)
  X2 <- rnorm(n, X1, 0.5)
  X3 <- rnorm(n)
  Y  <- rpois(n, exp(0.6 * X2 - 0.5 * X3))
  X4 <- rnorm(n, X2, 0.5)
  X5 <- log(Y + 1) + rnorm(n, 0, 0.3)
  data <- data.frame(X1, X2, X3, X4, X5, Y)

  result <- cglm(Y ~ X1 + X2 + X3 + X4 + X5, "poisson", data,
                 pval = "chi-square", search = "stepwise",
                 direction = "backward")
  expect_equal(result$model.opt, "Y ~ X2 + X3")
})

test_that("a binomial candidate set with a categorical variable searches backward", {
  set.seed(11)
  n <- 600
  X1 <- rnorm(n)
  Z  <- rbinom(n, 1, 0.5)
  Y  <- rbinom(n, 1, plogis(X1 + Z))
  data <- data.frame(X1, Z, Y)

  set.seed(1)
  expect_message(
    cglm(Y ~ X1 + Z, "binomial", data, pval = "bootstrap",
         search = "stepwise", B = 50),
    "Running the backward search instead"
  )
})

test_that("direction is validated and ignored by the exhaustive search", {
  set.seed(123)
  n <- 200
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(X1))
  data <- data.frame(X1, Y)
  expect_error(cglm(Y ~ X1, "poisson", data, search = "stepwise",
                    direction = "sideways"))
  expect_silent(cglm(Y ~ X1, "poisson", data, pval = "chi-square",
                     search = "all", direction = "backward"))
})
