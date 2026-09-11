# Internal helper functions for causalreg
# These are not exported and are used by causal_all() and causal_step()

# Cross-platform parallel lapply used by the model-search functions.
#
# The backend is chosen by getOption("causalreg.parallel"):
#   "auto"       (default) forking on Unix/macOS, PSOCK cluster on Windows
#   "fork"       force forking (parallel::mclapply)
#   "psock"      force a PSOCK cluster (portable; also used to test the
#                Windows path on Unix)
#   "sequential" no parallelism
# Forking is unavailable on Windows, so "auto"/"fork" fall back to PSOCK there.
# With ncores <= 1 the call is sequential regardless of backend.
.parallel_lapply <- function(X, FUN, ncores = 1L) {
  ncores <- as.integer(ncores)
  backend <- match.arg(getOption("causalreg.parallel", "auto"),
                       c("auto", "fork", "psock", "sequential"))

  can_fork <- .Platform$OS.type != "windows"
  use_fork <- (backend == "fork" || backend == "auto") && can_fork

  if (ncores <= 1L || backend == "sequential") {
    return(lapply(X, FUN))
  }

  if (use_fork) {
    return(parallel::mclapply(X, FUN, mc.cores = ncores, mc.set.seed = TRUE))
  }

  # PSOCK cluster: portable path (the default on Windows).
  cl <- parallel::makePSOCKcluster(ncores)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  # Workers must see the same library paths (e.g. the temporary lib used
  # during R CMD check) and load the package so internal helpers and the
  # compiled code resolve when the serialised closures are evaluated.
  parallel::clusterCall(cl, function(paths) .libPaths(paths), .libPaths())
  parallel::clusterEvalQ(cl, {
    loadNamespace("causalreg")
    loadNamespace("mgcv")
  })
  # Parallel-safe, reproducible RNG streams across workers.
  parallel::clusterSetRNGStream(cl)
  parallel::parLapply(cl, X, FUN)
}

# Fit a model using glm or gam
.fit_model <- function(formula, family, data, use_gam, ...) {
  if (use_gam) {
    gam(formula = formula, family = family, data = data, ...)
  } else {
    glm(formula = formula, family = family, data = data, ...)
  }
}

# Compute effective degrees of freedom
.compute_edf <- function(fitted_model, use_gam) {
  if (use_gam) {
    sa <- summary(fitted_model)
    length(sa$p.coeff) + sum(sa$edf)
  } else {
    length(coef(fitted_model))
  }
}

# Compute two-sided Pearson chi-square p-value
.pearson_chisq_pval <- function(pearson_stat, df) {
  2 * min(pchisq(pearson_stat, df), pchisq(pearson_stat, df, lower.tail = FALSE))
}

# Fit model and compute Pearson risk, p-value, and BIC
# ncores is passed to boot_pval for bootstrap parallelization
.fit_and_test <- function(formula, family, data, n, pval_method, B,
                          use_gam, ncores = 1L, use_cpp = TRUE,
                          fast_gam = FALSE, ...) {
  # Fast C++ path for GLM with supported families
  if (use_cpp && !use_gam && family %in% c("poisson", "binomial") &&
      length(list(...)) == 0) {
    mf <- model.frame(formula, data)
    X <- model.matrix(formula, mf)
    y <- as.numeric(model.response(mf))
    p <- ncol(X)

    fit_cpp <- fast_glm_fit(X, y, family)
    mu <- fit_cpp$fitted_values
    ps <- pearson_stat_cpp(y, mu, family)
    bic_val <- fast_glm_bic(y, mu, family, p, n)

    if (pval_method == "chi-square") {
      pv <- .pearson_chisq_pval(ps, n - p)
    } else {
      pv <- boot_pval_cpp(X, y, family, as.integer(B))
    }

    return(list(pearson = ps / n, pval = pv, bic = bic_val))
  }

  # Original R path
  fit <- .fit_model(formula, family, data, use_gam, ...)
  ps <- sum(residuals(fit, type = "pearson")^2)
  edf <- .compute_edf(fit, use_gam)
  bic_val <- BIC(fit)

  if (pval_method == "chi-square") {
    pv <- .pearson_chisq_pval(ps, n - edf)
  } else {
    sp_fixed <- if (use_gam && fast_gam) fit$sp else NULL
    pv <- boot_pval(formula, family = family, data = data, B = B,
                    use_gam = use_gam, ncores = ncores, use_cpp = use_cpp,
                    sp = sp_fixed, ...)
  }

  list(pearson = ps / n, pval = pv, bic = bic_val)
}

# Identify categorical and binary variables in data
.find_categorical <- function(data) {
  var_cat <- colnames(data)[vapply(data, function(x) !is.numeric(x), logical(1))]
  var_bin <- colnames(data)[vapply(data, function(x) length(unique(na.omit(x))) == 2, logical(1))]
  union(var_cat, var_bin)
}

# Categorical post-processing for binomial family (exhaustive search).
# If the selected model contains a single categorical variable and nothing
# else, its Pearson risk equals 1 by construction, so the invariance test
# carries no information: warn the user and return the model unchanged.
.handle_categorical_all <- function(mod_opt, family, data, alpha,
                                    response_name, mod_all, pv_all) {
  if (mod_opt == "no potential causal model found") return(mod_opt)
  if (family != "binomial") return(mod_opt)

  var_cat <- .find_categorical(data)
  var_mod <- attr(terms.formula(as.formula(mod_opt), data = data), "term.labels")
  var_noncat <- setdiff(var_mod, var_cat)

  if (length(var_noncat) == 0 && length(var_mod) == 1)
    message("The model found contains only one categorical variable. ",
            "Since the Pearson risk of this model is mathematically equal to 1, ",
            "the test is inconclusive.")

  mod_opt
}

# Handle categorical variable post-processing for binomial family (stepwise search)
# Fits models and computes p-values directly
.handle_categorical_step <- function(mod_opt, family, data, alpha, n, response_name,
                                     pval_method, B, use_gam, ncores = 1L,
                                     fast_gam = FALSE, ...) {
  if (family != "binomial") return(mod_opt)

  var_cat <- .find_categorical(data)
  var_mod <- attr(terms.formula(as.formula(mod_opt), data = data), "term.labels")
  var_noncat <- setdiff(var_mod, var_cat)

  if (length(var_noncat) >= length(var_mod)) return(mod_opt)

  # Try model without all categorical variables
  if (length(var_noncat) > 0) {
    mod_test <- reformulate(var_noncat, response = response_name)
    fmli <- update.formula(as.formula(mod_opt), mod_test)
    result <- .fit_and_test(fmli, family, data, n, pval_method, B, use_gam,
                            ncores = ncores, fast_gam = fast_gam, ...)
    if (result$pval > alpha) return(deparse1(fmli))
  }

  # Try removing categorical variables one at a time
  varc <- var_cat[var_cat %in% var_mod]
  for (i in seq_along(varc)) {
    fmli <- update.formula(as.formula(mod_opt), paste0("~. -", varc[i]))
    result <- .fit_and_test(fmli, family, data, n, pval_method, B, use_gam,
                            ncores = ncores, fast_gam = fast_gam, ...)
    if (result$pval > alpha) {
      mod_opt <- deparse1(fmli)
    }
  }

  # Same warning as the exhaustive search: a lone categorical covariate carries
  # no information, because the Pearson risk of that model equals 1 by
  # construction.
  var_final <- attr(terms.formula(as.formula(mod_opt), data = data), "term.labels")
  if (length(var_final) == 1L && length(setdiff(var_final, var_cat)) == 0L)
    message("The model found contains only one categorical variable. ",
            "Since the Pearson risk of this model is mathematically equal to 1, ",
            "the test is inconclusive.")

  mod_opt
}
