# Internal helper functions for causalreg
# These are not exported and are used by causal_all() and causal_step()

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
                          use_gam, ncores = 1L, ...) {
  fit <- .fit_model(formula, family, data, use_gam, ...)
  ps <- sum(residuals(fit, type = "pearson")^2)
  edf <- .compute_edf(fit, use_gam)
  bic_val <- BIC(fit)

  if (pval_method == "chi-square") {
    pv <- .pearson_chisq_pval(ps, n - edf)
  } else {
    pv <- boot_pval(formula, family = family, data = data, B = B,
                    use_gam = use_gam, ncores = ncores, ...)
  }

  list(pearson = ps / n, pval = pv, bic = bic_val)
}

# Identify categorical and binary variables in data
.find_categorical <- function(data) {
  var_cat <- colnames(data)[vapply(data, function(x) !is.numeric(x), logical(1))]
  var_bin <- colnames(data)[vapply(data, function(x) length(unique(na.omit(x))) == 2, logical(1))]
  union(var_cat, var_bin)
}

# Handle categorical variable post-processing for binomial family (exhaustive search)
# Uses pre-computed p-values from the full model list
.handle_categorical_all <- function(mod_opt, family, data, alpha,
                                    response_name, mod_all, pv_all) {
  if (mod_opt == "no potential causal model found") return(mod_opt)
  if (family != "binomial") return(mod_opt)

  var_cat <- .find_categorical(data)
  var_mod <- attr(terms.formula(as.formula(mod_opt), data = data), "term.labels")
  var_noncat <- setdiff(var_mod, var_cat)

  if (length(var_noncat) >= length(var_mod)) return(mod_opt)

  # Pre-compute deparsed model strings for matching
  mod_strs <- vapply(mod_all, deparse1, character(1))

  # Try model without all categorical variables
  if (length(var_noncat) > 0) {
    mod_test <- reformulate(var_noncat, response = response_name)
    fmli <- update.formula(as.formula(mod_opt), mod_test)
    fmli_str <- deparse1(fmli)
    match_idx <- which(mod_strs == fmli_str)

    if (length(match_idx) > 0 && pv_all[match_idx[1]] > alpha) {
      return(fmli_str)
    }
  }

  # Try removing categorical variables one at a time
  varc <- var_cat[var_cat %in% var_mod]
  for (i in seq_along(varc)) {
    fmli <- update.formula(as.formula(mod_opt), paste0("~. -", varc[i]))
    fmli_str <- deparse1(fmli)
    match_idx <- which(mod_strs == fmli_str)
    if (length(match_idx) > 0 && pv_all[match_idx[1]] > alpha) {
      mod_opt <- fmli_str
    }
  }

  mod_opt
}

# Handle categorical variable post-processing for binomial family (stepwise search)
# Fits models and computes p-values directly
.handle_categorical_step <- function(mod_opt, family, data, alpha, n, response_name,
                                     pval_method, B, use_gam, ncores = 1L, ...) {
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
                            ncores = ncores, ...)
    if (result$pval > alpha) return(deparse1(fmli))
  }

  # Try removing categorical variables one at a time
  varc <- var_cat[var_cat %in% var_mod]
  for (i in seq_along(varc)) {
    fmli <- update.formula(as.formula(mod_opt), paste0("~. -", varc[i]))
    result <- .fit_and_test(fmli, family, data, n, pval_method, B, use_gam,
                            ncores = ncores, ...)
    if (result$pval > alpha) {
      mod_opt <- deparse1(fmli)
    }
  }

  mod_opt
}
