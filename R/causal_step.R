# Unified stepwise search for causal submodels (replaces cglm_step and cgam_step)
causal_step <- function(formula, family, data, alpha = 0.05,
                        pval = "bootstrap", B = 100,
                        use_gam = FALSE, ncores = 1L,
                        use_cpp = TRUE, ...) {
  n <- nrow(data)
  vrs <- all.vars(formula)
  dip_name <- as.character(formula[[2]])

  # Determine variable names
  if (vrs[2] == ".") {
    var_names <- colnames(data)[colnames(data) != dip_name]
  } else {
    var_names <- attr(terms.formula(formula, data = data), "term.labels")
  }

  # Use GAM only when formula has explicit terms (not ".")
  effective_gam <- use_gam && (vrs[2] != ".")

  # Detect intercept from full model
  full_fit <- .fit_model(formula, family, data, effective_gam, ...)
  intercept <- names(full_fit$coefficients)[1] == "(Intercept)"

  p <- length(var_names)
  pval_method <- pval

  # Pre-allocate step tracking
  max_steps <- 2L * p + 2L
  mod_step <- vector("list", max_steps)
  pv_step <- numeric(max_steps)
  step_count <- 1L

  # Initialize with intercept/null model
  if (intercept) {
    mod_step[[1]] <- as.formula(paste0(dip_name, "~1"))
  } else {
    mod_step[[1]] <- as.formula(paste0(dip_name, "~-1"))
  }
  pv_step[1] <- 0

  var_available <- var_names    # candidates for inclusion
  var_in_model <- character(0)  # already in the model

  continue <- TRUE
  dots <- list(...)

  # === Forward phase ===
  while (continue && length(var_available) > 0) {
    n_cand <- length(var_available)
    current_mod <- mod_step[[step_count]]

    if (ncores > 1L) {
      # Parallel: evaluate candidates across cores
      eval_cand <- function(i) {
        fmli <- update.formula(current_mod, paste0("~. +", var_available[i]))
        do.call(.fit_and_test, c(list(formula = fmli, family = family,
                                      data = data, n = n, pval_method = pval_method,
                                      B = B, use_gam = effective_gam,
                                      ncores = 1L, use_cpp = use_cpp), dots))
      }
      cand_results <- parallel::mclapply(seq_len(n_cand), eval_cand,
                                          mc.cores = ncores, mc.set.seed = TRUE)
      pvals <- vapply(cand_results, function(r) r$pval, numeric(1))
    } else {
      pvals <- numeric(n_cand)
      for (i in seq_len(n_cand)) {
        fmli <- update.formula(current_mod, paste0("~. +", var_available[i]))
        result <- do.call(.fit_and_test, c(list(formula = fmli, family = family,
                                                data = data, n = n,
                                                pval_method = pval_method,
                                                B = B, use_gam = effective_gam,
                                                ncores = 1L, use_cpp = use_cpp), dots))
        pvals[i] <- result$pval
      }
    }

    i_best <- which.max(pvals)
    pv_best <- pvals[i_best]

    # Build the selected model with sorted terms
    mod_new <- update.formula(current_mod, paste0("~. +", var_available[i_best]))
    term_labels <- attr(terms.formula(mod_new), "term.labels")
    mod_new <- reformulate(sort(term_labels), response = dip_name, intercept = intercept)

    pval_diff <- pv_best - pv_step[step_count]
    continue <- (pval_diff > 0) | (pv_best > alpha)

    if (continue) {
      step_count <- step_count + 1L
      mod_step[[step_count]] <- mod_new
      pv_step[step_count] <- pv_best
      var_in_model <- c(var_in_model, var_available[i_best])
      var_available <- var_available[-i_best]
    }
  }

  # === Backward BIC phase ===
  model_current <- mod_step[[step_count]]
  bic_current <- BIC(.fit_model(model_current, family, data, effective_gam, ...))

  improve <- TRUE
  while (improve && length(var_in_model) >= 1) {
    n_vars <- length(var_in_model)

    if (ncores > 1L) {
      eval_drop <- function(i) {
        fmli <- update.formula(model_current, paste0("~. -", var_in_model[i]))
        do.call(BIC, list(do.call(.fit_model, c(list(formula = fmli, family = family,
                                                     data = data, use_gam = effective_gam),
                                                dots))))
      }
      bics <- unlist(parallel::mclapply(seq_len(n_vars), eval_drop,
                                         mc.cores = ncores))
    } else {
      bics <- numeric(n_vars)
      for (i in seq_len(n_vars)) {
        fmli <- update.formula(model_current, paste0("~. -", var_in_model[i]))
        bics[i] <- BIC(.fit_model(fmli, family, data, effective_gam, ...))
      }
    }

    i_best <- which.min(bics)
    bic_best <- bics[i_best]

    if (bic_best < bic_current) {
      model_current <- update.formula(model_current, paste0("~. -", var_in_model[i_best]))
      term_labels <- attr(terms.formula(model_current), "term.labels")
      if (length(term_labels) > 0) {
        model_current <- reformulate(sort(term_labels), response = dip_name, intercept = intercept)
      }
      bic_current <- bic_best
      var_in_model <- var_in_model[-i_best]
      step_count <- step_count + 1L
      mod_step[[step_count]] <- model_current
    } else {
      improve <- FALSE
    }
  }

  # === Handle categorical variables for binomial family ===
  mod_opt_str <- deparse1(mod_step[[step_count]])
  new_opt_str <- .handle_categorical_step(mod_opt_str, family, data, alpha, n, dip_name,
                                          pval_method, B, effective_gam,
                                          ncores = ncores, ...)

  if (new_opt_str != mod_opt_str) {
    step_count <- step_count + 1L
    mod_step[[step_count]] <- as.formula(new_opt_str)
  }

  # Trim and deparse
  mod_step <- mod_step[seq_len(step_count)]
  models_str <- vapply(mod_step, deparse1, character(1))

  list(models = as.list(models_str), model.opt = models_str[step_count])
}
