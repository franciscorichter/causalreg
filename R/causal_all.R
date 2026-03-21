# Unified exhaustive search for causal submodels (replaces cglm_all and cgam_all)
causal_all <- function(formula, family, data, alpha = 0.05,
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

  # Build all submodels (pre-allocate list)
  n_models <- 2^p - 1
  mod_all <- vector("list", n_models)
  idx <- 0L
  for (i in seq_len(p)) {
    vc <- combn(var_names, i)
    for (j in seq_len(ncol(vc))) {
      idx <- idx + 1L
      if (intercept) {
        mod_all[[idx]] <- as.formula(paste0(dip_name, "~", paste0(vc[, j], collapse = "+")))
      } else {
        mod_all[[idx]] <- as.formula(paste0(dip_name, "~", paste0(vc[, j], collapse = "+"), "-1"))
      }
    }
  }

  # Evaluate all models (pre-allocated vectors)
  pearson_all <- numeric(n_models)
  pv_all <- numeric(n_models)
  bic_all <- numeric(n_models)

  if (ncores > 1L) {
    # Parallel: evaluate models across cores, bootstrap within each is sequential
    dots <- list(...)
    eval_model <- function(j) {
      do.call(.fit_and_test, c(list(formula = mod_all[[j]], family = family,
                                    data = data, n = n, pval_method = pval,
                                    B = B, use_gam = effective_gam,
                                    ncores = 1L, use_cpp = use_cpp), dots))
    }
    all_results <- parallel::mclapply(seq_len(n_models), eval_model,
                                      mc.cores = ncores, mc.set.seed = TRUE)
    for (j in seq_len(n_models)) {
      pearson_all[j] <- all_results[[j]]$pearson
      pv_all[j] <- all_results[[j]]$pval
      bic_all[j] <- all_results[[j]]$bic
    }
  } else {
    for (j in seq_len(n_models)) {
      result <- .fit_and_test(mod_all[[j]], family, data, n, pval, B,
                              effective_gam, ncores = 1L, use_cpp = use_cpp, ...)
      pearson_all[j] <- result$pearson
      pv_all[j] <- result$pval
      bic_all[j] <- result$bic
    }
  }

  # Select optimal model: best BIC among causal models (p-value > alpha)
  if (any(pv_all > alpha)) {
    causal_mask <- pv_all > alpha
    best_idx <- which(causal_mask)[which.min(bic_all[causal_mask])]
    mod_opt <- deparse1(mod_all[[best_idx]])
  } else {
    mod_opt <- "no potential causal model found"
  }

  # Handle categorical variables for binomial family
  mod_opt <- .handle_categorical_all(mod_opt, family, data, alpha,
                                     dip_name, mod_all, pv_all)

  # Deparse formulas for output
  models_str <- vapply(mod_all, deparse1, character(1))

  list(models = as.list(models_str), pearsonrisk = pearson_all,
       pv = pv_all, bic = bic_all, model.opt = mod_opt)
}
