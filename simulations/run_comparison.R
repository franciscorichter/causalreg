#!/usr/bin/env Rscript
# =============================================================================
# Comparison: causalreg vs ACR (Adversarial Causal Regularization)
# =============================================================================
#
# causalreg: tests Pearson risk invariance, works on GLM/GAM data
# ACR:       adversarial weight learning for causal invariance, works on linear data
#
# We create multi-environment DGPs so ACR can discover latent environments.
# causalreg ignores the environment structure (doesn't need it).
# ACR uses MSE loss even on Poisson/Binomial data (suboptimal but feasible).

library(causalreg)
acr_path <- file.path(dirname(getwd()), "..", "experiment1_synthetic_sem", "acr_core.R")
if (!file.exists(acr_path)) {
  # Try absolute path
  acr_path <- "/Users/pancho/Library/CloudStorage/Dropbox/A2 - Research/07 - Causality and Invariance/code/experiment1_synthetic_sem/acr_core.R"
}
source(acr_path)

NCORES <- parallel::detectCores()
N_REPS <- 100
cat(sprintf("Using %d cores, %d reps per scenario\n\n", NCORES, N_REPS))

# =============================================================================
# Multi-environment DGPs (shifts in downstream variables only)
# =============================================================================

# DGP 1: Poisson GLM, p=2, with shift in X2 (downstream)
# Env 1: baseline. Env 2: X2 gets a mean shift
dgp_poisson_p2_env <- function(n, shift = 2) {
  n1 <- n %/% 2; n2 <- n - n1
  # Env 1
  X1_1 <- rnorm(n1); Y_1 <- rpois(n1, exp(X1_1))
  X2_1 <- log(Y_1 + 1) + rnorm(n1, 0, 0.3)
  # Env 2: shift in X2
  X1_2 <- rnorm(n2); Y_2 <- rpois(n2, exp(X1_2))
  X2_2 <- log(Y_2 + 1) + shift + rnorm(n2, 0, 0.3)

  list(data = data.frame(X1 = c(X1_1, X1_2), X2 = c(X2_1, X2_2), Y = c(Y_1, Y_2)),
       env = c(rep(0L, n1), rep(1L, n2)))
}

# DGP 2: Binomial GLM, p=2, with shift in X2
dgp_binomial_p2_env <- function(n, shift = 1) {
  n1 <- n %/% 2; n2 <- n - n1
  # Env 1
  X1_1 <- rnorm(n1); Y_1 <- rbinom(n1, 1, exp(X1_1) / (1 + exp(X1_1)))
  flip_1 <- rbinom(n1, 1, 0.1); X2_1 <- (1 - flip_1) * Y_1 + rnorm(n1, 0, 0.3)
  # Env 2: shift in X2
  X1_2 <- rnorm(n2); Y_2 <- rbinom(n2, 1, exp(X1_2) / (1 + exp(X1_2)))
  flip_2 <- rbinom(n2, 1, 0.1); X2_2 <- (1 - flip_2) * Y_2 + shift + rnorm(n2, 0, 0.3)

  list(data = data.frame(X1 = c(X1_1, X1_2), X2 = c(X2_1, X2_2), Y = c(Y_1, Y_2)),
       env = c(rep(0L, n1), rep(1L, n2)))
}

# DGP 3: Binomial GLM, p=5, with shifts in X4, X5
dgp_binomial_p5_env <- function(n, shift = 1.5) {
  n1 <- n %/% 2; n2 <- n - n1
  gen <- function(nn, s) {
    X1 <- rnorm(nn); X2 <- rnorm(nn, X1, 0.5); X3 <- rnorm(nn, 0, 1)
    X4 <- rnorm(nn, X2 + s, 0.5)
    Y <- rbinom(nn, 1, exp(0.8 * X2 - 0.9 * X3) / (1 + exp(0.8 * X2 - 0.9 * X3)))
    flip <- rbinom(nn, 1, 0.1)
    X5 <- (1 - flip) * Y + flip * (1 - Y) + s + rnorm(nn, 0, 0.3)
    data.frame(X1, X2, X3, X4, X5, Y)
  }
  d1 <- gen(n1, 0); d2 <- gen(n2, shift)
  list(data = rbind(d1, d2), env = c(rep(0L, n1), rep(1L, n2)))
}

# DGP G: Gaussian linear, p=5 (same DAG as DGP 3, both methods native)
dgp_gaussian_p5_env <- function(n, shift = 2) {
  n1 <- n %/% 2; n2 <- n - n1
  gen <- function(nn, s) {
    X1 <- rnorm(nn); X2 <- rnorm(nn, X1, 0.5); X3 <- rnorm(nn, 0, 1)
    X4 <- rnorm(nn, X2 + s, 0.5)
    Y <- 0.8 * X2 - 0.9 * X3 + rnorm(nn, 0, 0.5)
    X5 <- Y + s + rnorm(nn, 0, 0.3)
    data.frame(X1, X2, X3, X4, X5, Y)
  }
  d1 <- gen(n1, 0); d2 <- gen(n2, shift)
  list(data = rbind(d1, d2), env = c(rep(0L, n1), rep(1L, n2)))
}

# =============================================================================
# Helpers
# =============================================================================

extract_terms <- function(model_opt) {
  if (is.na(model_opt) || model_opt == "no potential causal model found")
    return(character(0))
  attr(terms.formula(as.formula(model_opt)), "term.labels")
}

# ACR variable selection: threshold on |beta|/max(|beta|)
acr_select <- function(beta, var_names, threshold = 0.15) {
  rel <- abs(beta) / max(abs(beta))
  var_names[rel > threshold]
}

# =============================================================================
# ACR configuration
# =============================================================================

acr_cfg <- list(
  varGamma = 5.0, kappa = 0.5, rho = 0.5, lr = 0.05,
  n_steps = 500L, ridge = 1e-3, momentum = 0.9,
  entropy_coeff = 0.001, eps = 1e-6
)
fit_gammas <- c(0, 0.5, 1, 2, 5, 10, 20)

# =============================================================================
# Simulation runner
# =============================================================================

run_one_rep <- function(dgp_fn, n, true_parents, var_names,
                        family, formula, use_causalreg, use_acr) {
  env_data <- dgp_fn(n)
  dat <- env_data$data
  results <- list()

  # --- causalreg ---
  if (use_causalreg) {
    t0 <- proc.time()
    cr_res <- tryCatch({
      pval_method <- if (family == "binomial") "bootstrap" else "chi-square"
      r <- cglm(formula, family, dat, pval = pval_method, search = "all", ncores = 1L)
      found <- extract_terms(r$model.opt)
      list(exact = setequal(found, true_parents),
           tp = sum(true_parents %in% found), fp = sum(!(found %in% true_parents)))
    }, error = function(e) list(exact = FALSE, tp = 0L, fp = 0L))
    t_cr <- (proc.time() - t0)[["elapsed"]]

    t0s <- proc.time()
    cr_step <- tryCatch({
      pval_method <- if (family == "binomial") "bootstrap" else "chi-square"
      r <- cglm(formula, family, dat, pval = pval_method, search = "stepwise", ncores = 1L)
      found <- extract_terms(r$model.opt)
      list(exact = setequal(found, true_parents),
           tp = sum(true_parents %in% found), fp = sum(!(found %in% true_parents)))
    }, error = function(e) list(exact = FALSE, tp = 0L, fp = 0L))
    t_cr_step <- (proc.time() - t0s)[["elapsed"]]

    results$cr_all <- c(cr_res, time = t_cr)
    results$cr_step <- c(cr_step, time = t_cr_step)
  }

  # --- ACR ---
  if (use_acr) {
    X <- as.matrix(dat[, var_names])
    y <- dat$Y

    t0 <- proc.time()
    acr_res <- tryCatch({
      phase1 <- run_acr(X, y, acr_cfg)
      phase2 <- cv_gamma_fit(X, y, phase1$weights, fit_gammas,
                             varGamma_cv = 2.0, ridge = 1e-3)
      found <- acr_select(phase2$beta, var_names)
      list(exact = setequal(found, true_parents),
           tp = sum(true_parents %in% found), fp = sum(!(found %in% true_parents)),
           beta = phase2$beta, best_gamma = phase2$best_gamma,
           auc = auc_invariant(env_data$env, phase1$weights))
    }, error = function(e) list(exact = FALSE, tp = 0L, fp = 0L,
                                beta = rep(NA, length(var_names)),
                                best_gamma = NA, auc = NA))
    t_acr <- (proc.time() - t0)[["elapsed"]]
    results$acr <- c(acr_res[c("exact", "tp", "fp", "auc", "best_gamma")], time = t_acr)
  }

  results
}

run_scenario <- function(label, dgp_fn, n, true_parents, var_names,
                         family, formula, use_causalreg, use_acr,
                         n_reps, ncores_sim) {
  cat(sprintf("%-55s", label)); flush.console()

  t0 <- proc.time()
  all_reps <- parallel::mclapply(seq_len(n_reps), function(i) {
    run_one_rep(dgp_fn, n, true_parents, var_names, family, formula,
                use_causalreg, use_acr)
  }, mc.cores = ncores_sim, mc.set.seed = TRUE)
  wall_time <- (proc.time() - t0)[["elapsed"]]

  # Aggregate results
  agg <- list()
  for (method in c("cr_all", "cr_step", "acr")) {
    vals <- lapply(all_reps, `[[`, method)
    vals <- vals[!sapply(vals, is.null)]
    if (length(vals) > 0) {
      agg[[method]] <- data.frame(
        recovery = mean(sapply(vals, `[[`, "exact")),
        tpr      = mean(sapply(vals, `[[`, "tp")) / max(length(true_parents), 1),
        avg_fp   = mean(sapply(vals, `[[`, "fp")),
        avg_time = mean(sapply(vals, `[[`, "time"))
      )
      if (method == "acr") {
        agg[[method]]$avg_auc <- mean(sapply(vals, `[[`, "auc"), na.rm = TRUE)
      }
    }
  }

  # Print summary
  parts <- c()
  for (m in names(agg)) {
    parts <- c(parts, sprintf("%s=%.0f%%", m, agg[[m]]$recovery * 100))
  }
  cat(sprintf(" %s  (%.1fs)\n", paste(parts, collapse = ", "), wall_time))
  flush.console()

  list(agg = agg, wall_time = wall_time)
}

# =============================================================================
# Run all comparisons
# =============================================================================

results_file <- "results/comparison_results.rds"

cat("=============================================================\n")
cat("COMPARISON: causalreg vs ACR\n")
cat("=============================================================\n\n")

all_comp <- list()

# --- DGP 1: Poisson GLM, p=2 ---
for (n in c(500, 1000, 2000)) {
  label <- sprintf("Poisson GLM (p=2) | n=%d", n)
  res <- run_scenario(label, dgp_poisson_p2_env, n,
                      true_parents = "X1", var_names = c("X1", "X2"),
                      family = "poisson", formula = Y ~ X1 + X2,
                      use_causalreg = TRUE, use_acr = TRUE,
                      n_reps = N_REPS, ncores_sim = NCORES)
  all_comp[[label]] <- res
}

cat("\n")

# --- DGP 2: Binomial GLM, p=2 ---
for (n in c(500, 1000, 2000)) {
  label <- sprintf("Binomial GLM (p=2) | n=%d", n)
  res <- run_scenario(label, dgp_binomial_p2_env, n,
                      true_parents = "X1", var_names = c("X1", "X2"),
                      family = "binomial", formula = Y ~ X1 + X2,
                      use_causalreg = TRUE, use_acr = TRUE,
                      n_reps = N_REPS, ncores_sim = NCORES)
  all_comp[[label]] <- res
}

cat("\n")

# --- DGP 3: Binomial GLM, p=5 ---
for (n in c(1000, 2000, 3000)) {
  label <- sprintf("Binomial GLM (p=5) | n=%d", n)
  res <- run_scenario(label, dgp_binomial_p5_env, n,
                      true_parents = c("X2", "X3"), var_names = paste0("X", 1:5),
                      family = "binomial", formula = Y ~ X1 + X2 + X3 + X4 + X5,
                      use_causalreg = TRUE, use_acr = TRUE,
                      n_reps = N_REPS, ncores_sim = NCORES)
  all_comp[[label]] <- res
}

cat("\n")

# --- DGP G: Gaussian linear, p=5 (ACR's native ground) ---
for (n in c(500, 1000, 2000)) {
  label <- sprintf("Gaussian linear (p=5) | n=%d", n)
  res <- run_scenario(label, dgp_gaussian_p5_env, n,
                      true_parents = c("X2", "X3"), var_names = paste0("X", 1:5),
                      family = "poisson", formula = Y ~ X1 + X2 + X3 + X4 + X5,
                      use_causalreg = FALSE, use_acr = TRUE,
                      n_reps = N_REPS, ncores_sim = NCORES)
  all_comp[[label]] <- res
}

saveRDS(all_comp, results_file)
cat("\nResults saved to", results_file, "\n")

# =============================================================================
# Generate comparison plots
# =============================================================================

cat("\nGenerating comparison plots...\n")

# Build data frame for plotting
plot_data <- do.call(rbind, lapply(names(all_comp), function(label) {
  parts <- strsplit(label, " \\| ")[[1]]
  dgp <- parts[1]
  n <- as.integer(gsub("n=", "", parts[2]))
  agg <- all_comp[[label]]$agg

  rows <- list()
  for (m in names(agg)) {
    method_label <- switch(m,
                           cr_all = "causalreg (all)",
                           cr_step = "causalreg (step)",
                           acr = "ACR")
    rows[[length(rows) + 1]] <- data.frame(
      dgp = dgp, n = n, method = method_label,
      recovery = agg[[m]]$recovery,
      tpr = agg[[m]]$tpr,
      avg_fp = agg[[m]]$avg_fp,
      avg_time = agg[[m]]$avg_time,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}))

# --- Figure: Recovery comparison ---
cols <- c("causalreg (all)" = "#E41A1C", "causalreg (step)" = "#377EB8", "ACR" = "#4DAF4A")
pchs <- c("causalreg (all)" = 16, "causalreg (step)" = 17, "ACR" = 15)

dgps <- unique(plot_data$dgp)
dgps <- dgps[dgps != "Gaussian linear (p=5)"]  # ACR only

pdf("figures/fig_comparison_recovery.pdf", width = 10, height = 4)
par(mfrow = c(1, length(dgps)), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))

for (d in dgps) {
  sub <- plot_data[plot_data$dgp == d, ]
  methods_here <- unique(sub$method)

  plot(NA, xlim = range(sub$n), ylim = c(0, 1),
       xlab = "Sample size (n)", ylab = "Exact recovery rate",
       main = d, las = 1)
  abline(h = seq(0, 1, 0.2), col = "gray90")

  for (m in methods_here) {
    s <- sub[sub$method == m, ]
    s <- s[order(s$n), ]
    lines(s$n, s$recovery, col = cols[m], lwd = 2, type = "b",
          pch = pchs[m], cex = 1.2)
  }
  legend("bottomright", legend = methods_here,
         col = cols[methods_here], pch = pchs[methods_here],
         lwd = 2, cex = 0.65, bg = "white")
}
mtext("Recovery Rate: causalreg vs ACR", outer = TRUE, cex = 1.2, font = 2)
dev.off()

# --- Figure: Computation time comparison ---
pdf("figures/fig_comparison_time.pdf", width = 10, height = 4)
par(mfrow = c(1, length(dgps)), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))

for (d in dgps) {
  sub <- plot_data[plot_data$dgp == d, ]
  methods_here <- unique(sub$method)
  ymax <- max(sub$avg_time, na.rm = TRUE) * 1.2

  plot(NA, xlim = range(sub$n), ylim = c(0, ymax),
       xlab = "Sample size (n)", ylab = "Avg. time per run (s)",
       main = d, las = 1)
  abline(h = pretty(c(0, ymax)), col = "gray90")

  for (m in methods_here) {
    s <- sub[sub$method == m, ]
    s <- s[order(s$n), ]
    lines(s$n, s$avg_time, col = cols[m], lwd = 2, type = "b",
          pch = pchs[m], cex = 1.2)
  }
  legend("topleft", legend = methods_here,
         col = cols[methods_here], pch = pchs[methods_here],
         lwd = 2, cex = 0.65, bg = "white")
}
mtext("Computation Time: causalreg vs ACR", outer = TRUE, cex = 1.2, font = 2)
dev.off()

# --- Figure: Gaussian p=5 (ACR only) ---
sub_gauss <- plot_data[plot_data$dgp == "Gaussian linear (p=5)", ]
if (nrow(sub_gauss) > 0) {
  pdf("figures/fig_gaussian_acr.pdf", width = 5, height = 4)
  par(mar = c(4, 4, 2.5, 1))
  plot(sub_gauss$n, sub_gauss$recovery, type = "b", col = cols["ACR"],
       pch = pchs["ACR"], lwd = 2, cex = 1.2,
       xlab = "Sample size (n)", ylab = "Exact recovery rate",
       main = "Gaussian Linear (p=5) — ACR only", las = 1, ylim = c(0, 1))
  abline(h = seq(0, 1, 0.2), col = "gray90")
  dev.off()
}

# --- LaTeX comparison table ---
cat("\nGenerating LaTeX comparison table...\n")
sink("results/table_comparison.tex")
cat("\\begin{tabular}{llrrrrr}\n")
cat("\\toprule\n")
cat("DGP & Method & $n$ & Recovery & TPR & Avg.\\ FP & Time (s) \\\\\n")
cat("\\midrule\n")

prev_dgp <- ""
for (i in seq_len(nrow(plot_data))) {
  r <- plot_data[i, ]
  if (r$dgp != prev_dgp && prev_dgp != "") cat("\\midrule\n")
  prev_dgp <- r$dgp
  cat(sprintf("%s & %s & %d & %.0f\\%% & %.2f & %.2f & %.2f \\\\\n",
              r$dgp, r$method, r$n,
              r$recovery * 100, r$tpr, r$avg_fp, r$avg_time))
}

cat("\\bottomrule\n")
cat("\\end{tabular}\n")
sink()

cat("\nDone! Figures and tables in figures/ and results/\n")
