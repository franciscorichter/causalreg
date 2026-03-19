#!/usr/bin/env Rscript
# =============================================================================
# Simulation study for causalreg: Recovery of true causal parents
# =============================================================================

library(causalreg)

NCORES <- parallel::detectCores()
N_REPS <- 100
cat(sprintf("Using %d cores, %d reps per scenario\n", NCORES, N_REPS))

# =============================================================================
# Data Generating Processes (from package examples)
# =============================================================================

# DGP 1: Poisson GLM, p=2. DAG: X1 -> Y -> X2
dgp_poisson_p2 <- function(n) {
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(X1))
  X2 <- log(Y + 1) + rnorm(n, 0, 0.3)
  data.frame(X1, X2, Y)
}

# DGP 2: Binomial GLM, p=2. DAG: X1 -> Y -> X2 (with flip noise)
dgp_binomial_p2 <- function(n) {
  X1 <- rnorm(n)
  Y  <- rbinom(n, 1, exp(X1) / (1 + exp(X1)))
  flip <- rbinom(n, 1, 0.1)
  X2 <- (1 - flip) * Y + rnorm(n, 0, 0.3)
  data.frame(X1, X2, Y)
}

# DGP 3: Binomial GLM, p=5. DAG: X1->X2->Y<-X3, X2->X4, Y->X5
dgp_binomial_p5 <- function(n) {
  X1 <- rnorm(n)
  X2 <- rnorm(n, X1, 0.5)
  X3 <- rnorm(n, 0, 1)
  X4 <- rnorm(n, X2, 0.5)
  Y  <- rbinom(n, 1, exp(0.8 * X2 - 0.9 * X3) / (1 + exp(0.8 * X2 - 0.9 * X3)))
  flip <- rbinom(n, 1, 0.1)
  X5 <- (1 - flip) * Y + flip * (1 - Y) + rnorm(n, 0, 0.3)
  data.frame(X1, X2, X3, X4, X5, Y)
}

# DGP 4: Poisson GAM, p=2. DAG: X1 -> Y -> X2 (nonlinear)
dgp_poisson_gam_p2 <- function(n) {
  X1 <- rnorm(n)
  Y  <- rpois(n, exp(sin(X1)))
  X2 <- log(Y + 1) + rnorm(n, 0, 0.5)
  data.frame(X1, X2, Y)
}

# DGP 5: Poisson GAM, p=7. DAG: X2,X3 -> m -> Z -> Y; X5,X6 downstream of Z
dgp_poisson_gam_p7 <- function(n) {
  X1 <- rnorm(n, sd = sqrt(0.04))
  X2 <- X1 + rnorm(n, sd = sqrt(0.04))
  X3 <- X1 + X2 + rnorm(n, sd = sqrt(0.04))
  m  <- sin(X2 * 5) + X3^3
  Z  <- m + rnorm(n, sd = sqrt(0.04))
  X4 <- X2 + rnorm(n, sd = sqrt(0.04))
  X5 <- Z + rnorm(n, sd = sqrt(0.04))
  X6 <- Z + rnorm(n, sd = sqrt(0.04))
  X7 <- X6 + rnorm(n, sd = sqrt(0.04))
  Y  <- qpois(pnorm(Z, mean = m, sd = sqrt(0.04)), lambda = exp(m))
  data.frame(X1, X2, X3, X4, X5, X6, X7, Y)
}

# =============================================================================
# Simulation runner
# =============================================================================

extract_terms <- function(model_opt) {
  if (is.na(model_opt) || model_opt == "no potential causal model found")
    return(character(0))
  attr(terms.formula(as.formula(model_opt)), "term.labels")
}

run_scenario <- function(dgp_fn, n, method_fn, true_parents, n_reps, ncores_sim) {
  results <- parallel::mclapply(seq_len(n_reps), function(rep) {
    dat <- dgp_fn(n)
    tryCatch({
      r <- method_fn(dat)
      found <- extract_terms(r$model.opt)
      list(
        exact = setequal(found, true_parents),
        superset = all(true_parents %in% found),
        subset = all(found %in% true_parents),
        tp = sum(true_parents %in% found),
        fp = sum(!(found %in% true_parents)),
        n_found = length(found)
      )
    }, error = function(e) {
      list(exact = FALSE, superset = FALSE, subset = FALSE,
           tp = 0L, fp = 0L, n_found = 0L)
    })
  }, mc.cores = ncores_sim, mc.set.seed = TRUE)

  n_true <- length(true_parents)
  data.frame(
    exact_recovery = mean(sapply(results, `[[`, "exact")),
    superset_rate  = mean(sapply(results, `[[`, "superset")),
    subset_rate    = mean(sapply(results, `[[`, "subset")),
    tpr            = mean(sapply(results, `[[`, "tp")) / max(n_true, 1),
    avg_fp         = mean(sapply(results, `[[`, "fp")),
    avg_found      = mean(sapply(results, `[[`, "n_found"))
  )
}

# =============================================================================
# Define all scenarios
# =============================================================================

scenarios <- list(
  # DGP 1: Poisson GLM, p=2
  list(name = "Poisson GLM (p=2)", dgp = dgp_poisson_p2,
       family = "poisson", type = "glm",
       formula = Y ~ X1 + X2, true_parents = "X1",
       methods = list(
         list(label = "all + chi-sq",  pval = "chi-square", search = "all"),
         list(label = "step + chi-sq", pval = "chi-square", search = "stepwise"),
         list(label = "all + boot",    pval = "bootstrap",  search = "all"),
         list(label = "step + boot",   pval = "bootstrap",  search = "stepwise")
       ),
       sample_sizes = c(200, 500, 1000, 2000)),

  # DGP 2: Binomial GLM, p=2
  list(name = "Binomial GLM (p=2)", dgp = dgp_binomial_p2,
       family = "binomial", type = "glm",
       formula = Y ~ X1 + X2, true_parents = "X1",
       methods = list(
         list(label = "all + boot",  pval = "bootstrap", search = "all"),
         list(label = "step + boot", pval = "bootstrap", search = "stepwise")
       ),
       sample_sizes = c(500, 1000, 2000)),

  # DGP 3: Binomial GLM, p=5
  list(name = "Binomial GLM (p=5)", dgp = dgp_binomial_p5,
       family = "binomial", type = "glm",
       formula = Y ~ X1 + X2 + X3 + X4 + X5, true_parents = c("X2", "X3"),
       methods = list(
         list(label = "all + boot",  pval = "bootstrap", search = "all"),
         list(label = "step + boot", pval = "bootstrap", search = "stepwise")
       ),
       sample_sizes = c(1000, 2000, 3000)),

  # DGP 4: Poisson GAM, p=2
  list(name = "Poisson GAM (p=2)", dgp = dgp_poisson_gam_p2,
       family = "poisson", type = "gam",
       formula = Y ~ s(X1) + s(X2), true_parents = "s(X1)",
       methods = list(
         list(label = "all + chi-sq",  pval = "chi-square", search = "all"),
         list(label = "step + chi-sq", pval = "chi-square", search = "stepwise")
       ),
       sample_sizes = c(500, 1000, 2000)),

  # DGP 5: Poisson GAM, p=7
  list(name = "Poisson GAM (p=7)", dgp = dgp_poisson_gam_p7,
       family = "poisson", type = "gam",
       formula = Y ~ s(X1) + s(X2) + s(X3) + s(X4) + s(X5) + s(X6) + s(X7),
       true_parents = c("s(X2)", "s(X3)"),
       methods = list(
         list(label = "step + chi-sq", pval = "chi-square", search = "stepwise")
       ),
       sample_sizes = c(500, 1000))
)

# =============================================================================
# Run simulations
# =============================================================================

results_file <- "results/sim_results.rds"

if (file.exists(results_file)) {
  cat("Loading cached results from", results_file, "\n")
  all_results <- readRDS(results_file)
} else {
  all_results <- data.frame()
  total_scenarios <- sum(sapply(scenarios, function(s) length(s$methods) * length(s$sample_sizes)))
  counter <- 0

  for (sc in scenarios) {
    for (meth in sc$methods) {
      for (n in sc$sample_sizes) {
        counter <- counter + 1
        label <- sprintf("[%d/%d] %s | %s | n=%d",
                         counter, total_scenarios, sc$name, meth$label, n)
        cat(label, "... ")
        flush.console()

        method_fn <- if (sc$type == "glm") {
          function(dat) cglm(sc$formula, sc$family, dat,
                             pval = meth$pval, search = meth$search, ncores = 1L)
        } else {
          function(dat) cgam(sc$formula, sc$family, dat,
                             pval = meth$pval, search = meth$search, ncores = 1L)
        }

        t0 <- proc.time()
        res <- run_scenario(sc$dgp, n, method_fn, sc$true_parents,
                            n_reps = N_REPS, ncores_sim = NCORES)
        elapsed <- (proc.time() - t0)[["elapsed"]]

        row <- cbind(
          data.frame(dgp = sc$name, method = meth$label,
                     search = meth$search, pval = meth$pval,
                     n = n, n_reps = N_REPS,
                     stringsAsFactors = FALSE),
          res
        )
        all_results <- rbind(all_results, row)

        cat(sprintf("recovery=%.0f%% (%.1fs)\n", res$exact_recovery * 100, elapsed))
        flush.console()
      }
    }
  }

  saveRDS(all_results, results_file)
  cat("\nResults saved to", results_file, "\n")
}

# =============================================================================
# Generate plots
# =============================================================================

cat("\nGenerating plots...\n")

# Color palette
cols <- c("all + chi-sq" = "#E41A1C", "step + chi-sq" = "#377EB8",
          "all + boot" = "#FF7F00", "step + boot" = "#984EA3")
pchs <- c("all + chi-sq" = 16, "step + chi-sq" = 17,
          "all + boot" = 15, "step + boot" = 18)

# --- Figure 1: Recovery rate by sample size, one panel per DGP ---
dgps <- unique(all_results$dgp)
n_dgps <- length(dgps)

pdf("figures/fig_recovery_by_n.pdf", width = 10, height = 8)
par(mfrow = c(ceiling(n_dgps / 2), 2), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))

for (d in dgps) {
  sub <- all_results[all_results$dgp == d, ]
  methods_here <- unique(sub$method)

  plot(NA, xlim = range(sub$n), ylim = c(0, 1),
       xlab = "Sample size (n)", ylab = "Exact recovery rate",
       main = d, las = 1)
  abline(h = seq(0, 1, 0.2), col = "gray90")

  for (m in methods_here) {
    s <- sub[sub$method == m, ]
    s <- s[order(s$n), ]
    lines(s$n, s$exact_recovery, col = cols[m], lwd = 2, type = "b",
          pch = pchs[m], cex = 1.2)
  }

  legend("bottomright", legend = methods_here,
         col = cols[methods_here], pch = pchs[methods_here],
         lwd = 2, cex = 0.7, bg = "white")
}
mtext("Exact Recovery Rate by Sample Size", outer = TRUE, cex = 1.2, font = 2)
dev.off()

# --- Figure 2: Detailed metrics for the 5-covariate case ---
sub5 <- all_results[all_results$dgp == "Binomial GLM (p=5)", ]
if (nrow(sub5) > 0) {
  pdf("figures/fig_binomial_p5_detail.pdf", width = 10, height = 4)
  par(mfrow = c(1, 3), mar = c(4, 4, 2.5, 1))

  for (metric in c("exact_recovery", "tpr", "avg_fp")) {
    ylab <- switch(metric,
                   exact_recovery = "Exact recovery rate",
                   tpr = "True positive rate",
                   avg_fp = "Avg. false positives")
    ylim <- if (metric == "avg_fp") c(0, max(sub5[[metric]]) * 1.2) else c(0, 1)

    plot(NA, xlim = range(sub5$n), ylim = ylim,
         xlab = "Sample size (n)", ylab = ylab,
         main = ylab, las = 1)
    abline(h = if (metric == "avg_fp") 0:5 else seq(0, 1, 0.2), col = "gray90")

    for (m in unique(sub5$method)) {
      s <- sub5[sub5$method == m, ]
      s <- s[order(s$n), ]
      lines(s$n, s[[metric]], col = cols[m], lwd = 2, type = "b",
            pch = pchs[m], cex = 1.2)
    }
    legend("topright", legend = unique(sub5$method),
           col = cols[unique(sub5$method)], pch = pchs[unique(sub5$method)],
           lwd = 2, cex = 0.7, bg = "white")
  }
  dev.off()
}

# --- Figure 3: chi-square vs bootstrap comparison (Poisson GLM p=2) ---
sub_comp <- all_results[all_results$dgp == "Poisson GLM (p=2)", ]
if (nrow(sub_comp) > 0) {
  pdf("figures/fig_chisq_vs_bootstrap.pdf", width = 6, height = 5)
  par(mar = c(4, 4, 2.5, 1))

  plot(NA, xlim = range(sub_comp$n), ylim = c(0, 1),
       xlab = "Sample size (n)", ylab = "Exact recovery rate",
       main = "Chi-square vs Bootstrap (Poisson GLM, p=2)", las = 1)
  abline(h = seq(0, 1, 0.2), col = "gray90")

  for (m in unique(sub_comp$method)) {
    s <- sub_comp[sub_comp$method == m, ]
    s <- s[order(s$n), ]
    lines(s$n, s$exact_recovery, col = cols[m], lwd = 2, type = "b",
          pch = pchs[m], cex = 1.2)
  }
  legend("bottomright", legend = unique(sub_comp$method),
         col = cols[unique(sub_comp$method)], pch = pchs[unique(sub_comp$method)],
         lwd = 2, cex = 0.7, bg = "white")
  dev.off()
}

# --- Generate LaTeX table ---
cat("\nGenerating LaTeX table...\n")
sink("results/table_results.tex")
cat("\\begin{tabular}{llrrrrr}\n")
cat("\\toprule\n")
cat("DGP & Method & $n$ & Recovery & TPR & Avg.\\ FP & Avg.\\ found \\\\\n")
cat("\\midrule\n")

for (i in seq_len(nrow(all_results))) {
  r <- all_results[i, ]
  dgp_short <- gsub("\\(", "(", gsub("\\)", ")", r$dgp))
  cat(sprintf("%s & %s & %d & %.0f\\%% & %.2f & %.2f & %.1f \\\\\n",
              dgp_short, r$method, r$n,
              r$exact_recovery * 100, r$tpr, r$avg_fp, r$avg_found))
  # Add midrule between DGPs
  if (i < nrow(all_results) && all_results$dgp[i + 1] != r$dgp)
    cat("\\midrule\n")
}

cat("\\bottomrule\n")
cat("\\end{tabular}\n")
sink()

cat("\nDone! Figures in figures/, results in results/\n")
