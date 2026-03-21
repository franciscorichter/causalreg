## Benchmark: C++ accelerated vs pure R implementation
## Compares correctness and timing across multiple scenarios

library(causalreg)
library(microbenchmark)

cat("====================================================\n")
cat("  causalreg: C++ vs R Benchmark\n")
cat("====================================================\n\n")

results <- list()

# ---------------------------------------------------------------------------
# Scenario 1: Poisson GLM, chi-square, exhaustive search, p=2
# ---------------------------------------------------------------------------
cat("--- Scenario 1: Poisson, chi-square, all, p=2 ---\n")
set.seed(123)
n <- 1000
X1 <- rnorm(n); Y <- rpois(n, exp(X1)); X2 <- log(Y+1) + rnorm(n, 0, 0.3)
dat1 <- data.frame(X1, X2, Y)

mb1 <- microbenchmark(
  R   = { set.seed(42); cglm(Y ~ X1+X2, "poisson", dat1, pval="chi-square", search="all", use_cpp=FALSE) },
  Cpp = { set.seed(42); cglm(Y ~ X1+X2, "poisson", dat1, pval="chi-square", search="all", use_cpp=TRUE) },
  times = 20
)
print(mb1)
results[[length(results)+1]] <- data.frame(
  scenario = "Poisson, chi-sq, all, p=2, n=1000",
  R_median_ms   = median(mb1$time[mb1$expr=="R"])   / 1e6,
  Cpp_median_ms  = median(mb1$time[mb1$expr=="Cpp"]) / 1e6
)

# ---------------------------------------------------------------------------
# Scenario 2: Poisson GLM, chi-square, exhaustive search, p=5
# ---------------------------------------------------------------------------
cat("\n--- Scenario 2: Poisson, chi-square, all, p=5 ---\n")
set.seed(123)
n <- 1000
X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5); X3 <- rnorm(n)
X4 <- rnorm(n, X2, 0.5); X5 <- rnorm(n)
Y <- rpois(n, exp(0.5*X1 - 0.3*X3))
dat2 <- data.frame(X1, X2, X3, X4, X5, Y)

mb2 <- microbenchmark(
  R   = { set.seed(42); cglm(Y ~ X1+X2+X3+X4+X5, "poisson", dat2, pval="chi-square", search="all", use_cpp=FALSE) },
  Cpp = { set.seed(42); cglm(Y ~ X1+X2+X3+X4+X5, "poisson", dat2, pval="chi-square", search="all", use_cpp=TRUE) },
  times = 10
)
print(mb2)
results[[length(results)+1]] <- data.frame(
  scenario = "Poisson, chi-sq, all, p=5, n=1000",
  R_median_ms   = median(mb2$time[mb2$expr=="R"])   / 1e6,
  Cpp_median_ms  = median(mb2$time[mb2$expr=="Cpp"]) / 1e6
)

# ---------------------------------------------------------------------------
# Scenario 3: Poisson GLM, chi-square, stepwise, p=5
# ---------------------------------------------------------------------------
cat("\n--- Scenario 3: Poisson, chi-square, stepwise, p=5 ---\n")
mb3 <- microbenchmark(
  R   = { set.seed(42); cglm(Y ~ X1+X2+X3+X4+X5, "poisson", dat2, pval="chi-square", search="stepwise", use_cpp=FALSE) },
  Cpp = { set.seed(42); cglm(Y ~ X1+X2+X3+X4+X5, "poisson", dat2, pval="chi-square", search="stepwise", use_cpp=TRUE) },
  times = 10
)
print(mb3)
results[[length(results)+1]] <- data.frame(
  scenario = "Poisson, chi-sq, step, p=5, n=1000",
  R_median_ms   = median(mb3$time[mb3$expr=="R"])   / 1e6,
  Cpp_median_ms  = median(mb3$time[mb3$expr=="Cpp"]) / 1e6
)

# ---------------------------------------------------------------------------
# Scenario 4: Binomial GLM, bootstrap (B=50), exhaustive, p=2
# ---------------------------------------------------------------------------
cat("\n--- Scenario 4: Binomial, bootstrap B=50, all, p=2 ---\n")
set.seed(123)
n <- 2000
X1 <- rnorm(n); Y <- rbinom(n, 1, exp(X1)/(1+exp(X1)))
flip <- rbinom(n, 1, 0.1); X2 <- (1-flip)*Y + rnorm(n, 0, 0.3)
dat4 <- data.frame(X1, X2, Y)

mb4 <- microbenchmark(
  R   = { set.seed(1); cglm(Y ~ X1+X2, "binomial", dat4, pval="bootstrap", B=50, search="all", use_cpp=FALSE) },
  Cpp = { set.seed(1); cglm(Y ~ X1+X2, "binomial", dat4, pval="bootstrap", B=50, search="all", use_cpp=TRUE) },
  times = 5
)
print(mb4)
results[[length(results)+1]] <- data.frame(
  scenario = "Binomial, boot B=50, all, p=2, n=2000",
  R_median_ms   = median(mb4$time[mb4$expr=="R"])   / 1e6,
  Cpp_median_ms  = median(mb4$time[mb4$expr=="Cpp"]) / 1e6
)

# ---------------------------------------------------------------------------
# Scenario 5: Binomial GLM, bootstrap (B=50), exhaustive, p=5
# ---------------------------------------------------------------------------
cat("\n--- Scenario 5: Binomial, bootstrap B=50, all, p=5 ---\n")
set.seed(12)
n <- 3000
X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5); X3 <- rnorm(n)
X4 <- rnorm(n, X2, 0.5)
Y <- rbinom(n, 1, exp(0.8*X2 - 0.9*X3) / (1 + exp(0.8*X2 - 0.9*X3)))
flip <- rbinom(n, 1, 0.1)
X5 <- (1-flip)*Y + flip*(1-Y) + rnorm(n, 0, 0.3)
dat5 <- data.frame(X1, X2, X3, X4, X5, Y)

mb5 <- microbenchmark(
  R   = { set.seed(1); cglm(Y ~ X1+X2+X3+X4+X5, "binomial", dat5, pval="bootstrap", B=50, search="all", use_cpp=FALSE) },
  Cpp = { set.seed(1); cglm(Y ~ X1+X2+X3+X4+X5, "binomial", dat5, pval="bootstrap", B=50, search="all", use_cpp=TRUE) },
  times = 3
)
print(mb5)
results[[length(results)+1]] <- data.frame(
  scenario = "Binomial, boot B=50, all, p=5, n=3000",
  R_median_ms   = median(mb5$time[mb5$expr=="R"])   / 1e6,
  Cpp_median_ms  = median(mb5$time[mb5$expr=="Cpp"]) / 1e6
)

# ---------------------------------------------------------------------------
# Scenario 6: Binomial GLM, bootstrap (B=50), stepwise, p=5
# ---------------------------------------------------------------------------
cat("\n--- Scenario 6: Binomial, bootstrap B=50, stepwise, p=5 ---\n")
mb6 <- microbenchmark(
  R   = { set.seed(1); cglm(Y ~ X1+X2+X3+X4+X5, "binomial", dat5, pval="bootstrap", B=50, search="stepwise", use_cpp=FALSE) },
  Cpp = { set.seed(1); cglm(Y ~ X1+X2+X3+X4+X5, "binomial", dat5, pval="bootstrap", B=50, search="stepwise", use_cpp=TRUE) },
  times = 3
)
print(mb6)
results[[length(results)+1]] <- data.frame(
  scenario = "Binomial, boot B=50, step, p=5, n=3000",
  R_median_ms   = median(mb6$time[mb6$expr=="R"])   / 1e6,
  Cpp_median_ms  = median(mb6$time[mb6$expr=="Cpp"]) / 1e6
)

# ---------------------------------------------------------------------------
# Scenario 7: Large n, Poisson, chi-square, all, p=3
# ---------------------------------------------------------------------------
cat("\n--- Scenario 7: Poisson, chi-square, all, p=3, n=10000 ---\n")
set.seed(123)
n <- 10000
X1 <- rnorm(n); X2 <- rnorm(n, X1, 0.5); X3 <- rnorm(n)
Y <- rpois(n, exp(0.5*X1))
dat7 <- data.frame(X1, X2, X3, Y)

mb7 <- microbenchmark(
  R   = { set.seed(42); cglm(Y ~ X1+X2+X3, "poisson", dat7, pval="chi-square", search="all", use_cpp=FALSE) },
  Cpp = { set.seed(42); cglm(Y ~ X1+X2+X3, "poisson", dat7, pval="chi-square", search="all", use_cpp=TRUE) },
  times = 10
)
print(mb7)
results[[length(results)+1]] <- data.frame(
  scenario = "Poisson, chi-sq, all, p=3, n=10000",
  R_median_ms   = median(mb7$time[mb7$expr=="R"])   / 1e6,
  Cpp_median_ms  = median(mb7$time[mb7$expr=="Cpp"]) / 1e6
)

# ---------------------------------------------------------------------------
# Compile results and generate LaTeX table
# ---------------------------------------------------------------------------
res_df <- do.call(rbind, results)
res_df$speedup <- res_df$R_median_ms / res_df$Cpp_median_ms

cat("\n====================================================\n")
cat("  Summary\n")
cat("====================================================\n")
print(res_df, digits = 3)

# Save CSV
write.csv(res_df, "simulations/benchmark_results.csv", row.names = FALSE)

# Generate LaTeX table
tex_lines <- c(
  "\\begin{tabular}{lrrr}",
  "\\toprule",
  "Scenario & R (ms) & C++ (ms) & Speedup \\\\",
  "\\midrule"
)
for (i in seq_len(nrow(res_df))) {
  tex_lines <- c(tex_lines, sprintf(
    "%s & %.1f & %.1f & %.1f$\\times$ \\\\",
    res_df$scenario[i],
    res_df$R_median_ms[i],
    res_df$Cpp_median_ms[i],
    res_df$speedup[i]
  ))
}
tex_lines <- c(tex_lines, "\\bottomrule", "\\end{tabular}")
writeLines(tex_lines, "simulations/results/table_benchmark.tex")

cat("\nBenchmark complete. Results saved to:\n")
cat("  simulations/benchmark_results.csv\n")
cat("  simulations/results/table_benchmark.tex\n")
