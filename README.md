# causalreg

> **Note:** This is a fork of the
> [`causalreg`](https://cran.r-universe.dev/causalreg) CRAN package by
> Vinciotti and Wit, refactored for code clarity and efficiency. The
> statistical method is unchanged.

Causal discovery in generalized linear models (GLMs) and generalized additive
models (GAMs) via **Pearson risk invariance**.

Given a response variable and a set of candidate covariates, `causalreg`
identifies the subset of covariates that are causal parents of the response
within a structural causal model. The key idea is that the Pearson risk
(expected sum of squared Pearson residuals divided by the sample size) equals 1
if and only if the model is correctly specified with respect to its causal
parents.

## Installation

Install the development version from GitHub:

```r
# install.packages("devtools")
devtools::install_github("franciscorichter/causalreg")
```

Or install from CRAN:

```r
install.packages("causalreg")
```

## Quick start

### Causal Poisson GLM

```r
library(causalreg)

# Simulate data: X1 is a cause of Y, X2 is an effect
n <- 1000
set.seed(123)
X1 <- rnorm(n)
Y  <- rpois(n, exp(X1))
X2 <- log(Y + 1) + rnorm(n, 0, 0.3)
data <- data.frame(X1, X2, Y)

# Exhaustive search with chi-square test
result <- cglm(Y ~ X1 + X2, "poisson", data, pval = "chi-square", search = "all")
result$model.opt
#> [1] "Y ~ X1"
```

### Causal logistic GLM

```r
n <- 2000
set.seed(123)
X1 <- rnorm(n)
Y  <- rbinom(n, 1, exp(X1) / (1 + exp(X1)))
flip <- rbinom(n, 1, 0.1)
X2 <- (1 - flip) * Y + rnorm(n, 0, 0.3)
data <- data.frame(X1, X2, Y)

# Stepwise search with bootstrap test
set.seed(1)
result <- cglm(Y ~ X1 + X2, "binomial", data, pval = "bootstrap", search = "stepwise")
result$model.opt
#> [1] "Y ~ X1"
```

### Causal Poisson GAM (nonlinear effects)

```r
n <- 1000
set.seed(123)
X1 <- rnorm(n)
Y  <- rpois(n, exp(sin(X1)))
X2 <- log(Y + 1) + rnorm(n, 0, 0.5)
data <- data.frame(X1, X2, Y)

result <- cgam(Y ~ s(X1) + s(X2), "poisson", data, pval = "chi-square", search = "all")
result$model.opt
#> [1] "Y ~ s(X1)"
```

## Main functions

| Function | Description |
|----------|-------------|
| `cglm()` | Causal discovery within a generalized linear model |
| `cgam()` | Causal discovery within a generalized additive model |

Both functions share the same interface:

- **`formula`** -- Model formula. For `cglm`, use standard R formula syntax
  (`Y ~ X1 + X2`). For `cgam`, you can include smooth terms (`Y ~ s(X1) + s(X2)`).
- **`family`** -- `"poisson"` or `"binomial"`.
- **`data`** -- A data frame.
- **`alpha`** -- Significance level (default 0.05). Models whose Pearson risk
  p-value exceeds `alpha` are considered causally valid.
- **`pval`** -- `"chi-square"` (fast, for Poisson) or `"bootstrap"` (general,
  required for binomial).
- **`B`** -- Number of bootstrap replicates (default 100).
- **`search`** -- `"all"` (exhaustive, evaluates all 2^p - 1 subsets) or
  `"stepwise"` (greedy forward selection + backward BIC pruning).

## Return values

### Exhaustive search (`search = "all"`)

| Element | Description |
|---------|-------------|
| `$model.opt` | Formula string of the selected causal model |
| `$models` | List of all evaluated model formula strings |
| `$pv` | P-values for the Pearson risk test (one per model) |
| `$bic` | BIC values (one per model) |
| `$pearsonrisk` | Pearson risk values (one per model) |

### Stepwise search (`search = "stepwise"`)

| Element | Description |
|---------|-------------|
| `$model.opt` | Formula string of the selected causal model |
| `$models` | List of model formula strings visited during the search |

## How it works

The method rests on a population-level identity. For a response Y and a set of
candidate parents **X**, the **Pearson risk** is the expected squared Pearson
residual

R_P = E[ (Y − μ(**X**))² / V(μ(**X**)) ]

where μ(**X**) is the conditional mean and V(·) is the family's variance
function. When the model is correctly specified with respect to the true causal
parents, the conditional variance of Y equals V(μ(**X**)), so each squared
Pearson residual has expectation 1 and therefore

R_P = 1   (exactly, for the causal model).

If the model includes non-causal variables or omits causal ones, the mean or
variance structure is misspecified and R_P ≠ 1.

Empirically, R_P is estimated by the average of the squared Pearson residuals,
(1/n) · Σ r̂ᵢ², and for each candidate subset the algorithm performs a
statistical test of the null hypothesis H₀: R_P = 1 using either:

- A **chi-square test** (fast, asymptotically valid for Poisson models), or
- A **bootstrap test** (general, works for any family including binomial).

Among all subsets that pass the test (p-value > alpha), the one with the
lowest BIC is selected.

## Choosing between search strategies

- **`search = "all"`**: Evaluates all 2^p - 1 non-empty subsets. Guaranteed to
  find the global optimum but becomes expensive for p > 15 covariates.
- **`search = "stepwise"`**: Forward greedy search (add the variable that
  maximizes the Pearson risk p-value) followed by backward BIC pruning. Much
  faster for large p, but may miss the global optimum.

## Choosing between p-value methods

- **`pval = "chi-square"`**: Uses the chi-square distribution of the Pearson
  statistic. Fast and appropriate for Poisson models.
- **`pval = "bootstrap"`**: Nonparametric bootstrap test. Required for
  binomial models and generally more robust. Controlled by the `B` parameter
  (more bootstrap samples = more precise but slower).

## Citation

If you use this package, please cite the paper describing the method:

> Polinelli A, Vinciotti V, Wit EC (2026). "Causal generalized linear models via
> Pearson risk invariance." *Journal of Causal Inference*, 14(1), 20240043.
> <https://doi.org/10.1515/jci-2024-0043>

```bibtex
@article{polinelli2026causal,
  title   = {Causal generalized linear models via {Pearson} risk invariance},
  author  = {Polinelli, A. and Vinciotti, Veronica and Wit, Ernst C.},
  journal = {Journal of Causal Inference},
  year    = {2026},
  volume  = {14},
  number  = {1},
  pages   = {20240043},
  doi     = {10.1515/jci-2024-0043},
}
```

## Changelog

### v0.2.1 — Faster GAM bootstrap (June 2026)

The v0.2.0 C++ acceleration covered only `cglm()`. `cgam()` with
`pval = "bootstrap"` stayed on pure R/mgcv and could be very slow, because every
bootstrap resample re-ran mgcv's smoothing-parameter (REML/GCV) selection. An
exhaustive search over 5 covariates (31 models, `B = 100`) on `n = 5000`
binomial data took roughly **43 minutes**.

- **New `fast_gam` argument in `cgam()`** (default `TRUE`). When enabled, the
  smoothing parameters chosen on the original data — once per candidate model —
  are held fixed across that model's bootstrap resamples, so each resample fit is
  a single penalized IRLS rather than a full smoothing-parameter search. This is
  an approximation of the fully re-selected bootstrap (individual bootstrap
  p-values can shift, more so at small `B`), but it produced the **same model
  selection** in all validation runs. Because it is now the default, GAM
  bootstrap results differ slightly from v0.2.0 — pass `fast_gam = FALSE` to
  recover the exact (re-selected) bootstrap.
- **`ncores` now matters for GAM too.** Model evaluations parallelize across
  cores via `parallel::mclapply` — this is exact (no approximation). The two
  levers compose.

Timings on the `n = 5000`, 5-covariate, `B = 100` example (Apple M1 Max);
all three configurations select the same model `Y ~ s(X2) + s(X3) + s(X5)`:

| Configuration | Time | Speedup |
|---|------:|------:|
| `ncores = 1, fast_gam = FALSE` (v0.2.0 behavior) | ~43 min | 1.0× |
| `ncores = 8, fast_gam = FALSE` (exact, parallel) | 20.0 min | 2.1× |
| `ncores = 8, fast_gam = TRUE`  (parallel + fixed sp) | 7.7 min | 5.6× |

```r
# fast_gam = TRUE is the default; add ncores for large searches
cgam(fml, "binomial", data, pval = "bootstrap", search = "all", ncores = 8)

# exact (re-selected) bootstrap, matching v0.2.0 — slower
cgam(fml, "binomial", data, pval = "bootstrap", search = "all",
     ncores = 8, fast_gam = FALSE)
```

- Method reference updated to the published version: Polinelli, Vinciotti & Wit
  (2026), *Journal of Causal Inference* 14(1), 20240043.

### v0.2.0 — C++ acceleration via Rcpp (March 2026)

**C++ fast path for GLM fitting (Rcpp/RcppArmadillo)**

- Added `src/fast_glm.cpp` implementing a bare-bones IRLS solver for Poisson
  (log link) and binomial (logit link) families, operating directly on design
  matrices to avoid formula parsing and model-frame overhead.
- Reimplemented Pearson chi-square statistic, log-likelihood, and BIC
  computation in C++ as single-pass routines (`pearson_stat_cpp`,
  `glm_loglik_cpp`, `fast_glm_bic`).
- Moved the full bootstrap resampling loop into C++ (`boot_pval_cpp`),
  eliminating per-replicate R interpreter round-trips.
- Added batch submodel evaluator (`eval_submodels_cpp`) for exhaustive search.
- Modified `helpers.R`, `boot_pval.R`, `causal_all.R`, `causal_step.R`, and
  `cglm.R` to auto-dispatch to the C++ fast path when applicable.
- New `use_cpp` parameter in `cglm()` (default `TRUE`); set to `FALSE` to
  restore pure-R behavior. `cgam()` is unaffected (always uses R/mgcv).
- Achieves 1.4--8.7× speedups depending on scenario (see benchmark table above).

**Benchmark**

- Added `simulations/benchmark.R` with 7 scenarios comparing C++ vs R.

### v0.1.2-fork — Refactored fork vs CRAN v0.1.2

The original CRAN package by Vinciotti and Wit
([source](https://cran.r-universe.dev/causalreg)) contained four near-identical
internal files (`cglm_all`, `cgam_all`, `cglm_step`, `cgam_step`) with
substantial code duplication. This fork refactors the codebase for clarity and
efficiency while preserving the statistical method exactly.

**Code cleanup and deduplication**

- Merged `cglm_all.R` and `cgam_all.R` into a single `causal_all.R` with a
  `use_gam` flag, eliminating ~95% duplicated code.
- Merged `cglm_step.R` and `cgam_step.R` into a single `causal_step.R`.
- Extracted shared logic into `helpers.R`: model fitting (`.fit_model`),
  effective degrees of freedom (`.compute_edf`), Pearson p-value computation
  (`.fit_and_test`), and categorical variable handling
  (`.handle_categorical_all`, `.handle_categorical_step`).
- Total: 4 implementation files reduced to 3 with shared internals.

**Bug fixes**

- Fixed `cgam_step.R` line 89: bootstrap p-value was computed on `fmli` (the
  last formula from the inner loop) instead of `mod.min` (the selected model).
- Fixed `cgam.R`: default `pval` argument was `"chi-squared"` (with trailing
  'd') but internal comparisons used `"chi-square"`, so the chi-square option
  silently fell through to bootstrap for `cgam()`.
- Fixed `cgam_all`: `pearsonrisk` output contained the raw Pearson statistic
  instead of the Pearson risk (divided by n), inconsistent with `cglm_all`.
- Added `match.arg()` validation for `pval` and `search` parameters in both
  `cglm()` and `cgam()` (previously, invalid values were silently accepted).

**Computational efficiency**

- Pre-allocated all numeric vectors (`pearson_all`, `pv_all`, `bic_all`,
  `pvals`, `bics`, bootstrap `pr`). The original code used `c(vec, val)` inside
  loops, causing O(n^2) memory allocation.
- Stepwise forward phase: reuses the stored p-value from the inner loop instead
  of refitting the selected model and recomputing the p-value (saves 1 + B
  model fits per forward step when using bootstrap).
- Stepwise backward BIC phase: reuses the stored BIC from the inner loop
  instead of refitting the selected model (saves 1 model fit per backward step).
- `boot_pval`: uses `sample.int(n)` instead of `sample(1:n)` and explicit
  `use_gam` dispatch instead of fragile `all.vars(formula)[2] == "."` heuristic.

**Packaging**

- Added README with full API documentation and usage guide.
- Added vignette (`vignettes/introduction.Rmd`) with worked examples.
- Added testthat tests (22 tests covering all function/family/search/pval
  combinations).
- Added `.gitignore`, `.Rbuildignore`.
- R CMD check passes with Status: OK.

## C++ Acceleration (Rcpp)

*Last updated: March 2026*

Core GLM fitting, Pearson statistic computation, and bootstrap resampling have
been reimplemented in C++ via RcppArmadillo. The fast path activates
automatically for `cglm()` with `"poisson"` or `"binomial"` families. Set
`use_cpp = FALSE` to use the original pure-R implementation. The C++ path does
**not** cover `cgam()` (GAMs are fit with R/mgcv); for fast GAM bootstrap
searches use `fast_gam = TRUE` and `ncores` instead (see v0.2.1 in the
changelog).

Benchmark results (median wall-clock time, Apple Silicon ARM64, R 4.4.2):

| Scenario | R (ms) | C++ (ms) | Speedup |
|----------|-------:|--------:|--------:|
| Poisson, χ², all, p=2, n=1,000 | 5.7 | 2.5 | 2.3× |
| Poisson, χ², all, p=5, n=1,000 | 51.8 | 14.9 | 3.5× |
| Poisson, χ², step, p=5, n=1,000 | 49.2 | 34.1 | 1.4× |
| Binomial, boot B=50, all, p=2, n=2,000 | 399.3 | 45.7 | 8.7× |
| Binomial, boot B=50, all, p=5, n=3,000 | 6,538.0 | 916.2 | 7.1× |
| Binomial, boot B=50, step, p=5, n=3,000 | 3,007.7 | 441.8 | 6.8× |
| Poisson, χ², all, p=3, n=10,000 | 80.3 | 25.3 | 3.2× |

## License

GPL-3
