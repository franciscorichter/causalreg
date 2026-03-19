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

The method is based on the following insight: in a correctly specified GLM/GAM
for a response Y given its true causal parents **X**_pa, the Pearson risk

R_P = (1/n) * sum of squared Pearson residuals

converges to 1. If the model includes non-causal variables or omits causal
ones, the Pearson risk deviates from 1.

The algorithm searches over subsets of the provided covariates, testing each
subset's Pearson risk against the null hypothesis R_P = 1 using either:

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

If you use this package, please cite:

> Vinciotti V, Wit EC (2026). *causalreg: Causal Generalized Linear Models*.
> R package version 0.1.2, <https://CRAN.R-project.org/package=causalreg>.

```bibtex
@Manual{,
  title = {causalreg: Causal Generalized Linear Models},
  author = {Veronica Vinciotti and Ernst C. Wit},
  year = {2026},
  note = {R package version 0.1.2},
  url = {https://CRAN.R-project.org/package=causalreg},
}
```

The underlying method is described in:

> Polinelli, A., V. Vinciotti and E.C. Wit. (2026). "Causal generalized linear
> models via Pearson risk invariance." *Journal of Causal Inference*.

## License

GPL-3
