# causalreg 0.3.0

* `cglm()` and `cgam()` gain a `direction` argument for the stepwise search,
  `"forward"` (the default and the previous behaviour) or `"backward"`. The
  backward search starts from the full model and removes, at each step, the
  variable whose removal gives the largest Pearson-risk p-value, stopping once
  the model is no longer rejected. Both directions then prune by BIC as before.
  Forward remains the default because it is cheaper.
* For a binomial response whose candidate set contains a categorical variable,
  the stepwise search now runs backward and says so. The Pearson risk of a
  binary regression on categorical covariates alone is mathematically equal
  to 1, so a forward search can enter such a variable at the first step and
  stop there.
* The stepwise search now emits the same "only one categorical variable"
  message that the exhaustive search already emitted.

# causalreg 0.2.2

* Parallel model search (`ncores > 1`) now works on **all platforms**. On
  Unix/macOS it continues to use forking (`parallel::mclapply`); on Windows it
  uses a PSOCK cluster (`parallel::parLapply`) with parallel-safe RNG streams.
  Previously `ncores` was silently ignored on Windows.
* Added the `causalreg.parallel` option to select the parallel backend
  (`"auto"`, `"fork"`, `"psock"`, `"sequential"`).

# causalreg 0.2.1

* `cgam()` fast bootstrap via fixed smoothing parameters (`fast_gam = TRUE`,
  the default): the smoothing parameters selected on the original data are held
  fixed across that model's bootstrap resamples, giving a 3-4x speedup.

# causalreg 0.2.0

* Added C++/Rcpp acceleration for GLM fitting, Pearson statistics, and the
  bootstrap (`use_cpp = TRUE`, the default for supported families).
* Added parallel model evaluation via the `ncores` argument.
