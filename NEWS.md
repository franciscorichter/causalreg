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
