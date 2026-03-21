// Fast GLM fitting via IRLS for Poisson and Binomial families
// Uses RcppArmadillo for efficient linear algebra

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// ---------- Internal helpers ----------

static vec clamp_mu(const vec& mu, const std::string& family) {
  if (family == "poisson") {
    return clamp(mu, 1e-10, 1e10);
  } else {
    return clamp(mu, 1e-10, 1.0 - 1e-10);
  }
}

// Single IRLS fit: returns fitted mu vector
// Returns empty vector on failure
static vec irls_fit(const mat& X, const vec& y, const std::string& family,
                    int max_iter = 25, double tol = 1e-8) {
  int n = X.n_rows;
  int p = X.n_cols;

  if (n <= p) return vec();  // underdetermined

  vec mu(n), eta(n), z(n), w(n);

  // Initialize
  if (family == "poisson") {
    mu = y + 0.1;
    eta = log(mu);
  } else {
    mu = (y + 0.5) / 2.0;
    eta = log(mu / (1.0 - mu));
  }

  vec beta(p, fill::zeros);
  vec beta_new(p);

  for (int iter = 0; iter < max_iter; iter++) {
    mu = clamp_mu(mu, family);

    // Compute working weights and response
    if (family == "poisson") {
      w = mu;
      z = eta + (y - mu) / mu;
    } else {
      vec var_mu = mu % (1.0 - mu);
      w = var_mu;
      z = eta + (y - mu) / var_mu;
    }

    // Clamp weights
    w = clamp(w, 1e-10, 1e10);

    // Weighted least squares via QR on sqrt(W)*X
    vec sqrt_w = sqrt(w);
    mat Xw = X.each_col() % sqrt_w;
    vec zw = z % sqrt_w;

    // Solve using QR decomposition
    bool ok = solve(beta_new, Xw, zw, solve_opts::no_approx);
    if (!ok) return vec();  // singular

    // Update
    eta = X * beta_new;
    if (family == "poisson") {
      mu = exp(eta);
    } else {
      mu = 1.0 / (1.0 + exp(-eta));
    }

    // Check convergence
    double beta_norm = norm(beta, 2);
    if (norm(beta_new - beta, 2) < tol * (beta_norm + tol)) {
      return clamp_mu(mu, family);
    }
    beta = beta_new;
  }

  return clamp_mu(mu, family);  // return even if not fully converged
}

// ---------- Exported functions ----------

//' Fast GLM fit via IRLS
//'
//' @param X Design matrix (n x p), including intercept column if needed
//' @param y Response vector (n)
//' @param family "poisson" or "binomial"
//' @param max_iter Maximum IRLS iterations (default 25)
//' @param tol Convergence tolerance (default 1e-8)
//' @return List with fitted_values (mu) and converged flag
// [[Rcpp::export]]
List fast_glm_fit(const arma::mat& X, const arma::vec& y,
                  const std::string& family,
                  int max_iter = 25, double tol = 1e-8) {
  vec mu = irls_fit(X, y, family, max_iter, tol);
  bool ok = mu.n_elem > 0;

  if (!ok) {
    // Return NAs on failure
    mu = vec(y.n_elem, fill::value(datum::nan));
  }

  return List::create(
    Named("fitted_values") = mu,
    Named("converged") = ok
  );
}

//' Compute Pearson chi-square statistic from y and mu
//'
//' @param y Response vector
//' @param mu Fitted values
//' @param family "poisson" or "binomial"
//' @return Pearson chi-square statistic (sum of squared Pearson residuals)
// [[Rcpp::export]]
double pearson_stat_cpp(const arma::vec& y, const arma::vec& mu,
                        const std::string& family) {
  int n = y.n_elem;
  double stat = 0.0;

  if (family == "poisson") {
    for (int i = 0; i < n; i++) {
      double mu_i = std::max(mu(i), 1e-10);
      double r = y(i) - mu_i;
      stat += r * r / mu_i;
    }
  } else {
    for (int i = 0; i < n; i++) {
      double mu_i = std::min(std::max(mu(i), 1e-10), 1.0 - 1e-10);
      double r = y(i) - mu_i;
      stat += r * r / (mu_i * (1.0 - mu_i));
    }
  }

  return stat;
}

//' Compute log-likelihood for Poisson or Binomial GLM
//'
//' @param y Response vector
//' @param mu Fitted values
//' @param family "poisson" or "binomial"
//' @return Log-likelihood value
// [[Rcpp::export]]
double glm_loglik_cpp(const arma::vec& y, const arma::vec& mu,
                      const std::string& family) {
  int n = y.n_elem;
  double ll = 0.0;

  if (family == "poisson") {
    for (int i = 0; i < n; i++) {
      double mu_i = std::max(mu(i), 1e-10);
      ll += y(i) * std::log(mu_i) - mu_i - std::lgamma(y(i) + 1.0);
    }
  } else {
    for (int i = 0; i < n; i++) {
      double mu_i = std::min(std::max(mu(i), 1e-10), 1.0 - 1e-10);
      ll += y(i) * std::log(mu_i) + (1.0 - y(i)) * std::log(1.0 - mu_i);
    }
  }

  return ll;
}

//' Compute BIC for a GLM fit
//'
//' @param y Response vector
//' @param mu Fitted values
//' @param family "poisson" or "binomial"
//' @param p Number of parameters
//' @param n Number of observations
//' @return BIC value
// [[Rcpp::export]]
double fast_glm_bic(const arma::vec& y, const arma::vec& mu,
                    const std::string& family, int p, int n) {
  double ll = glm_loglik_cpp(y, mu, family);
  return -2.0 * ll + std::log((double)n) * p;
}

//' Fit GLM and compute Pearson risk, BIC, and optionally bootstrap p-value
//'
//' @param X Design matrix
//' @param y Response vector
//' @param family "poisson" or "binomial"
//' @return List with pearson_stat, pearson_risk, bic
// [[Rcpp::export]]
List fast_fit_and_stat(const arma::mat& X, const arma::vec& y,
                       const std::string& family) {
  int n = X.n_rows;
  int p = X.n_cols;

  vec mu = irls_fit(X, y, family);
  bool ok = mu.n_elem > 0;

  double ps = ok ? pearson_stat_cpp(y, mu, family) : datum::nan;
  double bic = ok ? fast_glm_bic(y, mu, family, p, n) : datum::nan;

  return List::create(
    Named("pearson_stat") = ps,
    Named("pearson_risk") = ps / n,
    Named("bic") = bic,
    Named("converged") = ok
  );
}

//' Bootstrap p-value for Pearson risk = 1 test (fully in C++)
//'
//' @param X Design matrix
//' @param y Response vector
//' @param family "poisson" or "binomial"
//' @param B Number of bootstrap replicates
//' @return Two-sided bootstrap p-value
// [[Rcpp::export]]
double boot_pval_cpp(const arma::mat& X, const arma::vec& y,
                     const std::string& family, int B = 100) {
  int n = X.n_rows;
  int count_ge1 = 0;

  for (int b = 0; b < B; b++) {
    // Bootstrap sample indices
    uvec idx = randi<uvec>(n, distr_param(0, n - 1));
    mat Xb = X.rows(idx);
    vec yb = y.elem(idx);

    vec mu = irls_fit(Xb, yb, family);

    if (mu.n_elem == 0) continue;  // skip failed fits

    double pr = pearson_stat_cpp(yb, mu, family) / n;
    if (pr >= 1.0) count_ge1++;
  }

  double prob = (double)count_ge1 / B;
  return 2.0 * std::min(prob, 1.0 - prob);
}

//' Evaluate multiple submodels in C++ (for exhaustive search)
//'
//' Given a full design matrix and a list of column index vectors,
//' fit each submodel and return Pearson risk, p-value, BIC.
//'
//' @param X_full Full design matrix (with intercept)
//' @param y Response vector
//' @param family "poisson" or "binomial"
//' @param col_indices List of integer vectors, each specifying columns of X_full
//' @param pval_method "chi-square" or "bootstrap"
//' @param B Number of bootstrap replicates
//' @return List of vectors: pearson_risk, pval, bic
// [[Rcpp::export]]
List eval_submodels_cpp(const arma::mat& X_full, const arma::vec& y,
                        const std::string& family,
                        const List& col_indices,
                        const std::string& pval_method,
                        int B = 100) {
  int n = X_full.n_rows;
  int n_models = col_indices.size();

  vec pearson_risk(n_models);
  vec pval(n_models);
  vec bic(n_models);

  for (int m = 0; m < n_models; m++) {
    IntegerVector cols_r = col_indices[m];
    int p = cols_r.size();

    // Extract submodel columns (convert 1-based R indices to 0-based)
    uvec cols(p);
    for (int j = 0; j < p; j++) {
      cols(j) = cols_r[j] - 1;
    }
    mat Xsub = X_full.cols(cols);

    vec mu = irls_fit(Xsub, y, family);

    if (mu.n_elem == 0) {
      pearson_risk(m) = datum::nan;
      pval(m) = 0.0;
      bic(m) = datum::inf;
      continue;
    }

    double ps = pearson_stat_cpp(y, mu, family);
    pearson_risk(m) = ps / n;
    bic(m) = fast_glm_bic(y, mu, family, p, n);

    if (pval_method == "chi-square") {
      double df = n - p;
      if (df > 0) {
        // Two-sided chi-square p-value (use R's pchisq)
        double p_lower = R::pchisq(ps, df, 1, 0);
        double p_upper = R::pchisq(ps, df, 0, 0);
        pval(m) = 2.0 * std::min(p_lower, p_upper);
      } else {
        pval(m) = 0.0;
      }
    } else {
      pval(m) = boot_pval_cpp(Xsub, y, family, B);
    }
  }

  return List::create(
    Named("pearson_risk") = pearson_risk,
    Named("pval") = pval,
    Named("bic") = bic
  );
}
