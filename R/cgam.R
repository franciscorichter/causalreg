#' Causal generalized additive model
#'
#' This function does a search for a causal submodel within the generalized additive model provided.
#'
#' @param formula A formula object.
#' @param family A distributional family object. Currently supported options are: binomial and poisson.
#' @param data A data frame containing the variables in the model.
#' @param alpha Significance level for statistical test.
#' @param pval If pval="bootstrap", a bootstrap test is conducted to test whether Pearson risk is 1. When family="poisson" a chi-squared test can be conducted by setting pval="chi-square".
#' @param B Number of bootstrap samples when pval="bootstrap". Default is 100.
#' @param search If search="stepwise", a greedy forward stepwise search is conducted. Default is search="all", in which case all possible submodels are considered.
#' @param ncores Number of cores for parallel computation. Default is 1 (sequential). When ncores > 1, model evaluations are distributed across cores using forking (\code{parallel::mclapply}) on Unix/macOS and a PSOCK cluster (\code{parallel::parLapply}) on Windows, so parallelization is available on all platforms. The backend can be overridden with \code{options(causalreg.parallel = )} ("auto", "fork", "psock", or "sequential").
#' @param fast_gam Logical; only relevant when pval="bootstrap". If TRUE, the smoothing parameters selected on the original data (once per candidate model) are held fixed across that model's bootstrap resamples, so each resample fit skips mgcv's REML/GCV smoothing-parameter selection. This typically gives a 3-4x speedup (composable with ncores) and the same model selection, but it is an approximation: individual bootstrap p-values can differ from the fully re-selected version, more so at small B. Default is TRUE; set fast_gam=FALSE to re-select the smoothing parameters on every resample (the exact bootstrap used in earlier versions, slower). Has no effect on cglm or on the chi-square test.
#' @param ... Further arguments to be passed to the gam function.
#' @return A list containing the selected causal submodel and search diagnostics.
#' @references
#' Polinelli, A., V. Vinciotti and E.C. Wit. (2026). "Causal generalized linear models via Pearson risk invariance." Journal of Causal Inference, 14(1), 20240043. \doi{10.1515/jci-2024-0043}
#' @importFrom stats BIC as.formula coef glm pchisq reformulate residuals terms.formula update.formula na.omit
#' @importFrom utils combn
#' @importFrom mgcv gam
#' @examples
#' ##############################
#' #causal Poisson gam##########
#' n<-1000
#' set.seed(123)
#' X1<-rnorm(n,0,1)
#' Y<-rpois(n,exp(sin(X1)))
#' X2<-log(Y+1)+rnorm(n,0,0.5)
#' data<-data.frame(X1, X2, Y)
#' cm_all<-cgam(Y ~ s(X1)+s(X2),"poisson",data,pval="chi-square",search="all")
#' cm_all$model.opt
#' cm_step<-cgam(Y ~ s(X1)+s(X2),"poisson",data,pval="chi-square",search="stepwise")
#' cm_step$model.opt
#' \donttest{
#' #bigger simulation with 7 covariates
#' set.seed(123)
#' n<-1000
#' X1<-rnorm(n=n,sd=sqrt(0.04))
#' X2<-X1+rnorm(n=n,sd=sqrt(0.04))
#' X3<-X1+X2+rnorm(n=n,sd=sqrt(0.04))
#' m<-sin(X2*5)+X3^3
#' Z<-m+rnorm(n=n,sd=sqrt(0.04))
#' X4<-X2+rnorm(n=n,sd=sqrt(0.04))
#' X5<-Z+rnorm(n=n,sd=sqrt(0.04))
#' X6<-Z+rnorm(n=n,sd=sqrt(0.04))
#' X7<-X6+rnorm(n=n,sd=sqrt(0.04))
#' Y<-qpois(pnorm(Z, mean = m, sd = sqrt(0.04)), lambda=exp(m))
#' dat<-data.frame(X1, X2, X3, X4, X5, X6, X7,Y)
#' fml<- Y~s(X1)+s(X2)+s(X3)+s(X4)+s(X5)+s(X6)+s(X7)
#' mod.all <-cgam(fml,"poisson",dat,pval="chi-square",search="all")
#' mod.all$model.opt
#' mod.step <-cgam(fml,"poisson",dat,pval="chi-square",search="stepwise")
#' mod.step$model.opt
#' ####################################
#' #causal logistic gam################
#' n<-1000
#' set.seed(123)
#' X1<-rnorm(n,0,1)
#' Y<-rbinom(n,1,exp(X1)/(1+exp(X1)))
#' flip<-rbinom(n,1,0.1)
#' X2<-(1-flip)*Y+rnorm(n,0,0.3)
#' data<-data.frame(X1, X2, Y)
#' set.seed(1)
#' cm_all<-cgam(Y ~ s(X1)+s(X2),"binomial",data,pval="bootstrap",search="all")
#' cm_all$model.opt
#' set.seed(1)
#' cm_step<-cgam(Y ~ s(X1)+s(X2),"binomial",data,pval="bootstrap",search="stepwise")
#' cm_step$model.opt
#' }
#' @export
cgam <- function(formula, family, data, alpha = 0.05,
                 pval = c("bootstrap", "chi-square"), B = 100,
                 search = c("all", "stepwise"), ncores = 1L,
                 fast_gam = TRUE, ...) {
  pval <- match.arg(pval)
  search <- match.arg(search)

  if (search == "all") {
    causal_all(formula, family = family, data = data, alpha = alpha,
               pval = pval, B = B, use_gam = TRUE, ncores = ncores,
               fast_gam = fast_gam, ...)
  } else {
    causal_step(formula, family = family, data = data, alpha = alpha,
                pval = pval, B = B, use_gam = TRUE, ncores = ncores,
                fast_gam = fast_gam, ...)
  }
}
