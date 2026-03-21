#' Causal generalized linear model
#'
#' This function does a search for a causal submodel within the generalized linear model provided.
#'
#' @param formula A formula object.
#' @param family A distributional family object. Currently supported options are: binomial and poisson.
#' @param data A data frame containing the variables in the model.
#' @param alpha Significance level for statistical test
#' @param pval If pval="bootstrap", a bootstrap test is conducted to test whether Pearson risk is 1. When family="poisson" a chi-squared test can be conducted by setting pval="chi-square".
#' @param B Number of bootstrap samples when pval="bootstrap". Default is 100.
#' @param search If search="stepwise", a greedy forward stepwise search is conducted. Default is search="all", in which case all possible submodels are considered.
#' @param ncores Number of cores for parallel computation. Default is 1 (sequential). When ncores > 1, model evaluations are distributed across cores using \code{parallel::mclapply} (Unix/macOS). On Windows, parallelization is not supported and this parameter is ignored.
#' @param ... Further arguments to be passed to the glm function.
#' @return A list containing the selected causal submodel and search diagnostics.
#' @references
#' Polinelli, A., V. Vinciotti and E.C. Wit. (2024). "Causal generalized linear models via Pearson risk invariance" *arXiv preprint*.
#' @importFrom stats BIC as.formula coef glm pchisq reformulate residuals terms.formula update.formula na.omit
#' @importFrom utils combn
#' @examples
#' ###################################
#' #causal Poisson glm#################
#' n<-1000
#' set.seed(123)
#' X1<-rnorm(n,0,1)
#' Y<-rpois(n,exp(X1))
#' X2<-log(Y+1)+rnorm(n,0,0.3)
#' data<-data.frame(X1, X2, Y)
#' cm_all<-cglm(Y ~ X1+X2,"poisson",data,pval="chi-square",search="all")
#' cm_all$model.opt
#' cm_step<-cglm(Y ~ X1+X2,"poisson",data,pval="chi-square",search="stepwise")
#' cm_step$model.opt
#' \donttest{
#' ##########################
#' #causal logistic glm#######
#' n<-2000
#' set.seed(123)
#' X1<-rnorm(n,0,1)
#' Y<-rbinom(n,1,exp(X1)/(1+exp(X1)))
#' flip<-rbinom(n,1,0.1)
#' X2<-(1-flip)*Y+rnorm(n,0,0.3)
#' data<-data.frame(X1, X2, Y)
#' set.seed(1)
#' cm_all<-cglm(Y ~ X1+X2,"binomial",data,pval="bootstrap",search="all")
#' cm_all$model.opt
#' set.seed(1)
#' cm_step<-cglm(Y ~ X1+X2,"binomial",data,pval="bootstrap",search="stepwise")
#' cm_step$model.opt
#' #bigger simulation with 5 covariates
#' set.seed(12)
#' n<-3000
#' X1<-rnorm(n,0,1)
#' X2<-rnorm(n,X1,0.5)
#' X3<-rnorm(n,0,1)
#' X4<-rnorm(n,X2,.5)
#' Y<-rbinom(n,1,exp(.8*X2-.9*X3)/(1+exp(.8*X2-.9*X3)))
#' flip<-rbinom(n,1,0.1)
#' X5<-(1-flip)*Y+flip*(1-Y)+rnorm(n,0,.3)
#' dat<-data.frame(X1, X2, X3, X4, X5,Y)
#' set.seed(1)
#' mod.all <-cglm(Y~X1+X2+X3+X4+X5,"binomial",dat,pval="bootstrap",search="all")
#' mod.all$model.opt
#' set.seed(1)
#' mod.step <-cglm(Y~X1+X2+X3+X4+X5,"binomial",dat,pval="bootstrap",search="stepwise")
#' mod.step$model.opt
#' }
#' @param use_cpp Logical; if TRUE (default), use fast C++ implementations for supported families.
#' @export
cglm <- function(formula, family, data, alpha = 0.05,
                 pval = c("bootstrap", "chi-square"), B = 100,
                 search = c("all", "stepwise"), ncores = 1L,
                 use_cpp = TRUE, ...) {
  pval <- match.arg(pval)
  search <- match.arg(search)

  if (search == "all") {
    causal_all(formula, family = family, data = data, alpha = alpha,
               pval = pval, B = B, use_gam = FALSE, ncores = ncores,
               use_cpp = use_cpp, ...)
  } else {
    causal_step(formula, family = family, data = data, alpha = alpha,
                pval = pval, B = B, use_gam = FALSE, ncores = ncores,
                use_cpp = use_cpp, ...)
  }
}
