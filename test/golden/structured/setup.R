# Installs genlasso 1.6.1 from CRAN and glmgen 0.0.3 from GitHub at a fixed
# commit into the first library of .libPaths() (set R_LIBS); see README.md.
options(repos=c(CRAN="https://cloud.r-project.org"), Ncpus=1)
if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes")
genlasso_ok <- function() isTRUE(tryCatch(packageVersion("genlasso") == "1.6.1", error=function(e) FALSE))
if (!genlasso_ok()) remotes::install_version("genlasso", "1.6.1", upgrade="never")
stopifnot(genlasso_ok())
glmgen <- "9e6061503d6dd103712a0c07f6be7293c1c58aaf"
if (!requireNamespace("glmgen", quietly=TRUE))
  try(remotes::install_github("glmgen/glmgen", ref=glmgen, subdir="R_pkg/glmgen", upgrade="never"))
if (!requireNamespace("glmgen", quietly=TRUE))
  stop("could not install glmgen from GitHub at commit ", glmgen)
sha <- packageDescription("glmgen")$RemoteSha
if (!is.null(sha) && sha != glmgen) stop("glmgen is installed from commit ", sha, ", not ", glmgen)
stopifnot(as.character(packageVersion("glmgen")) == "0.0.3")
