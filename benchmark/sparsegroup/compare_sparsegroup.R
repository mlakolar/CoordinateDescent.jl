# sparsegl's sparse group lasso on the data of compare_sparsegroup.jl gen DIR,
# at the levels it wrote, and SGL's with a second argument "sgl" (SGL is slow:
# on the n = 200, p = 60 design its path at thresh = 1e-10 took over 20 minutes)
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
library(sparsegl)
have_sgl <- length(args) > 1 && args[2] == "sgl" && requireNamespace("SGL", quietly = TRUE)
rd <- function(name) read.csv(file.path(dir, name), header = FALSE)[, 1]
k <- 1
while (file.exists(file.path(dir, paste0("X", k, ".csv")))) {
  X <- as.matrix(read.csv(file.path(dir, paste0("X", k, ".csv")), header = FALSE))
  y <- rd(paste0("y", k, ".csv"))
  grp <- as.integer(rd(paste0("group", k, ".csv")))
  for (a in c(0.05, 0.5, 0.95)) {
    lam <- rd(sprintf("lambda%g_%d.csv", a, k))
    fit <- sparsegl(X, y, group = grp, family = "gaussian", asparse = a, lambda = lam, standardize = TRUE,
                    intercept = TRUE, eps = 1e-14, maxit = 1e8)
    write.table(as.matrix(coef(fit)), file.path(dir, sprintf("sparsegl%g_%d.csv", a, k)),
                row.names = FALSE, col.names = FALSE, sep = ",")
    if (have_sgl) {
      sfit <- SGL::SGL(list(x = X, y = y), index = grp, type = "linear", alpha = a, lambdas = lam,
                       nlam = length(lam), standardize = TRUE, thresh = 1e-10, maxit = 100000)
      # SGL's coefficients and intercept, on the scale of the data
      write.table(rbind(sfit$intercepts, sfit$beta), file.path(dir, sprintf("SGL%g_%d.csv", a, k)),
                  row.names = FALSE, col.names = FALSE, sep = ",")
    }
  }
  k <- k + 1
}
cat("sparsegl", as.character(packageVersion("sparsegl")), if (have_sgl) paste("and SGL", packageVersion("SGL")) else "",
    "done on", k - 1, "data sets\n")
