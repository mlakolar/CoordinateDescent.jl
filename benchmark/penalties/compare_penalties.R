# glmnet's elastic net and ncvreg's MCP and SCAD on the data of
# compare_penalties.jl gen DIR, at the levels it wrote
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
library(glmnet)
library(ncvreg)
rd <- function(name) read.csv(file.path(dir, name), header = FALSE)[, 1]
wr <- function(fit, name) write.table(as.matrix(coef(fit)), file.path(dir, name), row.names = FALSE, col.names = FALSE, sep = ",")
k <- 1
while (file.exists(file.path(dir, paste0("X", k, ".csv")))) {
  X <- as.matrix(read.csv(file.path(dir, paste0("X", k, ".csv")), header = FALSE))
  y <- rd(paste0("y", k, ".csv"))
  for (a in c(0.2, 0.5, 0.8)) {
    lam <- rd(sprintf("lambda_enet%g_%d.csv", a, k))
    wr(glmnet(X, y, alpha = a, lambda = lam, standardize = TRUE, thresh = 1e-18, maxit = 1e7), sprintf("glmnet_enet%g_%d.csv", a, k))
  }
  for (pen in c("MCP", "SCAD")) {
    lam <- rd(sprintf("lambda_%s_%d.csv", pen, k))
    wr(ncvreg(X, y, penalty = pen, gamma = if (pen == "MCP") 3 else 3.7, lambda = lam, eps = 1e-12, max.iter = 1e7),
       sprintf("ncvreg_%s_%d.csv", pen, k))
  }
  k <- k + 1
}
cat("glmnet", as.character(packageVersion("glmnet")), "and ncvreg", as.character(packageVersion("ncvreg")), "done on", k - 1, "data sets\n")
