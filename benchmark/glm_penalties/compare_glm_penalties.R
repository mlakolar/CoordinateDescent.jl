# glmnet's elastic net and ncvreg's MCP and SCAD for the binomial and Poisson
# families, on the data of compare_glm_penalties.jl gen DIR, at the levels it wrote
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
library(glmnet)
library(ncvreg)
rd <- function(name) read.csv(file.path(dir, name), header = FALSE)[, 1]
wr <- function(fit, name) write.table(as.matrix(coef(fit)), file.path(dir, name), row.names = FALSE, col.names = FALSE, sep = ",")
for (f in list.files(dir, pattern = "^X_.*\\.csv$")) {
  key <- sub("^X_(.*)\\.csv$", "\\1", f)               # family_k
  family <- sub("_.*$", "", key)
  X <- as.matrix(read.csv(file.path(dir, f), header = FALSE))
  y <- rd(paste0("y_", key, ".csv"))
  for (lf in list.files(dir, pattern = paste0("^lambda_", key, "_.*\\.csv$"))) {
    tag <- sub(paste0("^lambda_", key, "_(.*)\\.csv$"), "\\1", lf)   # enet0.5, MCP, SCAD
    lam <- rd(lf)
    if (startsWith(tag, "enet")) {
      a <- as.numeric(sub("enet", "", tag))
      fit <- glmnet(X, y, family = family, alpha = a, lambda = lam, standardize = TRUE, thresh = 1e-14, maxit = 1e7)
    } else {
      fit <- ncvreg(X, y, family = family, penalty = tag, gamma = if (tag == "MCP") 3 else 3.7, lambda = lam,
                    eps = 1e-12, max.iter = 1e7)
    }
    wr(fit, paste0("R_", key, "_", tag, ".csv"))
  }
}
cat("glmnet", as.character(packageVersion("glmnet")), "and ncvreg", as.character(packageVersion("ncvreg")), "done\n")
