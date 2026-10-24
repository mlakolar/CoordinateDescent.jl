# hdm's rlasso and rlassoEffect on the data of compare_hdm.jl gen DIR
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
library(hdm)
k <- 1
while (file.exists(file.path(dir, paste0("X", k, ".csv")))) {
  X <- as.matrix(read.csv(file.path(dir, paste0("X", k, ".csv")), header = FALSE))
  y <- read.csv(file.path(dir, paste0("y", k, ".csv")), header = FALSE)[, 1]
  d <- read.csv(file.path(dir, paste0("d", k, ".csv")), header = FALSE)[, 1]
  r <- rlasso(X, y)
  write.table(matrix(r$coefficients, ncol = 1), file.path(dir, paste0("rlasso", k, ".csv")), row.names = FALSE, col.names = FALSE, sep = ",")
  r0 <- rlasso(X, y, post = FALSE)
  write.table(matrix(r0$coefficients, ncol = 1), file.path(dir, paste0("rlasso_nopost", k, ".csv")), row.names = FALSE, col.names = FALSE, sep = ",")
  e1 <- rlassoEffect(X, y, d, method = "double selection")
  write.table(matrix(c(e1$alpha, e1$se), ncol = 1), file.path(dir, paste0("effect_doubleselection", k, ".csv")), row.names = FALSE, col.names = FALSE, sep = ",")
  e2 <- rlassoEffect(X, y, d, method = "partialling out")
  write.table(matrix(c(e2$alpha, e2$se), ncol = 1), file.path(dir, paste0("effect_partialling", k, ".csv")), row.names = FALSE, col.names = FALSE, sep = ",")
  k <- k + 1
}
cat("hdm", as.character(packageVersion("hdm")), "done on", k - 1, "data sets\n")
