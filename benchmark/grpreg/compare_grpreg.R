# grpreg's group lasso on the data of compare_grpreg.jl gen DIR
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
library(grpreg)
k <- 1
while (file.exists(file.path(dir, paste0("X", k, ".csv")))) {
  X <- as.matrix(read.csv(file.path(dir, paste0("X", k, ".csv")), header = FALSE))
  y <- read.csv(file.path(dir, paste0("y", k, ".csv")), header = FALSE)[, 1]
  group <- read.csv(file.path(dir, paste0("group", k, ".csv")), header = FALSE)[, 1]
  lambda <- read.csv(file.path(dir, paste0("lambda", k, ".csv")), header = FALSE)[, 1]
  fit <- grpreg(X, y, group, penalty = "grLasso", lambda = lambda, eps = 1e-10, max.iter = 1e6)
  write.table(as.matrix(coef(fit)), file.path(dir, paste0("grpreg", k, ".csv")), row.names = FALSE, col.names = FALSE, sep = ",")
  k <- k + 1
}
cat("grpreg", as.character(packageVersion("grpreg")), "done on", k - 1, "data sets\n")
