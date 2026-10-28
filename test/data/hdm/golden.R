# Pinned, unmodified hdm 0.3.2 and sandwich 3.1.1 references; see README.md.
# Numeric fixtures use a long CSV schema: fit,field,index,value. Empty index
# vectors are represented by a single index=0,value=0 row (length is zero).
# hdm lambda0 is retained on its native 2*n*Julia_lambda0 scale. hdm iter is
# retained verbatim (nonempty fits increment it once after the final fit).
library(hdm)
library(sandwich)
stopifnot(as.character(packageVersion("hdm")) == "0.3.2",
          as.character(packageVersion("sandwich")) == "3.1.1")
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1]] else "test/data/hdm"
options(OutDec = ".")
set.seed(11800) # Defaults below are X-independent and consume no randomness.

for (dataset in c("hetero", "highdim", "forced", "nearthreshold", "empty")) {
  dir <- file.path(root, dataset)
  read <- function(name) as.matrix(read.csv(file.path(dir, paste0(name, ".csv")),
                                           header = FALSE))
  X <- read("X"); y <- drop(read("y")); d <- drop(read("d"))
  I3 <- rep(FALSE, ncol(X))
  if (dataset == "forced") I3[drop(read("forced"))] <- TRUE
  rows <- character()
  put <- function(fit, field, x) {
    x <- as.numeric(x)
    idx <- seq_along(x)
    if (!length(x)) { idx <- 0L; x <- 0 }
    rows <<- c(rows, paste(fit, field, idx, sprintf("%.17g", x), sep = ","))
  }
  fits <- list()
  for (post in c(TRUE, FALSE)) for (response in c("y", "d")) {
    name <- paste0("rlasso_", response, "_", tolower(post))
    r <- rlasso(X, if (response == "y") y else d, post = post)
    fits[[name]] <- r
    put(name, "index", which(r$index))
    for (field in c("coefficients", "intercept", "residuals", "loadings", "lambda0", "iter"))
      put(name, field, r[[field]])
  }
  if (dataset == "forced")
    for (r in fits) stopifnot(!any(r$index[I3]))
  if (dataset == "empty")
    for (r in fits) stopifnot(!any(r$index))

  for (method in c("doubleselection", "partialling")) {
    name <- paste0("effect_", method)
    e <- if (method == "doubleselection")
      rlassoEffect(X, y, d, method = "double selection", I3 = I3, post = TRUE) else
      rlassoEffect(X, y, d, method = "partialling out", post = TRUE)
    for (field in c("alpha", "se", "t", "pval")) put(name, field, e[[field]])
    put(name, "index", which(as.logical(e$selection.index)))
    put(name, "epsilon", e$residuals$epsilon)
    put(name, "v", e$residuals$v)
    final <- paste0("final_", method)
    if (method == "partialling") {
      # These are hdm's own nuisance residuals, with its default fits.
      yr <- fits$rlasso_y_true$residuals; dr <- fits$rlasso_d_true$residuals
      reg <- lm(yr ~ dr)
      put(final, "yr", yr); put(final, "dr", dr)
      stopifnot(max(abs(residuals(reg) - e$residuals$epsilon)) < 1e-12,
                max(abs(dr - e$residuals$v)) < 1e-12)
    } else {
      I <- which(as.logical(e$selection.index))
      reg <- if (length(I)) lm(y ~ d + X[, I, drop = FALSE]) else lm(y ~ d)
    }
    put(final, "coefficient", coef(reg)[2])
    put(final, "vcov", vcov(reg)[2, 2])
    for (type in c("HC0", "HC1", "HC2", "HC3"))
      put(final, type, sandwich::vcovHC(reg, type = type)[2, 2])
    put(final, "hatvalues", hatvalues(reg))
    stopifnot(abs(coef(reg)[2] - e$alpha) < 1e-12)
  }
  writeLines(c("fit,field,index,value", rows), file.path(dir, "reference-r.csv"))
  metadata <- c(R = R.version.string, hdm = as.character(packageVersion("hdm")),
                sandwich = as.character(packageVersion("sandwich")),
                hdm_archive_sha256 = "3354b22752f0aa0c0d9c8c12b38810202b6380f2587e539be11d9a358412fe6a",
                data_script = "test/data/hdm/golden_data.jl",
                reference_script = "test/data/hdm/golden.R",
                baseline_script = "test/data/hdm/baseline.jl")
  writeLines(c("key,value", paste(names(metadata), metadata, sep = ",")),
             file.path(dir, "metadata.csv"))
  cat(dataset, "references written\n")
}
