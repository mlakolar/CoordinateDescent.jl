# The R side of benchmark/lla/compare_lla.jl: glmtlp's TLP and ncpen's MCP, SCAD
# and TLP paths, with an intercept, without and with standardization.
#
#     Rscript benchmark/lla/compare_lla.R DIR
suppressMessages({library(glmtlp); library(ncpen)})
dir <- commandArgs(trailingOnly = TRUE)[1]
TAU <- 0.25
rd <- function(f) as.matrix(read.csv(file.path(dir, f), header = FALSE))
wr <- function(A, f) write.table(A, file.path(dir, f), sep = ",", row.names = FALSE, col.names = FALSE)
log <- character(0)
for (d in c("g1", "g2")) {
  X <- rd(paste0(d, "_X.csv")); y <- as.numeric(rd(paste0(d, "_y.csv"))); p <- ncol(X)
  for (std in c(0, 1)) {
    pre <- paste0(d, "_std", std)
    lams <- as.numeric(rd(paste0(pre, "_levels.csv")))
    t0 <- proc.time()[3]
    fit <- glmtlp(X, y, family = "gaussian", penalty = "tlp", lambda = lams, tau = TAU, delta = 1,
                  tol = 1e-13, standardize = (std == 1), dc.maxit = 500, cd.maxit = 1e8)
    wr(rbind(fit$intercept, as.matrix(fit$beta)), paste0(pre, "_glmtlp.csv"))
    log <- c(log, sprintf("%s glmtlp %.2fs", pre, proc.time()[3] - t0))
    for (pen in c("mcp", "scad", "tlp")) {
      tau <- switch(pen, mcp = 3, scad = 3.7, tlp = TAU)
      t0 <- proc.time()[3]
      fit <- try(ncpen(y.vec = y, x.mat = X, family = "gaussian", penalty = pen, x.standardize = (std == 1),
                       intercept = TRUE, lambda = lams, tau = tau, df.max = p, cf.max = 1e10,
                       niter.max = 1000, qiter.max = 100, aiter.max = 1e5,
                       b.eps = 1e-13, k.eps = 1e-10, c.eps = 1e-10), silent = TRUE)
      if (inherits(fit, "try-error")) {
        log <- c(log, sprintf("%s ncpen %s failed: %s", pre, pen, as.character(fit)))
      } else {
        B <- as.matrix(fit$beta)          # with the intercept in the first row
        wr(B, paste0(pre, "_ncpen_", pen, ".csv"))
        log <- c(log, sprintf("%s ncpen %s %.2fs, %d x %d, last levels match %s", pre, pen, proc.time()[3] - t0,
                              nrow(B), ncol(B), isTRUE(all.equal(tail(as.numeric(fit$lambda), length(lams)), lams))))
      }
    }
  }
}
writeLines(c(paste("glmtlp", packageVersion("glmtlp"), "ncpen", packageVersion("ncpen")), log), file.path(dir, "r_log.txt"))
