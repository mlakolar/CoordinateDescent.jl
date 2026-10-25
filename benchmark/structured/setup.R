# Run under Slurm in the R 4.5.3 container; install into this task's library.
options(repos=c(CRAN="https://cloud.r-project.org"), Ncpus=1)
if (!requireNamespace("genlasso", quietly=TRUE)) install.packages("genlasso")
stopifnot(as.character(packageVersion("genlasso")) == "1.6.1")
