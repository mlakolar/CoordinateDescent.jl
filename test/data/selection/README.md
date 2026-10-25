# R golden data for selection procedures

Generated on Discovery by `benchmark/selection/golden.R` with R 4.5.3,
SLOPE 2.1.1, knockoff 0.3.6, stabs 0.7.1 and glmnet (see R-session.txt).
Inputs use seed 8203; Gaussian innovations use seed 8204. Fixed pairs are
saved in folds.csv. Julia reads these files in test/selection_golden.jl;
R is not needed to run the tests.

SLOPE uses gamma=1 (regularized fits), shared penalty multipliers and matched
preprocessing. Without an intercept, RMS scaling is explicit because R's SD
scaling uses centered variances whereas this package uses uncentered mean
squares. Gaussian knockoffs share both s and the normal innovations. Fixed-X
comparison uses Gram identities because orthogonal bases differ. R's adaptive
PSD safety adjustment can change the equicorrelated diagonal by about 1e-7;
the diagonal comparison therefore uses a 2e-7 maximum absolute tolerance.
Stability selection uses the shared fixed-lambda capped lasso selector.
