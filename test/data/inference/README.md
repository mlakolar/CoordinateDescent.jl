# Inference golden fixtures

Generated on Discovery in run `20261025-cd77-inference`, with R 4.5.3, hdi
0.1-10 from CRAN's Git mirror commit
`3eb71fb52b3c0f5213b1283d5f0317b91f6a133a`, glmnet 5.1, and the authors'
Javanmard--Montanari R implementation dated April 24, 2014. `R-session.txt`
records the loaded versions. Recreate with `benchmark/inference/references.R`
and its README recipe; no numerical values are hand-edited.

The Gaussian fixture fixes X, y, the initial slopes, sigma=1, initial lambda=.08
and nodewise lambda=.1. Nodewise fits use exact-lambda glmnet with no scaling;
columns already have population standard deviation one. `Z-raw.csv` holds raw
nodewise residuals. hdi normalizes these by `colMeans(Z*X)` internally.

The bootstrap uses the stored 80-by-199 resampled error matrix, fixed initial
fits and exact-lambda glmnet refits. Its complete hdi bootstrap path is exercised
with helpers replaced solely to control fitting/tuning and sampled errors.
The exported bootstrap distributions are hdi's centered distributions multiplied
by the observed standard errors; tests convert them to studentized draws.
`hdi-bootstrap.csv` includes hdi's equal-tail and max-T p-values. The package
exposes both its absolute-tail marginal p-values and `equal_tail_pvalues`.
This is matched-component agreement, not automatic-CV agreement between defaults.

`jm-theta.csv` uses the unchanged authors' QP row solver at mu=.12 with a tight
stopping threshold. Logistic/Poisson references use R's `glm` for the unpenalized
MLE, its model standard errors, and the restricted null score at beta[2]=.25.
