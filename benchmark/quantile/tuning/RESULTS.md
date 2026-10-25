# Quantile calibration and CV: short report

**Pivotal penalty calibration and check-loss cross-validation are implemented.**
Inference is a separate [issue #104](https://github.com/mlakolar/HighDimensionalModels.jl/issues/104).

```julia
using HighDimensionalModels, HiGHS, Random
lambda = quantile_lambda(X; tau=0.5, nsim=1000, rng=Xoshiro(17))
sol = quantile_lasso(X, y, lambda; tau=0.5)

cv = CVQuantileLassoPath(X, y; nfolds=5, nλ=40, rng=Xoshiro(18))
coef(cv; λ=:λmin)
dual_scores(cv; λ=:λmin)  # raw scores of the selected full-data fit
```

Calibration uses the fixed-quantile null score, the fit's effective loadings,
and explicit simulation/RNG controls. CV uses unsmoothed held-out check loss
for both exact and smoothed estimators. A penalty is selectable only when the
full-data fit and every fold converge. Full-path and selected-fit duals are
retained; no fold scores are averaged or substituted for the research duals.

Discovery passed **615 existing API assertions, 54 calibration assertions,
106 CV assertions and 24 independent R reference assertions**. The reference
checks common-uniform score quantiles, full coefficients and held-out fold
losses with offsets and training-fold standardization.
Final two-thread runs pass all 160 tuning assertions with the new backend and
136 with the older backend, whose tests check its documented eligibility
fallback. Warnings and the earlier failed compatibility assertions are retained.

The first R fixture exposed nonunique exact solutions: 54 fold/penalty pairs
agree on training objectives within 2.1e−12 and satisfy raw optimality checks,
yet some give different held-out predictions. That failed literal-equality
comparison is retained. A separate fixture avoiding integer empirical ranks
passes all reference checks. Older solvers can fail smooth-fold accuracy;
CV excludes the affected penalties rather than averaging fewer folds.

Repeated operation costs on three 1,000 × 40 inputs, two repetitions each:
**0.211s exact CV, 0.079s smoothed CV**, each with five folds and 12 penalties;
**0.0058s calibration** with 1,000 simulations. These are operation costs,
not a new competitor ranking. [Timing summary](results/discovery-2026-10-25/tuning-reference-v3/regular/results/timing-summary.md).

[Protocol and recreation](README.md), [run record](results/discovery-2026-10-25/run-record.json),
and [API documentation](../../../docs/src/quantile.md).
