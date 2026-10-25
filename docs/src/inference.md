# Debiased lasso and decorrelated scores

`debiased_lasso` corrects the bias of a penalized initial estimator. For a
Gaussian linear model, the estimate of the slope vector is
`beta + theta * Xc' * (y - intercept - X * beta) / n`, where `Xc` is centered
when an intercept is fitted. Inference concerns the original regression slopes,
including those set to zero by the lasso. It does not include the intercept.

Automatic tuning is experimental. A fresh Discovery holdout study with 100
datasets per design found active-coefficient coverage of 77.6% and 73.0% in
Gaussian designs, and 56.2% and 85.0% in logistic designs, for nominal 95%
intervals. All fits converged. This includes a moderate-correlation sparse
design, so the observed problem is broader than the severe stress case.
See `benchmark/inference/automatic-tuning/results/discovery-20261025/RESULTS.md`
for settings, uncertainty, diagnostics and matched computational comparisons.

A subsequent frozen restricted-LDPE candidate was validated on 200 new datasets
per design. Active coverage ranged from 94.2% to 95.2% in the four Gaussian
cases, and from 90.5% to 96.4% in the four logistic cases. It reduced remaining
bias, with longer intervals and lower power. Logistic undercoverage persisted
in the benign, moderate and random-support cases. Many selected penalties were
finite-grid endpoints. This candidate remains an explicit expert option;
it has not replaced the experimental CV defaults. See
`benchmark/inference/bias-tuning/results/discovery-20261025/RESULTS.md` and
`CURRENT_COMPARISON.md` there for uncertainty, grid diagnostics, the refreshed
paired competitor pilot, and conditional resource estimates.

The universal QP constraint is also an expert starting point, not a calibrated
inference default. In the refreshed paired 20-dataset pilot, its Gaussian
active coverage was 36% in the moderate design and 8% in the correlated design.
The programs passed the numerical feasibility/gap checks. That does not
control the remaining statistical bias. Keep this distinction when choosing
between the projection methods.

```julia
using HighDimensionalModels, Random
rng = Xoshiro(77)
X = randn(rng, 200, 30)
y = X[:, 1] + randn(rng, 200)
fit = debiased_lasso(X, y; rng)   # automatic initial and projection tuning
fit.converged || error("tighten the solver settings before inference")
coef(fit), stderror(fit), confint(fit)
joint = multiplier_bootstrap(fit; B = 1999, rng)
joint.adjusted_pvalues           # single-step Westfall--Young max-T
joint.intervals                  # simultaneous intervals
```

The omitted initial penalty is selected by five-fold cross-validation at the
minimum held-out MSE (Gaussian) or deviance (GLM), using a 40-level lasso path.
Gaussian noise is estimated from the residual sum of squares divided by
`n - nnz(beta) - intercept`. Supply `sigma` when an external noise estimate is
available. `initial_cv_folds` and `initial_cv_nlambda` control the search. Inference CV
uses `gaptol=0` unless explicitly overridden in `options`, so the selected
path fits use `options.opt_tol` rather than the generic prediction-path gap
target.

The default nodewise penalty is selected by pooled five-fold prediction CV over
20 randomly sampled predictor columns, or all columns when there are fewer.
Selection is independent of the requested targets and uses a common decreasing
20-level geometric grid from the largest sampled absolute correlation to 5%
of that value. Each training fold estimates its own centering and scaling.
`nodewise_cv_folds`, `nodewise_cv_nodes`, and `nodewise_cv_nlambda` control this
bounded search. The selected penalties are recorded in `fit.lambda` and
`fit.projection.lambda`; pass an explicit `rng` for reproducibility.

Numerical initial and nodewise penalties bypass the respective CV searches.
For Gaussian models, `initial_tuning=:scaled` with an omitted initial penalty
uses the original scaled lasso and its noise estimate; `nodewise_lambda=:universal`
uses the original universal nodewise level `sqrt(2log(max(p,2))/n)`.
QP projections retain their universal default constraint level; pooled CV
applies only to nodewise projections. These automatic prediction rules do not
guarantee confidence coverage, especially under strong correlation or dense
signals. Expert tuning and diagnostic output remain necessary in difficult cases.

A supplied initial penalty is on the objective
`sum(abs2, residuals)/(2n) + lambda*norm(beta,1)` on standardized columns,
unless `standardize_x=false`.

## Projection and reuse

```julia
P = nodewise_lasso(X, 0.12; lambda_path = [0.3, 0.2, 0.12], gram = :auto, rng)
fit = debiased_lasso(X, y, 0.1; projection = P, rng)
Q = debiasing_matrix(X; method = :qp, mu = 0.15, rng)
fit_qp = debiased_lasso(X, y, 0.1; projection = Q, rng)
```

The experimental bias-aware rule is available explicitly:

```julia
P = nodewise_lasso(X, :ldpe; ldpe_kappa=0.25, rng)
P.lambda                 # penalty chosen separately for each target
P.bias_factor            # working-design bias factors
P.noise_factor           # SD per unit noise on the original scale
P.reference_noise        # reference SD from stage1
P.bias_target_met        # feasibility of the original bias target
P.path_boundary          # whether a finite-grid endpoint was selected
fit = debiased_lasso(X, y; projection=P, rng)
```

This implements Zhang and Zhang's Table1 on a finite decreasing lasso grid.
It chooses the largest penalty meeting the bias-factor bound, then the smallest
penalty whose noise SD is within 25% of that reference. The default bound is
`sqrt(2log(max(p,2)))`; an infeasible original bound is recorded even if the
published adjustment produces a usable finite-grid score. A 25% SD increase
permits a 56.25% variance increase. Expert controls are `ldpe_kappa`,
`ldpe_bias_target`, `ldpe_relax`, `ldpe_nlambda`, `ldpe_minratio`, and an explicit
`lambda_path` grid. Noise factors are on the original coefficient scale;
bias factors use the working columns. `ldpe_unpenalized=m` removes the penalty
from the m predictors most correlated with each target on the working design.
This is Zhang and Zhang's restricted LDPE. It forces the score to be orthogonal
to those columns up to solver accuracy. The path starts at the largest conditional
correlation after regressing the target on those columns. The default m=0 retains
ordinary LDPE; m=4 is a development candidate following their simulation choice.
This rule is under development, and its GLM Hessian application needs separate
empirical validation. No choice here establishes finite-sample coverage.

For nodewise lasso, a row is the vector `(e_j - gamma_j) / tau_j²`, where
`tau_j² = X_j' * (X_j - X_-j*gamma_j)/n`. The normalization includes the
penalty contribution; using only the residual mean square is incorrect.
Both the inverse and the score vectors are returned on the original scale.
`targets=[1,3,7]` computes only those rows. The nodewise regressions share one
Gram matrix, and each worker has its own state and random generator.
`lambda_path` warm starts the solve at its final penalty. With `warm_start=false`
in the solver options, that reuse is disabled.

`gram=:auto` shares the Gram matrix when its dense storage fits the specified
`max_gram_bytes` (default 1 GiB). This heuristic applies to nodewise regressions,
which reuse the matrix across many fits; it differs from the single-response
lasso heuristic. `gram=false` avoids its quadratic storage. Sparse input is
converted to a dense working design; scores use `n * length(targets)` entries.
The inverse itself is sparse. Neither the Gram matrix nor the design is copied
per predictor. Keep BLAS threads at one when parallelizing the nodewise fits.

The QP minimizes `m'G*m/2` subject to `norm(G*m-e_j,Inf) <= mu` on the working
Gram matrix `G`. Coordinate descent solves its dual
`u'G*u/2-u_j+mu*norm(u,1)`; at an optimum, `m=u` is primal feasible. The solver
checks feasibility and primal-dual gap. It does not widen an infeasible `mu`.
Rows remain asymmetric and do not have unit diagonal normalization.

A saved projection may be reused for another response on exactly the same
design, centering and scaling. The caller is responsible for this match.
For GLMs the projection depends on the fitted Hessian, so it usually cannot
be reused across responses.

## Joint inference and resampling

The multiplier bootstrap draws one common Gaussian multiplier per observation
and projects it through all influence vectors. `robust=true` in the original
fit uses centered empirical score influences; the default uses homoskedastic
model influences. `multipliers` can supply an explicit n-by-B matrix. Returned
`draws` are studentized, with targets in rows and bootstrap samples in columns.
P-values compare absolute statistics and include an add-one correction.
`stepdown=true` computes stepdown adjusted p-values. Intervals always use the
single-step maximum, so they retain simultaneous interpretation. If B is too
small to resolve the requested confidence level, the interval is unbounded.

`bootstrap_lasso(X,y,fit; B=999, wild=false, rng)` implements the separate
residual bootstrap with lasso refitting at the fixed initial penalty. It reuses
the nodewise scores, re-estimates the residual noise level in each sample,
and refits under the complete null for the max-T adjustment. It corresponds
to `hdi::boot.lasso.proj` with `boot.shortcut=true` after matching the initial
fit, nodewise tuning, scaling and sampled errors. The default uses resampled
centered residuals; `wild=true` uses normal wild errors. `resamples` supplies
the already sampled n-by-B error matrix. The bootstrap is Gaussian-only.
Its `pvalues` use absolute tails; `equal_tail_pvalues` exposes hdi's convention
for direct comparison. These differ in finite Monte Carlo samples. The centered
distributions, equal-tail p-values and max-T adjustment are compared on matched
golden inputs. This method does not repeat CV within resamples.

## GLMs and the decorrelated score

```julia
binary_y = Float64.(rand(rng, size(X,1)) .< 0.5)
glmfit = debiased_lasso(X, binary_y; family = :binomial, rng)
test = decorrelated_score(X, binary_y, 1, 0.05; family = :binomial, null = 0, rng)
test.statistic, test.pvalue, test.converged
```

Logistic and Poisson debiasing uses the Hessian-weighted design, centered
at its weighted mean to project out an unpenalized intercept. Its score
correction uses the original covariates and `y - fitted_mean`; model variance
uses the Hessian, while the robust option uses empirical score products.
An omitted initial penalty uses automatic deviance CV. Binomial and Poisson dispersion is one.

The score test fits nuisance coefficients under the supplied null, then uses
Hessian-weighted penalized least squares to project the target on the nuisance
columns, following the penalized projection alternative of Ning and Liu.
Its information is the estimated Schur complement. A nonpositive value or a
degenerate Hessian is an error. `projection_lambda` is normalized by n, so its
meaning remains constant when Hessian weights have non-unit mean.

These are asymptotic procedures. Sparse coefficients, sufficient sample size,
nondegenerate information, and control of projection errors are substantive
requirements. Nodewise inference also needs suitably sparse precision rows;
the QP approach relaxes that requirement under its own assumptions. A solver
convergence flag establishes numerical convergence, not confidence coverage.
Strong correlation, dense signals, separation and poor tuning can invalidate
the normal approximation. The Discovery study records those failures without
changing the inferential claims.

## Sources and validation

The methods follow [Zhang and Zhang (2014)](https://arxiv.org/abs/1110.2563),
[van de Geer et al. (2014)](https://arxiv.org/abs/1303.0518),
[Javanmard and Montanari (2014)](https://jmlr.org/papers/v15/javanmard14a.html),
[Zhang and Cheng (2017)](https://arxiv.org/abs/1603.01295),
[Dezeure, Bühlmann and Zhang (2017)](https://arxiv.org/abs/1606.03940), and
[Ning and Liu (2017)](https://arxiv.org/abs/1412.8765).
Reference generation, replicated Discovery measurements, and coverage settings
are described in `benchmark/inference/README.md` in the source distribution.
`RESULTS.md` and `GLM_RESULTS.md` in that directory retain the observed
undercoverage and conservative score results; the default tuning is not an
empirically validated guarantee of nominal coverage across these designs.
