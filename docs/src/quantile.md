# Primal–dual quantile regression

The exact fit keeps the observation-level dual variables needed by research
procedures. Install and load the optional HiGHS backend, then fit one penalty or
a decreasing grid:

```julia
using HighDimensionalModels, HiGHS
sol = quantile_lasso(X, y, 0.05; tau=0.5)
path = QuantileLassoPath(X, y, [0.2, 0.1, 0.05]; tau=0.5)
B = coef(path)                  # coefficient columns, intercept first
U = dual_scores(path)           # n observations × penalty levels
xi = -U / nobs(path)            # QR.jl's row-dual sign and normalization
s = path[2]
```

At a valid solution, `dual_scores(s)` lies in `[tau-1,tau]`. A stopped exact
fit can retain an infeasible iterate with `converged=false`; its raw scores are
kept for diagnostics. With an intercept, valid scores sum to zero, and `X'u/n`
satisfies the weighted lasso stationarity conditions.
At nonzero residuals the score is determined by the sign; at zero residuals
preserve the optimizer's value. Different optimal bases can return different
valid duals. Compare their KKT conditions and the quantities your inference uses,
not necessarily the vectors entry by entry in a degenerate problem.

QR.jl used the equality `y - a - Xβ == rp - rn` and divided its loss by n.
Convert with `-dual_scores(s)/nobs(s)`, without projection. QR.jl
stored whole-path `xi` with levels in rows; transpose the new matrix for that
layout. The compatibility test compares against an independent primal LP with
that equality orientation. The historical JuMP/Gurobi binary stack is not used.

Each fit also has `certificate_dual`, a distinct feasible vector for a numerical
lower bound. Do not substitute it for the raw dual in inference. `duality_gap`
uses the raw pair and requires checking `dual_feasibility`; `certificate_gap`
uses the feasible vector. With zero penalties or zero loadings the conservative
certificate is zero and can be loose. `converged` is a numerical KKT/gap check;
`native_converged` records the optimizer flag separately.

The exact backend reuses its basis along the grid. Penalty factors can be zero
for unpenalized slopes; explicit λ=0 fits are supported. Standardization changes
penalty loadings, not the returned coefficient scale. An offset is supported;
observation weights and formula syntax are deferred. A finite `time_limit`
returns a partial path if needed. Check `complete`, `status` and each fit's
`converged` before using it. Saved dual vectors are independent snapshots.

Both backends accept `max_hat_s=Inf`, a nonnegative real support limit.
After saving the first fit with `count(!iszero, β) > max_hat_s`, the path
skips later levels. The intercept does not count; nonzero unpenalized slopes
do. The exceeding fit is retained, as in `LassoPath` and `GLMLassoPath`,
whereas QR.jl dropped it. If any levels are skipped, `complete=false` and
`status=:max_hat_s`; `λpath` contains the solved levels and `requested_λpath`
keeps the full grid. Exceeding only at the last level leaves the path complete
with its usual status. A single-level `quantile_lasso` fit is never cut short.
For exact fits, native non-`OPTIMAL` termination or a missing primal–dual pair
takes precedence. The support check follows each optimal saved fit, before
the next level's time-budget check. Per-fit `converged` still reports accuracy.

A smoothed path uses the production proximal Newton implementation:

```julia
smooth = QuantileLassoPath(X, y, [0.2, 0.1, 0.05];
                          method=:smooth, bandwidth=0.2)
```

At each penalty, proximal Newton builds a weighted least-squares approximation
to the smooth loss and solves its lasso subproblem by coordinate descent.
Backtracking accepts a step that does not increase the penalized objective.
The path uses warm starts, sequential strong screening and subsequent checks
of the excluded coordinates. With a ProximalSolvers backend supporting convolution
certificates, `tol` also sets the gap target, together with checks of the raw
scores during solving. A smaller explicit `options.gaptol` is honored.
Flat-tail curvature damping and absolute inner stopping reduce cancellation
from large working responses; they change the quadratic step, not the objective.
`options=CDOptions(gaptol=0)` selects the legacy objective-change route.
Older solver versions retain their original stopping behavior and receive the
same independent post-fit checks. On that route, external `tol` marks acceptance
without automatically refining a rejected fit.

The backend conservatively retains inner convergence failures. A returned pair
can therefore have `converged=true` after passing the external numerical checks
while `native_converged=false` and inner warnings remain visible.

Its dual vectors are Epanechnikov convolution scores. They have the same scaling
interface but are not automatically interchangeable with exact quantile scores
in an inference procedure. The explicit bandwidth is not a statistical default
recommendation. Inference and full parametric breakpoint enumeration remain
separate work. Quantile paths do not linearly
interpolate coefficients or duals between saved levels.

## Refitting a selected support

Use `quantile_refit` to remove the penalty while keeping a selected set of
columns. The result is a `QuantileRefit` with coefficients on the full design
and a restricted λ=0 fit in `r.fit`:

```julia
r = quantile_refit(X, y, [1, 3]; tau=0.5)
r = quantile_refit(X, y, sol)    # exact nonzeros of sol.β
coef(r)                        # full coefficients, intercept first if fitted
predict(r, Xnew; offset=offset_new)
u = dual_scores(r)             # raw scores from the restricted fit
xi = -u / nobs(r)               # QR.jl's row-dual convention
r.fit.converged                 # numerical KKT and raw-gap checks
```

An explicit support is copied, sorted and checked for distinct integer indices
in `1:size(X,2)`; it may be empty. Convert Bool masks with `findall`.
The solution form uses exact nonzeros without a threshold and inherits its
method, bandwidth and intercept setting. Every other column is fixed at zero.
A zero-loading column whose fitted coefficient is zero is excluded too;
list it explicitly to keep it. Penalty factors, standardization and
`max_hat_s` have no role and are not accepted as refit keywords. Sparse designs
remain sparse. Empty supports work with and without an intercept.

`r.β` has one entry per original column and is zero off `r.support`.
`r.fit.β` has one entry per selected column. `residuals(r)` and `nobs(r)`
refer to this restricted fit. `dof(r)` counts `length(r.support)` plus one
when fitting an intercept, even if a selected coefficient refits to zero.
All diagnostics and stationarity checks use the restricted design. At an
accurate optimum, `X[:,r.support]'*u/nobs(r)` is zero; correlations with
excluded columns need not vanish. At λ=0 the separate feasible certificate
is zero, so its gap may be loose. Numerical convergence uses the raw pair.

Several quantiles can share an exact model:

```julia
rs = quantile_refit(X, y, [1, 3]; tau=[0.7, 0.3, 0.7], offset=offset)
refits = refit_quantile_path(path, X, y; offset=offset)
r = refits[findall(!iszero, path[2].β)]
```

A scalar `tau` returns one result; a nonempty vector returns a vector in the
supplied order, including repeats. Exact refits reuse one solver model and its
current basis across quantiles, changing only the observation-score bounds.
Smoothed refits use the same Epanechnikov loss, tolerances and solver controls
as smoothed paths, with one independent fit per quantile and no warm starts.
Their scores are convolution derivatives, not exact check-loss scores.

`refit_quantile_path` returns a `Dict{Vector{Int},QuantileRefit}` and refits
each distinct support of the saved fits once, including nonconsecutive repeats.
The method and intercept setting come from the path. Each support inherits the
quantile and bandwidth of its first saved fit; a scalar `tau` overrides the
quantile. An empty path returns an empty dictionary. A truncated path uses
only its saved fits. Models are not reused across supports.

The solution form checks dimensions and requires the recomputed residuals
`y - offset - sol.intercept - X*sol.β` to match `sol.residuals` within
`sqrt(eps(Float64))*(1 + maximum(abs, y - offset))` in maximum norm.
Each distinct support's source fit receives the same check for path refits.
This catches incompatible responses, offsets or scaling. It does not prove
that the supplied design is the original one: changes with `ΔX*sol.β == 0`
pass. Supply the original data and offset.

Every requested refit is attempted, even after a non-optimal native
termination that returns a primal–dual pair. Such a result keeps the native
`r.fit.status` and `native_converged=false`; `converged` independently reports
numerical accuracy. A missing pair raises an error. An exact `time_limit`
is one deadline for the whole call, across all quantiles or supports.
The remaining time is passed to each solve. If the deadline has passed before
a requested solve, the call raises an error. Preparation and diagnostics can
overrun the deadline. Finite limits are rejected for smoothed refits.

Let `d = length(r.support) + r.fit.hasintercept`. When the restricted design
(with an intercept column if fitted) has full column rank, an exact vertex
solution has at least `d` zero residuals. If exactly `d` residuals are zero
and the design of those rows is nonsingular, stationarity uniquely determines
their scores. Scores and coefficients can be nonunique in degenerate cases:
more zero residuals than `d`, tied responses, rank-deficient or
overparameterized supports (`d ≥ n`), or dual-degenerate optima.
The refit keeps the optimizer's values. Compare objectives and KKT conditions
in such cases; entrywise comparisons need a checked nondegenerate fixture.
These routines provide computation, not post-selection inference; inference
is tracked in [issue #104](https://github.com/mlakolar/HighDimensionalModels.jl/issues/104).

## Penalty calibration and cross-validation

For a fixed quantile, simulate the pivotal score penalty, then use the same
loading and intercept settings in the fit:

```julia
using Random
lambda = quantile_lambda(X; tau=0.5, nsim=1000, rng=Xoshiro(17))
sol = quantile_lasso(X, y, lambda; tau=0.5)
```

The simulated statistic is `max_j |Z[:,j]'*(tau - (U < tau))|/(n*w[j])`,
where the uniforms are independent, `Z` is centered when fitting an intercept,
and `w` is the fit's effective loading vector. The returned penalty is `c`
times its empirical `1-gamma` quantile, with defaults `c=2`, `gamma=0.1` and
R-compatible type-7 quantiles. This translates the fixed-quantile simulation
of [Belloni and Chernozhukov](https://arxiv.org/abs/0904.2931) to our mean-loss
normalization. It uses no responses or noise estimate. It does not calibrate
a simultaneous quantile process or smoothing bias. Zero effective slope
loadings require a separate calibration; use an explicit penalty in that case.

Alternatively, select a penalty using held-out check loss:

```julia
cv = CVQuantileLassoPath(X, y; nfolds=5, nλ=40, rng=Xoshiro(18))
coef(cv)                       # one-standard-error selection, λ1se
coef(cv; λ=:λmin)               # minimum held-out check loss
predict(cv, Xnew; λ=:λmin)
u = dual_scores(cv; λ=:λmin)    # selected full-data fit's raw scores
U = dual_scores(cv.path)        # full-data path, observations × penalties
```

`folds` accepts explicit labels for grouped or blocked folds; an explicit
penalty grid can be supplied as the third positional argument. Centering and
standardization use each training fold. Automatic grids use the full-data
path as their shared grid; supply your own grid to avoid that dependence.
Both exact and smoothed paths are evaluated by the **unsmoothed check loss**.
An explicit `bandwidth` applies to every smoothed fold.

`max_hat_s` applies to the full-data path and every training-fold path.
The full-data stop truncates the shared grid. Folds that stop sooner leave
`NaN` errors at later levels, which are ineligible for selection and have
`cvm=Inf` and `cvse=NaN`, including levels that no fold reaches.

The result exposes `cvm`, `cvse`, `λmin`, `λ1se`, `folds`, `fold_errors`, and
the full-data `path`. Fold errors and their standard errors use fold-size
weighting as in `CVLassoPath`. Missing/inaccurate fold fits have `NaN` errors;
a penalty is eligible only when every fold and its full-data fit converge.
Ineligible levels have `cvm=Inf`; an entirely ineligible grid raises an error.
No folds are silently omitted to make a penalty look better. Prediction offsets
must be provided for new observations. Numerical tolerances can require tuning
for a difficult fold, and `time_limit` applies to each exact path separately.
An exact quantile fit can be nonunique, including intercept-only fits at an
integer empirical quantile rank. Different valid optimizers can then give
different held-out predictions despite agreeing on training objectives and KKT
conditions. CV retains the chosen backend's saved fits; it does not average
alternative optima or alter their raw duals.
With older ProximalSolvers versions, objective-based smoothed stopping may fail
the requested accuracy on some folds. CV excludes the affected levels. The
convolution gap/raw-score stopping backend is needed for the stricter behavior
demonstrated in the [solver follow-up benchmarks (snapshot)](https://github.com/mlakolar/HighDimensionalModels.jl/tree/abddaf56a790b86178de376b175a1185020c8381/benchmark/quantile/convergence).

Quantile inference is tracked separately in
[issue #104](https://github.com/mlakolar/HighDimensionalModels.jl/issues/104).

The `quantile` group of the [benchmark harness](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/benchmark/README.md) measures the
current fits; the larger benchmark campaign is tracked in
[issue #105](https://github.com/mlakolar/HighDimensionalModels.jl/issues/105).
Workspace development resolves ProximalSolvers master, including the merged
convolution-certification changes from PR #96. CI verifies this capability
explicitly. Migration version numbers remain unchanged; the feature-detected
older-release behavior described above is retained.
