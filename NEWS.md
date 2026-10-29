# HighDimensionalModels 0.1.0 (unreleased)

The rename from CoordinateDescent creates a new package identity and starts
versioning at 0.1.0. Existing CoordinateDescent releases keep their UUID.
Runtime dependencies resolve from General, with ProximalBase 0.5.4 and
ProximalSolvers 0.3 as the minimum supported releases. Unit tests use only
registered dependencies; examples and docs pin the unregistered KernelSmoothing
integration dependency to an explicit commit.

- Add `dirty_multitask`, `DirtyMultitaskPath` and `CVDirtyMultitaskPath` for
  separate shared and task-specific coefficient components. The default is
  Jalali et al.'s shared ℓ∞ norm; `shared_norm = :l2` is a variant. Fits return
  a canonical split, a recomputed two-component KKT residual and a relative
  duality gap. CV tunes ratio and level with common folds across ratios.

- Add `sparse_nuclear_regression` for nuclear and entrywise/row sparsity
  penalties on the same coefficient matrix, with duality-gap certificates and
  `SparseNuclearSolution`. `PenaltyGrid` and `CVPenaltyGrid` tune both levels
  with copied primal/dual warm starts and independent training-fold fits.

- Add row ℓ1/ℓ∞ (`row_norm = :linf`) and multitask sparse group (`alpha > 0`)
  fits and paths for `multitask_lasso` and `multi_lasso`, with verified
  screening and a row solver on the tasks' own designs. Add `CVMultiLassoPath`
  with observation folds shared across responses. The default row ℓ1/ℓ2 solver
  is unchanged. Both `CVMultitaskLassoPath` and `CVMultiLassoPath` now treat a
  nonconverged fold level as unreached (`NaN`); it cannot be selected.

- `CVLassoPath` and `CVGLMLassoPath` no longer throw when every fold path stops
  early (`max_hat_s`, `devmax`) before the last levels of the full path; those
  levels have `NaN` in `cvm` and `cvse`. `λmin` and `λ1se` of these and of
  `CVMultitaskLassoPath` are chosen among the levels that every fold reaches,
  so a level that only some fold paths reached is no longer selected. `cvm` is
  unchanged.

- Add `quantile_refit`, `QuantileRefit` and `refit_quantile_path` for unpenalized
  refits on selected supports, preserving raw dual scores and restricted-fit
  diagnostics. Exact refits reuse one model across requested quantiles.

- Add `max_hat_s` to exact and smoothed quantile paths and their cross-validation.
  Paths retain the first fit exceeding the nonzero-slope limit and skip later
  levels with `status=:max_hat_s`; unreached fold levels are ineligible.

- Add `slope_gaussian` and `calibration = :bh | :gaussian | :user` for `slope`
  and `SLOPEPath`, with recorded calibration metadata. Explicit weights take
  precedence; BH remains the default.

- Add `knockoff_diagonal` and `strategy = :equicorrelated | :mvr | :maxent | :user`
  for knockoff construction and `knockoff_filter`, with recorded construction
  metadata. MVR and maximum entropy require the optional Knockoffs.jl extension
  for diagonal optimization only; sampling stays native. Equicorrelation remains
  the default. `fixed_knockoffs` and `knockoff_filter(...; method=:fixed)` now
  raise `ArgumentError` for zero-norm columns or insufficient numerical column
  rank checked on the column-normalized design.

- Double selection supports `vcov` choices, final-stage diagnostics and
  precomputed nuisance fits. Forced controls are validated before fitting;
  invalid `forced` now raises `ArgumentError` instead of `BoundsError`.
  The positional constructors of `TreatmentEffect` and `RLassoSolution` changed.

- Partialling-out supports controlled nuisance `refit` modes and robust variance
  choices through `vcov`; it now rejects nonempty `forced` and `intercept=false`.

- `rlasso` accepts an `rng` keyword and retains the final penalized coefficients,
  their penalty/loadings and intercept, convergence flags, selected-design rank,
  and effective tuning in `RLassoSolution`.

Low-rank nuclear fits and paths accept `solver = :panoc` after loading the
optional ProximalAlgorithms dependency. Row-sparse fits and paths accept
`block_solver = :fista`; both options propagate through cross-validation.
Fixed-rank fits reuse existing factorizations for diagnostics and compute
residuals once, preserving numerical rank and degrees-of-freedom definitions.

- Float64 nuclear PANOC fits and paths reuse loss/gradient products and the final
  residual and singular values, preserving the objective and returned diagnostics.
  APG remains the default.

- Sparse reduced-rank fits and paths reuse sufficient statistics across penalties
  and compute rank/residual diagnostics from thin factors. CD results and all
  defaults are unchanged; FISTA results agree within the stopping tolerance.

- Sparse reduced-rank fits finish difficult group-lasso blocks with guarded
  Newton steps, improving convergence of ill-conditioned adaptive fits. Float32
  sparse defaults use a `100eps(T)` tolerance floor, and single fits warn on
  non-convergence with a conditioning diagnostic when applicable.

## Quantile primal–dual API

- Add exact HiGHS fit/path solvers and a convolution-smoothed proximal Newton route.
- Preserve raw observation-level duals through `dual_scores`, document QR.jl's sign/normalization conversion, and retain independent certificates and per-fit KKT diagnostics.
- Support weighted slope penalties, offsets, explicit zero penalties, sparse designs and partial exact paths; do not interpolate between quantile levels.

- Add `quantile_lambda` for fixed-quantile pivotal penalty simulation with explicit confidence, loading and RNG controls.
- Add `CVQuantileLassoPath` for held-out check-loss tuning, minimum/one-standard-error selection, and raw dual access from the selected full-data fit. Inaccurate or missing fold fits exclude that penalty.
- Enable certified smoothed stopping with the merged ProximalSolvers convolution backend; retain the documented older-release fallback. Active development and CI resolve workspace master sources, while release CI resolves registered runtime dependencies.

## Multivariate regression

Add rank-constrained least squares with the Bunea–She–Wegkamp rank selection
criterion, coefficient nuclear-norm regression with warm-started paths, and
row-sparse reduced-rank regression. Sparse singular-vector regression also
selects response loadings. `LowRankPath`, `CVLowRankPath` and `lowrank_ic`
provide rank/penalty grids, fold-specific tuning, and information criteria.
The methods support intercepts, rank-deficient designs, and StatsAPI.

## Multitask regression with task-specific designs

Add `multitask_lasso`, `MultitaskLassoPath` and `CVMultitaskLassoPath`, the row
ℓ1/ℓ2 penalty on `B = [b_1 … b_q]` when each task has its own design and
response, with the same predictors and possibly unequal sample sizes. Task
weights count each task once (`task_weights = :equal`, the default) or each
observation once (`:observations`); the columns are standardized within each
task, and the coefficients are returned on the original scale. With one task
the fit is `lasso`, and with a shared design and equal weights it is
`multi_lasso`. Cross-validation draws the folds within each task, refits on the
training observations alone, and reports the held-out error of each task
(`cvm_task`).

## Paths stop on a certified duality gap

With ProximalSolvers' gap stopping (ProximalSolvers #86), the paths (`LassoPath`, `CVLassoPath`,
`GLMLassoPath`, `GroupLassoPath`, `SparseGroupLassoPath`, `MultiLassoPath`)
and the last level of `warm_start = false` stop each level once its relative
duality gap is at most `1e-4`, for the losses and penalties that have a gap,
instead of solving every level to `opt_tol`. The coefficients are then within
about `1e-4` of the solution, relative to the largest, and within about
`5e-3` at worst. On the benchmark workloads of ProximalSolvers #86, `LassoPath`
and `CVLassoPath` took 19 to 34% less time, and `GLMLassoPath` between 27% less
and 9% more. The new option
`gaptol` of `CDOptions` sets the target: `nothing` (the default) takes
ProximalSolvers' `PATH_GAPTOL = 1e-4`, `1e-3` is the faster setting, and
`gaptol = 0` stops on the changes with `opt_tol`, as before. Tests that compare
paths with near-exact solutions pass `gaptol = 0`.

# Development changes

CoordinateDescent.jl is now HighDimensionalModels.jl. The Julia module is
`HighDimensionalModels`; replace `using CoordinateDescent` and qualified
`CoordinateDescent.` calls with the new name. The StatsModels extension is
`HighDimensionalModelsStatsModelsExt`. The renamed package has a new UUID and starts at version 0.1.0. Active development environments use local master checkouts of
ProximalBase, ProximalSolvers and KernelSmoothing.

# CoordinateDescent development history (previously labeled 0.8.0)

## Performance

`CVLassoPath` chooses the Gram form once on the full data and keeps that choice
for the training folds, avoiding a switch to residual updates when holding out
observations makes a square design wide. The folds still use the full-data
penalty grid. `GLMLassoPath` reuses its prepared model when it generates that
grid, instead of constructing the problem twice. GLM fits continue to use
proximal Newton.

## Breaking changes

The public API follows Julia's naming convention: types use `UpperCamelCase`;
functions, fields and keywords use lowercase, with underscores between words.
Mutating functions keep their trailing `!`. See the
[Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/#Use-naming-conventions-consistent-with-Julia-base/).

| Previous name | New name |
|---|---|
| `coordinateDescent!` | `coordinate_descent!` |
| `sqrtLasso`, `scaledLasso`, `scaledLasso!`, `feasibleLasso!` | `sqrt_lasso`, `scaled_lasso`, `scaled_lasso!`, `feasible_lasso!` |
| `refitLassoPath`, `adaptiveLasso` | `refit_lasso_path`, `adaptive_lasso` |
| `glmLasso`, `groupLasso`, `sparseGroupLasso`, `multiLasso` | `glm_lasso`, `group_lasso`, `sparse_group_lasso`, `multi_lasso` |
| `theoryLambda`, `postLasso`, `doubleSelection` | `theory_lambda`, `post_lasso`, `double_selection` |
| `varyingCoefficient`, `varyingCoefficientCV` | `varying_coefficient`, `varying_coefficient_cv` |
| `varyingCoefficientLasso`, `varyingCoefficientLassoCV` | `varying_coefficient_lasso`, `varying_coefficient_lasso_cv` |
| `numCoordinates`, `descendCoordinate!`, `fullGradient!` | `num_coordinates`, `descend_coordinate!`, `full_gradient!` |
| `coordinateScale`, `responseScale` | `coordinate_scale`, `response_scale` |
| `maxIter`, `optTol`, `warmStart`, `numSteps`, `relativeTol` | `max_iter`, `opt_tol`, `warm_start`, `num_steps`, `relative_tol` |
| `initProcedure`, `optionsCD`, `standardizeX` | `init_procedure`, `options_cd`, `standardize_x` |
| `newtonTol`, `newtonMaxIter`, `kktChecks`, `maxChange` | `newton_tol`, `newton_max_iter`, `kkt_checks`, `max_change` |

The previous function names forward with deprecation warnings for this release.
The previous keyword and property spellings also warn and forward; supplying
both spellings of a keyword is an error. Loss-interface names are deprecated
bindings to the same generic functions, so existing extensions continue to work.
Type names, including `IterLassoOptions` and `LassoPath`, are unchanged.

`coordinate_descent!` returns `ProximalSolvers.CDResults`, including `minimizer`,
`minimum`, `passes`, `converged`, `maxchange` and `tolerance`. The deprecated
`coordinateDescent!` keeps its previous return value, the iterate `x`, including
its five-argument form with `CDInfo`. Use `r = coordinate_descent!(x, f, g)` and
`r.minimizer` for the fitted iterate.

Integer fields use `Int` and integer inputs such as `degree` accept `Integer`.
The benchmark uses 0.05 and 0.01 times the smallest penalty with a zero solution.
