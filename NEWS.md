# Unreleased

Low-rank nuclear fits and paths accept `solver = :panoc` after loading the
optional ProximalAlgorithms dependency. Row-sparse fits and paths accept
`block_solver = :fista`; both options propagate through cross-validation.
Fixed-rank fits reuse existing factorizations for diagnostics and compute
residuals once, preserving numerical rank and degrees-of-freedom definitions.

## Quantile primal–dual API

- Add exact HiGHS fit/path solvers and a convolution-smoothed proximal Newton route.
- Preserve raw observation-level duals through `dual_scores`, document QR.jl's sign/normalization conversion, and retain independent certificates and per-fit KKT diagnostics.
- Support weighted slope penalties, offsets, explicit zero penalties, sparse designs and partial exact paths; do not interpolate between quantile levels.

- Add `quantile_lambda` for fixed-quantile pivotal penalty simulation with explicit confidence, loading and RNG controls.
- Add `CVQuantileLassoPath` for held-out check-loss tuning, minimum/one-standard-error selection, and raw dual access from the selected full-data fit. Inaccurate or missing fold fits exclude that penalty.
- Enable certified smoothed stopping with the merged ProximalSolvers convolution backend; retain the documented older-release fallback. Active development and CI resolve workspace master sources, with migration version numbers unchanged.

## Multivariate regression

Add rank-constrained least squares with the Bunea–She–Wegkamp rank selection
criterion, coefficient nuclear-norm regression with warm-started paths, and
row-sparse reduced-rank regression. Sparse singular-vector regression also
selects response loadings. `LowRankPath`, `CVLowRankPath` and `lowrank_ic`
provide rank/penalty grids, fold-specific tuning, and information criteria.
The methods support intercepts, rank-deficient designs, and StatsAPI.

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
`HighDimensionalModelsStatsModelsExt`. The package UUID and version remain
unchanged. Active development environments use local master checkouts of
ProximalBase, ProximalSolvers and KernelSmoothing.

# Version 0.8.0

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
