# Unreleased

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
