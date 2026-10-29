# Paths, screening and solver options

## Lasso paths

[`LassoPath`](@ref)`(X, y)` solves the lasso along a decreasing sequence of
penalty levels, each warm started from the previous solution. By default the
sequence has `nλ = 100` levels spaced evenly on the log scale from
``\lambda_{\max}``, the smallest level whose solution is zero, down to
`λminratio` times ``\lambda_{\max}``, with `λminratio` 0.01 when ``p > n`` and
``10^{-4}`` otherwise; `LassoPath(X, y, λpath)` takes levels of your own. Unlike
[`lasso`](@ref), a path standardizes the columns by default
(`standardize_x = true`).

- **Screening.** Each level ``\lambda_k`` is solved first over the coordinates
  kept by the sequential strong rule of Tibshirani et al. (2012), those with
  ``|X_j^\top r(\lambda_{k-1})| / n \ge (2\lambda_k - \lambda_{k-1})\,\omega_j``
  for the residuals ``r(\lambda_{k-1})`` at the previous level. The optimality
  conditions of the other coordinates are checked afterwards; those that
  violate them are added, and the level is solved again. `kkt_checks[k]` counts
  the checks at level ``k``.
- **Early stopping.** `max_hat_s` stops the path after the first solution with
  more nonzero coefficients.
- **The results.** `λpath`, `βpath` (sparse vectors) and `β0path`, with the
  passes and the convergence of coordinate descent at each level.
  [`coef`](@ref)`(path)` gives all coefficients as a sparse matrix,
  `coef(path, λ)` the coefficients at any ``\lambda`` in the range of the path,
  interpolated linearly in ``\lambda`` between levels as glmnet does, and
  [`predict`](@ref)`(path, X)` the fitted values along the path.
  [`fit`](@ref)`(LassoPath, X, y)` is the same as `LassoPath(X, y)`.
- **Refitting.** [`refit_lasso_path`](@ref) refits least squares on each distinct
  support of the path.

## Choosing λ

[`CVLassoPath`](@ref)`(X, y)` chooses the level by ``K``-fold
cross-validation (`nfolds = 10`), with the options and keywords of
[`LassoPath`](@ref). The path on all the data gives the levels, the path of
each fold is computed on the other folds at those levels, and the mean squared
error on the held-out fold estimates the prediction error at each level;
standardization and the intercept are computed on each training fold. `cvm`
averages the fold errors weighted by the fold sizes and `cvse` is its standard
error, as in glmnet; `λmin` minimizes `cvm`, and `λ1se` is the largest level
whose `cvm` is within `cvse` of the minimum, the one-standard-error rule.
`coef(cv)` and `predict(cv, X)` use the solution at `λ1se`, or at `λmin` with
`λ = :λmin`. The folds are a balanced random assignment drawn from `rng`, or
the fold labels `folds`, for example blocked folds for dependent data. They
run as tasks on the available threads, and the result does not depend on the
number of threads.

The information criteria [`aic`](@ref), [`bic`](@ref) and [`ebic`](@ref) of a
path, one value per level, choose the level without refitting:

```math
\mathrm{AIC} = n\log(\mathrm{RSS}/n) + 2\,\mathrm{df}, \qquad
\mathrm{BIC} = n\log(\mathrm{RSS}/n) + \log(n)\,\mathrm{df}, \qquad
\mathrm{EBIC} = \mathrm{BIC} + 2\gamma\log\binom{p}{s},
```

where ``\mathrm{df}`` is the number of nonzero coefficients, plus one for an
intercept, and the extended BIC of Chen and Chen (2008), for ``s`` nonzero
coefficients among ``p``, accounts for the size of the model space:
`path.λpath[argmin(bic(path, X, y))]` is the level that BIC chooses.

## Solver options

Every solver takes the options of coordinate descent as a [`CDOptions`](@ref)
argument, and its fields as keywords, which replace those of the options:
`lasso(X, y, λ; max_iter = 10_000, opt_tol = 1e-8)`. The scaled and the feasible
lasso take [`IterLassoOptions`](@ref), whose field `options_cd` holds the
options of their lasso fits.

- **The stopping rule.** Coordinate descent stops when no coordinate changes by
  more than `opt_tol` (by default ``10^{-7}``) in a pass, the change of
  coefficient ``j`` measured in units of ``\|y\| / \|X_j\|``, so that the result
  does not depend on the units of the data (`relative_tol = false` measures it
  in the units of the coefficients). It also stops after `max_iter` passes (by
  default 100,000), and then warns that the solution may be inaccurate.
- **The order.** Each pass visits the coordinates in a new random order
  (`order = :random`) drawn from the keyword `rng`, so a seeded generator, such
  as `rng = Xoshiro(1)`, reproduces the passes. `order = :cyclic` visits them
  in order.
- **Momentum.** With a random order, each pass over the active set starts from
  a point extrapolated from the last two (Nesterov momentum). It is on by
  default with a random order, a `SparseIterate` and the built-in losses. On
  the designs of [`order_anderson_bench.jl`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/abddaf56a790b86178de376b175a1185020c8381/benchmark/order_anderson_bench.jl)
  in the benchmark snapshot it computes 50-level paths 2.7 to 6.5 times faster
  than the plain random order, including designs whose columns share a common
  factor; short solves take about as long either way.
  `momentum = false` turns it off.
- **Anderson acceleration.** `anderson = 5`, with `order = :cyclic`,
  extrapolates from the last passes. It is faster than momentum when few passes
  are needed and when ``n \gg p``, and much slower when the columns share a
  common factor.
- **Warm starts.** [`coordinate_descent!`](@ref) starts from the iterate it is
  given. With `warm_start = false` it starts from zero and, for `NormL1` and
  `GroupNormL2`, solves along a path of `num_steps` levels from
  ``\lambda_{\max}`` down to the penalty.

[`coordinate_descent!`](@ref) and [`solve!`](@ref) run coordinate descent on a
loss and a penalty directly; `solve!` reports the passes and whether the
stopping rule was met.

## The Gram form

With `gram = true`, [`lasso`](@ref), [`sqrt_lasso`](@ref) and [`LassoPath`](@ref)
run coordinate descent on the Gram matrix of the centered columns,
``A = X^\top X / n``, instead of the residuals. A coordinate step then costs
``O(p)`` instead of ``O(n)``, after ``O(np^2)`` operations and ``p^2`` numbers of
memory to form ``A``. The loss, the stopping rule and so the solution are the
same, to the tolerance.

Forming ``A`` pays off over the many passes of a path when ``n \ge p``, and
almost never at one ``\lambda``. On one core of an x86-64 server CPU, these were the
ratios of the time on the Gram matrix to the time on the residuals for paths of
100 levels with an intercept, over independent and AR(1) columns, for ``n``
observations (rows) and ``p`` predictors (columns):

| ``n`` | ``p = 50`` | ``p = 200`` | ``p = 1000`` | ``p = 2000`` |
|---:|---:|---:|---:|---:|
| 500 | | | 1.03–1.06 | 2.1–2.2 |
| 1,000 | 0.12–0.14 | 0.17 | 0.56–0.60 | 1.3–1.4 |
| 2,000 | | | 0.34–0.37 | 0.66–0.93 |
| 4,000 | | | 0.11–0.14 | 0.37–0.38 |
| 5,000 | 0.03–0.04 | 0.05–0.08 | 0.12–0.15 | |
| 10,000 | | | 0.08–0.11 | 0.17–0.25 |
| 20,000 | 0.02–0.03 | 0.04–0.06 | 0.08–0.13 | |

At one ``\lambda`` the Gram matrix was slower in 93 of 94 cases, by up to 42
times. The default, `gram = :auto`, therefore uses it for paths on a dense
``X`` with ``n \ge p`` and ``p \le 2000``, and never at one ``\lambda``. These
measurements come from
[`gram_bench.jl`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/abddaf56a790b86178de376b175a1185020c8381/benchmark/gram_bench.jl)
in the benchmark snapshot. The `lasso` group of the
[benchmark harness](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/benchmark/README.md)
times current fits, including fits on the Gram matrix, but does not repeat
this comparison.
