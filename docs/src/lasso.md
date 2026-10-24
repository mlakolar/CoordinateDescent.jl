# The lasso and its tuning-free variants

## The lasso

[`lasso`](@ref)`(X, y, λ)` solves

```math
\underset{\beta_0,\,\beta}{\text{minimize}} \quad \frac{1}{2n}\,\|y - \beta_0 - X\beta\|^2 + \lambda \sum_{j=1}^p \omega_j |\beta_j|.
```

- **The scaling.** The loss is divided by ``2n``, as in glmnet, so that
  ``\lambda`` is on the scale of the gradient ``X_j^\top r / n`` and does not grow
  with the sample size.
- **The intercept.** With `intercept = true`, ``\beta_0`` is not penalized, and
  at the solution it is ``\bar y - \bar x^\top \hat\beta`` for the column means
  ``\bar x`` and the mean ``\bar y``. The columns and ``y`` are centered on the
  fly, so ``X`` is not copied. Without it, ``\beta_0 = 0``.
- **The loadings.** By default ``\omega_j = 1``. With `standardizeX = true`,
  ``\omega_j`` is the standard deviation of column ``j``,
  ``\sqrt{\sum_i (X_{ij} - \bar x_j)^2 / n}`` (centered only with an
  intercept), which penalizes the coefficients as if the columns were
  standardized while keeping them on the scale of the data.
  `lasso(X, y, λ, ω)` takes loadings of your own, as the adaptive lasso does; a
  loading of zero leaves a coefficient unpenalized.
- **The largest useful λ.** All coefficients are zero for
  ``\lambda \ge \lambda_{\max} = \max_j |X_j^\top (y - \bar y)| / (n \omega_j)``,
  with ``\bar y = 0`` without an intercept.

The theory of the lasso takes ``\lambda`` of the order of
``\sigma \sqrt{2 \log(p) / n}`` for noise with standard deviation ``\sigma`` and
standardized columns. That needs ``\sigma``. The variants below do not, and a
[`LassoPath`](@ref) with cross-validation chooses ``\lambda`` from the data (see
[Paths, screening and solver options](paths.md)).

[`lasso`](@ref) returns a [`CoordinateDescent.LassoSolution`](@ref): the
coefficients `x` (a `SparseIterate`), the `intercept`, the `residuals`,
``\hat\sigma`` (the standard deviation of the residuals), the passes of
coordinate descent and whether it converged. The functions of StatsAPI apply to
it: [`coef`](@ref) (intercept first), [`predict`](@ref), [`residuals`](@ref),
[`nobs`](@ref), [`dof`](@ref) (the number of nonzero coefficients, plus one for
the intercept) and [`deviance`](@ref).

## The elastic net, MCP and SCAD

[`lasso`](@ref) and [`LassoPath`](@ref) take other penalties of ProximalBase
in place of the ℓ1 norm, at the level ``\lambda``. `alpha < 1` gives the
elastic net of glmnet,

```math
\lambda\Big(\alpha\sum_j |\beta_j| + \frac{1 - \alpha}{2}\|\beta\|^2\Big), \qquad 0 < \alpha \le 1,
```

and `penalty` another penalty whose level ``\lambda`` replaces: `NormL1(1, w)`
and `NormL1Nonneg(1, w)`, the lasso and the nonnegative lasso with weights;
`ElasticNet(1, λ2, w)`, with the ridge part ``\lambda_2`` fixed;
`Scaled(ElasticNet(α, 1 - α, w), 1)`, the elastic net with weights on its ℓ1
part; and the nonconvex `MCP(1, a)` and `SCAD(1, a)` of ncvreg, whose
concavity ``a`` is kept. With `standardizeX`, the penalty applies to the
coefficients of the standardized columns, as in glmnet and ncvreg, and the
coefficients returned are those of the columns of ``X``. A nonconvex penalty
has local solutions, and the one found depends on the start: as in ncvreg,
`lasso` follows a path of levels from ``\lambda_{\max}`` down to ``\lambda``,
each warm started from the previous one. The path of the elastic net at a
fixed ``\alpha`` starts at ``\lambda_{\max}`` of the lasso divided by
``\alpha``.

The adaptive lasso of Zou (2006), [`adaptiveLasso`](@ref)`(X, y, λ)`,
minimizes

```math
\frac{1}{2n}\|y - \beta_0 - X\beta\|^2 + \lambda\sum_j \frac{|\beta_j|}{|\tilde\beta_j|^\gamma}
```

with loadings from a first estimate ``\tilde\beta``: the lasso (`init = :lasso`,
at `initλ`, by default ``\lambda``), ridge regression (`init = :ridge`) or a
vector of coefficients. Large coefficients are penalized less, so that they are
less biased, and a coefficient with ``\tilde\beta_j = 0`` stays at zero.

## The square-root lasso

[`sqrtLasso`](@ref)`(X, y, λ)` solves the square-root lasso of Belloni,
Chernozhukov and Wang (2011),

```math
\underset{\beta_0,\,\beta}{\text{minimize}} \quad \|y - \beta_0 - X\beta\| + \lambda \sum_{j=1}^p \omega_j |\beta_j|,
```

with the loadings of standardized columns by default (`standardizeX = true`).
The gradient of the loss at the true coefficients is
``-X^\top \varepsilon / \|\varepsilon\|``, which does not depend on the noise
level, and neither does a good ``\lambda``: with Gaussian noise,
``\lambda = 1.1\, \Phi^{-1}(1 - \alpha / (2p))`` keeps every coefficient at zero
with probability about ``1 - \alpha`` when ``\beta = 0``, whatever ``\sigma``. The
loss is not smooth where the residuals vanish, so each coordinate step
minimizes the objective over the coordinate exactly.

## The scaled lasso

[`scaledLasso`](@ref)`(X, y, λ, ω)` estimates the coefficients and the noise
level together, by the scaled lasso of Sun and Zhang (2012):

```math
\underset{\beta_0,\,\beta,\,\sigma > 0}{\text{minimize}} \quad \frac{1}{2n\sigma}\,\|y - \beta_0 - X\beta\|^2 + \frac{\sigma}{2} + \lambda \sum_{j=1}^p \omega_j |\beta_j|.
```

The minimum over ``\sigma`` is at ``\hat\sigma = \|y - \beta_0 - X\beta\| / \sqrt n``,
so ``\hat\beta`` solves a square-root lasso, with the loss
``\|y - \beta_0 - X\beta\| / \sqrt n``, which one run of coordinate descent
solves. ``\lambda`` is the penalty for unit noise, such as
``\sqrt{2 \log(p) / n}``: the lasso that ``\hat\beta`` solves has the penalty
level ``\lambda \hat\sigma``, which the solution reports in `penalty`.
[`scaledLasso!`](@ref) also solves it from sufficient statistics,
``X^\top X / n``, ``X^\top y / n`` and ``y^\top y / n``, and from a covariance
matrix, one variable on the others, as in the precision matrix estimator of
Sun and Zhang (2013).

## The feasible lasso

Under heteroscedastic noise, the penalty loadings of Belloni, Chen,
Chernozhukov and Hansen (2012) depend on the noise of each observation:

```math
\Gamma_j = \sqrt{\frac{1}{n} \sum_{i=1}^n (X_{ij} - \bar x_j)^2 \varepsilon_i^2}.
```

[`feasibleLasso!`](@ref)`(x, X, y, λ0)` alternates the lasso
``\|y - \beta_0 - X\beta\|^2 / (2n) + \lambda_0 \sum_j \Gamma_j |\beta_j|`` with
the update of ``\Gamma`` from its residuals, until the relative change of
``\Gamma`` is below `optTol` of [`IterLassoOptions`](@ref); `initProcedure`
chooses the initial residuals. The examples use
``\lambda_0 = 1.1\, \Phi^{-1}(1 - \gamma / (2p)) / \sqrt n`` with
``\gamma = 0.1 / \log n``.

## Observation weights and offsets

[`lasso`](@ref), [`sqrtLasso`](@ref), [`scaledLasso`](@ref) and
[`LassoPath`](@ref) take observation `weights` ``w_i \ge 0``, with a positive
sum, and an `offset` ``o``. The lasso then minimizes

```math
\frac{1}{2 \sum_i w_i} \sum_{i=1}^n w_i (y_i - o_i - \beta_0 - x_i^\top \beta)^2 + \lambda \sum_{j=1}^p \omega_j |\beta_j|.
```

Dividing by ``\sum_i w_i`` follows glmnet: ``\lambda`` means the same with and
without weights, unit weights give the unweighted fit, integer weights the fit
of replicated rows, and zero weights leave observations out. The intercept,
the loadings of `standardizeX` and ``\hat\sigma`` are weighted as well. The
square-root lasso takes the weighted norm ``\sqrt{\sum_i w_i r_i^2}`` of the
residuals ``r``, and the scaled lasso estimates
``\hat\sigma = \sqrt{\sum_i w_i r_i^2 / \sum_i w_i}``.

## The data

`X` can be any matrix and `y` any vector of real numbers. A `Matrix`, or a view
of a range of its rows or of some of its columns, is read in place. A
`SparseMatrixCSC` stays sparse; coordinate descent reads its columns entry by
entry, so for now it is slower than a dense matrix. Any other matrix, such as a
transpose, is copied once. The computations are in the floating point type of
`X` and `y`, so `Float32` data give `Float32` estimates.
