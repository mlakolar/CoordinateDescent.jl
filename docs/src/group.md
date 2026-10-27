# The group lasso

The group lasso of Yuan and Lin (2006) penalizes groups of coefficients
together,

```math
\underset{\beta}{\text{minimize}} \quad \frac{1}{2n}\,\|y - X\beta\|^2 + \lambda \sum_{k=1}^K w_k \|\beta_{G_k}\|_2,
```

for groups ``G_1, \dots, G_K`` that partition the coefficients, so that a group
is zero or nonzero as a whole.

Coordinate descent works on an `AtomIterate` of ProximalBase, whose coordinates
are the groups, with the penalty `GroupNormL2(λ, groups)`, whose weights are
``w_k = \sqrt{|G_k|}`` by default and `w` in `GroupNormL2(λ, groups, w)`. A
weight of zero leaves a group unpenalized. Each step updates one group at once,
with the largest eigenvalue of ``X_G^\top X_G / n`` as the curvature (Yang and
Zou, 2015). [`CDLeastSquaresLoss`](@ref), also with an intercept, and
[`CDQuadraticLoss`](@ref) support groups.

```@example group
using HighDimensionalModels, ProximalBase, LinearAlgebra, Random

rng = Xoshiro(1)
n, p = 200, 50
X = randn(rng, n, p)
y = X[:, 1:5] * ones(5) + randn(rng, n)

groups = [1:5, 6:10, 11:20, 21:50]
g = GroupNormL2(0.1, groups)         # weights √|Gₖ|
x = AtomIterate(p, groups)
coordinate_descent!(x, CDLeastSquaresLoss(y, X), g)
[norm(x[G]) for G in groups]         # the first group is selected
```

All groups are zero for ``\lambda \ge \max_k \|X_{G_k}^\top y\| / (n w_k)``. With
`warm_start = false`, [`coordinate_descent!`](@ref) solves along a path of
penalty levels from there, and screens the groups with the strong rule. The
example on the group lasso chooses ``\lambda`` on a validation set and checks the
optimality conditions.

## The estimators

[`group_lasso`](@ref)`(X, y, λ, groups)` and [`GroupLassoPath`](@ref)`(X, y, groups)`
are the group lasso with an unpenalized intercept and, by default
(`standardize = :orthonormal`), the penalty on the fitted values of each group,

```math
\frac{1}{2n}\|y - \beta_0 - X\beta\|^2 + \lambda\sum_k w_k\,\frac{\|\tilde X_{G_k}\beta_{G_k}\|_2}{\sqrt{n}},
```

with ``\tilde X_G`` the centered columns of group ``G`` and ``w_k = \sqrt{|G_k|}``
by default. This makes the estimate invariant to a change of basis within a
group (Simon and Tibshirani, 2012), as in R's grpreg. It is fitted on
orthonormalized groups: each ``\tilde X_G`` is factored by a pivoted QR as
``Z_G\tilde R`` with ``Z_G^\top Z_G/n = I``, the group lasso is solved in
``\theta_G = \tilde R\beta_G`` by block coordinate descent, whose steps are then
exact, and the coefficients are mapped back; columns of a group that are
combinations of its other columns get zero coefficients. With
`standardize = :none`, the penalty is ``\lambda\sum_k w_k\|\beta_{G_k}\|_2`` on
the columns as given. The groups must be disjoint and cover the columns; a
column alone is a group of its own, and a zero weight leaves a group
unpenalized. The solutions report the active groups.

[`sparse_group_lasso`](@ref)`(X, y, λ, groups)` and
[`SparseGroupLassoPath`](@ref)`(X, y, groups)` are the sparse group lasso of
Simon, Friedman, Hastie and Tibshirani (2013),

```math
\frac{1}{2n}\|y - \beta_0 - X\beta\|^2 + \lambda\Big(\alpha\sum_j |\tilde\beta_j| + (1 - \alpha)\sum_k w_k\|\tilde\beta_{G_k}\|_2\Big),
```

which selects groups and, within them, variables. With `standardize_x` (the
default), ``\tilde\beta = s\beta`` are the coefficients of the standardized
columns, as in R's SGL, whose default `alpha = 0.95` it shares; `alpha = 1` is
the lasso and `alpha = 0` the group lasso on the columns as given. The groups
are not orthonormalized, since the ℓ1 term is not invariant to a change of basis
within a group. Paths screen groups by the level at which each would be zero:
a group at zero is optimal while
``\lVert\mathrm{soft}(\nabla_{G_k}, \lambda\alpha)\rVert_2 \le \lambda(1 - \alpha) w_k``.

## The multi-response lasso

With several responses that share the predictors, [`multi_lasso`](@ref)`(X, Y, λ)`
and [`MultiLassoPath`](@ref)`(X, Y)` fit

```math
\frac{1}{2n}\|Y - 1\beta_0^\top - XB\|_F^2 + \lambda\sum_k \omega_k\|B_{k\cdot}\|_2,
```

the group lasso on the rows of the ``p \times q`` coefficient matrix ``B``,
which selects the same predictors for every response (glmnet's `mgaussian`).
The curvature of the loss over a row is a multiple of the identity, so each
block step on a row is exact. The solutions report ``B`` as a sparse matrix,
the intercepts of the responses and the active rows.

## Task-specific designs

When the responses do not share the observations, [`multitask_lasso`](@ref)`(Xs, ys, λ)`
and [`MultitaskLassoPath`](@ref)`(Xs, ys)` fit ``q`` regressions
``y_t = \beta_{0t} + X_t b_t + e_t``, each with its own ``n_t \times p``
design `Xs[t]` and response `ys[t]`. The designs have the same ``p``
predictors in the same order, and the sample sizes ``n_t`` may differ. With
``B = [b_1 \cdots b_q]`` and ``b_{tj}`` the coefficient of predictor ``j`` in
task ``t``, they minimize

```math
\sum_{t=1}^q \frac{c_t}{2n_t}\,\|y_t - \beta_{0t} - X_t b_t\|^2 + \lambda\sum_{j=1}^p \omega_j\,\big\|(s_{1j}b_{1j}, \dots, s_{qj}b_{qj})\big\|_2,
```

the row penalty of the multi-response lasso, which selects a predictor for all
the tasks or for none while its coefficients differ across tasks (Obozinski,
Taskar and Jordan, 2010). The coefficients are returned on the original scale.

- **Task weights.** `task_weights = :equal` (the default) takes ``c_t = 1``,
  so that the mean squared error of each task counts once, whatever its size.
  `task_weights = :observations` takes ``c_t = q n_t / N`` with
  ``N = \sum_t n_t``, so that each observation counts once: the loss is ``q``
  times the pooled loss ``\sum_t \|y_t - \beta_{0t} - X_t b_t\|^2 / (2N)``,
  scaled so that equal sizes give ``c_t = 1``. The two coincide when the tasks
  have the same size; otherwise `:observations` gives the larger tasks more
  weight. A vector of positive weights sets the ``c_t`` directly.
- **Standardization.** With `standardize_x = true` (the default), ``s_{tj}``
  is the standard deviation of column ``j`` within task ``t`` (with divisor
  ``n_t``, about the column mean with an intercept and about zero otherwise),
  and the penalty acts on the
  coefficients ``s_{tj} b_{tj}`` of the columns standardized within each task.
  A predictor is then penalized on the same scale in every task: rescaling its
  column in one task rescales its coefficient in that task and changes nothing
  else. With `standardize_x = false`, ``s_{tj} = 1``, and the units of each
  task's column enter the norm of the row. A column that is zero within a task,
  or constant within it with an intercept, gets a zero coefficient in that task
  and is left out of the fit; without an intercept, a nonzero constant column
  is an ordinary predictor.
- **Penalty factors.** `penalty_factors` are the ``\omega_j``, ones by default,
  finite and nonnegative, with at least one positive. A zero factor leaves a
  predictor unpenalized. It is allowed with levels of your own only: no level
  makes the coefficients of an unpenalized predictor zero, so the automatic
  grid below needs positive factors.
- **Intercepts.** Each task has an unpenalized intercept
  ``\beta_{0t} = \bar y_t - \bar x_t^\top b_t`` (`intercept = true`, the
  default); with `intercept = false` they are zero and nothing is centered.
- **Reductions.** With one task, the problem is [`lasso`](@ref)`(X, y, λ)` with
  the same `intercept` and `standardize_x` (whose defaults in `lasso` are
  `false`), and with a weight ``c`` it is the lasso at ``\lambda / c``. When
  every task has the same design ``X`` and all the weights are ``c``,
  ``s_{tj} = s_j`` and, with the default penalty factors, the problem is
  [`multi_lasso`](@ref)`(X, Y, λ / c)` with ``Y = [y_1 \cdots y_q]``; a dense
  shared design is fitted with the model of `multi_lasso`. Its `nobs` is then ``qn``, where `multi_lasso` reports
  ``n``.
- **Method.** The tasks are centered and standardized each on its own, the
  rows of task ``t`` are scaled by ``\sqrt{N c_t / n_t}``, and the blocks are
  placed on the diagonal of a sparse ``N \times pq`` design over
  ``\operatorname{vec}(B)``, whose groups are the rows of ``B``; group
  coordinate descent solves the least-squares problem on it, with coordinate
  orders drawn from `rng`. The columns of different tasks are orthogonal, so
  the curvature over a row is diagonal. With standardization and equal weights
  it is the identity and each block step is exact; otherwise a step minimizes
  a majorization, and convergence takes more passes.
- **Sparse designs.** A sparse ``X_t`` stays sparse without an intercept. With
  an intercept, centering fills it in: its block in the stacked design is
  dense, about ``n_t p`` stored entries.
- **The results.** A [`MultitaskLassoSolution`](@ref) holds ``B`` as a sparse
  ``p \times q`` matrix, the intercepts, the active rows, `nobs` (``N``),
  `task_nobs` (the ``n_t``) and the task weights. [`coef`](@ref) gives the
  ``(p + 1) \times q`` matrix with the intercepts in the first row,
  [`predict`](@ref)`(sol, Xs_new)` a vector with the predictions of each task
  at its own new design, and [`dof`](@ref) the number of nonzero coefficients
  plus ``q`` with an intercept.

[`MultitaskLassoPath`](@ref)`(Xs, ys)` solves along a decreasing sequence of
levels, each warm started from the previous solution, with the sequential
strong rule on the rows as in [`MultiLassoPath`](@ref). By default it has
`nλ = 100` levels spaced evenly on the log scale from
``\lambda_{\max}(1 + \sqrt{\varepsilon})``, where

```math
\lambda_{\max} = \max_j \frac{1}{\omega_j}\,\Big\|\Big(\frac{c_t\,\tilde x_{tj}^\top \tilde y_t}{n_t\,s_{tj}}\Big)_{t:\,s_{tj} > 0}\Big\|_2
```

is the smallest level whose solution is ``B = 0`` (``\tilde x_{tj}`` and
``\tilde y_t`` are centered with an intercept), down to `λminratio` times
``\lambda_{\max}``, with `λminratio` 0.01 when ``N < pq`` and ``10^{-4}``
otherwise: the grid of [`LassoPath`](@ref) for one task and of `MultiLassoPath`
for a shared design. `MultitaskLassoPath(Xs, ys, λpath)` takes levels of your
own, and `max_rows` stops the path after the first level with more nonzero
rows. `coef(path)` gives a matrix per level, `coef(path, λ)` one matrix,
interpolated in ``\lambda``, and `predict(path, Xs_new)` a vector of ``q``
matrices, of size ``n_t \times L`` for ``L`` levels. `fit(MultitaskLassoPath, Xs, ys)`
and `fit(CVMultitaskLassoPath, Xs, ys)` are the constructors.

[`CVMultitaskLassoPath`](@ref)`(Xs, ys)` chooses the level by ``K``-fold
cross-validation (`nfolds = 10`), with the folds drawn within each task: the
observations of each task are assigned to the ``K`` folds in a balanced way, as
[`CVLassoPath`](@ref) assigns those of one data set, so that every fold holds
out part of every task. `folds` takes one vector of labels ``1, \dots, K`` per
task instead, with every label used in every task. The path on all the data
gives the levels, and each training fold refits the same procedure on its own
observations: the centering, the scales, the `:observations` weights and the
stacked design are computed again, so nothing leaks from the held-out part.
Every training fold must keep at least two observations of each task with an
intercept, and one without. The held-out error of fold ``k`` at a level is

```math
e_k = \frac{1}{w_k}\sum_{t=1}^q \frac{c_t}{n_t}\sum_{i \in \mathrm{test}(t, k)} (y_{ti} - \hat y_{ti})^2, \qquad
w_k = \sum_{t=1}^q \frac{c_t\, m_{tk}}{n_t},
```

with ``m_{tk}`` the number of observations of task ``t`` in fold ``k``, and
``c_t``, ``n_t`` those of the full data. `cvm` averages the ``e_k`` with the weights ``w_k``,
and `cvse`, `λmin` and `λ1se` follow as for `CVLassoPath`; a fold path that
stops early leaves the remaining levels out of the averages. At the levels
every fold reaches, `cvm` is ``\sum_t c_t \mathrm{MSE}_t / \sum_t c_t``, the
held-out counterpart of the training loss, where ``\mathrm{MSE}_t``, the
held-out mean squared error of task ``t``, is `cvm_task[t, :]`. With one task
and the same folds, `cvm`, `cvse`, `λmin` and `λ1se` are those of
`CVLassoPath(X, y; intercept = true)`. `nrows` counts the nonzero
rows of `path` at each level, and `coef(cv)` and `predict(cv, Xs_new)` use the
solution at `λ1se`, or at `λmin` with `λ = :λmin`. The folds run in parallel on
the available threads, each with a generator seeded from `rng`, and the result
does not depend on the number of threads. The example on multitask regression
fits three tasks of different sizes and compares the selection with separate
lasso fits.
