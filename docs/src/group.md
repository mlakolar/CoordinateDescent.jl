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
which selects rows across responses (glmnet's `mgaussian`). Entries within a
selected row can still be zero.
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

the row penalty of the multi-response lasso, which selects predictors across
tasks while allowing zero entries within selected rows (Obozinski, Taskar and
Jordan, 2010). The coefficients are returned on the original scale.

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
  grid below, and hence cross-validation, needs positive factors.
- **Intercepts.** Each task has an unpenalized intercept
  ``\beta_{0t} = \bar y_t - \bar x_t^\top b_t`` (`intercept = true`, the
  default); with `intercept = false` they are zero and nothing is centered.
- **Reductions.** With one task, the problem is [`lasso`](@ref)`(X, y, λ)` with
  the same `intercept` and `standardize_x` (whose defaults in `lasso` are
  `false`), and with a weight ``c`` it is the lasso at ``\lambda / c``. When
  every task has the same design ``X`` and all the weights are ``c``,
  ``s_{tj} = s_j`` and, with the default penalty factors, the problem is
  [`multi_lasso`](@ref)`(X, Y, λ / c)` with ``Y = [y_1 \cdots y_q]``; a dense
  shared design is fitted with the model of `multi_lasso`. Its `nobs` is then
  ``qn``, where `multi_lasso` reports ``n``.
- **Method.** The tasks are centered and standardized each on its own, the
  rows of task ``t`` are scaled by ``\sqrt{N c_t / n_t}``, and the blocks are
  placed on the diagonal of a sparse ``N \times pq`` design over
  ``\operatorname{vec}(B)``, whose groups are the rows of ``B``; group
  coordinate descent solves the least-squares problem on it, with coordinate
  orders drawn from `rng`. The columns of different tasks are orthogonal, so
  the curvature over a row is diagonal. With standardization and equal weights
  it is a multiple of the identity and each block step is exact; otherwise a
  step minimizes a majorization, and convergence takes more passes. Momentum
  is on by default here: each pass over the active rows starts from a point
  extrapolated from the last two, which saves passes at small penalties.
  `momentum = false` or `order = :cyclic` turns it off (see
  [`CDOptions`](@ref)). A dense shared design keeps the options as
  `multi_lasso` takes them, without momentum unless `momentum = true`. In
  [`MultitaskLassoPath`](@ref), and so in cross-validation, the block of a
  task with more rows ``n_t`` than fitted columns ``p_t``, with ``p_t`` at
  most 2000 and at least half of the block stored, is replaced by the
  ``p_t \times p_t`` triangular factor ``R_t`` of its QR decomposition
  ``Q_t R_t``, and its rows of the response by the first ``p_t`` entries of
  ``Q_t^\top \tilde y_t``; the norm of the other entries goes to one extra
  row. The least-squares loss is the same for every ``B``, so the solutions,
  the duality gaps and the stopping rule are unchanged up to rounding, while a
  pass reads at most ``p_t(p_t + 1)/2`` entries of the block instead of up to
  ``n_t p_t``. The QR costs about ``2 n_t p_t^2`` operations before the first
  level; a single fit keeps the stacked design.
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
``c_t``, ``n_t`` those of the full data. `cvm` averages the ``e_k`` with the
weights ``w_k``, and `cvse`, `λmin` and `λ1se` follow as for `CVLassoPath`. A
fold path that stops early leaves the remaining levels out of the averages;
nonconverged fold levels are also left out and cannot be selected,
and the levels that no fold reaches have `NaN` in `cvm`, `cvse` and
`cvm_task` and are not candidates for `λmin` and `λ1se`. At the levels
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

## Other multitask row penalties

`multitask_lasso`, `MultitaskLassoPath`, `CVMultitaskLassoPath`, `multi_lasso`
and `MultiLassoPath` accept `row_norm = :l2` and `alpha = 0` by default. They
penalize one coefficient matrix. Write ``U_{jt} = s_{tj} B_{jt}``, with the
scales and task weights defined above. The objective is

```math
\sum_t \frac{c_t}{2n_t}\|y_t-\beta_{0t}-X_tb_t\|^2 + \lambda\sum_j h_j(U_{j\cdot}),
\qquad
h_j(u)=\begin{cases}
\omega_j\|u\|_\infty, & \texttt{row\_norm = :linf},\\
\alpha\|u\|_1+(1-\alpha)\omega_j\|u\|_2, & \texttt{row\_norm = :l2}.
\end{cases}
```

`alpha` must lie in ``[0,1]``; `row_norm = :linf` requires `alpha = 0`.
The default is the row ℓ1/ℓ2 penalty. `row_norm = :linf` gives row ℓ1/ℓ∞,
which often has several entries tied at the largest absolute value in a row.
With `row_norm = :l2, alpha > 0`, the sparse group penalty combines row
selection with entrywise sparsity. At `alpha = 1` the tasks are separate
lassos at ``\lambda/c_t``. Row selection allows zero entries within a selected
row for all three penalties; it does not force each selected predictor to
have a nonzero coefficient in every task.

The factors ``\omega_j`` weight the row term only. A zero factor under the
sparse group leaves the entrywise term ``\alpha\lambda\|u\|_1`` in place.
The automatic grid needs finite ``\lambda_{\max}``; unpenalized rows that
have a nonzero gradient at zero require explicit levels. The `multi_lasso`
family has no penalty factors or task weights: it uses one shared design,
``c_t=1``, and loss ``\|Y-1\beta_0^\top-XB\|_F^2/(2n)``. Its `nobs` is ``n``.

Let ``g_j`` be the gradient of the loss with respect to row ``U_{j\cdot}``,
restricted to coordinates with ``s_{tj}>0``. At zero coefficients, with the
intercepts profiled, the thresholds are

```math
\lambda_{\max}^{(\infty)}=\max_j\frac{\|g_j(0)\|_1}{\omega_j},\qquad
\lambda_{\max}^{(\mathrm{sg})}=\max_j\rho_j,
\quad \|\operatorname{soft}(g_j(0),\alpha\rho_j)\|_2=(1-\alpha)\omega_j\rho_j.
```

The first maximum uses positive factors; a zero-factor row with a nonzero
correlation has no finite threshold. For sparse group with ``\omega_j=0``
and ``\alpha>0``, ``\rho_j=\|g_j(0)\|_\infty/\alpha``. The roots are computed
exactly between the breakpoints of soft thresholding.

A zero row is optimal when ``\|g_j\|_1\le\lambda\omega_j`` for ℓ∞, or
``\|\operatorname{soft}(g_j,\alpha\lambda)\|_2\le(1-\alpha)\lambda\omega_j``
for sparse group. On a nonzero ℓ∞ row, ``\|g_j\|_1=\lambda\omega_j``; the
gradient is supported on entries of maximal absolute coefficient, with
opposite signs. On a nonzero sparse group row,

```math
0\in g_j+(1-\alpha)\lambda\omega_j\frac{U_{j\cdot}}{\|U_{j\cdot}\|_2}
       +\alpha\lambda\,\partial\|U_{j\cdot}\|_1.
```

Each task's residuals sum to zero when intercepts are fitted. Screening uses
the sequential strong rule: a row is kept if its gradient's dual norm at the
previous solution is at least ``2\lambda_k-\lambda_{k-1}``. For sparse group
that norm is the root above. Every excluded row is checked against its zero-row
optimality condition after solving; violators are added and the level is
solved again. `kkt_checks` records these rounds.

The row ℓ1/ℓ∞ and sparse-group penalties use row coordinate descent on the
tasks' own designs.
No stacked design is built, and a shared design is stored once. The solver
keeps per-task residuals and centered column squared norms. A row uses the
majorizing curvature
``L_j=\max_{t:s_{tj}>0} c_t\|\tilde x_{tj}\|^2/(n_t s_{tj}^2)`` and a proximal
step of size ``1/L_j``. With standardized columns and equal weights, this
step is exact. A full pass costs ``O(p\sum_t n_t)`` for dense designs, plus
``O(pq\log q)`` for the ℓ∞ proximal steps (``O(pq)`` for sparse group).
The residual cache costs ``O(\sum_t n_t)`` and column/coefficient caches
``O(pq)``. Sparse designs are centered on the fly, including with intercepts,
without densifying them.

Full passes alternate with passes on active rows, in cyclic order or random
order from `rng`. A full pass confirms convergence. With `relative_tol`,
each change ``\Delta U_{jt}`` is measured in fitted-value units as
``|\Delta U_{jt}|\sqrt{a_{tj}}``, where
``a_{tj}=c_t\|\tilde x_{tj}\|^2/(n_t s_{tj}^2)``. The largest such change is
compared with ``\texttt{opt\_tol}\,r``, with response scale
``r=\sqrt{\sum_t c_t\|\tilde y_t\|^2/n_t}``. Columns and responses are centered
with an intercept. The scale has a positive numerical floor; when ``r=0``,
zero coefficients solve the problem exactly. Without `relative_tol`,
`opt_tol` bounds the absolute change of ``U``.
A positive `gaptol` also requires a relative duality gap below that target.
Paths default to `max(ProximalSolvers.PATH_GAPTOL, 64eps(T))`; single fits
require the gap only with an explicit positive `gaptol`, except continuation
with `warm_start = false`, which uses the path default. `gaptol = 0` uses the
change rule alone. The dual point scales the weighted residuals into all row
dual balls. No gap is used at ``\lambda=0``, with a completely unpenalized
row, or when the primal objective is zero. Reaching `max_iter` passes gives
`converged = false`. The default ℓ1/ℓ2 solver keeps its existing behavior.

[`sparse_group_lasso`](@ref) has one response and user-defined groups of
predictors. Here a group is one predictor across tasks, with a common entrywise
penalty. A dirty model decomposes ``B = A + S`` with different penalties on the
two components; it is not one of these penalties.

### Cross-validation with a shared design

[`CVMultiLassoPath`](@ref)`(X, Y)` uses the same observation folds for all
responses. It accepts positional `CDOptions`, `nfolds = 10`, explicit `folds`,
`rng`, `threads`, `nλ`, `λminratio`, and the keywords of `MultiLassoPath`.
Every training fold refits centering, scales and coefficients on its own rows
at the full-data grid. If fold ``k`` has ``m_k`` observations, its error is
``e_k=\sum_{i\in k,t}(Y_{it}-\hat Y_{it})^2/(m_kq)``. `cvm` averages these
errors with weights ``m_k``; `cvse` is their standard error, as for
`CVLassoPath`. `cvm_response` holds the held-out MSE for each response.
With one response and the same folds, fits and standardization, these summaries
and selected levels agree with `CVLassoPath(X, y; intercept = true,
standardize_x = true)`.

Both multitask CV types treat a nonconverged fold level as an unreached level:
its fold error is `NaN`, and it is never selected. Means use the available
folds; `λmin` and `λ1se` compare levels that converged in every fold.
`coef(cv)` and `predict(cv, Xnew)` use `λ1se`, or `λmin` with `λ = :λmin`.
Fold generators are seeded separately from `rng`, so threading does not change
the results. See `examples/13_multitask_penalties.jl` for a comparison of the
three penalties on shared and task-specific supports.

## The dirty multitask model

[`dirty_multitask`](@ref) fits the model of
[Jalali, Ravikumar, Sanghavi and Ruan (2010)](https://users.ece.utexas.edu/~sanghavi/mtdm_final.pdf).
It estimates a total coefficient matrix ``B=A+S`` with separate penalties on
its components:

```math
\min_{A,S,\beta_0}\;
\sum_t\frac{c_t}{2n_t}\|y_t-\beta_{0t}-X_t(a_t+s_t)\|^2
+\sum_j\omega_j\bigl(\lambda_b\|\widetilde A_{j\cdot}\|_\infty
                         +\lambda_s\|\widetilde S_{j\cdot}\|_1\bigr).
```

The arguments `λ_shared` and `λ_specific` are the positional levels
``\lambda_b`` and ``\lambda_s``. The default `shared_norm = :linf` is the
paper's shared norm; `shared_norm = :l2` is a variant. Designs may be shared
(`dirty_multitask(X, Y, λb, λs)`) or task-specific (`Xs, ys`). The intercepts,
scales and task weights follow `multitask_lasso`. Write
``E_j=\{t:s_{tj}>0\}``, ``q_j=|E_j|`` and
``v=(s_{tj}B_{jt})_{t\in E_j}``. Both components stay zero outside ``E_j``.
With `task_weights = :equal`, each task has loss ``\|r_t\|^2/(2n_t)``.

### Reduction and computation

The loss depends on the total ``B``. Minimizing over its split gives the row
penalty and its dual ball

```math
h_j(v)=\omega_j\min_{a+s=v}
       \{\lambda_b\|a\|+\lambda_s\|s\|_1\},\qquad
C_j=\omega_j\{z:\|z\|_*\le\lambda_b,\ \|z\|_\infty\le\lambda_s\},
```

where the dual norm ``\|\cdot\|_*`` is ℓ1 for the shared ℓ∞ norm and ℓ2 for
the shared ℓ2 norm. Equivalently, ``h_j(v)=\sup_{z\in C_j}\langle z,v\rangle``.
For ℓ∞, this is the sorted ℓ1 penalty

```math
h_j(v)=\omega_j\sum_{i=1}^{q_j}
\min\{\lambda_s,\max(\lambda_b-(i-1)\lambda_s,0)\}\,|v|_{(i)},
\qquad |v|_{(1)}\ge\cdots\ge|v|_{(q_j)}.
```

ProximalBase's `SLOPE` computes its value and proximal step, with buffers
owned by each fit. The ℓ2 variant uses the proximal step ``u-\Pi_{C_j}(u)``.
Projection onto the intersection of the ℓ2 ball of radius ``c`` and the ℓ∞
ball of radius ``d`` has entries
``z_i=\operatorname{clip}(u_i/(1+\mu),-d,d)``. Here ``\mu=0`` if clipping is
feasible, otherwise its positive root enforces ``\|z\|_2=c``. Zero radii are
handled directly. Both variants use the row solver on the tasks' designs,
including sparse designs with intercepts, without a stacked design.

The penalty factors multiply the **whole** row penalty. Thus ``\omega_j=0``
leaves the row unpenalized. In the multitask sparse group above, a zero factor
leaves its entrywise penalty in place. The dirty model also differs from
`sparse_group_lasso`, which has one response and groups of predictors, and
from multitask sparse group, which applies both penalties to a single ``B``.
With one task, applying levels ``a,b`` to the same coefficient gives
``(a+b)|B|``; the dirty model gives ``\min(a,b)|B|``.

### Regimes and the returned components

For positive levels, let ``r=\lambda_b/\lambda_s``. On each row's fitted
coordinates the ℓ∞ estimator reduces to row ℓ1/ℓ∞ when ``r\le1`` and to the
entrywise lasso when ``r\ge q_j``. For ℓ2 the corresponding regimes are the
group lasso when ``r\le1`` and the entrywise lasso when ``r\ge\sqrt{q_j}``.
Factors do not change these regimes. Interior ratios allow both components;
they need not both be nonzero.

The returned split is computed in standardized coordinates and then divided
by the scales. It follows these conventions, in order:

- If ``\lambda_b=0``, return ``A=B``. If only ``\lambda_s=0``, return ``S=B``.
  Both zero gives ``A=B``. A zero level makes the row penalty zero.
- For ℓ∞, return ``A=0`` when ``r\ge q_j``. For ℓ2, return ``A=0`` when
  ``r\ge\sqrt{m_v}``, where ``m_v`` counts the nonzero entries of ``v``.
- Otherwise, return ``S=0`` when ``r\le1``.
- In the interior, return ``a_i=\operatorname{clip}(v_i,-\tau,\tau)`` and
  ``s=v-a``. For ℓ∞, ``\tau=|v|_{(\lceil r\rceil)}``. At an integer
  ``r=k``, every threshold in ``[|v|_{(k+1)},|v|_{(k)}]`` minimizes the row
  penalty; this rule takes ``|v|_{(k)}``, the most shared choice.
  For ℓ2, ``\tau`` is the unique positive root of
  ``\sum_i\min(|v_i|/\tau,1)^2=r^2``. Sorted intervals give
  ``\tau^2=\sum_{i>k}|v|_{(i)}^2/(r^2-k)`` on
  ``[|v|_{(k+1)},|v|_{(k)}]`` (upper endpoint infinite for ``k=0``).

An estimator's reduction, the existence of a pure optimal split, and the split
returned by this convention are different statements. For example, the ℓ2
rule can return a pure specific split for a particular row before the global
lasso regime. Components need not be unique even when total coefficients
are unique. `sharedrows` counts the nonzero rows of this canonical ``A``;
it does not identify shared features or a latent decomposition from the data.

### Paths, tuning and diagnostics

[`DirtyMultitaskPath`](@ref)`(Xs, ys; ratio = r)` holds ``r`` fixed and varies
``\lambda=\lambda_s``, with ``\lambda_b=r\lambda``. Its automatic grid starts
at ``\lambda_{\max}(1+\sqrt{\mathrm{eps}})``, where

```math
d_j(g)=\frac{\max\{\|g\|_*/r,\|g\|_\infty\}}{\omega_j},\qquad
\lambda_{\max}(r)=\max_j d_j(g_j(0)).
```

The gradient is on the fitted standardized coordinates, with intercepts
profiled. An unpenalized row with a nonzero gradient has infinite threshold
and needs explicit levels. Screening compares ``d_j`` with
``2\lambda_k-\lambda_{k-1}``, then checks every excluded row and repairs any
violations. Total coefficients warm start the next level; components are
split anew. `coef(path, λ)` requires a fitted level.

[`CVDirtyMultitaskPath`](@ref) compares ratios and levels by the held-out
score of `CVMultitaskLassoPath`. Defaults are five geometric ratios on
``[1,q]`` for ℓ∞ or ``[1,\sqrt q]`` for ℓ2. Per-task folds are common across
ratios; with a shared design, whole observation rows are held out across
responses. Each training fit refits centering, scales and task weights.
`cvm` is a `nratios × nλ` matrix, with `NaN` padding for shorter paths.
Selection requires convergence on the full data and every fold, minimizes
`cvm`, and breaks ties by larger ratio, then larger specific level. There is
no two-dimensional one-standard-error rule. `coef(cv)` and `predict(cv, Xnew)`
use the selected full-data fit.

A fit's `kkt` recomputes the two-component conditions at the returned point.
Writing ``z_j=-\nabla_j f(B)``, these are membership in ``C_j`` and

```math
\langle z_j,\widetilde A_{j\cdot}\rangle
 =\omega_j\lambda_b\|\widetilde A_{j\cdot}\|,\qquad
\langle z_j,\widetilde S_{j\cdot}\rangle
 =\omega_j\lambda_s\|\widetilde S_{j\cdot}\|_1,
```

together with intercept stationarity, zero dropped coordinates and ``B=A+S``.
`kkt` is the largest violation divided by the maximum of one and the
magnitudes of the terms being compared. `gap` is the **relative** duality gap,
using residual scaling ``\min(1,\lambda/\max_j d_j)``; it is `NaN` when
unavailable, including zero levels, unpenalized rows and zero objective.
Convergence requires the row solver's change rule and any requested available
gap target. `nobs` is ``\sum_t n_t``. `dof` counts nonzero entries in total
``B`` plus fitted intercepts; it is not effective degrees of freedom.
See `examples/15_multitask_dirty.jl` for a small comparison with row penalties
and separate lasso fits on shared and task-specific supports.
