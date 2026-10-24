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
using CoordinateDescent, ProximalBase, LinearAlgebra, Random

rng = Xoshiro(1)
n, p = 200, 50
X = randn(rng, n, p)
y = X[:, 1:5] * ones(5) + randn(rng, n)

groups = [1:5, 6:10, 11:20, 21:50]
g = GroupNormL2(0.1, groups)         # weights √|Gₖ|
x = AtomIterate(p, groups)
coordinateDescent!(x, CDLeastSquaresLoss(y, X), g)
[norm(x[G]) for G in groups]         # the first group is selected
```

All groups are zero for ``\lambda \ge \max_k \|X_{G_k}^\top y\| / (n w_k)``. With
`warmStart = false`, [`coordinateDescent!`](@ref) solves along a path of
penalty levels from there, and screens the groups with the strong rule. The
example on the group lasso chooses ``\lambda`` on a validation set and checks the
optimality conditions.

## The estimators

[`groupLasso`](@ref)`(X, y, λ, groups)` and [`GroupLassoPath`](@ref)`(X, y, groups)`
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

## The multi-response lasso

With several responses that share the predictors, [`multiLasso`](@ref)`(X, Y, λ)`
and [`MultiLassoPath`](@ref)`(X, Y)` fit

```math
\frac{1}{2n}\|Y - 1\beta_0^\top - XB\|_F^2 + \lambda\sum_k \omega_k\|B_{k\cdot}\|_2,
```

the group lasso on the rows of the ``p \times q`` coefficient matrix ``B``,
which selects the same predictors for every response (glmnet's `mgaussian`).
The curvature of the loss over a row is a multiple of the identity, so each
block step on a row is exact. The solutions report ``B`` as a sparse matrix,
the intercepts of the responses and the active rows.
