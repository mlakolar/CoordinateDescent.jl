# Varying-coefficient models

In a varying-coefficient model the coefficients change with an index variable
``z``, such as time or age:

```math
y_i = \sum_{j=1}^p X_{ij}\, \beta_j(z_i) + \varepsilon_i.
```

[`locpoly`](@ref) estimates ``\beta(z_0)`` at a point ``z_0`` by local polynomial
regression. Each ``\beta_j(z)`` is approximated by a polynomial of degree ``q``
in ``z - z_0``, which expands the design to
``\tilde X_i = X_i \otimes (1, z_i - z_0, \dots, (z_i - z_0)^q)``, and the
observations are weighted by a kernel ``K_h(z_i, z_0)`` with bandwidth ``h``,
a [`GaussianKernel`](@ref) or an [`EpanechnikovKernel`](@ref). The
coefficients of variable ``j`` are its value and its derivatives at ``z_0``, up
to factorials.

[`locpolyl1`](@ref) also selects the variables. At each point ``z_0`` of a grid
it solves the weighted lasso

```math
\underset{\beta}{\text{minimize}} \quad \frac{1}{2n} \sum_{i=1}^n w_i (y_i - \tilde X_i^\top \beta)^2 + \lambda_0 \sum_k s_k |\beta_k|, \qquad w_i = K_h(z_i, z_0),
```

with loadings ``s_k = \sqrt{\sum_i w_i \tilde X_{ik}^2 / n}`` that standardize the
expanded columns, warm started from the solution at the previous point, and it
can refit weighted least squares on the selected variables.

The bandwidth can be chosen by leave-one-out cross-validation,
[`CoordinateDescent.lvocv_locpoly`](@ref) and
[`CoordinateDescent.lvocv_locpolyl1`](@ref), or on a validation set,
[`CoordinateDescent.split_locpoly`](@ref).
[`CoordinateDescent.getStandardError`](@ref) and
[`CoordinateDescent.getStandardErrorHEW`](@ref) give standard errors of the
estimates. The example on varying-coefficient models goes through them.
