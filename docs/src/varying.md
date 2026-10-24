# Varying-coefficient models

In a varying-coefficient model the coefficients change with an index variable
``z``, such as time or age:

```math
y_i = \sum_{j=1}^p X_{ij}\, \beta_j(z_i) + \varepsilon_i.
```

A column of ones in ``X`` gives a varying intercept.

## Local polynomial fits

[`varying_coefficient`](@ref)`(X, z, y, zgrid; kernel, bandwidth, degree)`
estimates the coefficient functions at each point ``z_0`` of `zgrid` by local
polynomial regression. Near ``z_0`` each ``\beta_j`` is a polynomial of degree
``q`` in ``z - z_0``, and the polynomials are fitted by weighted least squares,

```math
\underset{\theta}{\text{minimize}} \quad \sum_{i=1}^n w_i K_h(z_i - z_0)\Big(y_i - \sum_{j=1}^p X_{ij}\sum_{r=0}^q \theta_{jr}(z_i - z_0)^r\Big)^2,
\qquad K_h(d) = k(d/h)/h,
```

with observation weights ``w_i``. Then ``r!\,\theta_{jr}`` estimates the derivative
``\beta_j^{(r)}(z_0)``: `coef(fit)` returns the estimates of ``\beta_j(z_0)``, and
`coef(fit; derivative = r)` those of the derivatives.

The kernel ``k`` is a function of the scaled distance ``u = (z_i - z_0)/h``: the
kernels of [KernelSmoothing.jl](https://github.com/mlakolar/KernelSmoothing.jl)
(`EpanechnikovKernel()`, `GaussianKernel()`, whose bandwidth is a standard
deviation, `UniformKernel()`, and others), or any function of ``u``. With a
compactly supported kernel only the observations within ``h`` of ``z_0`` enter;
when they do not determine the polynomials, the fit of minimum norm is taken
and the `rank` of the fit records it.

With `se = true`, the fit also gives the fitted values and residuals of the local
fits at the observations (one more local fit per observation), and
heteroskedasticity-robust (sandwich) standard errors, `stderror(fit)` and
`confint(fit)`. These describe the variance of the local
estimate, not its bias, so an interval covers ``\beta_j(z_0)`` when the
bandwidth is small enough for the bias to be negligible. `predict(fit, X, z)`
interpolates the estimates linearly between the points of the grid.

[`varying_coefficient_cv`](@ref) chooses the bandwidth by leave-one-out
cross-validation. The local fit is linear in ``y``, so the prediction of ``y_i``
without observation ``i`` comes from the fit with it,
``(\hat y_i - H_{ii} y_i)/(1 - H_{ii})``, at the cost of one local fit per
observation.

## Selecting the variables

[`varying_coefficient_lasso`](@ref)`(X, z, y, zgrid, λ; kernel, bandwidth)` adds a
penalty that selects, at each point, the variables whose coefficient functions
are not zero nearby:

```math
\underset{\theta}{\text{minimize}} \quad \frac{\sum_i v_i\big(y_i - \sum_j X_{ij}\sum_r \theta_{jr}((z_i - z_0)/h)^r\big)^2}{2\sum_i v_i} + \lambda\, P(\theta),
\qquad v_i = w_i K_h(z_i - z_0).
```

The loss is the mean squared error on the local sample, so ``\lambda`` is on the
scale of an ordinary lasso. With `penalty = :group`, the default, ``P`` is the
group lasso over the ``q + 1`` coefficients of each variable, on the fitted
values of its locally weighted columns (the orthonormalized groups of
[`group_lasso`](@ref)), so that a variable enters with all its polynomial terms.
With `penalty = :lasso`, ``P`` is the ℓ1 norm with the loadings of the
standardized local columns. With `refit`, the default, the selected variables are
refitted at each point by the unpenalized local fit, which removes the
shrinkage. [`varying_coefficient_lasso_cv`](@ref) chooses the bandwidth by
leave-one-out cross-validation, one local lasso per observation and bandwidth.

The example on varying-coefficient models goes through these functions.
