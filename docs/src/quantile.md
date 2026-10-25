# Primal–dual quantile regression

The exact fit keeps the observation-level dual variables needed by research
procedures. Install and load the optional HiGHS backend, then fit one penalty or
a decreasing grid:

```julia
using HighDimensionalModels, HiGHS
sol = quantile_lasso(X, y, 0.05; tau=0.5)
path = QuantileLassoPath(X, y, [0.2, 0.1, 0.05]; tau=0.5)
B = coef(path)                  # coefficient columns, intercept first
U = dual_scores(path)           # n observations × penalty levels
xi = qr_dual(path)              # QR.jl's row-dual sign and normalization
s = path[2]
```

`dual_scores(s)` returns the raw scores in `[tau-1,tau]`. With an intercept they
sum to zero, and `X'u/n` satisfies the weighted lasso stationarity conditions.
At nonzero residuals the score is determined by the sign; at zero residuals
preserve the optimizer's value. Different optimal bases can return different
valid duals. Compare their KKT conditions and the quantities your inference uses,
not necessarily the vectors entry by entry in a degenerate problem.

QR.jl used the equality `y - a - Xβ == rp - rn` and divided its loss by n.
`qr_dual(s)` therefore returns `-dual_scores(s)/n`, without projection. QR.jl
stored whole-path `xi` with levels in rows; transpose the new matrix for that
layout. The compatibility test compares against an independent primal LP with
that equality orientation. The historical JuMP/Gurobi binary stack is not used.

Each fit also has `certificate_dual`, a distinct feasible vector for a numerical
lower bound. Do not substitute it for the raw dual in inference. `duality_gap`
uses the raw pair and requires checking `dual_feasibility`; `certificate_gap`
uses the feasible vector. With zero penalties or zero loadings the conservative
certificate is zero and can be loose. `converged` is a numerical KKT/gap check;
`native_converged` records the optimizer flag separately.

The exact backend reuses its basis along the grid. Penalty factors can be zero
for unpenalized slopes; explicit λ=0 fits are supported. Standardization changes
penalty loadings, not the returned coefficient scale. An offset is supported;
observation weights and formula syntax are deferred. A finite `time_limit`
returns a partial path if needed. Check `complete`, `status` and each fit's
`converged` before using it. Saved dual vectors are independent snapshots.

A smoothed path uses the production proximal Newton implementation:

```julia
smooth = QuantileLassoPath(X, y, [0.2, 0.1, 0.05];
                          method=:smooth, bandwidth=0.2)
```

Its dual vectors are Epanechnikov convolution scores. They have the same scaling
interface but are not automatically interchangeable with exact quantile scores
in an inference procedure. The explicit bandwidth is not a statistical default
recommendation. Pivotal penalty calibration, CV, inference and full parametric
breakpoint enumeration remain separate work. Quantile paths do not linearly
interpolate coefficients or duals between saved levels.
