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
xi = -U / nobs(path)            # QR.jl's row-dual sign and normalization
s = path[2]
```

At a valid solution, `dual_scores(s)` lies in `[tau-1,tau]`. A stopped exact
fit can retain an infeasible iterate with `converged=false`; its raw scores are
kept for diagnostics. With an intercept, valid scores sum to zero, and `X'u/n`
satisfies the weighted lasso stationarity conditions.
At nonzero residuals the score is determined by the sign; at zero residuals
preserve the optimizer's value. Different optimal bases can return different
valid duals. Compare their KKT conditions and the quantities your inference uses,
not necessarily the vectors entry by entry in a degenerate problem.

QR.jl used the equality `y - a - Xβ == rp - rn` and divided its loss by n.
Convert with `-dual_scores(s)/nobs(s)`, without projection. QR.jl
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

At each penalty, proximal Newton builds a weighted least-squares approximation
to the smooth loss and solves its lasso subproblem by coordinate descent.
Backtracking accepts a step that does not increase the penalized objective.
The path uses warm starts, sequential strong screening and subsequent checks
of the excluded coordinates. With a ProximalSolvers backend supporting convolution
certificates, `tol` also sets the gap target, together with checks of the raw
scores during solving. A smaller explicit `options.gaptol` is honored.
Flat-tail curvature damping and absolute inner stopping reduce cancellation
from large working responses; they change the quadratic step, not the objective.
`options=CDOptions(gaptol=0)` selects the legacy objective-change route.
Older solver versions retain their original stopping behavior and receive the
same independent post-fit checks. On that route, external `tol` marks acceptance
without automatically refining a rejected fit.

The backend conservatively retains inner convergence failures. A returned pair
can therefore have `converged=true` after passing the external numerical checks
while `native_converged=false` and inner warnings remain visible.

Its dual vectors are Epanechnikov convolution scores. They have the same scaling
interface but are not automatically interchangeable with exact quantile scores
in an inference procedure. The explicit bandwidth is not a statistical default
recommendation. Pivotal penalty calibration, CV, inference and full parametric
breakpoint enumeration remain separate work. Quantile paths do not linearly
interpolate coefficients or duals between saved levels.
