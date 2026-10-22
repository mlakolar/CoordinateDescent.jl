# CoordinateDescent.jl

[![CI](https://github.com/mlakolar/CoordinateDescent.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/mlakolar/CoordinateDescent.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![codecov](https://codecov.io/gh/mlakolar/CoordinateDescent.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/CoordinateDescent.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Implements coordinate descent for a smooth function plus a penalty that decomposes across coordinates.

Currently a naive version of the active-set coordinate descent is implemented that works for the L1 and weighted L1 penalty.

Examples:
* Lasso
* Sqrt-Lasso
* Scaled-Lasso
* Feasible (heteroscedastic) Lasso
* Lasso path and refitting
* Varying-coefficient models via local polynomial regression with an L1 penalty

```julia
using CoordinateDescent, ProximalBase

n, p = 200, 50
X = randn(n, p)
y = X[:, 1:5] * ones(5) + randn(n)

sol = lasso(X, y, 0.1)          # minimize ‖y - Xβ‖² / (2n) + 0.1 ‖β‖₁
sol.x                           # sparse coefficient vector
sol.converged, sol.iterations   # stopping rule met?, passes over the coordinates

# an unpenalized intercept (sol.intercept) and penalties scaled by the
# standard deviations of the columns
sol = lasso(X, y, 0.1; intercept=true, standardizeX=true)

# solutions along 50 values of λ, from the smallest one with a zero solution
path = LassoPath(X, y; nλ=50, intercept=true)
path.λpath, path.βpath, path.β0path

# lower-level interface
x = SparseIterate(p)
f = CDLeastSquaresLoss(y, X)
g = ProxL1(0.1)
info = CDInfo()
coordinateDescent!(x, f, g, CDOptions(; optTol=1e-8), info)
info.converged, info.iterations
```

Coordinate descent stops when no coordinate changes by more than `optTol` in a
pass, the change of coefficient `k` measured in units of `‖y‖ / ‖X[:, k]‖`, so
the result does not depend on the units of the data (`CDOptions(; relativeTol =
false)` measures it in the units of the coefficients). It also stops after
`maxIter` passes (default 100 000), and then warns that the solution may be
inaccurate.

Package depends on [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl).
