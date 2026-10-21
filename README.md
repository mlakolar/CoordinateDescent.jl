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

# lower-level interface
x = SparseIterate(p)
f = CDLeastSquaresLoss(y, X)
g = ProxL1(0.1)
coordinateDescent!(x, f, g, CDOptions(; optTol=1e-8))
```

Package depends on [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl).
