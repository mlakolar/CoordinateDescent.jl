# CoordinateDescent.jl

[![Build Status](https://travis-ci.org/mlakolar/CoordinateDescent.jl.svg?branch=master)](https://travis-ci.org/mlakolar/CoordinateDescent.jl) [![codecov](https://codecov.io/gh/mlakolar/CoordinateDescent.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mlakolar/CoordinateDescent.jl)

Implements coordinate descent for a smooth function plus penalty that decomposes across coordinates. 

Curently a naive version of the active-set coordinate descent is implemented that works for L1 and weighted L1 penalty.

Examples:
* Lasso
* Sqrt-Lasso
* Scaled-Lasso

Package depends on [ProximalBase.jl](https://github.com/mlakolar/ProximalBase.jl)
