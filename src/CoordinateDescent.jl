"""
    CoordinateDescent

Lasso estimators computed by coordinate descent: [`lasso`](@ref),
[`sqrtLasso`](@ref), [`scaledLasso`](@ref), [`feasibleLasso!`](@ref),
[`LassoPath`](@ref), and the varying-coefficient estimators [`locpoly`](@ref)
and [`locpolyl1`](@ref). They run the coordinate descent, the penalty paths and
the reweighting of ProximalOPT.jl on the losses and penalties of
ProximalBase.jl, as does [`coordinateDescent!`](@ref), which minimizes
`f(x) + g(x)` for a loss `f` (a [`CoordinateDifferentiableFunction`](@ref) or a
loss that ProximalOPT knows) and a penalty `g` that separates across
coordinates, such as the weighted ℓ1 norm `NormL1`. The lasso solvers accept
real data and penalties of any type and convert them to a common floating point
type: the element type of `X` and `y` promoted to a float, or that of the
iterate passed to `scaledLasso!` and `feasibleLasso!`.
"""
module CoordinateDescent

using ProximalBase
using ProximalOPT
using ProximalOPT: CDModel, CDResults, ReweightResults, residuals
using DataStructures: nlargest
using SparseArrays
using Statistics
using LinearAlgebra

export
  lasso, sqrtLasso, feasibleLasso!, scaledLasso, scaledLasso!,
  LassoPath, refitLassoPath,
  IterLassoOptions,

  # CD
  CoordinateDifferentiableFunction,
  CDLeastSquaresLoss, CDWeightedLSLoss, CDQuadraticLoss, CDSqrtLassoLoss,
  CDOptions, CDInfo,
  coordinateDescent!,

  # var coef
  GaussianKernel, SmoothingKernel, EpanechnikovKernel, evaluate, createKernel,
  locpoly, locpolyl1

include("utils.jl")
include("losses.jl")
include("coordinate_descent.jl")
include("lasso.jl")
include("varying_coefficient_lasso.jl")

end
