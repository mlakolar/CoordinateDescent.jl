"""
    CoordinateDescent

Coordinate descent for problems of the form

    minimize  f(x) + g(x)

where `f` is a smooth loss (a [`CoordinateDifferentiableFunction`](@ref)) and
`g` is a penalty that separates across coordinates, such as the weighted ℓ1
norm `ProxL1` from ProximalBase.jl. The solver works on sparse iterates and
alternates between full passes and passes over the active set.

High-level interfaces: [`lasso`](@ref), [`sqrtLasso`](@ref),
[`scaledLasso`](@ref), [`feasibleLasso!`](@ref), [`LassoPath`](@ref), and the
varying-coefficient estimators [`locpoly`](@ref) and [`locpolyl1`](@ref). The
lasso solvers accept real data and penalties of any type and convert them to a
common floating point type: the element type of `X` and `y` promoted to a
float, or that of the iterate passed to `scaledLasso!` and `feasibleLasso!`.
"""
module CoordinateDescent

using ProximalBase
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
  CDLeastSquaresLoss, CDWeightedLSLoss, CDQuadraticLoss, CDSqrtLassoLoss, CDSqrtQuadraticLoss,
  CDOptions, CDInfo,
  coordinateDescent!,

  # var coef
  GaussianKernel, SmoothingKernel, EpanechnikovKernel, evaluate, createKernel,
  locpoly, locpolyl1

include("utils.jl")
include("atom_iterator.jl")
include("cd_differentiable_function.jl")
include("anderson.jl")
include("momentum.jl")
include("coordinate_descent.jl")
include("lasso.jl")
include("varying_coefficient_lasso.jl")

end
