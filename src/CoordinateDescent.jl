"""
    CoordinateDescent

Lasso estimators computed by coordinate descent: [`lasso`](@ref),
[`sqrtLasso`](@ref), [`scaledLasso`](@ref), [`feasibleLasso!`](@ref),
[`LassoPath`](@ref), the penalized generalized linear models [`glmLasso`](@ref) and
[`GLMLassoPath`](@ref), the group lasso [`groupLasso`](@ref) and
[`GroupLassoPath`](@ref), the multi-response lasso [`multiLasso`](@ref) and
[`MultiLassoPath`](@ref), the inference after selection of R's hdm
([`rlasso`](@ref), [`postLasso`](@ref), [`doubleSelection`](@ref)), and the
varying-coefficient estimators [`locpoly`](@ref)
and [`locpolyl1`](@ref). They run the coordinate descent, the penalty paths and
the reweighting of ProximalSolvers.jl on the losses and penalties of
ProximalBase.jl, as does [`coordinateDescent!`](@ref), which minimizes
`f(x) + g(x)` for a loss `f` (a [`CoordinateDifferentiableFunction`](@ref) or a
loss that ProximalSolvers knows) and a penalty `g` that separates across
coordinates, such as the weighted ℓ1 norm `NormL1`. The lasso solvers accept
real data and penalties of any type and convert them to a common floating point
type: the element type of `X` and `y` promoted to a float, or that of the
iterate passed to `scaledLasso!` and `feasibleLasso!`.
"""
module CoordinateDescent

using ProximalBase
using ProximalSolvers
using ProximalSolvers: CDModel, CDResults, ReweightResults, SqrtQuadModel
using StatsAPI: StatsAPI, aic, bic, coef, confint, deviance, dof, fit, fitted, nobs, nulldeviance, predict, residuals,
  stderror, vcov
using Random: AbstractRNG, Xoshiro, default_rng, randperm
using SparseArrays
using SpecialFunctions: erfc, erfcinv
using Statistics
using LinearAlgebra

export
  lasso, sqrtLasso, feasibleLasso!, scaledLasso, scaledLasso!,
  LassoPath, refitLassoPath,
  adaptiveLasso,
  IterLassoOptions,

  # the StatsAPI interface of the solutions
  coef, predict, residuals, nobs, dof, deviance, fit, fitted, nulldeviance, stderror, vcov, confint,

  # cross-validation and information criteria along a path
  CVLassoPath,
  aic, bic, ebic,

  # CD
  CoordinateDifferentiableFunction,
  CDLeastSquaresLoss, CDWeightedLSLoss, CDQuadraticLoss, CDSqrtLassoLoss, CDSqrtQuadraticLoss,
  CDOptions, CDInfo,
  coordinateDescent!, solve!,

  # generalized linear models
  glmLasso,
  GLMLassoPath,

  # the group lasso
  groupLasso,
  GroupLassoPath,

  # the multi-response lasso
  multiLasso,
  MultiLassoPath,

  # inference after selection, as in R's hdm
  theoryLambda,
  rlasso,
  postLasso,
  doubleSelection,

  # var coef
  GaussianKernel, SmoothingKernel, EpanechnikovKernel, evaluate, createKernel,
  locpoly, locpolyl1

include("utils.jl")
include("losses.jl")
include("coordinate_descent.jl")
include("lasso.jl")
include("penalties.jl")
include("glm.jl")
include("group.jl")
include("multiresponse.jl")
include("cv.jl")
include("hdm.jl")
include("statsapi.jl")
include("varying_coefficient_lasso.jl")

include("precompile.jl")

end
