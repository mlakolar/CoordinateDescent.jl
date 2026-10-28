"""
    HighDimensionalModels

Lasso estimators computed by coordinate descent: [`lasso`](@ref),
[`sqrt_lasso`](@ref), [`scaled_lasso`](@ref), [`feasible_lasso!`](@ref),
[`LassoPath`](@ref), the penalized generalized linear models [`glm_lasso`](@ref) and
[`GLMLassoPath`](@ref), the group lasso [`group_lasso`](@ref) and
[`GroupLassoPath`](@ref), the multi-response lasso [`multi_lasso`](@ref) and
[`MultiLassoPath`](@ref), the multitask lasso with task-specific designs
[`multitask_lasso`](@ref), [`MultitaskLassoPath`](@ref) and
[`CVMultitaskLassoPath`](@ref), the inference after selection of R's hdm
([`rlasso`](@ref), [`post_lasso`](@ref), [`double_selection`](@ref)), and the
varying-coefficient models [`varying_coefficient`](@ref) and
[`varying_coefficient_lasso`](@ref), with the kernels of KernelSmoothing.jl. They run the coordinate descent, the penalty paths and
the reweighting of ProximalSolvers.jl on the losses and penalties of
ProximalBase.jl, as does [`coordinate_descent!`](@ref), which minimizes
`f(x) + g(x)` for a loss `f` (a [`CoordinateDifferentiableFunction`](@ref) or a
loss that ProximalSolvers knows) and a penalty `g` that separates across
coordinates, such as the weighted ℓ1 norm `NormL1`. Low-rank multivariate fits use [`reduced_rank`](@ref),
[`nuclear_regression`](@ref), [`sparse_reduced_rank`](@ref),
[`sparse_svd_regression`](@ref), and [`sparse_nuclear_regression`](@ref), with
[`LowRankPath`](@ref), [`CVLowRankPath`](@ref), [`PenaltyGrid`](@ref) and
[`CVPenaltyGrid`](@ref) for tuning. The lasso solvers accept
real data and penalties of any type and convert them to a common floating point
type: the element type of `X` and `y` promoted to a float, or that of the
iterate passed to `scaled_lasso!` and `feasible_lasso!`.

Penalties on ordered coefficients and graphs are fitted by [`fused_lasso`](@ref),
[`trend_filter`](@ref) and [`generalized_lasso`](@ref), using exact proximal
steps, proximal gradient or specialized ADMM rather than coordinate descent.
"""
module HighDimensionalModels

using ProximalBase
using ProximalSolvers
using ProximalSolvers: CDModel, CDResults, ReweightResults, SqrtQuadModel
using StatsAPI: StatsAPI, aic, bic, coef, coefnames, confint, deviance, dof, fit, fitted, nobs, nulldeviance, predict,
  residuals, responsename, stderror, vcov
using Random: AbstractRNG, Xoshiro, default_rng, randperm, randn
using SparseArrays
using SpecialFunctions: erfc, erfcinv
using Statistics
using LinearAlgebra

export
  lasso, sqrt_lasso, feasible_lasso!, scaled_lasso, scaled_lasso!,
  LassoPath, refit_lasso_path,
  adaptive_lasso,
  IterLassoOptions,

  # the StatsAPI interface of the solutions
  coef, predict, residuals, nobs, dof, deviance, fit, fitted, nulldeviance, stderror, vcov, confint,

  # models fitted from a formula (with StatsModels)
  FormulaFit, coefnames, responsename,

  # cross-validation and information criteria along a path
  CVLassoPath,
  aic, bic, ebic,

  # CD
  CoordinateDifferentiableFunction,
  CDLeastSquaresLoss, CDWeightedLSLoss, CDQuadraticLoss, CDSqrtLassoLoss, CDSqrtQuadraticLoss,
  CDOptions, CDInfo,
  coordinate_descent!, solve!,
  LLA,                 # ProximalSolvers' local linear approximation, for lla

  # primal–dual quantile regression
  quantile_lasso, QuantileLassoSolution, QuantileLassoPath, dual_scores,
  quantile_refit, QuantileRefit, refit_quantile_path,
  quantile_lambda, CVQuantileLassoPath,

  # generalized linear models
  glm_lasso,
  GLMLassoPath,
  CVGLMLassoPath,

  # the group lasso and the sparse group lasso
  group_lasso,
  GroupLassoPath,
  sparse_group_lasso,
  SparseGroupLassoPath,

  # the multi-response lasso
  multi_lasso,
  MultiLassoPath,

  # the multitask lasso with task-specific designs
  multitask_lasso, MultitaskLassoSolution, MultitaskLassoPath, CVMultitaskLassoPath,

  # penalties on ordered coefficients and graphs
  fused_lasso, generalized_lasso, trend_filter,
  difference_matrix, graph_difference,
  StructuredLassoSolution, GeneralizedLassoPath, CVGeneralizedLassoPath,
  fused_lasso_path, trend_filter_path, cv_fused_lasso, cv_trend_filter,
  # low-rank multivariate regression
  reduced_rank, select_rank, nuclear_regression, sparse_reduced_rank,
  sparse_svd_regression, LowRankSolution, LowRankPath, CVLowRankPath, lowrank_ic,
  sparse_nuclear_regression, SparseNuclearSolution, PenaltyGrid, CVPenaltyGrid,

  # inference after selection, as in R's hdm
  theory_lambda,
  rlasso,
  post_lasso,
  double_selection,

  # selection with error control
  slope, slope_bh, slope_gaussian, SLOPEPath, SLOPESolution,
  equicorrelated_s, gaussian_knockoffs, fixed_knockoffs,
  knockoff_threshold, lasso_coefdiff, knockoff_filter, KnockoffResult,
  knockoff_diagonal, KnockoffDiagonal,
  complementary_pairs, stability_bound, stability_selection, StabilitySelectionResult,

  # varying-coefficient models (kernels from KernelSmoothing.jl)
  varying_coefficient, varying_coefficient_lasso, varying_coefficient_cv, varying_coefficient_lasso_cv

include("utils.jl")
include("losses.jl")
include("coordinate_descent.jl")
include("lasso.jl")
include("penalties.jl")
include("glm.jl")
include("quantile.jl")
include("group.jl")
include("sparsegroup.jl")
include("multiresponse.jl")
include("cv.jl")
include("multitask.jl")
include("quantile_tuning.jl")
include("structured.jl")
include("structured_cv.jl")
include("lowrank.jl")
include("sparse_svd.jl")
include("lowrank_sparse.jl")
include("penalty_grid.jl")
include("hdm.jl")
include("statsapi.jl")
include("formula.jl")
include("varying.jl")

include("slope.jl")
include("knockoff_construction.jl")
include("knockoffs.jl")
include("stability.jl")

include("deprecated.jl")

include("precompile.jl")

end
