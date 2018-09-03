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
  CDLeastSquaresLoss, CDWeightedLSLoss, CDQuadraticLoss, CDSqrtLassoLoss,
  CDOptions,
  coordinateDescent, coordinateDescent!,

  # var coef
  GaussianKernel, SmoothingKernel, EpanechnikovKernel, evaluate, createKernel,
  locpoly, locpolyl1

include("utils.jl")
include("atom_iterator.jl")
include("cd_differentiable_function.jl")
include("coordinate_descent.jl")
include("lasso.jl")
include("varying_coefficient_lasso.jl")

end
