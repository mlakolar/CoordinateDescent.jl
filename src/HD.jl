module HD

using ProximalBase

export
  lasso, sqrtLasso,
  computeLassoPath, refitLassoPath,
  ScaledLassoOptions, scaledLasso, scaledLasso!,

  # CD
  CoordinateDifferentiableFunction,
  CDLeastSquaresLoss, CDQuadraticLoss, CDSqrtLassoLoss,
  CDOptions,
  coordinateDescent, coordinateDescent!



include("coordinate_descent.jl")
include("lasso.jl")


end
