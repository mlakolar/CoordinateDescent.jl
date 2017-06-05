module CoordinateDescent

using ProximalBase
using DataStructures: binary_maxheap

export
  lasso, sqrtLasso,
  computeLassoPath, refitLassoPath,
  ScaledLassoOptions, scaledLasso, scaledLasso!,

  # CD
  CoordinateDifferentiableFunction,
  CDLeastSquaresLoss, CDQuadraticLoss, CDSqrtLassoLoss,
  CDOptions,
  coordinateDescent, coordinateDescent!


include("utils.jl")
include("atom_iterator.jl")
include("cd_differentiable_function.jl")
include("coordinate_descent.jl")
include("lasso.jl")

end
