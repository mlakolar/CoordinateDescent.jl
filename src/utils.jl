####################################
#
# options
#
####################################

struct CDOptions
  maxIter::Int64
  optTol::Float64
  randomize::Bool           # coordinate are visitied in a randomized order or not
  warmStart::Bool           # when running CD, should we do path following or not
  numSteps::Int64           # when pathFollowing, how many points are there on the path
end

CDOptions(;
  maxIter::Int64=2000,
  optTol::Float64=1e-7,
  randomize::Bool=true,
  warmStart::Bool=true,
  numSteps::Int=50) = CDOptions(maxIter, optTol, randomize, warmStart, numSteps)



####################################
#
# helper functions
#
####################################


"""
Helper function that finds an initial estimate for σ that is needed
for Lasso and ScaledLasso procedures.

The procedure works as follows:

* s input variables that are most correlated with the response y are found
* y is regressed on those s features
* σ is estimated based on the residuals, which gives an upper bound on the true sigma
"""
function findInitSigma(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  n, p = size(X)
  if s > p
    s = p
  end
  c = (X'*y) / n
  @. c = abs(c)

  # find value of s-th largest element in abs(c)
  h = binary_maxheap(T)
  for i=1:p
      push!(h, c[i])
  end

  val = zero(T)
  for i=1:s
    val = pop!(h)
  end

  # solve OLS with s most correlated values
  S = c .> val
  βh = X[:, S] \ y
  std(y - X[:,S]*βh)
end





# ###
#
# # find the most correlated columns of X with Y
# #
# # this runs marginal rergession of Y on X_j for every j
# #
# function findCorrelatedColumns(tildeX::Array{Float64, 2}, y::Array{Float64, 1},
#                                q::Int64,                                          # order of the polynomial
#                                numCorrCol::Int64,
#                                kernelWeights::Array{Float64, 1})
#
#   (n, qp) = size(X)
#   p = Int(qp / q)
#   sqKernelWeights = sqrt(kernelWeights)
#   wX = zeros(n, (q+1))             # sqrt(kernelWeights) * [1, (z - z0), ..., (z-z0)^q] ⊗ X_j
#   wY = sqKernelWeights .* y        # sqrt(kernelWeights) * Y
#
#   resSq = zeros(p)                     # placeholder for residuals squared
#
#   for j=1:p
#
#     # transform features for X_j
#     for i=1:n
#       for l=0:q
#         @inbounds wX[i, l + 1] = X[i, (j - 1) * (q + 1) + l + 1] * sqKernelWeights[i]
#       end
#     end
#
#     # compute hat_beta
#     βh = wX \ wY
#     # compute residuals squared
#     res[j] = sumabs2(wY - wX * βh)
#
#   end
#
#   sortperm(res)[1:numCorrCol]
# end
