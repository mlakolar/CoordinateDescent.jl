
### Kernel functions
abstract type SmoothingKernel end


struct GaussianKernel{T} <: SmoothingKernel
  h::T
end
evaluate(k::GaussianKernel{T}, x::T, y::T) where {T <: AbstractFloat} = exp(-(x-y)^2. / k.h) / k.h



######################
#
#
#
######################


function _locpoly!(w::Vector{T}, X::Matrix{T}, z::Vector{T}, y::Vector{T},
                 z0::T,
                 degree::Int64,                                     # degree of the polynomial
                 kernel::SmoothingKernel) where {T <: AbstractFloat}

    n, p = size(X)
    @. w = sqrt(evaluate(kernel, z, z0))   # square root of kernel weights
    tX = _expand_X(w, X, z, z0, degree)    # √w ⋅ x ⊗ [1 (zi - z0) ... (zi-z0)^q]
    @. w *= y                              # √w ⋅ y
    tX \ w
end


locpoly(X::Matrix{T}, z::Vector{T}, y::Vector{T}, z0::T, degree::Int64, kernel::SmoothingKernel=GaussianKernel(one(T))) where {T <: AbstractFloat} = _locpoly!(similar(y), X, z, y, z0, degree, kernel)

function locpoly(X::Matrix{T}, z::Vector{T}, y::Vector{T},
                 zgrid::Vector{T},
                 degree::Int64,                                     # degree of the polynomial
                 kernel::SmoothingKernel=GaussianKernel(one(T))) where {T <: AbstractFloat}

  out = zeros(size(X, 2)*(degree+1), length(zgrid))
  ind = 0
  w = similar(y)
  for z0 in zgrid
    ind += 1
    out[:, ind] = _locpoly!(w, X, z, y, z0, degree, kernel)
  end
  out
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



#
# function var_coef_lasso!(beta::SparseVector{Float64, Int64},
#                     X::Array{Float64, 2}, y::Array{Float64, 1}, z::Array{Float64, 1},
#                     lambda::Array{Float64, 1},
#                     z0::Float64,
#                     q::Int64,                                          # order of the polynomial
#                     h::Float64;                                        # bandwidth
#                     maxIter::Int64=2000, maxInnerIter::Int64=1000, optTol::Float64=1e-7,
#                     kernel::SmoothingKernelFunction=GaussianKernel(),
#                     sizeInitS::Int64=5)
#
#   (n, p) = size(X)
#   tildeX = zeros(n, p*(q+1))
#   get_data_matrix!(tildeX, X, z, z0, q)
#
#   kernelWeights = zeros(n)
#   evalKernel!(kernel, kernelWeights, z, z0, h)
#
#   # find largest correlations
#   initS = findCorrelatedColumns(tildeX, y, q, sizeInitS, kernelWeights)
#
#   # compute regression with the selected S
#   XX = zeros((q+1)*sizeInitS, (q+1)*sizeInitS)
#   for a=1:(q+1)*sizeInitS
#     for b=a:(q+1)*sizeInitS
#       if a == b
#         k = initS[a]
#         for i=1:n
#           @inbounds XX[a,a] = XX[a,a] + tildeX[i, k]^2 * kernelWeights[i]
#         end
#       else
#         k = initS[a]
#         l = initS[b]
#         for i=1:n
#           @inbounds XX[a,b] = XX[a,b] + tildeX[i, k] * kernelWeights[i] * tildeX[i, l]
#         end
#         @inbounds XX[b,a] = XX[a,b]
#       end
#     end
#   end
#
#
#   XX = zeros((q+1)*p, (q+1)*p)
#
#
#
#   lasso!(beta, XX, Xy, lambda; maxIter=maxIter, maxInnerIter=maxInnerIter, optTol=optTol)
#
#   nothing
# end

######################################
#
# utils
#
######################################

# expands the data matrix X
# each row becomes
# X[i, :] ⊗ [1, (z - z0), ..., (z-z0)^q]
function _expand_X(w::Vector{T}, X::Matrix{T}, z::Vector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}
  n, p = size(X)
  tX = zeros(n, p*(degree+1))
  for j=1:p
    for i=1:n
      v = X[i, j]
      df = z[i] - z0
      col = (j-1)*(degree+1) + 1
      @inbounds tX[i, col] = v * w[i]
      for l=1:degree
        v *= df
        @inbounds tX[i, col + l] = v * w[i]
      end
    end
  end
  return qrfact!(tX)
end
