
### Kernel functions
abstract type SmoothingKernel{T} end


struct GaussianKernel{T} <: SmoothingKernel{T}
  h::T
end
evaluate(k::GaussianKernel{T}, x::T, y::T) where {T <: AbstractFloat} = exp(-(x-y)^2. / k.h) / k.h



######################
#
#
#
######################


function _locpoly!(
  wX::Matrix{T}, w::Vector{T},
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int64, kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  @. w = sqrt(evaluate(kernel, z, z0))    # square root of kernel weights
  _expand_wX!(wX, w, X, z, z0, degree)    # √w ⋅ x ⊗ [1 (zi - z0) ... (zi-z0)^q]
  @. w *= y                               # √w ⋅ y
  qrfact!(wX) \ w
end

locpoly(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int64, kernel::SmoothingKernel=GaussianKernel(one(T))) where {T <: AbstractFloat} =
    _locpoly!(Array{T}(length(y), size(X, 2) * (degree+1)), similar(y), X, z, y, z0, degree, kernel)

function locpoly(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int64,                                     # degree of the polynomial
  kernel::SmoothingKernel{T}=GaussianKernel(one(T))) where {T <: AbstractFloat}

  n, p = size(X)
  ep = p * (degree + 1)
  out = Array{T}(ep, length(zgrid))
  w = Array{T}(n)
  wX = Array{T}(n, ep)

  ind = 0
  for z0 in zgrid
    ind += 1
    out[:, ind] = _locpoly!(wX, w, X, z, y, z0, degree, kernel)
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

"""
Computes matrix whose each row is equal to
w[i] ⋅ (X[i, :] ⊗ [1, (z - z0), ..., (z-z0)^q])
where q is the degree of the polynomial.

The otput matrix is preallocated.
"""
function _expand_wX!(
  wX::Matrix{T},
  w::Vector{T}, X::Matrix{T},
  z::Vector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  # wX = zeros(n, p*(degree+1))
  for j=1:p
    for i=1:n
      v = X[i, j] * w[i]
      df = z[i] - z0
      col = (j-1)*(degree+1) + 1
      @inbounds wX[i, col] = v
      for l=1:degree
        v *= df
        @inbounds wX[i, col + l] = v
      end
    end
  end
  # return qrfact!(tX)
  wX
end


# expands the data matrix X
# each row becomes
# (X[i, :] ⊗ [1, (z - z0), ..., (z-z0)^q])
function _expand_X!(
  tX::Matrix{T},
  X::Matrix{T},
  z::Vector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  for j=1:p
    for i=1:n
      v = X[i, j]
      df = z[i] - z0
      col = (j-1)*(degree+1) + 1
      @inbounds tX[i, col] = v
      for l=1:degree
        v *= df
        @inbounds tX[i, col + l] = v
      end
    end
  end
  tX
end


function _expand_Xt_w_X!(
  Xt_w_X::Matrix{T},
  w::Vector{T}, X::Matrix{T},
  z::Vector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  fill!(Xt_w_X, zero(T))
  @inbounds for j=1:p
    for k=j:p
      for i=1:n
        v1 = X[i, j] * w[i]
        df1 = z[i] - z0

        col=(j-1)*(degree+1)+1
        for jj=0:degree
          v2 = X[i, k]
          df2 = z[i] - z0
          if k != j
            krange = 0:degree
          else
            krange = jj:degree
            v2 = v2 * df2^jj
          end

          row = (k-1)*(degree+1)+1
          for kk=krange
            # need X[i, j] * (z[i] - z0)^(jj+kk) * X[i, k] * w[i]
            # v2 = (z[i] - z0)^kk * X[i, k]
            # v1 = (z[i] - z0)^jj * X[i, k] * w[i]
            Xt_w_X[row+kk, col+jj] += v2 * v1
            v2 *= df2
          end
          v1 *= df1
        end
      end
    end
  end

  @inbounds for c=1:size(Xt_w_X, 2)
    for r=c+1:size(Xt_w_X, 1)
      Xt_w_X[c, r] = Xt_w_X[r, c]
    end
  end

  Xt_w_X
end

function _expand_Xt_w_Y!(
  Xt_w_Y::Vector{T},
  w::Vector{T}, X::Matrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  fill!(Xt_w_Y, zero(T))
  @inbounds for j=1:p
    for i=1:n
      v = X[i, j] * w[i] * y[i]
      df = z[i] - z0

      col=(j-1)*(degree+1)+1
      for jj=0:degree
        # need X[i, j] * (z[i] - z0)^jj * Y[i, k] * w[i]
        Xt_w_Y[col+jj] += v
        v *= df
      end
    end
  end

  Xt_w_Y
end
