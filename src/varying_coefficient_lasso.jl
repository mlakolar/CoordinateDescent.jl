
### Kernel functions
abstract type SmoothingKernel{T} end


struct GaussianKernel{T} <: SmoothingKernel{T}
  h::T
end
evaluate(k::GaussianKernel{T}, x::T, y::T) where {T <: AbstractFloat} = exp(-(x-y)^2. / k.h) / k.h



############################################################
#
# local polynomial regression with lasso
#
############################################################

function locpolyl1(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int64,
  kernel::SmoothingKernel{T},
  λ0::T,
  options::CDOptions=CDOptions()) where {T <: AbstractFloat}

  # construct inner options because we do not want warmStart = false
  # we want to start from the previous iteration since the points
  # on the grid should be close to each other
  opt = CDOptions(options.maxIter, options.optTol, options.randomize, true, options.numSteps)

  n, p = size(X)
  ep = p * (degree + 1)
  out = Array{SparseVector{T, Int64}}(length(zgrid))

  # temporary storage
  w = Array{T}(n)
  wX = Array{T}(n, ep)
  stdX = ones(T, ep)
  f = CDWeightedLSLoss(y, wX, w)        # inner parts of f will be modified in a loop
  g = AProxL1(λ0, stdX)
  β = SparseIterate(ep)

  ind = 0
  for z0 in zgrid
    ind += 1

    # the following two should update f
    @. w = evaluate(kernel, z, z0)
    _expand_X!(wX, X, z, z0, degree)
    # compute std for each column
    @inbounds for j=1:ep
      v = zero(T)
      @simd for i=1:n
        v += wX[i, j] * wX[i, j] * w[i]
      end
      stdX[j] = sqrt(v / n)
    end

    # solve for β
    coordinateDescent!(β, f, g, opt)
    out[ind] = convert(SparseVector, β)
  end
  out
end



############################################################
#
# local polynomial regression low dimensions
#
############################################################


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


############################################################
#
# utils
#
############################################################

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
