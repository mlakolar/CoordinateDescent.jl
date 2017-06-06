##

function _locpoly_alt1!(
  eX::Matrix{T}, w::Vector{T},
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int64, kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  @. w = evaluate(kernel, z, z0)
  _expand_X!(eX, X, z, z0, degree)

  ((eX' * diagm(w)) * eX) \ ((eX' * diagm(w)) * y)
end

function locpoly_alt1(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int64,                                     # degree of the polynomial
  kernel::SmoothingKernel{T}=GaussianKernel(one(T))) where {T <: AbstractFloat}

  n, p = size(X)
  ep = p * (degree + 1)
  out = Array{T}(ep, length(zgrid))
  w = Array{T}(n)
  eX = Array{T}(n, ep)

  ind = 0
  for z0 in zgrid
    ind += 1
    out[:, ind] = _locpoly_alt1!(eX, w, X, z, y, z0, degree, kernel)
  end
  out
end

##

function _locpoly_alt!(
  Xt_w_X::Matrix{T}, Xt_w_Y::Vector{T}, w::Vector{T},
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int64, kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  @. w = evaluate(kernel, z, z0)
  _expand_Xt_w_X!(Xt_w_X, w, X, z, z0, degree)
  _expand_Xt_w_Y!(Xt_w_Y, w, X, z, y, z0, degree)

  cholfact!(Xt_w_X) \ Xt_w_Y
end

function locpoly_alt(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int64,                                     # degree of the polynomial
  kernel::SmoothingKernel{T}=GaussianKernel(one(T))) where {T <: AbstractFloat}

  n, p = size(X)
  ep = p * (degree + 1)
  out = Array{T}(ep, length(zgrid))
  w = Array{T}(n)
  Xt_w_X = Array{T}(ep, ep)
  Xt_w_Y = Array{T}(ep)

  ind = 0
  for z0 in zgrid
    ind += 1
    out[:, ind] = _locpoly_alt!(Xt_w_X, Xt_w_Y, w, X, z, y, z0, degree, kernel)
  end
  out
end


function _expand_Xt_w_X_alt!(
  Xt_w_X::Matrix{T},
  w::Vector{T}, X::Matrix{T},
  z::Vector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  fill!(Xt_w_X, zero(T))
  @inbounds for j=1:p, jj=0:degree
    col=(j-1)*(degree+1)+1+jj
    for k=j:p
      if k != j
        krange = 0:degree
      else
        krange = jj:degree
      end
      for kk=krange
        row = (k-1)*(degree+1)+1+kk

        # compute Xt_w_X[row, col]
        for i=1:n
          Xt_w_X[row, col] += X[i, j] * (z[i] - z0)^(jj+kk) * X[i, k] * w[i]
        end
        if row != col
          Xt_w_X[col, row] = Xt_w_X[row, col]
        end
      end
    end
  end
  Xt_w_X
end


function genData(n, p)
  X = randn(n, p)
  Z = rand(n)
  ɛ = randn(n)
  Y = zeros(n)
  betaMat = zeros(p, n)

  randBeta = [rand([2,4,6,8]) for j=1:p]
  for i=1:n
    betaMat[:, i] = sin.(randBeta * Z[i])
    Y = dot(betaMat[:, i], X[i, :]) + ɛ
  end
  Y, X, Z, betaMat
end


# using BenchmarkTools

####
#
# n, p = 100, 2
# Y, X, Z, betaMat = genData(n, p)
# zgrid = collect(0.01:0.1:0.99)
#
# gk = GaussianKernel(0.2)
# degree = 3
#
# @time o1 = locpoly(X, Z, Y, zgrid, 1, gk)
# @time o2 = locpoly_alt(X, Z, Y, zgrid, 1, gk)
# @time o3 = locpoly_alt1(X, Z, Y, zgrid, 1, gk)
#
# maximum(abs.(o1-o2))
# maximum(abs.(o1-o3))
#
# @benchmark locpoly($X, $Z, $Y, $zgrid, $degree, $gk)
# @benchmark locpoly_alt($X, $Z, $Y, $zgrid, $degree, $gk)
# @benchmark locpoly_alt1($X, $Z, $Y, $zgrid, $degree, $gk)


####

# p = 10
# X = randn(100, p)
# z = rand(100)
# w = zeros(100)
# k = GaussianKernel(0.2)
# @. w = evaluate(k, z, 0.5)
#
#
# degree = 2
# cp = p*(degree+1)
#
# eX = zeros(100, cp)
# _expand_X!(eX, X, z, 0.5, degree)
#
# Xt_w_X = zeros(cp, cp)
# Xt_w_X1 = zeros(cp, cp)
# o1 = _expand_Xt_w_X!(Xt_w_X, w, X, z, 0.5, degree)
# o2 = _expand_Xt_w_X_alt!(Xt_w_X1, w, X, z, 0.5, degree)
# o3 = (eX'*diagm(w))*eX
#
# maximum(abs.(o1 - o2))
# maximum(abs.(o1 - o3))
#
# using BenchmarkTools
# @benchmark _expand_Xt_w_X!($Xt_w_X, $w, $X, $z, 0.5, degree)
# @benchmark _expand_Xt_w_X_alt!($Xt_w_X1, $w, $X, $z, 0.5, degree)



















##
