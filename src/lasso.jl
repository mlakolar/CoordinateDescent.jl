######################################################################
#
#   Lasso Interface
#
######################################################################

lasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) =
    coordinateDescent!(SparseIterate(size(X, 2)), CDLeastSquaresLoss(y,X), ProxL1(λ), options)

lasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions()) =
    coordinateDescent!(SparseIterate(size(X, 2)), CDLeastSquaresLoss(y,X), AProxL1(λ, ω), options)

lasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::Array{T},
  options::CDOptions=CDOptions()) =
    coordinateDescent!(SparseIterate(size(X, 2)), CDLeastSquaresLoss(y,X), AProxL1(1., λ), options)


######################################################################
#
#   Sqrt-Lasso Interface
#
######################################################################


sqrtLasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) =
    coordinateDescent!(SparseIterate(size(X, 2)), CDSqrtLassoLoss(y,X), ProxL1(λ), options)

sqrtLasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions()) =
    coordinateDescent!(SparseIterate(size(X, 2)), CDSqrtLassoLoss(y,X), AProxL1(λ, ω), options)


######################################################################
#
#   Scaled Lasso Interface
#
######################################################################


struct ScaledLassoOptions
  maxIter::Int64
  optTol::Float64
  σinit::Float64
  optionsCD::CDOptions
end

ScaledLassoOptions(;
  maxIter::Int64=20,
  optTol::Float64=1e-2,
  σinit::Float64=1.,
  optionsCD::CDOptions=CDOptions()) = ScaledLassoOptions(maxIter, optTol, σinit, optionsCD)


scaledLasso{T<:AbstractFloat}(
    X::AbstractMatrix{T},
    y::AbstractVector{T},
    λ::T,
    ω::Array{T},
    optionsScaledLasso::ScaledLassoOptions=ScaledLassoOptions()
    ) = scaledLasso!(SparseIterate(size(X, 2)), X, y, λ, ω, optionsScaledLasso)


function scaledLasso!{T<:AbstractFloat}(
  β::SparseIterate{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::Array{T},
  optionsScaledLasso::ScaledLassoOptions=ScaledLassoOptions()
  )

  n, p = size(X)
  f = CDLeastSquaresLoss(y,X)
  σ = optionsScaledLasso.σinit

  for iter=1:optionsScaledLasso.maxIter
    g = AProxL1(λ * σ, ω)
    coordinateDescent!(β, f, g, optionsScaledLasso.optionsCD)
    σnew = sqrt( sum(abs2, f.r) / n )

    if abs(σnew - σ) / σ < optionsScaledLasso.optTol
      break
    end
    σ = σnew
  end
  β, σ
end


######################################################################
#
#   Lasso Path Interface
#
######################################################################


struct LassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  βpath::Vector{SparseIterate{T,1}}
end

function refitLassoPath{T<:AbstractFloat}(
  path::LassoPath{T},
  X::StridedMatrix{T},
  Y::StridedVector{T})

  λpath = path.λpath
  βpath = path.βpath

  out = Dict{Vector{Int64},Vector{Float64}}()
  for i=1:length(λpath)
    S = find(βpath[i])
    if haskey(out, S)
      continue
    end
    out[S] = X[:, S] \ Y
  end
  out
end


# λArr is in decreasing order
function LassoPath(
  X::StridedMatrix{T},
  Y::StridedVector{T},
  λpath::Vector{T},
  options=CDOptions();
  max_hat_s=Inf, standardizeX::Bool=true) where {T<:AbstractFloat}

  n, p = size(X)
  loadingX = ones(p)
  if standardizeX
    @inbounds for i=1:p
      loadingX[i] = std(view(X, :, i))
    end
  end

  β = SparseIterate(T, p)
  f = CDLeastSquaresLoss(Y, X)

  numλ  = length(λpath)
  βpath = Vector{SparseIterate{T}}(numλ)

  for indλ=1:numλ
    coordinateDescent!(β, f, AProxL1(λpath[indλ], loadingX), options)
    βpath[indλ] = copy(β)
    if nnz(β) > max_hat_s
      resize!(λpath, indλ)
      break
    end
  end

  LassoPath{T}(copy(λpath), βpath)
end
