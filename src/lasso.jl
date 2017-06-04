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
  λarr::Vector{T}
  β::Vector{SparseIterate{T}}
end

function refitLassoPath{T<:AbstractFloat}(
  path::LassoPath{T},
  X::StridedMatrix{T},
  Y::StridedVector{T})

  λArr = path.λArr

  tmpDict = Dict()
  for i=1:length(λArr)
    S = find(lasso_path.β)
    if haskey(tmpDict, S)
      continue
    end
    tmpDict[S] = X[:, S] \ Y
  end
  tmpDict
end


# λArr is in decreasing order
function computeLassoPath{T<:AbstractFloat}(
  X::StridedMatrix{T},
  Y::StridedVector{T},
  λarr::StridedVector{T},
  options=CDOptions();
  max_hat_s=Inf, intercept=false)

  n, p = size(X)
  loadingX = zeros(p)
  j = intercept ? 2 : 1
  for i=j:p
    loadingX[i] = vecnorm(view(X, :, i)) / sqrt(n)
  end

  β = SparseIterate(p)
  f = CDLeastSquaresLoss(Y, X)

  _λArr = copy(λArr)
  numλ  = length(λArr)
  hβ = Vector{SparseVector{T}}(numλ)

  for indλ=1:numλ
    g = AProxL1(loadingX*λArr[indλ])
    coordinateDescent!(β, f, g)
    hβ[indλ] = copy(β)         # need to create a constructor for this
    if nnz(β) > max_hat_s
      _λArr = λArr[1:indλ-1]   # todo: use resize?
      hβ = hβ[1:indλ-1]
      break
    end
  end

  LassoPath(_λArr, hβ)
end
