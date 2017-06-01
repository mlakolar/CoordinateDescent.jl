######################################################################
#
#   Lasso Interface
#
######################################################################

lasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) = coordinateDescent(CDLeastSquaresLoss(y,X), ProxL1(λ), options)

lasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::Array{T},
  options::CDOptions=CDOptions()) = coordinateDescent(CDLeastSquaresLoss(y,X), AProxL1(λ), options)


sqrtLasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) = coordinateDescent(CDSqrtLassoLoss(y,X), ProxL1(λ), options)

sqrtLasso{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::Array{T},
  options::CDOptions=CDOptions()) = coordinateDescent(CDSqrtLassoLoss(y,X), AProxL1(λ), options)


struct ScaledLassoOptions
  maxIter::Int64
  optTol::Float64
  optionsCD::CDOptions
end

ScaledLassoOptions(;
  maxIter::Int64=100,
  optTol::Float64=1e-6,
  optionsCD::CDOptions=CDOptions()) = ScaledLassoOptions(maxIter, optTol, optionsCD)


scaledLasso{T<:AbstractFloat}(
    X::StridedMatrix{T},
    y::StridedVector{T},
    λ::Array{T},
    optionsScaledLasso::ScaledLassoOptions=ScaledLassoOptions()
    ) = scaledLasso!(SparseIterate(size(X, 2)), X, y, λ, optionsScaledLasso)

function scaledLasso!{T<:AbstractFloat}(
  β::SparseIterate{T},
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::Array{T},
  optionsScaledLasso::ScaledLassoOptions=ScaledLassoOptions()
  )

  n, p = size(X)
  f = CDSqrtLassoLoss(y,X)
  σ = one(T)
  g = AProxL1(λ * σ)
  σnew = one(T)

  for iter=1:optionsScaledLasso.maxIter
    coordinateDescent!(β, f, g, optionsScaledLasso.optionsCD)
    σnew = sqrt( sum(abs2, f.r) / n )

    if abs(σnew - σ) / σ < optionsScaledLasso.optTol
      break
    end
    σ = σnew
    @. g.λ = λ * σ
  end
  β, σnew
end


######################################################################
#
#   Lasso Path Interface
#
######################################################################


struct LassoPath{T<:AbstractFloat}
  λarr::Vector{T}
  β::Vector{SparseVector{T}}
end

function refitLassoPath{T<:AbstractFloat}(
  path::LassoPath{T},
  X::StridedMatrix{T},
  Y::StridedVector{T})

  λArr = path.λArr

  tmpDict = Dict()
  for i=1:length(λArr)
    support_nz = nonzeroinds(lasso_path.β[i])
    if haskey(tmpDict, support_nz)
      continue
    end
    tmpDict[support_nz] = X[:, support_nz] \ Y
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
