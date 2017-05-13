######################################################################
#
#   Lasso Interface
#
######################################################################

lasso_raw{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) = lasso_raw(X, y, λ*ones(size(X,2)), options)

lasso_raw{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::StridedVector{T},
  options::CDOptions=CDOptions()) = coordinateDescent(CDLeastSquaresLoss(y,X), λ, options)

# minimize x'Ax / 2 + b'x + λ⋅|x|
lasso{T<:AbstractFloat}(
  A::StridedMatrix{T},
  b::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) = lasso(A, b, λ*ones(length(b)), options)

lasso{T<:AbstractFloat}(
  XX::StridedMatrix{T},
  Xy::StridedVector{T},
  λ::Array{Float64, 1},
  options::CDOptions=CDOptions()) = coordinateDescent(CDQuadraticLoss(A, b), λ)


######################################################################
#
#   Lasso Path Interface
#
######################################################################


struct LassoPath{T<:AbstractFloat}
  λarr::Vector{T}
  β::Vector{SparseVector{T}}
end

function compute_lasso_path_refit{T<:AbstractFloat}(
  lasso_path::LassoPath{T},
  A::StridedMatrix{T},
  b::StridedVector{T})
  λArr = lasso_path.λArr

  tmpDict = Dict()
  for i=1:length(λArr)
    support_nz = nonzeroinds(lasso_path.β[i])
    if haskey(tmpDict, support_nz)
      continue
    end
    tmpDict[support_nz] = - A[support_nz, support_nz] \ b[support_nz]
  end
  tmpDict
end


# λArr is in decreasing order
function compute_lasso_path{T<:AbstractFloat}(
  A::StridedMatrix{T},
  b::StridedVector{T},
  λarr::StridedVector{T};
  max_hat_s=Inf, zero_thr=1e-4, intercept=false)

  loadingX = sqrt(diag(A))
  if intercept
    loadingX[1] = 0.
  end

  curβ = spzeros(p)
  f = CDQuadraticLoss(A, b)
  p = size(A, 1)

  _λArr = copy(λArr)
  numλ  = length(λArr)
  hβ = Vector{SparseVector{T}}(numλ)

  for indλ=1:numλ
    coordinateDescent!(curβ, f, λArr[indλ] * loadingX)
    hβ[indλ] = copy(curβ)
    if nnz(curβ) > max_hat_s
      _λArr = λArr[1:indλ-1]
      hβ = hβ[1:indλ-1]
      break
    end
  end

  LassoPath(_λArr, hβ)
end
