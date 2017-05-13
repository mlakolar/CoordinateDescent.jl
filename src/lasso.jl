######################################################################
#
#   Lasso
#
######################################################################

struct CDOptions
  maxIter::Int64
  maxInnerIter::Int64
  optTol::Float64
  kktTol::Float64
end

CDOptions(;
  maxIter::Int64=2000,
  maxInnerIter::Int64=1000,
  optTol::Float64=1e-7,
  kktTol::Float64=1e-7) = CDOptions(maxIter, maxInnerIter, optTol, kktTol)

  ######################################################################
  #
  #   Lasso Helper Functions
  #
  ######################################################################


  # helper function for Active Shooting implementation of Lasso
  # iterates over the active set
  #
  # β is a sparse vector that contains information about the active set
  # when adding an element to the active set, we set that element of β to eps()
  #
  # TODO: add logging capabilities
  function minimize_active_set!{T<:AbstractFloat}(
    β::SparseVector{T},
    XX::StridedMatrix{T},
    Xy::StridedVector{T},
    λ::StridedVector{T},
    options::CDOptions=CDOptions())

    nzval = SparseArrays.nonzeros(β)
    rowval = SparseArrays.nonzeroinds(β)

    maxInnerIter = options.maxInnerIter
    optTol = options.optTol

    for iter=1:maxInnerIter
      fDone = true
      for j = 1:length(rowval)
        ci = rowval[j]
        # Compute the Shoot and Update the variable
        S0 = compute_residual(XX, Xy, β, ci) - XX[ci,ci]*nzval[j]
        oldVal = nzval[j]
        nzval[j] = shrink(-S0 / XX[ci,ci], λ[ci] / XX[ci,ci])
        if abs(oldVal - nzval[j]) > optTol
          fDone = false
        end
      end
      if fDone
        break
      end
    end
    dropzeros!(β)
  end

  # computes  (X^T)_k Y - sum_{j in active_set} (X^T)_j X_k β_k
  function compute_residual(XX::Array{Float64, 2}, Xy::Array{Float64, 1}, β::SparseVector{Float64, Int64}, k::Int64)
    nzval = β.nzval
    rowval = β.nzind

    S0 = -Xy[k]
    for rInd=1:length(rowval)
      S0 += XX[rowval[rInd],k] * nzval[rInd]
    end
    return S0
  end

  # finds index to add to the active_set
  function add_violating_index!{T<:AbstractFloat}(
    β::SparseVector{T},
    XX::StridedMatrix{T},
    Xy::StridedVector{T},
    λ::StridedVector{T})

    p = size(XX, 1)
    nzval = SparseArrays.nonzeros(β)
    rowval = SparseArrays.nonzeroinds(β)

    val = zero(T)
    ind = 0
    for j = setdiff(1:p, rowval)
      S0 = abs(compute_residual(XX, Xy, β, j))
      if S0 > λ[j]
        if S0 > val
          val = S0;
          ind = j;
        end
      end
    end
    if ind != 0
      β[ind] = eps()
    end
    return ind
  end


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

function lasso_raw{T<:AbstractFloat}(
  X::StridedMatrix{T},
  y::StridedVector{T},
  λ::StridedVector{T},
  options::CDOptions=CDOptions())

  n, p = size(X)
  XX = (X'X) / n
  Xy = (X'y) / n

  lasso!(spzeros(p), XX, Xy, λ, options)
end

lasso!{T<:AbstractFloat}(
  β::SparseVector{T},
  XX::StridedMatrix{T},
  Xy::StridedVector{T},
  λ::T,
  options::CDOptions=CDOptions()) = lasso!(β, XX, Xy, λ*ones(size(X,2)), options)

function lasso!{T<:AbstractFloat}(
  β::SparseVector{T},
  XX::StridedMatrix{T},
  Xy::StridedVector{T},
  λ::Array{Float64, 1},
  options::CDOptions=CDOptions())

  p = size(XX, 2)
  length(λ) == p || throw(DimensionMismatch())

  if iszero(β)
    add_violating_index!(β, XX, Xy, λ) != 0 || return β
  end

  for iter=1:options.maxIter
    minimize_active_set!(β, XX, Xy, λ, options)
    add_violating_index!(β, XX, Xy, λ) != 0 || return β
  end
  β
end


######################################################################
#
#   Lasso Path Interface
#
######################################################################


type LassoPath
  λArr
  β
end

function compute_lasso_path_refit(lasso_path::LassoPath, XX::Array{Float64, 2}, Xy::Array{Float64, 1})
  λArr = lasso_path.λArr

  tmpDict = Dict()
  for i=1:length(λArr)
    support_nz = find(lasso_path.β[i])
    if haskey(tmpDict, support_nz)
      continue
    end
    tmpDict[support_nz] = XX[support_nz, support_nz] \ Xy[support_nz]
  end
  tmpDict
end


# λArr is in decreasing order
function compute_lasso_path(XX::Array{Float64, 2}, Xy::Array{Float64, 1},
                            λArr::Array{Float64, 1};
			    max_hat_s=Inf, zero_thr=1e-4, intercept=false)

  p = size(XX, 1)
  loadingX = sqrt(diag(XX))
  if intercept
    loadingX[1] = 0.
  end

  curβ = spzeros(p, 1)

  _λArr = copy(λArr)
  numλ  = length(λArr)
  hβ = cell(numλ)

  for indλ=1:numλ
    lasso!(curβ, XX, Xy, λArr[indλ] * loadingX)
    hβ[indλ] = copy(curβ)
    if nnz(curβ) > max_hat_s
      _λArr = λArr[1:indλ-1]
      hβ = hβ[1:indλ-1]
      break
    end
  end

  LassoPath(_λArr, hβ)
end
