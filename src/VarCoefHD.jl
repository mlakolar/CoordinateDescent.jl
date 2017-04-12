
### Kernel functions
abstract type SmoothingKernelFunction end
immutable GaussianKernel <: SmoothingKernelFunction end


evalKernel(::GaussianKernel, x::Float64, y::Float64, h::Float64) = exp(-(x-y)^2. / h) / h
function evalKernel!(::GaussianKernel, buffer::Array{Float64, 1}, x::Array{Float64, 1}, y::Float64, h::Float64)
  assert(length(buffer) == length(x))
  @simd for i=1:length(x)
    @inbounds buffer[i] = evalKernel(GaussianKernel(), x[i], y, h)
  end
end

###

# find the most correlated columns of X with Y
#
# this runs marginal rergession of Y on X_j for every j
#
function findCorrelatedColumns(tildeX::Array{Float64, 2}, y::Array{Float64, 1},
                               q::Int64,                                          # order of the polynomial
                               numCorrCol::Int64,
                               kernelWeights::Array{Float64, 1})

  (n, qp) = size(X)
  p = Int(qp / q)
  sqKernelWeights = sqrt(kernelWeights)
  wX = zeros(n, (q+1))             # sqrt(kernelWeights) * [1, (z - z0), ..., (z-z0)^q] ⊗ X_j
  wY = sqKernelWeights .* y        # sqrt(kernelWeights) * Y

  resSq = zeros(p)                     # placeholder for residuals squared

  for j=1:p

    # transform features for X_j
    for i=1:n
      for l=0:q
        @inbounds wX[i, l + 1] = X[i, (j - 1) * (q + 1) + l + 1] * sqKernelWeights[i]
      end
    end

    # compute hat_beta
    βh = wX \ wY
    # compute residuals squared
    res[j] = sumabs2(wY - wX * βh)

  end

  sortperm(res)[1:numCorrCol]
end

function varCoeffReg(X::Array{Float64, 2}, y::Array{Float64, 1}, z::Array{Float64, 1},
                    z0::Float64,
                    q::Int64,                                          # order of the polynomial
                    h::Float64;                                        # bandwidth
                    kernel::SmoothingKernelFunction=GaussianKernel())

    (n, p) = size(X)
    tildeX = zeros(n, p*(q+1))
    get_data_matrix!(tildeX, X, z, z0, q)

    kernelWeights = zeros(n)
    evalKernel!(kernel, kernelWeights, z, z0, h)
    sw = sqrt(kernelWeights)

    (sw .* tildeX) \ (sw .* y)
end

# expands the data matrix X
# each row becomes
# [1, (z - z0), ..., (z-z0)^q] ⊗ X[i, :]
function get_data_matrix!(tildeX::Array{Float64, 2}, X::Array{Float64, 2}, z::Array{Float64, 1}, z0::Float64, q::Float64)
  (n,p) = size(X)
  for j=1:p
    for i=1:n
      for l=0:q
        @inbounds tildeX[i, (j-1)*(q+1) + l + 1] = X[i, j] * (z[i] - z0)^l
      end
    end
  end
end

function var_coef_lasso!(beta::SparseVector{Float64, Int64},
                    X::Array{Float64, 2}, y::Array{Float64, 1}, z::Array{Float64, 1},
                    lambda::Array{Float64, 1},
                    z0::Float64,
                    q::Int64,                                          # order of the polynomial
                    h::Float64;                                        # bandwidth
                    maxIter::Int64=2000, maxInnerIter::Int64=1000, optTol::Float64=1e-7,
                    kernel::SmoothingKernelFunction=GaussianKernel(),
                    sizeInitS::Int64=5)

  (n, p) = size(X)
  tildeX = zeros(n, p*(q+1))
  get_data_matrix!(tildeX, X, z, z0, q)

  kernelWeights = zeros(n)
  evalKernel!(kernel, kernelWeights, z, z0, h)

  # find largest correlations
  initS = findCorrelatedColumns(tildeX, y, q, sizeInitS, kernelWeights)

  # compute regression with the selected S
  XX = zeros((q+1)*sizeInitS, (q+1)*sizeInitS)
  for a=1:(q+1)*sizeInitS
    for b=a:(q+1)*sizeInitS
      if a == b
        k = initS[a]
        for i=1:n
          @inbounds XX[a,a] = XX[a,a] + tildeX[i, k]^2 * kernelWeights[i]
        end
      else
        k = initS[a]
        l = initS[b]
        for i=1:n
          @inbounds XX[a,b] = XX[a,b] + tildeX[i, k] * kernelWeights[i] * tildeX[i, l]
        end
        @inbounds XX[b,a] = XX[a,b]
      end
    end
  end


  XX = zeros((q+1)*p, (q+1)*p)



  lasso!(beta, XX, Xy, lambda; maxIter=maxIter, maxInnerIter=maxInnerIter, optTol=optTol)

  nothing
end
