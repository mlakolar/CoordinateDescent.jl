
function locpoly_alt(X::Matrix{T}, z::Vector{T}, y::Vector{T},
                 z0::T,
                 degree::Int64,                                           # degree of the polynomial
                 kernel::SmoothingKernel=GaussianKernel(one(T))) where {T <: AbstractFloat}

    n, p = size(X)
    tX = zeros(n, p*(degree+1))
    _expand_X_alt!(tX, X, z, z0, degree)

    w = zeros(T, n)
    @. w = evaluate(kernel, z, z0)
    (tX' * diagm(w) * tX) \ (tX' * diagm(w) * y)
end

function locpoly_alt(X::Matrix{T}, z::Vector{T}, y::Vector{T},
                 zgrid::Vector{T},
                 degree::Int64,                                     # degree of the polynomial
                 kernel::SmoothingKernel=GaussianKernel(one(T))) where {T <: AbstractFloat}

  out = zeros(size(X, 2)*(degree+1), length(zgrid))
  ind = 0
  for z0 in zgrid
    ind += 1
    out[:, ind] = locpoly_alt(X, z, y, z0, degree, kernel)
  end
  out
end

function _expand_X_alt!(tX::Matrix{T}, X::Matrix{T}, z::Vector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}
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

n, p = 1000, 3
Y, X, Z, betaMat = genData(n, p)
zgrid = collect(0.01:0.05:0.99)

gk = GaussianKernel(0.2)
tX = zeros(n, 2 * p)

@time o1 = locpoly(X, Z, Y, zgrid, 1, gk)
@time o2 = locpoly_alt(X, Z, Y, zgrid, 1, gk)
maximum(abs.(o1-o2))
