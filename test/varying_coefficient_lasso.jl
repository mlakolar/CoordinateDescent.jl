module VarCoeffLassoTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

Random.seed!(1)



@testset "kernels" begin
  x = 0.3
  y = 0.4
  k = GaussianKernel(1.)
  @test evaluate(k, x, y) ≈ exp(-0.01)

  function evaluate!(g::GaussianKernel{T}, buffer::Array{T}, x::Array{T}, y::Float64) where {T <: AbstractFloat}
    size(buffer) == size(x) || throw(DimensionMismatch())

    @inbounds @simd for i=1:length(x)
       buffer[i] = evaluate(g, x[i], y)
    end
    buffer
  end

  x = rand(100, 1000)
  y = rand()
  k = GaussianKernel(0.5)

  o1 = zero(x)
  o2 = zero(x)
  o1 = evaluate.(Ref(k), x, Ref(y))
  evaluate!(k, o2, x, y)

  @test o1 ≈ o2
end

@testset "expand_X" begin
  X = reshape(collect(1.:6.), 2, 3)
  z = [0.2, 0.4]
  z0 = 0.3

  tX = similar(X)
  @test CoordinateDescent._expand_X!(tX, X, z, z0, 0) == X

  tX = zeros(2, 6)
  tX1 = zeros(2, 6)
  Q = [1. -0.1; 1. 0.1]
  for i=1:2
    tX1[i, :] = kron(X[i,:], Q[i, :])
  end
  @test CoordinateDescent._expand_X!(tX, X, z, z0, 1) ≈ tX1

  tX = zeros(2, 9)
  tX1 = zeros(2, 9)
  Q = [1. -0.1 0.01; 1. 0.1 0.01]
  for i=1:2
    tX1[i, :] = kron(X[i,:], Q[i, :])
  end
  @test CoordinateDescent._expand_X!(tX, X, z, z0, 2) ≈ tX1
end

@testset "expand_X multiplications" begin

  p = 10
  X = randn(100, p)
  Y = randn(100)
  z = rand(100)
  w = zeros(100)
  k = GaussianKernel(0.2)
  w = evaluate.(Ref(k), z, Ref(0.5))

  for degree=0:2
    cp = p*(degree+1)
    eX = zeros(100, cp)
    ewX = zeros(100, cp)

    CoordinateDescent._expand_X!(eX, X, z, 0.5, degree)
    @test CoordinateDescent._expand_wX!(ewX, w, X, z, 0.5, degree) ≈ Diagonal(w)*eX

    Xt_w_Y = zeros(cp)
    @test CoordinateDescent._expand_Xt_w_Y!(Xt_w_Y, w, X, z, Y, 0.5, degree) ≈ eX' * Diagonal(w) * Y

    Xt_w_X = zeros(cp, cp)
    @test CoordinateDescent._expand_Xt_w_X!(Xt_w_X, w, X, z, 0.5, degree) ≈ (eX'*Diagonal(w))*eX
  end
end
@testset "epanechnikov kernel" begin
  k = EpanechnikovKernel(0.5)
  @test evaluate(k, 0.1, 0.3) ≈ 0.75 * (1 - 0.4^2) / 0.5
  @test evaluate(k, 0.1, 0.7) == 0.
  @test createKernel(EpanechnikovKernel{Float64}, 0.5) == k
  @test createKernel(GaussianKernel{Float64}, 0.5) == GaussianKernel(0.5)
end

@testset "locpoly" begin
  n, p = 500, 2
  X = randn(n, p)
  z = rand(n)
  y = sin.(4 * z) .* X[:, 1] + cos.(2 * z) .* X[:, 2] + 0.1 * randn(n)
  zgrid = collect(0.1:0.2:0.9)
  gk = GaussianKernel(0.1)

  for degree=0:2
    cp = p*(degree+1)
    eX = zeros(n, cp)
    B = locpoly(X, z, y, zgrid, degree, gk)
    @test size(B) == (cp, length(zgrid))

    for (ind, z0) in enumerate(zgrid)
      CoordinateDescent._expand_X!(eX, X, z, z0, degree)
      w = evaluate.(Ref(gk), z, Ref(z0))
      β = (eX' * Diagonal(w) * eX) \ (eX' * Diagonal(w) * y)
      @test B[:, ind] ≈ β
      @test locpoly(X, z, y, z0, degree, gk) ≈ β
    end
  end
end

@testset "locpolyl1" begin
  n, s, p = 500, 2, 10
  X = randn(n, p)
  z = rand(n)
  y = sin.(4 * z) .* X[:, 1] + cos.(2 * z) .* X[:, 2] + 0.1 * randn(n)
  zgrid = collect(0.1:0.2:0.9)
  gk = GaussianKernel(0.1)
  opt = CDOptions(;maxIter=5000, optTol=1e-10)

  for degree=0:2
    cp = p*(degree+1)
    eX = zeros(n, cp)
    stdX = zeros(cp)

    λ0 = 0.05
    B, BR = locpolyl1(X, z, y, zgrid, degree, gk, λ0, true, opt)
    @test size(B) == size(BR) == (cp, length(zgrid))

    for (ind, z0) in enumerate(zgrid)
      w = evaluate.(Ref(gk), z, Ref(z0))
      CoordinateDescent._expand_X!(eX, X, z, z0, degree)
      CoordinateDescent._stdX!(stdX, w, eX)

      # KKT conditions of the weighted lasso at z0
      r = y - eX * B[:, ind]
      @test maximum(abs.(eX' * (w .* r) / n) ./ (λ0 * stdX)) ≈ 1. atol=1e-5

      # refit is weighted least squares on the selected groups
      S = CoordinateDescent.get_nonzero_coordinates(B[:, ind], p, degree, true)
      @test any(S)
      Xs = eX[:, S]
      @test Vector(BR[S, ind]) ≈ (Xs' * Diagonal(w) * Xs) \ (Xs' * Diagonal(w) * y)
      @test iszero(BR[.!S, ind])
    end
  end
end


end
