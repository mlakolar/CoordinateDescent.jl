module VarCoeffLassoTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays
using Statistics

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

@testset "get_beta!" begin
  zgrid = [0., 0.5, 1.]
  B = [0. 1. 3.;
       2. 2. 0.]
  out = zeros(2)

  @test CoordinateDescent.get_beta!(out, zgrid, B, 0.5) == [1., 2.]       # grid point
  @test CoordinateDescent.get_beta!(out, zgrid, B, 0.1) ≈ [0.2, 2.]       # interpolation
  @test CoordinateDescent.get_beta!(out, zgrid, B, 0.75) ≈ [2., 1.]
  @test CoordinateDescent.get_beta!(out, zgrid, B, -0.2) == [0., 2.]      # outside the grid
  @test CoordinateDescent.get_beta!(out, zgrid, B, 1.3) == [3., 0.]
  @test CoordinateDescent.get_beta!(out, zgrid, sparse(B), 0.1) ≈ [0.2, 2.]
end

@testset "get_nonzero_coordinates" begin
  p, degree = 3, 1
  β = SparseIterate(6)
  β[2] = 1.                           # variable 1 (derivative only)
  β[5] = -1.                          # variable 3
  @test CoordinateDescent.get_nonzero_coordinates(β, p, degree, false) == [true, false, true]
  @test CoordinateDescent.get_nonzero_coordinates(β, p, degree, true) == [true, true, false, false, true, true]
  @test CoordinateDescent.get_nonzero_coordinates(sparsevec(Vector(β)), p, degree, false) == [true, false, true]
end

@testset "varying coefficient utilities" begin
  n, p = 200, 3
  X = randn(n, p)
  z = rand(n)
  y = sin.(4 * z) .* X[:, 1] + cos.(2 * z) .* X[:, 2] + 0.1 * randn(n)
  gk = GaussianKernel(0.1)
  z0 = 0.4

  @testset "standard errors" begin
    for degree=0:2
      cp = p*(degree+1)
      eX = zeros(n, cp)
      CoordinateDescent._expand_X!(eX, X, z, z0, degree)
      w = evaluate.(Ref(gk), z, Ref(z0))
      A = inv(eX' * Diagonal(w) * eX)
      ind = 1:(degree+1):cp

      σ2 = 2.5
      V = σ2 * A * (eX' * Diagonal(w.^2) * eX) * A
      @test CoordinateDescent.getStandardError(X, z, σ2, z0, degree, gk) ≈ diag(V)[ind]
      @test CoordinateDescent.getStandardError(view(X, :, 1:p), z, σ2, z0, degree, gk) ≈ diag(V)[ind]

      ϵ2 = rand(n)
      V = A * (eX' * Diagonal(w.^2 .* ϵ2) * eX) * A
      @test CoordinateDescent.getStandardErrorHEW(X, z, ϵ2, z0, degree, gk) ≈ diag(V)[ind]
    end
  end

  @testset "residuals" begin
    zgrid = collect(0.:0.1:1.)
    degree = 1
    B = locpoly(X, z, y, zgrid, degree, gk)
    ϵ = CoordinateDescent.getResiduals!(zeros(n), X, z, y, zgrid, B, degree)

    βi = zeros(p*(degree+1))
    for i=1:n
      CoordinateDescent.get_beta!(βi, zgrid, B, z[i])
      @test ϵ[i] ≈ y[i] - dot(X[i, :], βi[1:2:end])
    end
    @test sqrt(mean(abs2, ϵ)) < 0.3
  end

  @testset "leave-one-out" begin
    m = 40
    Xs, zs, ys = X[1:m, :], z[1:m], y[1:m]
    hArr = [0.05, 0.2]
    degree = 1
    MSE = CoordinateDescent.lvocv_locpoly(Xs, zs, ys, degree, hArr, GaussianKernel{Float64})

    for (indH, h) in enumerate(hArr)
      err = 0.
      for i=1:m
        keep = setdiff(1:m, i)
        β = locpoly(Xs[keep, :], zs[keep], ys[keep], zs[i], degree, GaussianKernel(h))
        err += (ys[i] - dot(Xs[i, :], β[1:2:end]))^2
      end
      @test MSE[indH] ≈ err
    end
  end

  @testset "split" begin
    ntest = 50
    Xt, zt = randn(ntest, p), rand(ntest)
    yt = sin.(4 * zt) .* Xt[:, 1] + cos.(2 * zt) .* Xt[:, 2]
    zgrid = collect(0.:0.1:1.)
    hArr = [0.05, 0.1]
    MSE = CoordinateDescent.split_locpoly(X, z, y, Xt, zt, yt, zgrid, 1, hArr, GaussianKernel{Float64})

    for (indH, h) in enumerate(hArr)
      B = locpoly(X, z, y, zgrid, 1, GaussianKernel(h))
      βi = zeros(2p)
      err = 0.
      for i=1:ntest
        CoordinateDescent.get_beta!(βi, zgrid, B, zt[i])
        err += (yt[i] - dot(Xt[i, :], βi[1:2:end]))^2
      end
      @test MSE[indH] ≈ err
    end
  end
end

@testset "locpolyl1 refit and cross-validation" begin
  n, p = 300, 8
  X = randn(n, p)
  z = rand(n)
  y = sin.(4 * z) .* X[:, 1] + cos.(2 * z) .* X[:, 2] + 0.1 * randn(n)
  gk = GaussianKernel(0.1)
  degree = 1

  B, _ = locpolyl1(X, z, y, [0.5], degree, gk, 0.05, false)
  βr, S = CoordinateDescent.refit_locpolyl1(X, z, y, 0.5, degree, gk, B[:, 1])
  @test S == CoordinateDescent.get_nonzero_coordinates(B[:, 1], p, degree, false)
  @test S[1] && S[2]
  @test βr ≈ locpoly(X[:, S], z, y, 0.5, degree, gk)

  m = 60
  MSE = CoordinateDescent.lvocv_locpolyl1(X[1:m, :], z[1:m], y[1:m], degree, [0.1, 0.3], GaussianKernel{Float64}, 0.1)
  @test length(MSE) == 2
  @test all(isfinite, MSE) && all(>(0), MSE)
  @test MSE[1] / m < var(y[1:m])
end

end
