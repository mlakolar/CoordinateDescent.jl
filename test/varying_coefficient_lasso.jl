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
#
# @testset "locpoly" begin
#
#   n, p = 500, 2
#   Y, X, Z, betaMat = genData(n, p)
#   zgrid = collect(0.01:0.2:0.99)
#
#   gk = GaussianKernel(0.4)
#
#   w = zeros(n)
#
#   for degree=0:2
#     @test locpoly(X, Z, Y, zgrid, degree, gk) ≈ locpoly_alt(X, Z, Y, zgrid, degree, gk)
#
#     z0 = 0.5
#     cp = p*(degree+1)
#
#     eX = zeros(n, cp)
#     _expand_X!(eX, X, Z, z0, degree)
#
#     @. w = evaluate(gk, Z, z0)
#
#     @test locpoly(X, Z, Y, z0, degree, gk) ≈ (eX' * diagm(w) * eX)\(eX' * diagm(w) * Y)
#   end
# end
#
#
#
# @testset "locpolyl1" begin
#
#   n, s, p = 500, 10, 50
#   gk = GaussianKernel(0.1)
#   zgrid = collect(0.01:0.1:0.99)
#   opt = CDOptions(;randomize=false)
#
#   for i=1:NUMBER_REPEAT
#     Y, X, Z, betaT = genData(n, s)
#     X = [X zeros(n, p-s)]
#
#     λ0 = rand() / 10
#
#     for degree=0:2
#       o1 = locpolyl1(X,Z,Y,zgrid,degree,gk,λ0, opt)
#       o2 = locpolyl1_alt(X,Z,Y,zgrid,degree,gk,λ0, opt)
#
#       @test maximum( maximum(abs.(o1[i] - o2[i])) for i=1:length(zgrid) ) ≈ 0. atol=1e-4
#     end
#   end
#
# end



end
