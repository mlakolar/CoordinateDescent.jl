module CoordinateDescentTest

const  NUMBER_REPEAT = 1

using CoordinateDescent
using Test
using ProximalBase
using Random
using LinearAlgebra
using SparseArrays

Random.seed!(1)


  # check that the warm start and non warm start produce the same result
@testset "ProxL1" begin
    for i=1:NUMBER_REPEAT
      n = 500
      p = 500
      s = 50

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      λ = 0.01
      g = ProximalBase.ProxL1(λ)
      f = CDLeastSquaresLoss(Y, X)

      opt1 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=false)
      opt2 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=true)
      opt3 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=false)
      opt4 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=true)

      x1 = SparseIterate(sprand(p, 0.6))
      x2 = SparseIterate(sprand(p, 0.6))
      x3 = SparseIterate(sprand(p, 0.6))
      x4 = SparseIterate(sprand(p, 0.6))

      coordinateDescent!(x1, f, g, opt1)
      coordinateDescent!(x2, f, g, opt2)
      coordinateDescent!(x3, f, g, opt3)
      coordinateDescent!(x4, f, g, opt4)

      @test Vector(x1) ≈ Vector(x2) atol=1e-5
      @test Vector(x3) ≈ Vector(x2) atol=1e-5
      @test Vector(x4) ≈ Vector(x2) atol=1e-5
    end
end
#
#   context("AProxL1") do
#     for i=1:NUMBER_REPEAT
#       n = 500
#       p = 500
#       s = 50
#
#       X = randn(n, p)
#       β = randn(s)
#       Y = X[:,1:s] * β + randn(n)
#
#       λ = 0.01
#       g = ProximalBase.ProxL1(λ, rand(p))
#       f = CDLeastSquaresLoss(Y, X)
#
#       opt1 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=false)
#       opt2 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=true)
#       opt3 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=false)
#       opt4 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=true)
#
#       x1 = convert(SparseIterate, sprand(p, 0.6))
#       x2 = convert(SparseIterate, sprand(p, 0.6))
#       x3 = convert(SparseIterate, sprand(p, 0.6))
#       x4 = convert(SparseIterate, sprand(p, 0.6))
#
#       coordinateDescent!(x1, f, g, opt1)
#       coordinateDescent!(x2, f, g, opt2)
#       coordinateDescent!(x3, f, g, opt3)
#       coordinateDescent!(x4, f, g, opt4)
#
#       @fact full(x1) ≈ full(x2); atol=1e-5)
#       @fact full(x3) ≈ full(x2); atol=1e-5)
#       @fact full(x4) ≈ full(x2); atol=1e-5)
#     end
#   end
# end
#
# facts("weighted least squares loss") do
#
#   n, s, p = 500, 2, 50
#   gk = GaussianKernel(0.3)
#   w = zeros(n)
#   sw = zeros(n)
#
#   for i=1:NUMBER_REPEAT
#     for degree=0:2
#       Y, X, Z, betaMat = genData(n, s)
#       X = [X zeros(n, p-s)]
#       z0 = rand()
#
#       cp = p*(degree+1)
#       eX = zeros(n, cp)
#       _expand_X!(eX, X, Z, z0, degree)
#
#       @. w = evaluate(gk, Z, z0)
#       @. sw = sqrt(w)
#
#       λ = 0.001
#       g = ProximalBase.ProxL1(λ)
#       f1 = CDLeastSquaresLoss(diagm(sw) * Y, diagm(sw)*X)
#       f2 = CDWeightedLSLoss(Y, X, w)
#
#       opt1 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=false)
#       opt2 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=true)
#       opt3 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=false)
#       opt4 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=true)
#
#       x1 = convert(SparseIterate, sprand(p, 0.6))
#       x2 = convert(SparseIterate, sprand(p, 0.6))
#       x3 = convert(SparseIterate, sprand(p, 0.6))
#       x4 = convert(SparseIterate, sprand(p, 0.6))
#       x5 = convert(SparseIterate, sprand(p, 0.6))
#       x6 = convert(SparseIterate, sprand(p, 0.6))
#       x7 = convert(SparseIterate, sprand(p, 0.6))
#       x8 = convert(SparseIterate, sprand(p, 0.6))
#
#       coordinateDescent!(x1, f1, g, opt1)
#       coordinateDescent!(x2, f1, g, opt2)
#       coordinateDescent!(x3, f1, g, opt3)
#       coordinateDescent!(x4, f1, g, opt4)
#
#       coordinateDescent!(x5, f2, g, opt1)
#       coordinateDescent!(x6, f2, g, opt2)
#       coordinateDescent!(x7, f2, g, opt3)
#       coordinateDescent!(x8, f2, g, opt4)
#
#       @fact full(x1) ≈ full(x2); atol=1e-5)
#       @fact full(x3) ≈ full(x2); atol=1e-5)
#       @fact full(x4) ≈ full(x2); atol=1e-5)
#       @fact full(x5) ≈ full(x2); atol=1e-5)
#       @fact full(x6) ≈ full(x2); atol=1e-5)
#       @fact full(x7) ≈ full(x2); atol=1e-5)
#       @fact full(x8) ≈ full(x2); atol=1e-5)
#     end
#   end
#
# end

end
