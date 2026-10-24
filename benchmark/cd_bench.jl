# Times the lasso solvers on one problem with n = 3000, p = 5000 and 100 nonzero
# coefficients. CD_BENCH_N and CD_BENCH_P set a smaller problem, as a quick check
# that the script runs:
#
#     CD_BENCH_N=300 CD_BENCH_P=500 julia --project=benchmark benchmark/cd_bench.jl

import ProximalBase
import CoordinateDescent
using LinearAlgebra, Printf, Random, Statistics

n = parse(Int, get(ENV, "CD_BENCH_N", "3000"))
p = parse(Int, get(ENV, "CD_BENCH_P", "5000"))
s = min(100, p ÷ 10)

Random.seed!(123)
X = randn(n, p)
Y = X[:,1:s] * (randn(s) .* (1. .+ rand(s))) + 6. * randn(n)

stdX = vec(std(X, dims=1))

options = CoordinateDescent.IterLassoOptions(;optTol=1e-3, maxIter=50)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time CoordinateDescent.scaledLasso!(x, X, Y, λ, stdX, options)


@show σinit = std(CoordinateDescent._findInitResiduals!(X, Y, 30, zeros(n)))
options = CoordinateDescent.IterLassoOptions(;optTol=1e-2, maxIter=10, initProcedure=:InitStd, σinit=σinit)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time CoordinateDescent.scaledLasso!(x, X, Y, λ, stdX, options)


# The lasso at fractions of λmax, the smallest penalty with a zero solution
# (for unit and for standardized loadings), from zero and along a path of 50
# penalty levels from λmax
function lasso_table(X, Y, stdX)
  n, p = size(X)
  f = CoordinateDescent.CDLeastSquaresLoss(Y, X)
  c = abs.(X' * Y) ./ n
  println("  λ / λmax  loadings  start              time (s)  passes  nonzeros")
  for frac in (0.05, 0.01),
      (loadings, g) in (("unit", ProximalBase.NormL1(frac * maximum(c))),
                        ("stdX", ProximalBase.NormL1(frac * maximum(c ./ stdX), stdX))),
      (start, options) in (("from zero", CoordinateDescent.CDOptions(; warmStart=true)),
                           ("path, 50 levels", CoordinateDescent.CDOptions(; warmStart=false, numSteps=50)))
    x = ProximalBase.SparseIterate(p)
    t = @elapsed r = CoordinateDescent.solve!(x, f, g, options)
    @printf("  %8.2f  %-8s  %-16s  %9.2f  %6d  %8d\n", frac, loadings, start, t, r.passes, count(!iszero, x))
  end
end

lasso_table(X, Y, stdX)
