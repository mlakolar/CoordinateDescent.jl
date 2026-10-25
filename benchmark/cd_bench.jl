# Times the lasso solvers on one problem with n = 3000, p = 5000 and 100 nonzero
# coefficients. CD_BENCH_N and CD_BENCH_P set a smaller problem, as a quick check
# that the script runs:
#
#     CD_BENCH_N=300 CD_BENCH_P=500 julia --project=benchmark benchmark/cd_bench.jl

import ProximalBase
import HighDimensionalModels
using LinearAlgebra, Printf, Random, Statistics

n = parse(Int, get(ENV, "CD_BENCH_N", "3000"))
p = parse(Int, get(ENV, "CD_BENCH_P", "5000"))
s = min(100, p ÷ 10)

Random.seed!(123)
X = randn(n, p)
Y = X[:,1:s] * (randn(s) .* (1. .+ rand(s))) + 6. * randn(n)

stdX = vec(std(X, dims=1))

options = HighDimensionalModels.IterLassoOptions(;opt_tol=1e-3, max_iter=50)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time HighDimensionalModels.scaled_lasso!(x, X, Y, λ, stdX, options)


@show σinit = std(HighDimensionalModels._find_init_residuals!(X, Y, 30, zeros(n)))
options = HighDimensionalModels.IterLassoOptions(;opt_tol=1e-2, max_iter=10, init_procedure=:InitStd, σinit=σinit)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time HighDimensionalModels.scaled_lasso!(x, X, Y, λ, stdX, options)


# The lasso at fractions of λmax, the smallest penalty with a zero solution
# (for unit and for standardized loadings), from zero and along a path of 50
# penalty levels from λmax
function lasso_table(X, Y, stdX)
  n, p = size(X)
  f = HighDimensionalModels.CDLeastSquaresLoss(Y, X)
  c = abs.(X' * Y) ./ n
  println("  λ / λmax  loadings  start              time (s)  passes  nonzeros")
  for frac in (0.05, 0.01),
      (loadings, g) in (("unit", ProximalBase.NormL1(frac * maximum(c))),
                        ("stdX", ProximalBase.NormL1(frac * maximum(c ./ stdX), stdX))),
      (start, options) in (("from zero", HighDimensionalModels.CDOptions(; warm_start=true)),
                           ("path, 50 levels", HighDimensionalModels.CDOptions(; warm_start=false, num_steps=50)))
    x = ProximalBase.SparseIterate(p)
    t = @elapsed r = HighDimensionalModels.solve!(x, f, g, options)
    @printf("  %8.2f  %-8s  %-16s  %9.2f  %6d  %8d\n", frac, loadings, start, t, r.passes, count(!iszero, x))
  end
end

lasso_table(X, Y, stdX)
