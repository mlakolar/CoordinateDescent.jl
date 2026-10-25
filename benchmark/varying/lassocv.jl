# Leave-one-out cross-validation of the varying-coefficient lasso at a moderate
# and a small penalty level, the case #38 found slow: n = 100 observations,
# p = 60 variables, degree 1 (120 columns in each local fit) and three
# bandwidths. The old lvocv_locpolyl1 solves a scaled lasso (penalty λ σ̂), the
# new varyingCoefficientLassoCV a lasso or group lasso, so the times are
# compared, not the errors.
#
#     julia --project=benchmark/varying/old benchmark/varying/lassocv.jl old
#     julia --project=benchmark/varying/new benchmark/varying/lassocv.jl new

if Base.find_package("HighDimensionalModels") !== nothing
  @eval using HighDimensionalModels
  const CoordinateDescent = HighDimensionalModels
else
  @eval using CoordinateDescent
end
using Printf, Random
const NEW = ARGS[1] == "new"
NEW && @eval using KernelSmoothing: EpanechnikovKernel as KSEpa

rng = Xoshiro(4)
n, p = 100, 60
z = rand(rng, n)
X = [ones(n) randn(rng, n, p - 1)]
y = X[:, 1] .* sin.(2π .* z) .+ X[:, 2] .* (4 .* z .* (1 .- z)) .+ 0.3 .* randn(rng, n)
hs = [0.2, 0.3, 0.4]

if NEW
  run(λ, penalty) = varyingCoefficientLassoCV(X, z, y, hs, λ; kernel = KSEpa(), penalty).cvm
  run(0.1, :lasso)
  for λ in (0.1, 0.003), penalty in (:lasso, :group)
    t = @elapsed cvm = run(λ, penalty)
    @printf "new %-7s λ = %-6g %7.2f s  errors %s\n" penalty λ t round.(cvm; digits = 3)
  end
else
  run(λ) = CoordinateDescent.lvocv_locpolyl1(X, z, y, 1, hs, EpanechnikovKernel{Float64}, λ) ./ n
  run(0.1)
  for λ in (0.1, 0.003)
    t = @elapsed cvm = run(λ)
    @printf "old scaled  λ = %-6g %7.2f s  errors %s\n" λ t round.(cvm; digits = 3)
  end
end
