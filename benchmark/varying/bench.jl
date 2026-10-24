# The varying-coefficient code before and after it moved onto the kernels of
# KernelSmoothing.jl (#38), on the same data: locpoly against varyingCoefficient,
# lvocv_locpoly against varyingCoefficientCV, and locpolyl1 against
# varyingCoefficientLasso. The old EpanechnikovKernel(h) gives the same weights
# as KernelSmoothing's EpanechnikovKernel() with bandwidth h, so the unpenalized
# estimates and the cross-validation errors must agree. For the lasso the script
# compares the number of selected variables, summed over the grid points.
#
#     julia --project=benchmark/varying/old benchmark/varying/bench.jl old OUT
#     julia --project=benchmark/varying/new benchmark/varying/bench.jl new OUT

using CoordinateDescent, Printf, Random, Serialization
const NEW = ARGS[1] == "new"
NEW && @eval using KernelSmoothing: EpanechnikovKernel as KSEpa

function data(n, p; seed = 1)
  rng = Xoshiro(seed)
  z = rand(rng, n)
  X = [ones(n) randn(rng, n, p - 1)]
  y = X[:, 1] .* sin.(2π .* z) .+ X[:, 2] .* (4 .* z .* (1 .- z)) .+ 0.3 .* randn(rng, n)
  X, z, y
end

besttime(f; reps = 5) = (f(); minimum(@elapsed(f()) for _ in 1:reps))

results = Dict{String, Any}()
for (n, p) in ((500, 3), (2000, 3), (2000, 10))
  X, z, y = data(n, p)
  zgrid = collect(range(0.05, 0.95; length = 50))
  h = 0.15
  if NEW
    f() = varyingCoefficient(X, z, y, zgrid; kernel = KSEpa(), bandwidth = h)
    B = coef(f())
    results["fit $n $p"] = (besttime(f), B)
    results["fit+se $n $p"] = (besttime(() -> varyingCoefficient(X, z, y, zgrid; kernel = KSEpa(), bandwidth = h, se = true)), nothing)
  else
    g() = locpoly(X, z, y, zgrid, 1, EpanechnikovKernel(h))
    B = g()[1:2:end, :]
    results["fit $n $p"] = (besttime(g), B)
  end
end
for n in (300, 1000)
  X, z, y = data(n, 3; seed = 2)
  hs = [0.1, 0.2, 0.3]
  if NEW
    c() = varyingCoefficientCV(X, z, y, hs; kernel = KSEpa())
    results["cv $n"] = (besttime(c; reps = 2), c().cvm)
  else
    d() = CoordinateDescent.lvocv_locpoly(X, z, y, 1, hs, EpanechnikovKernel{Float64})
    results["cv $n"] = (besttime(d; reps = 2), d() ./ n)
  end
end
let n = 500, p = 50
  X, z, y = data(n, p; seed = 3)
  zgrid = collect(range(0.1, 0.9; length = 20))
  if NEW
    l1() = varyingCoefficientLasso(X, z, y, zgrid, 0.1; kernel = KSEpa(), bandwidth = 0.2, penalty = :lasso)
    l2() = varyingCoefficientLasso(X, z, y, zgrid, 0.1; kernel = KSEpa(), bandwidth = 0.2, penalty = :group)
    results["lasso :lasso"] = (besttime(l1; reps = 3), sum(l1().selected))
    results["lasso :group"] = (besttime(l2; reps = 3), sum(l2().selected))
  else
    l0() = locpolyl1(X, z, y, zgrid, 1, EpanechnikovKernel(0.2), 0.1, true)
    nsel() = (B = l0()[1]; sum(any(!iszero, B[2j-1:2j, k]) for j in 1:p, k in 1:20))
    results["lasso :lasso"] = (besttime(l0; reps = 3), nsel())
  end
end
serialize(ARGS[2], results)
for k in sort(collect(keys(results)))
  @printf "%-6s %-16s %.4f s\n" ARGS[1] k results[k][1]
end
