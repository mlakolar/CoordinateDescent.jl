# Issue #73: run this script in environments pointing at the base and patched
# CoordinateDescent checkouts, with identical ProximalBase/ProximalSolvers.
# Includes construction, one BLAS thread, fixed data and solver seeds, and a
# warm-up before the reported repetitions. No cluster is needed.
#
#   julia --project=benchmark benchmark/path_setup_bench.jl
#   CD_PATH_BENCH_QUICK=1 julia --project=benchmark benchmark/path_setup_bench.jl
#
# CD_PATH_BENCH_REPS sets the number of measured repetitions (default 3).
# CD_PATH_BENCH_OUTPUT optionally saves the CV grid and errors for comparison.
using CoordinateDescent, LinearAlgebra, Random, Statistics, Printf

BLAS.set_num_threads(1)
const QUICK = get(ENV, "CD_PATH_BENCH_QUICK", "0") == "1"
const REPS = parse(Int, get(ENV, "CD_PATH_BENCH_REPS", "3"))

function design(n, p; seed = 100 + n + p)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  for j in 2:p
    @views X[:, j] .= 0.5 .* X[:, j-1] .+ sqrt(0.75) .* X[:, j]
  end
  β = zeros(p)
  β[randperm(rng, p)[1:min(20, p)]] .= rand(rng, (-1.0, 1.0), min(20, p))
  signal = X * β
  y = signal .+ sqrt(sum(abs2, signal) / n / 4) .* randn(rng, n)
  yb = Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-0.5 .* signal)))
  X, y, yb
end

function measure(name, f)
  f()
  times, bytes = Float64[], Int[]
  local result
  for _ in 1:REPS
    GC.gc()
    stats = @timed f()
    result = stats.value
    push!(times, stats.time)
    push!(bytes, stats.bytes)
  end
  @printf("%-26s median %.6f s, %.3f MB; times = %s\n", name, median(times), median(bytes) / 1e6, times)
  flush(stdout)
  result
end

function main()
  X, y, _ = design(QUICK ? 120 : 1000, QUICK ? 120 : 1000)
  cv = measure("CVLassoPath square", () -> CVLassoPath(X, y; threads = false, rng = Xoshiro(1)))
  @printf("  levels=%d, λmin=%.10g, λ1se=%.10g, minimum CV MSE=%.10g\n",
          length(cv.λpath), cv.λmin, cv.λ1se, minimum(cv.cvm))
  if haskey(ENV, "CD_PATH_BENCH_OUTPUT")
    open(ENV["CD_PATH_BENCH_OUTPUT"], "w") do io
      println(io, "lambda,cvm,cvse")
      for k in eachindex(cv.λpath)
        @printf(io, "%.17g,%.17g,%.17g\n", cv.λpath[k], cv.cvm[k], cv.cvse[k])
      end
    end
  end

  Xg, _, yg = design(QUICK ? 100 : 200, QUICK ? 300 : 10000)
  # One level isolates setup; the elastic net also exercises column scaling.
  for alpha in (1.0, 0.7)
    path = measure("GLMLassoPath setup α=$alpha", () ->
      GLMLassoPath(Xg, yg; nλ = 1, alpha, rng = Xoshiro(1)))
    @assert all(path.converged)
  end
  path = measure("GLMLassoPath 100 levels", () -> GLMLassoPath(Xg, yg; rng = Xoshiro(1)))
  @printf("  levels=%d, all converged=%s, final deviance ratio=%.10g\n",
          length(path.λpath), all(path.converged), last(path.devratio))
end

main()
