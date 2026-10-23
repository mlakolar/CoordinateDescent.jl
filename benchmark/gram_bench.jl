# Compares lasso and LassoPath on the Gram matrix (gram = true) and on the
# residuals (gram = false), to calibrate gram = :auto. For each design, the lasso
# with an intercept at 0.1 and 0.02 λmax, and a 100-level LassoPath; the shortest
# time of three interleaved runs of each form, the passes and the largest
# difference of the coefficients. GRAM_BENCH_QUICK=1 runs small sizes, as a check
# that the script runs.
#
#     julia --project=benchmark benchmark/gram_bench.jl

using CoordinateDescent, LinearAlgebra, Printf, Random, Statistics

LinearAlgebra.BLAS.set_num_threads(1)

const QUICK = get(ENV, "GRAM_BENCH_QUICK", "0") == "1"
const NS = QUICK ? (200, 600) : (1000, 5000, 20000)
const PS = QUICK ? (20, 100) : (50, 200, 1000)

# n × p columns, independent or AR(1) with correlation 0.5 between neighbors,
# and ten nonzero coefficients
function design(rng, n, p, kind)
  X = randn(rng, n, p)
  if kind == :ar1
    for j in 2:p
      @views X[:, j] .= 0.5 .* X[:, j-1] .+ sqrt(0.75) .* X[:, j]
    end
  end
  β = zeros(p)
  β[round.(Int, range(1, p; length = min(10, p)))] .= 1
  X, 2.0 .+ X * β + randn(rng, n)
end

function besttime(f)
  t = Inf
  local r
  for _ in 1:3
    t = min(t, @elapsed(r = f()))
  end
  r, t
end

function run(rng)
  @printf("%6s %5s %-5s %-10s %10s %10s %6s %8s %8s %9s %5s\n", "n", "p", "X", "problem", "resid (s)", "gram (s)",
          "ratio", "passes", "passes", "max |Δβ|", "auto")
  for n in NS, p in PS, kind in (:iid, :ar1)
    X, y = design(rng, n, p, kind)
    λmax = maximum(abs, X' * (y .- mean(y))) / n
    auto1 = CoordinateDescent._usegram(:auto, X, false)
    autoP = CoordinateDescent._usegram(:auto, X, true)
    # warm-up of both forms at this size, outside the timings
    lasso(X, y, λmax / 2; intercept = true, gram = false); lasso(X, y, λmax / 2; intercept = true, gram = true)
    for frac in (0.1, 0.02)
      s1, t1 = besttime(() -> lasso(X, y, frac * λmax; intercept = true, gram = false))
      s2, t2 = besttime(() -> lasso(X, y, frac * λmax; intercept = true, gram = true))
      @printf("%6d %5d %-5s %-10s %10.4f %10.4f %6.2f %8d %8d %9.1e %5s\n", n, p, kind, "λ = $(frac)λmax", t1, t2,
              t2 / t1, s1.iterations, s2.iterations, norm(Vector(s1.x) - Vector(s2.x), Inf), auto1)
    end
    P1, t1 = besttime(() -> LassoPath(X, y; nλ = 100, intercept = true, gram = false))
    P2, t2 = besttime(() -> LassoPath(X, y; nλ = 100, intercept = true, gram = true))
    m = min(length(P1.βpath), length(P2.βpath))
    d = maximum(k -> norm(Vector(P1.βpath[k]) - Vector(P2.βpath[k]), Inf), 1:m)
    @printf("%6d %5d %-5s %-10s %10.4f %10.4f %6.2f %8d %8d %9.1e %5s\n", n, p, kind, "path 100", t1, t2, t2 / t1,
            sum(P1.iterations), sum(P2.iterations), d, autoP)
    flush(stdout)
  end
end

run(Xoshiro(2026))
