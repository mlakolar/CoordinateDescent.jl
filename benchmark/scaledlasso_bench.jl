# The scaled lasso: the algorithm of version 0.6.0, which alternated lasso fits
# with updates of σ̂, against the square-root lasso solved in one run of
# coordinate descent, from the data and from sufficient statistics
# (CDSqrtQuadraticLoss).
#
#   julia --project=benchmark benchmark/scaledlasso_bench.jl [quick]
#
# Part 1, regression: n × p designs with independent or AR(1) columns, 10
# nonzero coefficients, unit noise, λ = √(2 log p / n) and the column standard
# deviations as loadings. Part 2, the precision matrix estimator of Sun and
# Zhang (2013): the scaled lasso of every variable on the others, for AR(1),
# band and Erdős–Rényi precision matrices; from the data it copies the other
# columns for every variable, from sufficient statistics all the columns share
# S = X'X/n, with an infinite weight on the response. Times are the best of up
# to 5 runs within a budget (one run for long ones), on one thread; the
# accuracy is the largest difference of the coefficients from a solution at
# optTol = 1e-12 (relative to the largest coefficient), and of σ̂.

using CoordinateDescent, ProximalBase, ProximalSolvers
using LinearAlgebra, Printf, Random, SparseArrays, Statistics

const QUICK = "quick" in ARGS
BLAS.set_num_threads(1)

function best_time(f; runs=5, budget=2.0, warmup=true)
  v = warmup ? f() : nothing
  best, total = Inf, 0.0
  for _ in 1:runs
    t = @elapsed v = f()
    best = min(best, t)
    total += t
    total > budget && break
  end
  v, best
end

# the scaledLasso! of CoordinateDescent 0.6.0: lasso fits at the penalty λσ̂,
# alternated with σ̂ = ‖y - Xβ̂‖/√n until the relative change of σ̂ is below
# optTol, starting from the residual standard deviation of least squares on the
# 5 columns most correlated with y
function alternating!(x, X, y, λ, ω, optTol; maxIter=100, optionsCD=CDOptions())
  n = size(X, 1)
  f = CDLeastSquaresLoss(y, X)
  σ = std(CoordinateDescent._findInitResiduals!(X, y, 5, copy(y)))
  info = CDInfo()
  passes = 0
  for _ in 1:maxIter
    coordinateDescent!(x, f, NormL1(λ * σ, ω), optionsCD, info)
    passes += info.iterations
    σnew = norm(ProximalSolvers.residuals(f)) / sqrt(n)
    done = abs(σnew - σ) / σ < optTol
    σ = σnew
    done && break
  end
  x, σ, passes
end

# columns with correlations ρ^|j - k|
function regression(n, p, ρ, rng)
  X = randn(rng, n, p)
  for j in 2:p
    X[:, j] .= ρ .* X[:, j-1] .+ sqrt(1 - ρ^2) .* X[:, j]
  end
  β = zeros(p)
  β[1:10] .= 1
  y = X * β + randn(rng, n)
  X, y
end

function part1()
  println("## Regression\n")
  println("Seconds, and in parentheses the largest difference of the coefficients from the solution:\n")
  println("| n | p | ρ | 0.6.0, optTol = 1e-2 | 0.6.0, optTol = 1e-6 | scaledLasso | from A, b, c | A, b, c | passes 0.6.0 / new |")
  println("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
  rng = Xoshiro(1)
  sizes = QUICK ? ((300, 100),) : ((1000, 200), (5000, 500), (1000, 1000), (200, 1000), (200, 5000))
  for (n, p) in sizes, ρ in (0.0, 0.5, 0.9)
    X, y = regression(n, p, ρ, rng)
    λ = sqrt(2 * log(p) / n)
    ω = vec(sqrt.(sum(abs2, X; dims=1) / n))
    tight = IterLassoOptions(; optionsCD=CDOptions(; optTol=1e-12, maxIter=100_000))
    ref = scaledLasso(X, y, λ, ω, tight)
    scale = maximum(abs, ref.x)
    err(x) = maximum(abs, Vector(x) - Vector(ref.x)) / scale
    (x1, _, passes1), t1 = best_time(() -> alternating!(SparseIterate(p), X, y, λ, ω, 1e-2))
    (x2, _, _), t2 = best_time(() -> alternating!(SparseIterate(p), X, y, λ, ω, 1e-6))
    s3, t3 = best_time(() -> scaledLasso(X, y, λ, ω))
    (A, b, c), tabc = best_time(() -> (X'X / n, -X'y / n, dot(y, y) / n))
    s4, t4 = best_time(() -> scaledLasso!(SparseIterate(p), CDSqrtQuadraticLoss(A, b, c), λ, ω))
    @printf("| %d | %d | %.1f | %.3g (%.0e) | %.3g (%.0e) | %.3g (%.0e) | %.3g (%.0e) | %.3g | %d / %d |\n",
            n, p, ρ, t1, err(x1), t2, err(x2), t3, err(s3.x), t4, err(s4.x), tabc, passes1, s3.iterations)
  end
end

function precision(model, p, rng)
  if model == "ar1"
    ρ = 0.6
    Ω = Matrix(SymTridiagonal(fill(1 + ρ^2, p), fill(-ρ, p - 1))) ./ (1 - ρ^2)
    Ω[1, 1] = Ω[p, p] = 1 / (1 - ρ^2)
    return Ω
  elseif model == "band"
    Ω = Matrix(1.0I, p, p)
    for i in 1:p-1; Ω[i, i+1] = Ω[i+1, i] = 0.6; end
    for i in 1:p-2; Ω[i, i+2] = Ω[i+2, i] = 0.3; end
    return Ω
  end
  B = zeros(p, p)
  for c in 1:p, r in c+1:p
    if rand(rng) < min(1.0, 4 / p)
      B[r, c] = B[c, r] = 0.4 + 0.4 * rand(rng)
    end
  end
  B + (abs(min(eigmin(Symmetric(B)), 0)) + 0.05) * I
end

# the coefficients and σ̂ of every column, by a method f(j, others) that returns
# the coefficients of the other variables, in the order of others, and σ̂
function columns(f, p)
  B = zeros(p, p)
  σ = zeros(p)
  for j in 1:p
    others = [1:j-1; j+1:p]
    x, σ[j] = f(j, others)
    B[others, j] .= x
  end
  B, σ
end

function part2()
  println("\n## The precision matrix estimator of Sun and Zhang (2013): all p columns\n")
  println("Seconds, and in parentheses the largest difference of the coefficients from the solution:\n")
  println("| problem | 0.6.0, optTol = 1e-2 | 0.6.0, optTol = 1e-6 | scaledLasso | from S |")
  println("|---|---:|---:|---:|---:|")
  rng = Xoshiro(2)
  sizes = QUICK ? ((50, 25),) : ((200, 100), (200, 400), (500, 250), (500, 1000), (1000, 500), (1000, 2000))
  for (p, n) in sizes, model in ("ar1", "band", "er")
    Ω = precision(model, p, rng)
    X = randn(rng, n, p) / cholesky(Symmetric(Ω)).U'
    S = X'X / n
    λ = sqrt(2 * log(p) / n)
    w = sqrt.(diag(S))
    # from the data: the regression of x_j on the other columns, copied
    function data(j, others, solve)
      x, σ = solve(X[:, others], X[:, j], w[others])
      Vector(x), σ
    end
    function fromS(j, others; options=CDOptions())
      ω = copy(w)
      ω[j] = Inf
      s = scaledLasso!(SparseIterate(p), CDSqrtQuadraticLoss(S, -S[:, j], S[j, j]; check=false), λ, ω, options)
      Vector(s.x)[others], s.σ
    end
    ref, _ = columns((j, others) -> fromS(j, others; options=CDOptions(; optTol=1e-12, maxIter=100_000)), p)
    scale = maximum(abs, ref)
    err(B) = maximum(abs, B - ref) / scale
    (B1, _), t1 = best_time(() -> columns((j, o) -> data(j, o, (Xo, y, ω) -> begin
        x, σ, _ = alternating!(SparseIterate(p - 1), Xo, y, λ, ω, 1e-2); (x, σ) end), p); runs=1, budget=0, warmup=false)
    (B2, _), t2 = best_time(() -> columns((j, o) -> data(j, o, (Xo, y, ω) -> begin
        x, σ, _ = alternating!(SparseIterate(p - 1), Xo, y, λ, ω, 1e-6); (x, σ) end), p); runs=1, budget=0, warmup=false)
    (B3, _), t3 = best_time(() -> columns((j, o) -> data(j, o, (Xo, y, ω) -> begin
        s = scaledLasso(Xo, y, λ, ω); (s.x, s.σ) end), p); runs=1, budget=0, warmup=false)
    (B4, _), t4 = best_time(() -> columns(fromS, p); runs=3, budget=5.0)
    name = Dict("ar1" => "AR(1)", "band" => "band", "er" => "Erdős–Rényi")[model]
    @printf("| %s, p = %d, n = %d | %.3g (%.0e) | %.3g (%.0e) | %.3g (%.0e) | %.3g (%.0e) |\n",
            name, p, n, t1, err(B1), t2, err(B2), t3, err(B3), t4, err(B4))
  end
end

println("Julia $VERSION, CoordinateDescent $(pkgversion(CoordinateDescent)), one thread\n")
part1()
part2()
