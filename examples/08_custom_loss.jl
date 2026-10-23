# A loss of your own: ℓ1-penalized logistic regression
#
#     minimize  ∑ᵢ [log(1 + exp(ηᵢ)) - yᵢ ηᵢ] / n + λ ∑ⱼ ωⱼ |βⱼ|,   η = Xβ, yᵢ ∈ {0, 1}
#
# coordinateDescent! minimizes any subtype of CoordinateDifferentiableFunction
# that implements
#
#     numCoordinates(f)                the number of coordinates
#     initialize!(f, x)                compute the cache of f (here η = Xx) for x
#     gradient(f, x, k)                ∂f/∂xₖ at x, from the cache
#     descendCoordinate!(f, g, x, k)   update x[k] and the cache; return the change
#
# and optionally coordinateScale(f, k) and responseScale(f), for a stopping
# rule in the units of the problem, and fullGradient!(grad, f, x), a faster
# way to compute the whole gradient. These functions are not exported: extend
# them as CoordinateDescent.initialize! and so on. Coordinate descent itself is
# ProximalOPT's, which calls descendCoordinate! for each step.
#
# The logistic loss is built into ProximalBase as well (MyLogisticLoss, a
# GLMLoss), and ProximalOPT minimizes it by coordinate descent or by proximal
# Newton; the last section checks the loss written here against them.
#
#     julia --project=examples examples/08_custom_loss.jl

using CoordinateDescent, ProximalBase, ProximalOPT
using LinearAlgebra, Printf, Random, Statistics

sigmoid(t) = 1 / (1 + exp(-t))

struct MyLogisticLoss{T<:AbstractFloat} <: CoordinateDifferentiableFunction
  y::Vector{T}          # responses in {0, 1}
  X::Matrix{T}
  η::Vector{T}          # the cache: the linear predictor Xx
  a::Vector{T}          # curvature bounds ‖X[:, k]‖² / (4n), see descendCoordinate!
end

MyLogisticLoss(y::Vector{T}, X::Matrix{T}) where {T} =
  MyLogisticLoss(y, X, zeros(T, length(y)), vec(sum(abs2, X; dims = 1)) / (4 * length(y)))

CoordinateDescent.numCoordinates(f::MyLogisticLoss) = size(f.X, 2)

# coordinateDescent! calls initialize! before its first pass, and x need not
# be zero
function CoordinateDescent.initialize!(f::MyLogisticLoss, x)
  f.η .= f.X * x
  nothing
end

# ∂f/∂xₖ = X[:, k]'(σ(η) - y) / n
function CoordinateDescent.gradient(f::MyLogisticLoss{T}, x, k) where {T}
  s = zero(T)
  @inbounds for i in eachindex(f.η)
    s += f.X[i, k] * (sigmoid(f.η[i]) - f.y[i])
  end
  s / length(f.η)
end

# Along coordinate k, f lies below the quadratic with curvature
# a = ‖X[:, k]‖² / (4n) that touches it at x, because σ' ≤ 1/4. The step
# minimizes that quadratic plus the penalty, a proximal step
# x[k] ← prox(x[k] - ∂f/∂xₖ / a) with step 1/a, so it never increases the
# objective. prox_entry is the proximal operator of the penalty on one entry,
# which makes the loss work with every separable penalty: NormL1,
# ZeroFunction, ElasticNet, MCP, …, and those of 09_custom_penalty.jl.
function CoordinateDescent.descendCoordinate!(f::MyLogisticLoss, g, x, k)
  a = f.a[k]
  old = x[k]
  h = prox_entry(g, old - CoordinateDescent.gradient(f, x, k) / a, k, 1 / a) - old
  if !iszero(h)                         # write x[k], and keep the cache current: η += h X[:, k]
    x[k] = old + h
    @inbounds for i in eachindex(f.η)
      f.η[i] += h * f.X[i, k]
    end
  end
  h
end

# With relativeTol (the default) the change Δxₖ is measured as
# |Δxₖ| coordinateScale(f, k) and compared with optTol responseScale(f).
# ‖X[:, k]‖ / √n is the root mean square change of η, and the default
# responseScale(f) = 1 compares it with optTol on the scale of the log-odds.
CoordinateDescent.coordinateScale(f::MyLogisticLoss, k) = sqrt(4 * f.a[k])

# the whole gradient X'(σ(η) - y) / n in one product, used by the screening
# and to find λmax when warmStart = false
function CoordinateDescent.fullGradient!(grad, f::MyLogisticLoss, x)
  mul!(grad, transpose(f.X), sigmoid.(f.η) .- f.y)
  grad ./= length(f.η)
end

# --- Using it -----------------------------------------------------------------

Random.seed!(8)

# 100 predictors, of which five matter, and an intercept of -0.5. The first
# column of X1 is the intercept, which the loadings ω₁ = 0 leave unpenalized.
n, p = 600, 100
X = randn(n, p)
β = zeros(p)
β[1:5] = [2.0, -1.5, 1.0, -1.0, 0.5]
y = Float64.(rand(n) .< sigmoid.(-0.5 .+ X * β))
X1 = [ones(n) X]
ω = [0.0; ones(p)]

println("== One λ")

f = MyLogisticLoss(y, X1)
λ = 0.05
x = SparseIterate(p + 1)
info = CDInfo()
coordinateDescent!(x, f, NormL1(λ, ω), CDOptions(), info)
println("selects ", findall(!iszero, x[2:end]), " in $(info.iterations) passes; intercept ", round(x[1]; digits = 2))

# the solution satisfies |∂f/∂βⱼ| ≤ λωⱼ where βⱼ = 0, and ∂f/∂βⱼ = -λωⱼ sign(βⱼ)
# elsewhere
grad = CoordinateDescent.fullGradient!(zeros(p + 1), f, x)
kkt = maximum(j -> iszero(x[j]) ? max(abs(grad[j]) - λ * ω[j], 0.0) : abs(grad[j] + λ * ω[j] * sign(x[j])), 1:p+1)
@printf("largest violation of the optimality conditions: %.1e\n", kkt)

# With warmStart = false, coordinate descent follows a path from λmax, which
# it finds with gradient, and screens each level with the strong rule, which
# uses fullGradient!: a user-defined loss gets both
xp = SparseIterate(p + 1)
coordinateDescent!(xp, f, NormL1(λ, ω), CDOptions(; warmStart = false), info)
@printf("from zero along a path: %d passes, solutions differ by %.1e\n", info.iterations, norm(xp - x, Inf))

println("\n== Choosing λ on a validation set")

# A path of λ, each fit started from the previous solution, scored by the
# log-loss on held-out data. It starts at λmax, the smallest λ at which only
# the intercept is nonzero: the largest |∂f/∂βⱼ| at the intercept-only fit
# β₀ = log(ȳ / (1 - ȳ)). The gradient is read from the cache, so f is first
# initialized at that point.
Xv = randn(2000, p)
yv = Float64.(rand(2000) .< sigmoid.(-0.5 .+ Xv * β))
logloss(η, y) = mean(@. log(1 + exp(η)) - y * η)
x = SparseIterate(p + 1)
x[1] = log(mean(y) / (1 - mean(y)))
CoordinateDescent.initialize!(f, x)
λmax = maximum(abs, CoordinateDescent.fullGradient!(zeros(p + 1), f, x)[2:end])
println("       λ  nonzeros  validation log-loss")
best = (Inf, 0.0)
for λk in λmax * exp10.(range(0, -2; length = 12))
  coordinateDescent!(x, f, NormL1(λk, ω))
  ll = logloss(x[1] .+ Xv * x[2:end], yv)
  @printf("  %6.4f  %8d  %19.4f\n", λk, count(!iszero, x[2:end]), ll)
  ll < best[1] && (global best = (ll, λk))
end
println("smallest validation log-loss at λ = ", round(best[2]; digits = 4))

println("\n== No penalty, checked against Newton's method")

# On the five relevant variables and the intercept, ZeroFunction() gives the
# maximum likelihood estimate, which Newton's method also finds
function newton(X, y; iterations = 25)
  b = zeros(size(X, 2))
  for _ in 1:iterations
    μ = sigmoid.(X * b)
    b += (X' * (μ .* (1 .- μ) .* X)) \ (X' * (y - μ))
  end
  b
end
Xs = X1[:, 1:6]
xs = SparseIterate(6)
coordinateDescent!(xs, MyLogisticLoss(y, Xs), ZeroFunction(), CDOptions(; optTol = 1e-10))
bn = newton(Xs, y)
println("coordinate descent ", round.(Vector(xs); digits = 3))
println("Newton             ", round.(bn; digits = 3))
@printf("difference %.1e\n", norm(xs - bn, Inf))

println("\n== The built-in logistic loss")

# (1/n) * LogisticLoss(y, X1) is the loss above. coordinateDescent! takes it
# too, and runs ProximalOPT's coordinate descent with the same curvature bound.
# ProximalOPT's proximal Newton instead fits a weighted lasso to a quadratic
# model of the loss at each step, as glmnet does.
x = SparseIterate(p + 1)
coordinateDescent!(x, f, NormL1(λ, ω), CDOptions(; optTol = 1e-10))
xb = SparseIterate(p + 1)
coordinateDescent!(xb, (1 / n) * LogisticLoss(y, X1), NormL1(λ, ω), CDOptions(; optTol = 1e-10))
xn = SparseIterate(p + 1)
solve!(xn, (1 / n) * LogisticLoss(y, X1), NormL1(λ, ω), ProxNewton())
@printf("built-in loss: difference %.1e; proximal Newton: difference %.1e\n", norm(xb - x, Inf), norm(xn - x, Inf))
