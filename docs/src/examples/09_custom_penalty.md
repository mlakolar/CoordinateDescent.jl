```@meta
EditURL = "../../../examples/09_custom_penalty.jl"
```

# A penalty of your own: the elastic net and the minimax concave penalty

A penalty g(x) = ∑ₖ gₖ(xₖ) that separates across coordinates plugs into
coordinateDescent! as a subtype of ProximalBase.AbstractFunction that says it
is separable and has the proximal operator of one entry, and the value of
the penalty, g(x):

    ProximalBase.isseparable(::Type{<:G}) = true
    ProximalBase.prox_entry(g, v, k, γ):   argmin_z gₖ(z) + (z - v)² / (2γ)

CDLeastSquaresLoss, CDWeightedLSLoss and CDQuadraticLoss take a proximal
step with it, and so does any loss whose coordinate step calls prox_entry,
such as the logistic loss of 08_custom_loss.jl. Path following from λmax
(warmStart = false) and screening are implemented for NormL1 and GroupNormL2
only: with another penalty, warmStart = false starts from zero and solves at
g directly, and a path is a loop over penalties with warm starts, as below.

ProximalBase has both penalties of this example built in, ElasticNet and
MCP; the example checks the ones written here against them.

````@example 09_custom_penalty
using CoordinateDescent, ProximalBase
import ProximalBase: isseparable, isproximable, prox_entry
using LinearAlgebra, Printf, Random, Statistics
````

## The elastic net, λ (α ‖x‖₁ + (1 - α) ‖x‖² / 2)

````@example 09_custom_penalty
struct MyElasticNet{T<:AbstractFloat} <: ProximalBase.AbstractFunction
  λ::T
  α::T
end

isseparable(::Type{<:MyElasticNet}) = true
isproximable(::Type{<:MyElasticNet}) = true
````

soft-thresholding followed by a shrinkage

````@example 09_custom_penalty
prox_entry(g::MyElasticNet, v, k, γ) = shrink(v, γ * g.λ * g.α) / (1 + γ * g.λ * (1 - g.α))
(g::MyElasticNet)(x) = g.λ * (g.α * norm(x, 1) + (1 - g.α) * sum(abs2, x) / 2)
````

## The minimax concave penalty of Zhang (2010)


    gₖ(z) = λ|z| - z² / (2γ) for |z| ≤ γλ, and γλ² / 2 otherwise,

which penalizes small coefficients like the lasso and large ones not at
all. It is not convex, and its proximal operator with step t is defined for
t < γ: zero for |v| ≤ tλ, (|v| - tλ) / (1 - t/γ) with the sign of v up to
γλ, and v beyond. The step of the least squares loss is n / ‖X[:, k]‖², so
with columns scaled to ‖X[:, k]‖² = n it is 1, and γ = 3 is a common choice.

````@example 09_custom_penalty
struct MyMCP{T<:AbstractFloat} <: ProximalBase.AbstractFunction
  λ::T
  γ::T
end

isseparable(::Type{<:MyMCP}) = true
isproximable(::Type{<:MyMCP}) = true

function prox_entry(g::MyMCP, v, k, t)
  t < g.γ || throw(ArgumentError("the MCP needs γ > $(round(t; sigdigits = 3)), the step of coordinate descent"))
  abs(v) <= t * g.λ ? zero(v) :
    abs(v) <= g.γ * g.λ ? sign(v) * (abs(v) - t * g.λ) / (1 - t / g.γ) : v
end
(g::MyMCP)(x) = sum(z -> abs(z) <= g.γ * g.λ ? g.λ * abs(z) - z^2 / (2 * g.γ) : g.γ * g.λ^2 / 2, x)
````

## The elastic net on correlated predictors

````@example 09_custom_penalty
Random.seed!(9)
````

three groups of five predictors with correlation 0.9 within a group, every
predictor of the groups with coefficient 1, and 85 predictors that do not
matter; the columns are scaled to ‖X[:, k]‖² = n

````@example 09_custom_penalty
n, p = 200, 100
X = randn(n, p)
for G in (1:5, 6:10, 11:15)
  X[:, G] = sqrt(0.9) * randn(n) .+ sqrt(0.1) * X[:, G]
end
X ./= sqrt.(sum(abs2, X; dims = 1) / n)
β = [ones(15); zeros(p - 15)]
y = X * β + 2 * randn(n)
````

## The elastic net

The lasso tends to pick a few predictors of a correlated group and drop the
others; the ridge part of the elastic net spreads the coefficients more
evenly over the group (Zou and Hastie, 2005). α = 1 is the lasso. At the
same λ, a smaller α also selects more of the predictors that do not matter:
in practice λ and α are chosen together, for example by cross-validation.

````@example 09_custom_penalty
f = CDLeastSquaresLoss(y, X)
println("  α   nonzeros  of the 15  β̂[1:5]                          sd of β̂[1:15]")
for α in (1.0, 0.5, 0.1)
  x = SparseIterate(p)
  coordinateDescent!(x, f, MyElasticNet(0.3, α))
  @printf("  %.1f  %8d  %9d  %-30s  %13.3f\n", α, count(!iszero, x), count(!iszero, x[1:15]),
          string(round.(x[1:5]; digits = 2)), std(x[1:15]))
end
````

The elastic net is the lasso on the data augmented with √(nλ(1 - α)) I,
whose loss divides by n + p instead of n: a check of the proximal operator

````@example 09_custom_penalty
λ, α = 0.3, 0.5
xe = SparseIterate(p)
coordinateDescent!(xe, f, MyElasticNet(λ, α), CDOptions(; optTol = 1e-12))
xa = SparseIterate(p)
coordinateDescent!(xa, CDLeastSquaresLoss([y; zeros(p)], [X; sqrt(n * λ * (1 - α)) * I(p)]),
                   NormL1(n / (n + p) * λ * α), CDOptions(; optTol = 1e-12))
@printf("the lasso on augmented data agrees to %.1e\n", norm(xe - xa, Inf))
````

the quadratic loss takes the penalty too

````@example 09_custom_penalty
xq = SparseIterate(p)
coordinateDescent!(xq, CDQuadraticLoss(X' * X / n, -X' * y / n), MyElasticNet(λ, α), CDOptions(; optTol = 1e-12))
@printf("so does CDQuadraticLoss(X'X / n, -X'y / n): %.1e\n", norm(xq - xa, Inf))
@printf("objective at the solution: %.6f\n", sum(abs2, y - X * xe) / (2n) + MyElasticNet(λ, α)(xe))
````

ProximalBase's ElasticNet(λ1, λ2) is λ1 ‖x‖₁ + (λ2/2) ‖x‖²

````@example 09_custom_penalty
xb = SparseIterate(p)
coordinateDescent!(xb, f, ElasticNet(λ * α, λ * (1 - α)), CDOptions(; optTol = 1e-12))
@printf("ProximalBase's ElasticNet agrees to %.1e\n", norm(xb - xe, Inf))
````

## The minimax concave penalty

Four large coefficients and independent columns. Along a path of λ from
λmax down, each fit warm started from the previous one, as is recommended
for nonconvex penalties (Breheny and Huang, 2011), the lasso shrinks the
large coefficients by about λ, while the MCP does not penalize coefficients
larger than γλ: once it selects the right variables, it gives least squares
on them.

````@example 09_custom_penalty
X = randn(n, p)
X ./= sqrt.(sum(abs2, X; dims = 1) / n)
β = [3.0, -2.0, 2.0, -3.0, zeros(p - 4)...]
y = X * β + randn(n)
f = CDLeastSquaresLoss(y, X)
λmax = maximum(abs, X' * y) / n
λs = λmax * exp10.(range(0, -1.5; length = 10))
xl, xm = SparseIterate(p), SparseIterate(p)
kept = nothing
println("       λ   lasso: nonzeros  ‖β̂ - β‖   MCP: nonzeros  ‖β̂ - β‖")
for (k, λk) in enumerate(λs)
  coordinateDescent!(xl, f, NormL1(λk))
  coordinateDescent!(xm, f, MyMCP(λk, 3.0))
  @printf("  %6.3f  %15d  %8.3f  %13d  %8.3f\n", λk, count(!iszero, xl), norm(xl - β),
          count(!iszero, xm), norm(xm - β))
  k == 6 && (global kept = (λk, Vector(xl), Vector(xm)))
end
λk, bl, bm = kept
@printf("at λ = %.3f: lasso %s, MCP %s, least squares on 1:4 %s\n", λk, round.(bl[1:4]; digits = 2),
        round.(bm[1:4]; digits = 2), round.(X[:, 1:4] \ y; digits = 2))
````

ProximalBase's MCP(λ, a) is the same penalty with γ = a, along the same path

````@example 09_custom_penalty
xb = SparseIterate(p)
for λj in λs[1:6]
  coordinateDescent!(xb, f, MCP(λj, 3.0))
end
@printf("ProximalBase's MCP agrees to %.1e\n", norm(Vector(xb) - bm, Inf))
````

a step longer than γ makes firm thresholding undefined, which prox_entry
above reports: here the columns are scaled by 1/2, so the step is 4 > γ = 3.
ProximalBase's MCP takes the global minimizer of the nonconvex step instead.

````@example 09_custom_penalty
try
  coordinateDescent!(SparseIterate(p), CDLeastSquaresLoss(y, X / 2), MyMCP(0.1, 3.0))
catch err
  println("with ‖X[:, k]‖² = n / 4: ", sprint(showerror, err))
end
````

