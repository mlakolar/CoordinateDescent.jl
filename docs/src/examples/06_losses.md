```@meta
EditURL = "../../../examples/06_losses.jl"
```

# The losses and the low-level interface

    coordinateDescent!(x, f, g, options = CDOptions(); kwargs...)
    r = solve!(x, f, g, options = CDOptions(); kwargs...)

minimizes f(x) + g(x) for a loss f and a penalty g from ProximalBase:
NormL1(λ) for λ ‖x‖₁, NormL1(λ, ω) for λ ∑ⱼ ωⱼ |xⱼ|, ZeroFunction() for no
penalty, and GroupNormL2 for the group lasso (05_group_lasso.jl). x is a
SparseIterate, or a dense Vector that is overwritten with the solution. The
losses, models of ProximalSolvers, whose coordinate descent does the work, are

    CDLeastSquaresLoss(y, X; intercept)   ‖y - Xβ‖² / (2n)
    CDWeightedLSLoss(y, X, w)             ∑ᵢ wᵢ (yᵢ - Xᵢ'β)² / (2n)
    CDQuadraticLoss(A, b)                 x'Ax / 2 + b'x
    CDSqrtLassoLoss(y, X; intercept)      ‖y - Xβ‖

lasso, sqrtLasso and the other high-level functions are built on them.
Calling coordinateDescent! directly lets you reuse a loss and warm start
from any point, and it takes penalties the high-level functions do not offer
(09_custom_penalty.jl).

````@example 06_losses
using CoordinateDescent, ProximalBase, ProximalSolvers
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(6)

n, p = 300, 40
X = randn(n, p)
β = zeros(p)
β[1:4] = [1.5, -1.0, 0.75, 0.5]
y = X * β + randn(n)
support(x) = findall(!iszero, x)
````

## Least squares

````@example 06_losses
f = CDLeastSquaresLoss(y, X)
x = SparseIterate(p)
coordinateDescent!(x, f, NormL1(0.1))
println("NormL1(0.1) selects ", support(x))
````

the objective; the functions of ProximalBase are callable, g(x) is the penalty

````@example 06_losses
g = NormL1(0.1)
@printf("objective %.6f\n", sum(abs2, y - X * x) / (2n) + g(x))
````

no penalty: least squares by coordinate descent

````@example 06_losses
xls = SparseIterate(p)
coordinateDescent!(xls, f, ZeroFunction(), CDOptions(; optTol = 1e-12))
@printf("ZeroFunction() matches X \\ y to %.1e\n", norm(xls - X \ y, Inf))
````

loadings ω, here zero for the first column, which is then not penalized:
for a λ that sets every other coefficient to zero, it is still there

````@example 06_losses
ω = [0.0; ones(p - 1)]
x = SparseIterate(p)
coordinateDescent!(x, f, NormL1(5.0, ω))
println("NormL1(5.0, ω) with ω₁ = 0 keeps ", support(x))
````

with an intercept, the columns and y are centered on the fly; the intercept
that goes with x is ProximalSolvers.intercept(f, x), ȳ - x̄'x

````@example 06_losses
fi = CDLeastSquaresLoss(y .+ 5, X; intercept = true)
x = SparseIterate(p)
coordinateDescent!(x, fi, NormL1(0.1))
@printf("with intercept = true: β̂₀ = %.2f, selects %s\n", ProximalSolvers.intercept(fi, x), support(x))
````

a dense starting point is converted, and overwritten with the solution

````@example 06_losses
xd = zeros(p)
coordinateDescent!(xd, f, NormL1(0.1))
println("dense Vector: ", typeof(xd), " with ", count(!iszero, xd), " nonzeros")
````

## Weighted least squares

With noise variances σᵢ², the weights wᵢ = 1 / σᵢ² give efficient
estimates. CDWeightedLSLoss(y, X, w) has the same minimizer as least squares
on the rows scaled by √wᵢ, without forming them.

````@example 06_losses
σ = exp.(randn(n))
yh = X * β + σ .* randn(n)
w = 1 ./ σ .^ 2
w ./= mean(w)                  # the loss divides by n, so weights of mean one keep λ comparable
xw = SparseIterate(p)
coordinateDescent!(xw, CDWeightedLSLoss(yh, X, w), NormL1(0.05), CDOptions(; optTol = 1e-10))
xs = SparseIterate(p)
coordinateDescent!(xs, CDLeastSquaresLoss(sqrt.(w) .* yh, sqrt.(w) .* X), NormL1(0.05), CDOptions(; optTol = 1e-10))
xu = SparseIterate(p)
coordinateDescent!(xu, CDLeastSquaresLoss(yh, X), NormL1(0.05))
@printf("weighted ‖β̂ - β‖ = %.3f (scaled rows agree to %.1e), unweighted ‖β̂ - β‖ = %.3f\n",
        norm(xw - β), norm(xw - xs, Inf), norm(xu - β))
````

The losses keep references to y, X and w, and coordinateDescent! starts by
recomputing their caches, so changing the data in place and solving again
fits the new data. Here the weights are redrawn in place for a weighted
bootstrap of the lasso, each fit warm started from the previous one.

````@example 06_losses
wb = ones(n)
fb = CDWeightedLSLoss(y, X, wb)
xb = SparseIterate(p)
counts = zeros(Int, p)
B = 100
for b in 1:B
  wb .= -log.(rand(n))            # Exp(1) weights
  coordinateDescent!(xb, fb, NormL1(0.1))
  counts .+= .!iszero.(xb)
end
println("selection frequencies in $B bootstrap fits: ",
        join(["$j: $(counts[j])" for j in sortperm(counts; rev = true)[1:6]], ", "))
````

## A quadratic loss

With many observations, X'X / n and X'y / n summarize the data:
CDQuadraticLoss(X'X / n, -X'y / n) has the same minimizer as the least
squares loss, and its updates cost O(p) instead of O(n). Forming X'X costs
O(np²) once (and the loss copies it), so it pays off when the same data are
fit many times: long paths, many penalties, the bootstrap.

````@example 06_losses
nbig = 200_000
Xbig = randn(nbig, p)
ybig = Xbig * β + randn(nbig)
λs = exp.(range(log(0.5), log(0.001); length = 20))
function lassopath!(x, f, λs)
  for λ in λs
    coordinateDescent!(x, f, NormL1(λ))
  end
  x
end
randn(10, 2)' * randn(10, 2)                                           # compile
tA = @elapsed fq = CDQuadraticLoss(Xbig' * Xbig / nbig, -Xbig' * ybig / nbig)
fl = CDLeastSquaresLoss(ybig, Xbig)
lassopath!(SparseIterate(p), fq, λs[1:2]); lassopath!(SparseIterate(p), fl, λs[1:2])   # compile
xq, xl = SparseIterate(p), SparseIterate(p)
tq = @elapsed lassopath!(xq, fq, λs)
tl = @elapsed lassopath!(xl, fl, λs)
@printf("path of %d values: %.2f ms with the quadratic loss, after %.1f ms to form X'X / n once,\n",
        length(λs), 1000tq, 1000tA)
@printf("and %.1f ms with least squares; the solutions differ by %.1e\n", 1000tl, norm(xq - xl, Inf))
````

A quadratic loss also comes from a covariance matrix alone. In a Gaussian
vector with precision matrix Ω, regressing variable j on the others gives
coefficients -Ω[j, others] / Ω[j, j], which are nonzero exactly for the
neighbors of j in its graph (Meinshausen and Bühlmann, 2006). From the
sample covariance S, the lasso for that regression is
CDQuadraticLoss(S[others, others], -S[others, j]), with λ of the order of
√(2 log(d) / n) times the noise level. Here Ω is tridiagonal, so the
neighbors of j are j - 1 and j + 1.

````@example 06_losses
d = 30
Ω = SymTridiagonal(fill(1.0, d), fill(0.4, d - 1))
Zg = randn(2000, d) * inv(cholesky(Matrix(Ω)).U)'
S = cov(Zg)
for j in (1, 10, 20)
  others = setdiff(1:d, j)
  xn = SparseIterate(d - 1)
  coordinateDescent!(xn, CDQuadraticLoss(S[others, others], -S[others, j]), NormL1(0.15))
  println("neighbors of variable $j: ", others[support(xn)])
end
````

## The square-root lasso loss

CDSqrtLassoLoss takes NormL1 only: its coordinate step is the exact minimizer
for the ℓ1 penalty. sqrtLasso(X, y, λ; standardizeX = false) does the same.

````@example 06_losses
xr = SparseIterate(p)
coordinateDescent!(xr, CDSqrtLassoLoss(y, X), NormL1(2.0), CDOptions(; optTol = 1e-10))
sol = sqrtLasso(X, y, 2.0, CDOptions(; optTol = 1e-10); standardizeX = false)
@printf("selects %s, as sqrtLasso does (difference %.1e)\n", support(xr), norm(xr - sol.x, Inf))
````

