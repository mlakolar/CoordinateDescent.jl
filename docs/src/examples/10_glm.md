```@meta
EditURL = "../../../examples/10_glm.jl"
```

# Penalized generalized linear models

    glmLasso(X, y, λ, loss = ProximalBase.Logistic(), options = CDOptions();
             intercept = true, standardizeX = true, weights, offset)
    GLMLassoPath(X, y, loss = ProximalBase.Logistic(); nλ = 100, λminratio, kwargs...)

minimize ∑ᵢ wᵢ ℓ(β₀ + xᵢ'β + oᵢ, yᵢ) / ∑ᵢ wᵢ + λ ∑ⱼ ωⱼ |βⱼ| for a scalar loss ℓ
of ProximalBase, with glmnet's conventions and defaults: an unpenalized
intercept, loadings ωⱼ equal to the standard deviations of the columns,
observation weights w and an offset o. They run proximal Newton (glmnet's
method): each step minimizes the weighted least-squares model of the loss by
coordinate descent, then backtracks.

````@example 10_glm
using CoordinateDescent, ProximalBase
using LinearAlgebra, Printf, Random, Statistics

Random.seed!(10)
logistic(z) = 1 / (1 + exp(-z))

n, p = 400, 60
X = randn(n, p)
β = zeros(p)
β[1:5] = [1.2, -1.0, 0.8, -0.6, 0.5]
y = Float64.(rand(n) .< logistic.(-0.5 .+ X * β))
support(x) = findall(!iszero, x)
````

## Logistic regression

````@example 10_glm
sol = glmLasso(X, y, 0.03)
@printf "intercept %.3f, selected %s\n" sol.intercept support(sol.x)
@printf "explained deviance %.3f, proximal Newton took %d coordinate descent passes\n" (1 - sol.deviance / sol.nulldeviance) sol.iterations
````

the optimality conditions: |∇ⱼ| ≤ λωⱼ at the zeros, ∇ⱼ = -λωⱼ sign(βⱼ) elsewhere,
and the gradient over the intercept is zero

````@example 10_glm
μ = logistic.(sol.linearpredictor)
G = X' * (μ .- y) / n
ω = vec(std(X; dims = 1, corrected = false))
viol = maximum(j -> sol.x[j] == 0 ? max(abs(G[j]) - 0.03ω[j], 0) : abs(G[j] + 0.03ω[j] * sign(sol.x[j])), 1:p)
@printf "largest violation of the optimality conditions %.1e, mean residual %.1e\n" viol mean(μ .- y)
````

## The path, and λ chosen on a validation set

````@example 10_glm
train, valid = 1:300, 301:400
path = GLMLassoPath(X[train, :], y[train]; nλ = 50)
````

the validation deviance of each solution

````@example 10_glm
function deviance(η, y)
  2 * sum(log1p.(exp.(η)) .- y .* η)
end
dev = [deviance(path.β0path[k] .+ X[valid, :] * path.βpath[k], y[valid]) for k in eachindex(path.λpath)]
k = argmin(dev)
@printf "%d levels, explaining up to %.1f%% of the deviance (a path stops at 99.9%%)\n" length(path.λpath) 100 * last(path.devratio)
@printf "best λ = %.4f (level %d): %d nonzero coefficients, validation deviance %.1f\n" path.λpath[k] k count(!iszero, path.βpath[k]) dev[k]
````

## Poisson regression with an exposure offset

Poisson draws by Knuth's method, to avoid a dependency on Distributions

````@example 10_glm
function rpois(μ)
  k, t, L = 0, rand(), exp(-μ)
  while t > L
    k += 1
    t *= rand()
  end
  k
end
exposure = rand(n) .* 4 .+ 1
counts = Float64.(rpois.(exposure .* exp.(0.2 .+ 0.4 .* X[:, 1] .- 0.3 .* X[:, 2])))
````

λmax, the first level of a path, is the scale of the penalty for this loss

````@example 10_glm
λmaxp = GLMLassoPath(X, counts, ProximalBase.Poisson(); nλ = 1, offset = log.(exposure)).λpath[1]
solp = glmLasso(X, counts, 0.2 * λmaxp, ProximalBase.Poisson(); offset = log.(exposure))
@printf "λmax %.3f; at 0.2 λmax: intercept %.3f, selected %s\n" λmaxp solp.intercept support(solp.x)
````

observation weights: integer weights are replicated rows

````@example 10_glm
w = Float64.(rand(1:3, n))
solw = glmLasso(X, y, 0.03; weights = w)
rows = reduce(vcat, [fill(i, Int(w[i])) for i in 1:n])
solr = glmLasso(X[rows, :], y[rows], 0.03)
@printf "weighted fit and replicated rows agree to %.1e\n" norm(Vector(solw.x) - Vector(solr.x), Inf)
````

