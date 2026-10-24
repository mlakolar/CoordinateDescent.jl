# glmLasso paths against GLMNet.jl (glmnet's Fortran) and Lasso.jl (Julia
# coordinate descent), on logistic and Poisson lasso paths with an intercept
# and standardized columns: the same objective in all three, the same 100
# levels from λmax down to 0.01 λmax. Each package runs at its default
# tolerances; at those, glmnet and Lasso.jl stop much earlier than glmLasso, so
# glmLasso also runs at looser tolerances (optTol 1e-4, newtonTol 1e-5) and
# glmnet at a tighter one (1e-10). The accuracy of each is the largest
# difference from a reference path that glmnet computes at the tolerance 1e-13,
# relative to the largest reference coefficient.
#
#     julia --project=benchmark/glm benchmark/glm/glm_bench.jl [smoke]
#
# Times are the minimum over interleaved repetitions, after a warm-up run of
# each package on the same problem; one BLAS thread.

import GLMNet, Lasso
using Distributions: Binomial, Poisson
using CoordinateDescent, ProximalBase
using LinearAlgebra, Printf, Random
BLAS.set_num_threads(1)

const SMOKE = "smoke" in ARGS
const SIZES = SMOKE ? [(200, 30)] : [(1000, 100), (1000, 2000), (10000, 100), (10000, 2000)]
const REPS = SMOKE ? 1 : 3
const NLAMBDA = 100

logistic(z) = 1 / (1 + exp(-z))

# Gaussian columns with AR(1) correlation 0.5 between neighbors, and 10
# coefficients of ±0.5 at random positions
function design(n, p, family; seed = 2026)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  for j in 2:p
    @views X[:, j] .= 0.5 .* X[:, j-1] .+ sqrt(0.75) .* X[:, j]
  end
  β = zeros(p)
  β[randperm(rng, p)[1:10]] .= rand(rng, (-0.5, 0.5), 10)
  η = X * β
  y = family === :logistic ? Float64.(rand(rng, n) .< logistic.(η .- 0.3)) :
                             Float64.(rand.(rng, Poisson.(exp.(0.5 .+ η))))
  X, y
end

ours(X, y, λs, loss) = GLMLassoPath(X, y, λs, loss; devmax = 1)
ours_loose(X, y, λs, loss) = GLMLassoPath(X, y, λs, loss, CDOptions(; optTol = 1e-4); devmax = 1, newtonTol = 1e-5)
glmnet(X, y, λs, family; tol = 1e-7) = family === :logistic ?
  GLMNet.glmnet(X, [1 .- y y], Binomial(); lambda = λs, tol = tol) :
  GLMNet.glmnet(X, y, Poisson(); lambda = λs, tol = tol)
lassojl(X, y, λs, family) = Lasso.fit(Lasso.LassoPath, X, y, family === :logistic ? Binomial() : Poisson();
                                      λ = λs, standardize = true)

# the coefficients of each level as a matrix with the intercept in row 1
coefs(p::CoordinateDescent.GLMLassoPath) = [p.β0path'; reduce(hcat, Vector.(p.βpath))]
coefs(g::GLMNet.GLMNetPath) = [g.a0'; Matrix(g.betas)]
coefs(l::Lasso.LassoPath) = Matrix(Lasso.coef(l))

function relerr(B, R)
  k = min(size(B, 2), size(R, 2))
  maximum(abs, B[:, 1:k] .- R[:, 1:k]) / maximum(abs, R[:, 1:k])
end

function bench(n, p, family)
  X, y = design(n, p, family)
  loss = family === :logistic ? ProximalBase.Logistic() : ProximalBase.Poisson()
  λmax = GLMLassoPath(X, y, loss; nλ = 1).λpath[1]
  λs = [λmax * 0.01^(k / (NLAMBDA - 1)) for k in 0:NLAMBDA-1]
  R = coefs(glmnet(X, y, λs, family; tol = 1e-13))
  fits = (ours = () -> ours(X, y, λs, loss), ours_loose = () -> ours_loose(X, y, λs, loss),
          glmnet = () -> glmnet(X, y, λs, family), glmnet_tight = () -> glmnet(X, y, λs, family; tol = 1e-10),
          lassojl = () -> lassojl(X, y, λs, family))
  res = Dict(k => f() for (k, f) in pairs(fits))          # warm-up, and the accuracy
  times = Dict(k => Inf for k in keys(fits))
  for _ in 1:REPS, (k, f) in pairs(fits)
    times[k] = min(times[k], @elapsed f())
  end
  nl(k) = size(coefs(res[k]), 2)
  err(k) = relerr(coefs(res[k]), R)
  @printf "%-8s n=%5d p=%5d |" family n p
  for (k, name) in ((:ours, "glmLasso"), (:ours_loose, "glmLasso loose"), (:glmnet, "glmnet"),
                    (:glmnet_tight, "glmnet 1e-10"), (:lassojl, "Lasso.jl"))
    @printf " %s %.3f s (%d, %.0e) |" name times[k] nl(k) err(k)
  end
  @printf " glmLasso/glmnet %.2f, loose/glmnet %.2f, glmLasso/glmnet 1e-10 %.2f, glmLasso/Lasso.jl %.2f\n" times[:ours] / times[:glmnet] times[:ours_loose] / times[:glmnet] times[:ours] / times[:glmnet_tight] times[:ours] / times[:lassojl]
  flush(stdout)
end

for family in (:logistic, :poisson), (n, p) in SIZES
  bench(n, p, family)
end
