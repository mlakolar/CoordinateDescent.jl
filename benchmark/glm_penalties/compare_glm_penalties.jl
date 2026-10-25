# GLMLassoPath with the elastic net (alpha) against R's glmnet, and with MCP and
# SCAD (penalty) against R's ncvreg, for logistic and Poisson regression, on
# the same data and the same levels of λ. Both standardize the columns and put
# the penalty on the coefficients of the standardized columns. Along a
# nonconvex path, the two programs can reach different local solutions after a
# level where the solution bifurcates; the first level where they differ is
# reported with the objectives and optimality conditions of both.
#
# For logistic regression ncvreg's coordinate steps majorize the loss with the
# curvature 1/4, and are exact only for a > 4 (MCP) and a > 5 (SCAD) (Breheny
# and Huang, 2011); at the defaults a = 3 and 3.7 its solutions need not be
# stationary. So MCP and SCAD are compared at the defaults and at a = 6 and 7,
# and the largest violation of the stationarity conditions of each program is
# reported. ncvreg stops a path early when the model is saturated (p > n); the
# levels it returned are compared.
#
#     julia --project=benchmark/glm_penalties benchmark/glm_penalties/compare_glm_penalties.jl gen DIR
#     Rscript benchmark/glm_penalties/compare_glm_penalties.R DIR      # needs glmnet and ncvreg
#     julia --project=benchmark/glm_penalties benchmark/glm_penalties/compare_glm_penalties.jl compare DIR

using HighDimensionalModels, DelimitedFiles, Printf, Random, Statistics
using ProximalBase: MCP, SCAD, Logistic, Poisson

# AR(ρ) columns on different scales, an intercept and six nonzero
# coefficients; the last binomial design has more columns than rows
const SETTINGS = [(family = "binomial", n = 400, p = 40, ρ = 0.5), (family = "binomial", n = 600, p = 100, ρ = 0.0),
                  (family = "binomial", n = 150, p = 300, ρ = 0.3), (family = "poisson", n = 400, p = 40, ρ = 0.3)]
const PENALTIES = [("enet0.2", (alpha = 0.2,)), ("enet0.5", (alpha = 0.5,)), ("enet0.8", (alpha = 0.8,)),
                   ("MCP", (penalty = MCP(1, 3),)), ("SCAD", (penalty = SCAD(1, 3.7),)),
                   ("MCP6", (penalty = MCP(1, 6),)), ("SCAD7", (penalty = SCAD(1, 7),))]
loss(family) = family == "binomial" ? Logistic() : Poisson()

function dataset(s; seed)
  rng = Xoshiro(seed)
  X = randn(rng, s.n, s.p)
  for j in 2:s.p
    X[:, j] .= s.ρ .* X[:, j-1] .+ sqrt(1 - s.ρ^2) .* X[:, j]
  end
  c = 0.5 .+ 2 .* rand(rng, s.p)
  X .*= transpose(c)
  β = zeros(s.p)
  β[1:6] .= (s.family == "binomial" ? [1.0, -0.8, 0.6, -0.6, 0.4, 0.3] : [0.4, -0.3, 0.3, -0.2, 0.2, 0.1]) ./ c[1:6]
  η = 0.2 .+ X * β
  y = s.family == "binomial" ? Float64.(rand(rng, s.n) .< 1 ./ (1 .+ exp.(-η))) :
      Float64.([sum(cumsum(-log.(rand(rng, 60))) .< exp(e)) for e in η])      # Poisson(exp(η)) by exponential gaps
  X, y
end

# 25 levels from λmax down to 5% of it, as GLMLassoPath computes them
levels(X, y, family; kw...) = GLMLassoPath(X, y, loss(family); nλ = 25, λminratio = 0.05, devmax = 1, kw...).λpath

function gen(dir)
  mkpath(dir)
  for (k, s) in enumerate(SETTINGS)
    key = "$(s.family)_$k"
    X, y = dataset(s; seed = 500 + k)
    writedlm(joinpath(dir, "X_$key.csv"), X, ','); writedlm(joinpath(dir, "y_$key.csv"), y, ',')
    for (tag, kw) in PENALTIES
      writedlm(joinpath(dir, "lambda_$(key)_$tag.csv"), levels(X, y, s.family; kw...), ',')
    end
  end
end

coefs(path) = [reshape(path.β0path, 1, :); reduce(hcat, Vector.(path.βpath))]

# the objective and the largest violation of the local optimality conditions
# for the coefficients b of the data, on the standardized columns
function local_check(X, y, family, b0, b, pen, λ)
  n = size(X, 1)
  sd = vec(sqrt.(mean((X .- mean(X; dims = 1)) .^ 2; dims = 1)))
  b̃ = b .* sd
  η = b0 .+ X * b
  μ = family == "binomial" ? 1 ./ (1 .+ exp.(-η)) : exp.(η)
  nll = family == "binomial" ? sum(log1p.(exp.(η)) .- y .* η) / n : sum(exp.(η) .- y .* η) / n
  a = pen.a
  val(t) = pen isa MCP ? (abs(t) <= a * λ ? λ * abs(t) - t^2 / (2a) : a * λ^2 / 2) :
           (abs(t) <= λ ? λ * abs(t) : abs(t) <= a * λ ? (2a * λ * abs(t) - t^2 - λ^2) / (2(a - 1)) : λ^2 * (a + 1) / 2)
  der(t) = pen isa MCP ? sign(t) * max(λ - abs(t) / a, 0.0) : sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))
  grad = (X ./ transpose(sd))' * (μ .- y) / n
  kkt = maximum(j -> b̃[j] == 0 ? max(abs(grad[j]) - λ, 0.0) : abs(grad[j] + der(b̃[j])), eachindex(b̃))
  nll + sum(val, b̃), kkt
end

function compare(dir)
  for (k, s) in enumerate(SETTINGS)
    key = "$(s.family)_$k"
    X = readdlm(joinpath(dir, "X_$key.csv"), ','); y = vec(readdlm(joinpath(dir, "y_$key.csv"), ','))
    for (tag, kw) in PENALTIES
      λs = vec(readdlm(joinpath(dir, "lambda_$(key)_$tag.csv"), ','))
      R = readdlm(joinpath(dir, "R_$(key)_$tag.csv"), ',')
      path = GLMLassoPath(X, y, λs, loss(s.family), CDOptions(; opt_tol = 1e-12); devmax = 1, newton_tol = 1e-14, kw...)
      ours = coefs(path)[:, axes(R, 2)]           # ncvreg may stop early
      same = all(findall(!iszero, ours[2:end, j]) == findall(!iszero, R[2:end, j]) for j in axes(R, 2))
      @printf "%-8s %-8s n=%3d p=%3d | same supports at all %2d levels: %-5s | max |Δcoef| %.1e (largest %.2f)\n" s.family tag s.n s.p size(R, 2) same maximum(abs, ours .- R) maximum(abs, R)
      if haskey(kw, :penalty)
        vo = maximum(j -> local_check(X, y, s.family, ours[1, j], ours[2:end, j], kw.penalty, λs[j])[2], axes(R, 2))
        vr = maximum(j -> local_check(X, y, s.family, R[1, j], R[2:end, j], kw.penalty, λs[j])[2], axes(R, 2))
        @printf "    largest stationarity violation over the levels: %.1e here, %.1e ncvreg\n" vo vr
        j = findfirst(j -> maximum(abs, ours[:, j] .- R[:, j]) > 1e-5, axes(R, 2))
        if j !== nothing
          oo, ko = local_check(X, y, s.family, ours[1, j], ours[2:end, j], kw.penalty, λs[j])
          or, kr = local_check(X, y, s.family, R[1, j], R[2:end, j], kw.penalty, λs[j])
          @printf "    the same to %.0e through level %d; at level %d: objective %.6f here, %.6f ncvreg; optimality violations %.0e and %.0e\n" (j > 1 ? maximum(abs, ours[:, 1:j-1] .- R[:, 1:j-1]) : NaN) (j - 1) j oo or ko kr
        end
      end
    end
  end
end

ARGS[1] == "gen" ? gen(ARGS[2]) : compare(ARGS[2])
