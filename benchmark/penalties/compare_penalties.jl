# LassoPath with the elastic net (alpha) against R's glmnet, and with MCP and
# SCAD (penalty) against R's ncvreg, on the same data and the same levels of λ.
#
# glmnet's gaussian family scales y to unit standard deviation internally,
# which divides the ridge part of its penalty by sd(y): it solves
# λ(α‖β‖₁ + (1 - α)‖β‖²/(2 sd(y))), the penalty Scaled(ElasticNet(α, (1 - α)/sd(y)), λ),
# rather than the λ(α‖β‖₁ + (1 - α)‖β‖²/2) of its documentation and of alpha.
# Both are compared. Along a nonconvex path, the two programs can reach
# different local solutions after a level where the solution bifurcates; the
# first such level, the objectives there and the optimality conditions of
# both solutions are printed.
#
#     julia --project=benchmark/penalties benchmark/penalties/compare_penalties.jl gen DIR
#     Rscript benchmark/penalties/compare_penalties.R DIR             # needs glmnet and ncvreg
#     julia --project=benchmark/penalties benchmark/penalties/compare_penalties.jl compare DIR

using CoordinateDescent, DelimitedFiles, Printf, Random, Statistics
using ProximalBase: MCP, SCAD, ElasticNet, Scaled

# AR(ρ) columns on different scales, an intercept and eight nonzero
# coefficients; the last design has more columns than rows
const SETTINGS = [(n = 200, p = 50, ρ = 0.5), (n = 500, p = 100, ρ = 0.0), (n = 100, p = 300, ρ = 0.3)]
const ALPHAS = (0.2, 0.5, 0.8)
const PENALTIES = (("MCP", MCP(1, 3)), ("SCAD", SCAD(1, 3.7)))

function dataset(n, p, ρ; seed)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  for j in 2:p
    X[:, j] .= ρ .* X[:, j-1] .+ sqrt(1 - ρ^2) .* X[:, j]
  end
  c = 0.5 .+ 2 .* rand(rng, p)
  X .*= transpose(c)
  β = zeros(p)
  β[1:8] .= [2.0, -1.5, 1.0, -1.0, 0.8, -0.6, 0.5, 0.4] ./ c[1:8]
  X, 1.0 .+ X * β .+ randn(rng, n)
end

# 30 levels from λmax down to 1% of it (5% with p > n), as the paths compute them
levels(X, y; kw...) = LassoPath(X, y; nλ = 30, λminratio = size(X, 1) < size(X, 2) ? 0.05 : 0.01, intercept = true, kw...).λpath

function gen(dir)
  mkpath(dir)
  for (k, s) in enumerate(SETTINGS)
    X, y = dataset(s.n, s.p, s.ρ; seed = 300 + k)
    writedlm(joinpath(dir, "X$k.csv"), X, ','); writedlm(joinpath(dir, "y$k.csv"), y, ',')
    for a in ALPHAS
      writedlm(joinpath(dir, "lambda_enet$(a)_$k.csv"), levels(X, y; alpha = a), ',')
    end
    for (name, pen) in PENALTIES
      writedlm(joinpath(dir, "lambda_$(name)_$k.csv"), levels(X, y; penalty = pen), ',')
    end
  end
end

coefs(path) = [reshape(path.β0path, 1, :); reduce(hcat, Vector.(path.βpath))]

function report(label, s, ours, R)
  same = all(findall(!iszero, ours[2:end, j]) == findall(!iszero, R[2:end, j]) for j in axes(R, 2))
  @printf "%-34s n=%3d p=%3d | same supports at all %d levels: %-5s | max |Δcoef| %.1e (largest coefficient %.2f)\n" label s.n s.p size(R, 2) same maximum(abs, ours .- R) maximum(abs, R)
end

# the objective and the largest violation of the local optimality conditions
# of MCP or SCAD on the standardized columns, for the coefficients b of the data
function local_check(X, y, b, pen, λ)
  n = size(X, 1)
  xbar = vec(mean(X; dims = 1)); sd = vec(sqrt.(mean((X .- transpose(xbar)) .^ 2; dims = 1)))
  Xc = (X .- transpose(xbar)) ./ transpose(sd); yc = y .- mean(y)
  b̃ = b .* sd
  a = pen.a
  val(t) = pen isa MCP ? (abs(t) <= a * λ ? λ * abs(t) - t^2 / (2a) : a * λ^2 / 2) :
           (abs(t) <= λ ? λ * abs(t) : abs(t) <= a * λ ? (2a * λ * abs(t) - t^2 - λ^2) / (2(a - 1)) : λ^2 * (a + 1) / 2)
  der(t) = pen isa MCP ? sign(t) * max(λ - abs(t) / a, 0.0) :
           sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))
  grad = -Xc' * (yc - Xc * b̃) / n
  kkt = maximum(j -> b̃[j] == 0 ? max(abs(grad[j]) - λ, 0.0) : abs(grad[j] + der(b̃[j])), eachindex(b̃))
  sum(abs2, yc - Xc * b̃) / (2n) + sum(val, b̃), kkt
end

function report_local(X, y, λs, pen, ours, R)
  j = findfirst(j -> maximum(abs, ours[:, j] .- R[:, j]) > 1e-6, axes(R, 2))
  j === nothing && return
  oo, ko = local_check(X, y, ours[2:end, j], pen, λs[j])
  or, kr = local_check(X, y, R[2:end, j], pen, λs[j])
  @printf "    the same to %.0e through level %d; at level %d: objective %.6f here, %.6f ncvreg; optimality violations %.0e and %.0e\n" maximum(abs, ours[:, 1:j-1] .- R[:, 1:j-1]) (j - 1) j oo or ko kr
end

function compare(dir)
  for (k, s) in enumerate(SETTINGS)
    X = readdlm(joinpath(dir, "X$k.csv"), ','); y = vec(readdlm(joinpath(dir, "y$k.csv"), ','))
    sdy = sqrt(mean(abs2, y .- mean(y)))
    for a in ALPHAS
      λs = vec(readdlm(joinpath(dir, "lambda_enet$(a)_$k.csv"), ','))
      R = readdlm(joinpath(dir, "glmnet_enet$(a)_$k.csv"), ',')
      path = LassoPath(X, y, λs; alpha = a, intercept = true, opt_tol = 1e-12)
      report("alpha = $a (glmnet)", s, coefs(path), R)
      path = LassoPath(X, y, λs; penalty = Scaled(ElasticNet(a, (1 - a) / sdy), 1), intercept = true, opt_tol = 1e-12)
      report("alpha = $a, ridge/sd(y) (glmnet)", s, coefs(path), R)
    end
    for (name, pen) in PENALTIES, order in (:random, :cyclic)
      λs = vec(readdlm(joinpath(dir, "lambda_$(name)_$k.csv"), ','))
      R = readdlm(joinpath(dir, "ncvreg_$(name)_$k.csv"), ',')
      path = LassoPath(X, y, λs; penalty = pen, intercept = true, opt_tol = 1e-12, order, rng = Xoshiro(k))
      report("$name, $order order (ncvreg)", s, coefs(path), R)
      report_local(X, y, λs, pen, coefs(path), R)
    end
  end
end

ARGS[1] == "gen" ? gen(ARGS[2]) : compare(ARGS[2])
