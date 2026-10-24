# LassoPath with the elastic net (alpha) against R's glmnet, and with MCP and
# SCAD (penalty) against R's ncvreg, on the same data and the same levels of λ.
#
#     julia --project=benchmark/penalties benchmark/penalties/compare_penalties.jl gen DIR
#     Rscript benchmark/penalties/compare_penalties.R DIR             # needs glmnet and ncvreg
#     julia --project=benchmark/penalties benchmark/penalties/compare_penalties.jl compare DIR

using CoordinateDescent, DelimitedFiles, Printf, Random
using ProximalBase: MCP, SCAD

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
  @printf "%-28s n=%3d p=%3d | same supports at all %d levels: %-5s | max |Δcoef| %.1e (largest coefficient %.2f)\n" label s.n s.p size(R, 2) same maximum(abs, ours .- R) maximum(abs, R)
end

function compare(dir)
  for (k, s) in enumerate(SETTINGS)
    X = readdlm(joinpath(dir, "X$k.csv"), ','); y = vec(readdlm(joinpath(dir, "y$k.csv"), ','))
    for a in ALPHAS
      λs = vec(readdlm(joinpath(dir, "lambda_enet$(a)_$k.csv"), ','))
      path = LassoPath(X, y, λs; alpha = a, intercept = true, optTol = 1e-12)
      report("elastic net α = $a (glmnet)", s, coefs(path), readdlm(joinpath(dir, "glmnet_enet$(a)_$k.csv"), ','))
    end
    for (name, pen) in PENALTIES, order in (:random, :cyclic)
      λs = vec(readdlm(joinpath(dir, "lambda_$(name)_$k.csv"), ','))
      path = LassoPath(X, y, λs; penalty = pen, intercept = true, optTol = 1e-12, order, rng = Xoshiro(k))
      report("$name, $order order (ncvreg)", s, coefs(path), readdlm(joinpath(dir, "ncvreg_$(name)_$k.csv"), ','))
    end
  end
end

ARGS[1] == "gen" ? gen(ARGS[2]) : compare(ARGS[2])
