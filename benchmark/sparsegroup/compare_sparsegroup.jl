# SparseGroupLassoPath against R's sparsegl (and SGL when it is installed), for
# α ∈ {0.05, 0.5, 0.95} (sparsegl's asparse, SGL's alpha). All penalize the
# coefficients of standardized columns, but sparsegl and SGL scale the columns
# to unit Euclidean norm, √n times their standard deviation, so their level λ
# is our √n λ: gen writes our 25 levels divided by √n, at which R is run, and
# SparseGroupLassoPath is run at √n times them, its own levels. SGL returns the coefficients of its standardized columns and
# no intercept; they are mapped back to the columns of X, with the intercept of
# the centered fit. The KKT violation of each solution for the objective of
# sparse_group_lasso is reported too.
#
#     julia --project=benchmark/sparsegroup benchmark/sparsegroup/compare_sparsegroup.jl gen DIR
#     Rscript benchmark/sparsegroup/compare_sparsegroup.R DIR            # needs sparsegl (SGL optional)
#     julia --project=benchmark/sparsegroup benchmark/sparsegroup/compare_sparsegroup.jl compare DIR

using CoordinateDescent, DelimitedFiles, LinearAlgebra, Printf, Random, Statistics

const SETTINGS = [(n = 200, p = 60, ρ = 0.5, gsize = 5), (n = 400, p = 120, ρ = 0.0, gsize = 10),
                  (n = 100, p = 200, ρ = 0.3, gsize = 8)]
const ALPHAS = (0.05, 0.5, 0.95)

function dataset(s; seed)
  rng = Xoshiro(seed)
  X = randn(rng, s.n, s.p)
  for j in 2:s.p
    X[:, j] .= s.ρ .* X[:, j-1] .+ sqrt(1 - s.ρ^2) .* X[:, j]
  end
  c = 0.5 .+ 2 .* rand(rng, s.p)
  X .*= transpose(c)
  β = zeros(s.p)
  β[1:3] .= [1.5, -1.0, 0.8] ./ c[1:3]                                # a sparse first group
  β[s.gsize+1:2s.gsize] .= 0.4 .* randn(rng, s.gsize) ./ c[s.gsize+1:2s.gsize]   # a dense second group
  groups = [collect(g:min(g + s.gsize - 1, s.p)) for g in 1:s.gsize:s.p]
  X, 1.0 .+ X * β .+ randn(rng, s.n), groups
end

grouplabels(groups, p) = (l = zeros(Int, p); for (k, G) in enumerate(groups); l[G] .= k; end; l)
levels(X, y, groups, α) = SparseGroupLassoPath(X, y, groups; nλ = 25, λminratio = 0.05, alpha = α).λpath

function gen(dir)
  mkpath(dir)
  for (k, s) in enumerate(SETTINGS)
    X, y, groups = dataset(s; seed = 700 + k)
    writedlm(joinpath(dir, "X$k.csv"), X, ','); writedlm(joinpath(dir, "y$k.csv"), y, ',')
    writedlm(joinpath(dir, "group$k.csv"), grouplabels(groups, s.p), ',')
    for α in ALPHAS
      writedlm(joinpath(dir, "lambda$(α)_$k.csv"), levels(X, y, groups, α) ./ sqrt(s.n), ',')
    end
  end
end

coefs(path) = [reshape(path.β0path, 1, :); reduce(hcat, Vector.(path.βpath))]
soft(v, t) = sign.(v) .* max.(abs.(v) .- t, 0)

# the KKT violation of the coefficients [β₀; β] for the objective of sparse_group_lasso
function kkt(X, y, groups, b, λ, α)
  n = size(X, 1); xbar = vec(mean(X; dims = 1)); s = vec(sqrt.(mean((X .- transpose(xbar)) .^ 2; dims = 1)))
  Xs = (X .- transpose(xbar)) ./ transpose(s)
  t = s .* b[2:end]
  G = -Xs' * (y .- mean(y) .- Xs * t) / n
  v = 0.0
  for gr in groups
    w = sqrt(length(gr))
    if all(iszero, t[gr])
      v = max(v, norm(soft(G[gr], λ * α)) - λ * (1 - α) * w)
    else
      nG = norm(t[gr])
      for j in gr
        v = max(v, t[j] == 0 ? max(abs(G[j]) - λ * α, 0.0) : abs(G[j] + λ * α * sign(t[j]) + λ * (1 - α) * w * t[j] / nG))
      end
    end
  end
  v
end

function compare(dir)
  for (k, s) in enumerate(SETTINGS), α in ALPHAS
    X = readdlm(joinpath(dir, "X$k.csv"), ','); y = vec(readdlm(joinpath(dir, "y$k.csv"), ','))
    l = Int.(vec(readdlm(joinpath(dir, "group$k.csv"), ',')))
    groups = [findall(==(g), l) for g in 1:maximum(l)]
    λs = sqrt(size(X, 1)) .* vec(readdlm(joinpath(dir, "lambda$(α)_$k.csv"), ','))
    ours = coefs(SparseGroupLassoPath(X, y, λs, groups, CDOptions(; opt_tol = 1e-12); alpha = α))
    ko = maximum(j -> kkt(X, y, groups, ours[:, j], λs[j], α), axes(ours, 2))
    xbar = vec(mean(X; dims = 1)); nrm = vec(sqrt.(sum((X .- transpose(xbar)) .^ 2; dims = 1)))
    for prog in ("sparsegl", "SGL")
      f = joinpath(dir, "$(prog)$(α)_$k.csv")
      isfile(f) || continue
      R = readdlm(f, ',')
      if prog == "SGL"                       # the standardized coefficients, without the intercept
        B = R ./ nrm
        R = [reshape(mean(y) .- B' * xbar, 1, :); B]
      end
      kr = maximum(j -> kkt(X, y, groups, R[:, j], λs[j], α), axes(R, 2))
      same = all(findall(!iszero, ours[2:end, j]) == findall(!iszero, R[2:end, j]) for j in axes(R, 2))
      @printf "%-8s α=%.2f n=%3d p=%3d | same supports: %-5s | max |Δcoef| %.1e (largest %.2f) | KKT %.0e here, %.0e %s\n" prog α s.n s.p same maximum(abs, ours .- R) maximum(abs, R) ko kr prog
    end
  end
end

ARGS[1] == "gen" ? gen(ARGS[2]) : compare(ARGS[2])
