# GroupLassoPath (orthonormalized groups) against R's grpreg on the same data
# and the same levels of λ.
#
#     julia --project=benchmark/grpreg benchmark/grpreg/compare_grpreg.jl gen DIR
#     Rscript benchmark/grpreg/compare_grpreg.R DIR                  # needs grpreg
#     julia --project=benchmark/grpreg benchmark/grpreg/compare_grpreg.jl compare DIR

using CoordinateDescent, DelimitedFiles, Printf, Random

# groups of sizes 1 to 5; correlated columns within groups and on different
# scales, so that orthonormalization matters; four groups in the model
const SETTINGS = [(n = 100, J = 20), (n = 300, J = 60), (n = 200, J = 150)]

function dataset(n, J; seed)
  rng = Xoshiro(seed)
  sizes = [1 + (k - 1) % 5 for k in 1:J]
  X = reduce(hcat, [randn(rng, n, 1) .+ 0.6 .* randn(rng, n, s) for s in sizes])
  X .*= transpose(1 .+ 3 .* rand(rng, size(X, 2)))
  group = reduce(vcat, [fill(k, s) for (k, s) in enumerate(sizes)])
  β = zeros(size(X, 2))
  for k in (1, 2, 4, 5)
    β[group .== k] .= 0.5 .* randn(rng, sizes[k]) ./ (1 .+ 3 .* rand(rng, sizes[k]))
  end
  X, 1.0 .+ X * β .+ randn(rng, n), group
end

λs(k) = exp.(range(log(0.5), log(0.01); length = 10))

function gen(dir)
  mkpath(dir)
  for (k, s) in enumerate(SETTINGS)
    X, y, group = dataset(s.n, s.J; seed = 200 + k)
    writedlm(joinpath(dir, "X$k.csv"), X, ','); writedlm(joinpath(dir, "y$k.csv"), y, ',')
    writedlm(joinpath(dir, "group$k.csv"), group, ','); writedlm(joinpath(dir, "lambda$k.csv"), λs(k), ',')
  end
end

function compare(dir)
  for (k, s) in enumerate(SETTINGS)
    X = readdlm(joinpath(dir, "X$k.csv"), ','); y = vec(readdlm(joinpath(dir, "y$k.csv"), ','))
    group = vec(Int.(readdlm(joinpath(dir, "group$k.csv"), ',')))
    groups = [findall(==(g), group) for g in 1:maximum(group)]
    path = GroupLassoPath(X, y, λs(k), groups, CDOptions(; opt_tol = 1e-12))
    R = readdlm(joinpath(dir, "grpreg$k.csv"), ',')               # (p + 1) × nλ, the intercept first
    ours = [reshape(path.β0path, 1, :); reduce(hcat, Vector.(path.βpath))]
    same = all(findall(!iszero, ours[2:end, j]) == findall(!iszero, R[2:end, j]) for j in axes(R, 2))
    @printf "n=%4d groups=%3d p=%3d | same supports at all %d levels: %-5s | max |Δcoef| %.1e (largest coefficient %.2f)\n" s.n s.J size(X, 2) size(R, 2) same maximum(abs, ours .- R) maximum(abs, R)
  end
end

ARGS[1] == "gen" ? gen(ARGS[2]) : compare(ARGS[2])
