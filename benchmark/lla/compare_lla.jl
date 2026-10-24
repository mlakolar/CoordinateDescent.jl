# Paths of nonconvex penalties by local linear approximation (lla, #69) against
# R's glmtlp (the TLP, CappedL1) and ncpen (MCP, SCAD and TLP), gaussian, with
# an intercept, on the columns as given and standardized. See benchmark/README.md.
#
#     julia --project=benchmark/lla benchmark/lla/compare_lla.jl gen DIR       # data and levels
#     Rscript benchmark/lla/compare_lla.R DIR                                  # needs glmtlp and ncpen
#     julia --project=benchmark/lla benchmark/lla/compare_lla.jl compare DIR   # our fits against R's
using CoordinateDescent, ProximalBase, DelimitedFiles, Random, Printf, Statistics

const TAU = 0.25
mode, dir = ARGS[1], ARGS[2]
rd(f) = readdlm(joinpath(dir, f), ',', Float64)
wr(f, A) = writedlm(joinpath(dir, f), A, ',')

function data(seed, n, p)
  rng = Xoshiro(seed)
  X = randn(rng, n, p) .* transpose(0.5 .+ 1.5 .* rand(rng, p))     # columns on different scales
  β = zeros(p); β[1:6] .= [1.5, -1.2, 1.0, -0.8, 0.6, 0.4]
  y = 1.5 .+ X * β .+ randn(rng, n)
  X, y
end

# the levels: 30 from λmax of the lasso (with an intercept, on the columns as
# given or standardized) down to rmin λmax
function levels(X, y, std, rmin)
  n = size(X, 1)
  Xc = X .- mean(X; dims = 1)
  std && (Xc ./= sqrt.(mean(Xc .^ 2; dims = 1)))
  λmax = maximum(abs.(Xc' * (y .- mean(y)))) / n
  [λmax * rmin^(k / 29) for k in 0:29]
end

sets = (("g1", 101, 200, 50, 0.05), ("g2", 102, 100, 300, 0.1))
if mode == "gen"
  for (name, seed, n, p, rmin) in sets
    X, y = data(seed, n, p)
    wr("$(name)_X.csv", X); wr("$(name)_y.csv", y)
    for std in (false, true)
      wr("$(name)_std$(Int(std))_levels.csv", levels(X, y, std, rmin))
    end
  end
else
  coefmat(path) = hcat([Vector(b) for b in path.βpath]...)
  function report(label, B, b0, R)
    K = size(B, 2)
    Bm, b0m = R[2:end, end-K+1:end], R[1, end-K+1:end]     # R: the intercept in the first row; ncpen may prepend levels
    d = maximum(abs, B .- Bm); d0 = maximum(abs, b0 .- b0m)
    same = count(k -> (abs.(B[:, k]) .> 1e-10) == (abs.(Bm[:, k]) .> 1e-10), 1:K)     # entries beyond rounding
    @printf "%-32s max |Δβ| %.1e  max |Δβ₀| %.1e  same support %d/%d\n" label d d0 same K
  end
  for (name, seed, n, p, rmin) in sets
    X, y = rd("$(name)_X.csv"), vec(rd("$(name)_y.csv"))
    for std in (false, true)
      λs = vec(rd("$(name)_std$(Int(std))_levels.csv"))
      tag = "$(name) std=$(std)"
      # glmtlp: TLP from the lasso at each level, its DC steps to convergence
      f = joinpath(dir, "$(name)_std$(Int(std))_glmtlp.csv")
      if isfile(f)
        R = rd(basename(f))
        P = LassoPath(X, y, λs; penalty = CappedL1(1, TAU), intercept = true, standardizeX = std, optTol = 1e-13)
        report("$tag glmtlp tlp", coefmat(P), P.β0path, R)
        # lasso at single levels gives the same as the path
        ks = (10, 20, 30)
        dl = maximum(k -> maximum(abs, Vector(lasso(X, y, λs[k]; penalty = CappedL1(1, TAU), intercept = true,
                                                    standardizeX = std, optTol = 1e-13).x) .- R[2:end, k]), ks)
        @printf "%-32s lasso at levels %s: max |Δβ| %.1e\n" "$tag glmtlp tlp" ks dl
      end
      # ncpen: CCCP warm started along the path = LLA from the previous level, to convergence
      for (pen, g) in (("mcp", MCP(1, 3)), ("scad", SCAD(1, 3.7)), ("tlp", CappedL1(1, TAU)))
        f = joinpath(dir, "$(name)_std$(Int(std))_ncpen_$(pen).csv")
        isfile(f) || (println("$tag ncpen $pen: no R result"); continue)
        R = rd(basename(f))
        P = LassoPath(X, y, λs; penalty = g, lla = LLA(; stages = 1000, start = :previous), intercept = true,
                      standardizeX = std, optTol = 1e-13)
        report("$tag ncpen $pen", coefmat(P), P.β0path, R)
      end
    end
  end
end
