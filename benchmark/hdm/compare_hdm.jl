# rlasso and doubleSelection against R's hdm (rlasso, rlassoEffect) on the same
# data, and the coverage of doubleSelection's confidence intervals.
#
#     julia --project=benchmark/hdm benchmark/hdm/compare_hdm.jl gen DIR       # data in DIR
#     Rscript benchmark/hdm/compare_hdm.R DIR                                  # hdm's results in DIR (needs hdm)
#     julia --project=benchmark/hdm benchmark/hdm/compare_hdm.jl compare DIR   # the comparison
#     julia --project=benchmark/hdm benchmark/hdm/compare_hdm.jl coverage      # 500 replications

using CoordinateDescent, DelimitedFiles, Printf, Random, Statistics

# heteroskedastic designs with a treatment d that depends on the controls that
# also enter y, so that selecting on y alone would bias the estimate of α = 0.5
const SETTINGS = [(n = 100, p = 50), (n = 200, p = 100), (n = 300, p = 300), (n = 500, p = 200), (n = 1000, p = 500)]

function dataset(n, p; seed)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  for j in 2:p
    @views X[:, j] .= 0.5 .* X[:, j-1] .+ sqrt(0.75) .* X[:, j]
  end
  d = X[:, 1:4] * [1.0, -0.5, 0.5, 0.25] .+ randn(rng, n)
  y = 0.5 .* d .+ X[:, 1:4] * [0.5, 0.5, -0.5, 0.25] .+ (0.5 .+ 0.5 .* abs.(X[:, 5])) .* randn(rng, n)
  X, y, d
end

function gen(dir)
  mkpath(dir)
  for (k, s) in enumerate(SETTINGS)
    X, y, d = dataset(s.n, s.p; seed = 100 + k)
    writedlm(joinpath(dir, "X$k.csv"), X, ',')
    writedlm(joinpath(dir, "y$k.csv"), y, ',')
    writedlm(joinpath(dir, "d$k.csv"), d, ',')
  end
end

function compare(dir)
  for (k, s) in enumerate(SETTINGS)
    X = readdlm(joinpath(dir, "X$k.csv"), ','); y = vec(readdlm(joinpath(dir, "y$k.csv"), ','))
    d = vec(readdlm(joinpath(dir, "d$k.csv"), ','))
    for post in (true, false)
      r = rlasso(X, y; post)
      R = vec(readdlm(joinpath(dir, "rlasso$(post ? "" : "_nopost")$k.csv"), ','))    # intercept, β
      selR = findall(!iszero, R[2:end])
      @printf "n=%4d p=%3d rlasso post=%-5s | selected: same %-5s (ours %d, hdm %d) | max |Δcoef| %.1e\n" s.n s.p post (r.selected == selR) length(r.selected) length(selR) maximum(abs, [r.intercept; Vector(r.x)] .- R)
    end
    for method in (:doubleselection, :partialling)
      e = doubleSelection(y, d, X; method)
      R = vec(readdlm(joinpath(dir, "effect_$(method)$k.csv"), ','))                 # α, se
      @printf "n=%4d p=%3d %-15s | α ours %.6f hdm %.6f | se ours %.6f hdm %.6f | Δα %.1e Δse %.1e\n" s.n s.p method e.estimate R[1] e.se R[2] abs(e.estimate - R[1]) abs(e.se - R[2])
    end
  end
end

# the design of the simulations of Belloni, Chernozhukov and Hansen (2014):
# AR(1) correlation 0.5 between the controls, and coefficients 1/j² of both the
# outcome and the treatment on control j, so that the confounding is
# approximately sparse
function bch2014(n, p; seed)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  for j in 2:p
    @views X[:, j] .= 0.5 .* X[:, j-1] .+ sqrt(0.75) .* X[:, j]
  end
  b = [1 / j^2 for j in 1:p]
  d = X * b .+ randn(rng, n)
  X, 0.5 .* d .+ X * b .+ randn(rng, n), d
end

function coverage(; reps = 500)
  for (name, gen, n, p) in (("BCH 2014, 1/j² coefficients", bch2014, 100, 200), ("BCH 2014, 1/j² coefficients", bch2014, 200, 100),
                            ("the comparison design, weak confounders", dataset, 200, 100))
    for method in (:doubleselection, :partialling)
      cover, est, ses = 0, Float64[], Float64[]
      for r in 1:reps
        X, y, d = gen(n, p; seed = 10_000 + r)
        e = doubleSelection(y, d, X; method)
        cover += e.ci[1] <= 0.5 <= e.ci[2]
        push!(est, e.estimate); push!(ses, e.se)
      end
      @printf "coverage %-40s %-15s n=%d p=%d reps=%d: 95%% CI covers α = 0.5 in %.3f; mean estimate %.4f, sd %.4f, mean se %.4f\n" name method n p reps cover / reps mean(est) std(est) mean(ses)
      flush(stdout)
    end
  end
end

mode = isempty(ARGS) ? "coverage" : ARGS[1]
mode == "gen" ? gen(ARGS[2]) : mode == "compare" ? compare(ARGS[2]) : coverage()
