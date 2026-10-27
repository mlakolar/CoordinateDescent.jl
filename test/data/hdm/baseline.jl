# U1 regression guard for the unmodified source at
# d6174f5c148ed7e08a4732c915aa691ecb83cb8a. Never regenerate after changing
# an estimator merely to make a baseline test pass.
# For dataset k in the order below: rlasso y/true, d/true, y/false, d/false
# use seeds 11900+10k+1:4; double selection and partialling use +5 and +6.
# Random.seed! is immediately before EVERY estimator call because the
# default coordinate order is random. Double selection fits d then y;
# partialling fits y then d. Forced controls enter double selection only.
# CSV schema and the empty-vector sentinel match golden.R.

using HighDimensionalModels, DelimitedFiles, LinearAlgebra, Printf, Random
BLAS.set_num_threads(1)
root = isempty(ARGS) ? (@__DIR__) : only(ARGS)

for (k, dataset) in enumerate(("hetero", "highdim", "forced", "nearthreshold", "empty"))
  dir = joinpath(root, dataset)
  X = readdlm(joinpath(dir, "X.csv"), ',', Float64)
  y = vec(readdlm(joinpath(dir, "y.csv"), ',', Float64))
  d = vec(readdlm(joinpath(dir, "d.csv"), ',', Float64))
  forced = dataset == "forced" ? vec(readdlm(joinpath(dir, "forced.csv"), ',', Int)) : Int[]
  open(joinpath(dir, "baseline-julia.csv"), "w") do io
    println(io, "fit,field,index,value")
    function put(fit, field, x)
      a = x isa Number ? [x] : collect(x)
      if isempty(a)
        println(io, fit, ',', field, ",0,0")
      else
        for (i, value) in enumerate(a)
          @printf(io, "%s,%s,%d,%.17g\n", fit, field, i, value)
        end
      end
    end
    for (j, (post, response)) in enumerate(((true, :y), (true, :d), (false, :y), (false, :d)))
      target = response == :y ? y : d
      Random.seed!(11900 + 10k + j)
      r = rlasso(X, target; post)
      name = "rlasso_$(response)_$post"
      put(name, "selected", r.selected)
      put(name, "intercept", r.intercept)
      put(name, "coefficients", Vector(r.x))
      for (field, value) in (("residuals", r.residuals), ("loadings", r.loadings),
                             ("lambda0", r.λ0), ("iterations", r.iterations), ("sigma", r.σ))
        put(name, field, value)
      end
    end
    for (j, method) in enumerate((:doubleselection, :partialling))
      force = method == :doubleselection ? forced : Int[]
      Random.seed!(11900 + 10k + 4 + j)
      r = double_selection(y, d, X; method, forced = force)
      name = "effect_$method"
      for field in (:estimate, :se, :t, :pvalue, :ci, :selected)
        put(name, string(field), getproperty(r, field))
      end
    end
  end
  println(dataset, " baseline written")
end
