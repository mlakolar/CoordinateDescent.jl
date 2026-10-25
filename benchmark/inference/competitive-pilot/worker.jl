# One independent process per fit, under Slurm. No local Julia execution.
using CoordinateDescent, ProximalBase, Random, LinearAlgebra, DelimitedFiles, Statistics
BLAS.set_num_threads(1)
engine, mode, scenario, rep = ARGS
dest = joinpath("data", scenario, rep)
out = joinpath("outputs", join((engine, mode, scenario, rep), '-'))
meta = readdlm(joinpath(dest, "config.csv"), ',', String)
config = Dict(meta[1,k] => meta[2,k] for k in axes(meta,2))
X = readdlm(joinpath(dest,"X.csv"), ',', Float64)
y = vec(readdlm(joinpath(dest,"y.csv"), ',', Float64))
targets = vec(readdlm(joinpath(dest,"targets.csv"), Int))
family = Symbol(config["family"])
lam = parse(Float64, config["lambda_main"])
node = parse(Float64, config["lambda_node"])
gen() = Xoshiro(770000 + parse(Int,rep))

function fit(A,b,ts)
  kw = (; family, targets=ts, threads=false, rng=gen())
  engine == "oursqp" && (kw = (;kw..., method=:qp))
  if mode == "matched"
    return debiased_lasso(A,b,lam;kw...,nodewise_lambda=node,
                         options=CDOptions(;max_iter=100000,opt_tol=1e-7))
  else
    # The public GLM API has no automatic initial lambda. Supply the predeclared
    # required value, leaving its other statistical defaults unchanged.
    return family == :gaussian ? debiased_lasso(A,b;kw...) : debiased_lasso(A,b,lam;kw...)
  end
end

function main()
  # Compile each selected path on a bounded independent warm-up problem.
  rg = Xoshiro(177)
  A = randn(rg,80,12)
  b = family == :gaussian ? randn(rg,80) : Float64.(rand(rg,80) .< .5)
  fit(A,b,[1,2])
  GC.gc()
  started = time_ns()
  r = fit(X,y,targets)
  elapsed = (time_ns()-started)/1e9
  ci = confint(r)
  open(out*".csv","w") do io
    println(io,"target,estimate,se,lower,upper,pvalue")
    writedlm(io,hcat(targets,coef(r),stderror(r),ci,r.pvalues),',')
  end
  open(out*".meta","w") do io
    println(io,"status=",r.converged ? "ok" : "nonconverged")
    println(io,"fit_seconds=",elapsed)
    println(io,"lambda=",r.lambda)
    println(io,"node_lambda=",first(r.projection.lambda))
    println(io,"sigma=",r.sigma)
    println(io,"actual_inferred_targets=",length(targets))
    println(io,"passes=",sum(r.projection.iterations))
    println(io,"required_glm_lambda=",family != :gaussian)
  end
end
try
  main()
catch e
  open(out*".meta","w") do io
    println(io,"status=error")
    println(io,"error=",replace(sprint(showerror,e),'\n'=>' '))
  end
  showerror(stderr,e,catch_backtrace()); println(stderr)
end
