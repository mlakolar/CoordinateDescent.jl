# Replicated design-based solves and paths, under Slurm on Discovery.
using CoordinateDescent, LinearAlgebra, Random, Statistics
BLAS.set_num_threads(1)
seed, out = parse(Int,ARGS[1]), ARGS[2]
families = length(ARGS) >= 3 && ARGS[3] == "admm" ? (:graph,:trend) : (:fused,:graph,:trend)
mkpath(out)
rng = Xoshiro(seed)
function problem(rng,n,p)
  X = randn(rng,n,p)
  for j in 2:p
    @views X[:,j] .= .5 .* X[:,j-1] .+ sqrt(.75) .* X[:,j]
  end
  β = [j <= p÷3 ? 0. : j <= 2p÷3 ? .5 : -.5 for j in 1:p]
  X, X*β + .3randn(rng,n)
end
function solve_case(family, X,y,ls,aspath)
  p = size(X,2)
  opts = (;abstol=1e-7,reltol=1e-6,max_iter=20000)
  if family == :fused
    aspath ? fused_lasso_path(X,y,ls;l1_ratio=.25,opts...) :
      fused_lasso(X,y,.25ls[end],.75ls[end];opts...)
  elseif family == :graph
    D = graph_difference(p,[(j,j+1) for j in 1:p-1])
    aspath ? GeneralizedLassoPath(X,y,D,ls;opts...) : generalized_lasso(X,y,D,ls[end];opts...)
  else
    aspath ? trend_filter_path(X,y,ls;order=1,opts...) : trend_filter(X,y,ls[end];order=1,opts...)
  end
end
levels = [.1,.05,.02,.01]
X0,y0 = problem(rng,32,8)
for f in families, path in (false,true)
  solve_case(f,X0,y0,levels,path)
end
open(joinpath(out,"regression-$seed.tsv"),"w") do io
  println(io,"seed\tn\tp\tfamily\tpath\tround\tseconds\tbytes\titerations\tconverged")
  for (n,p) in ((300,80),(1000,200))
    X,y = problem(rng,n,p)
    for f in families, path in (false,true), round in 1:3
      GC.gc()
      m = @timed solve_case(f,X,y,levels,path)
      fits = path ? m.value.solutions : [m.value]
      ok = all(s->s.converged,fits)
      println(io,join((seed,n,p,f,path,round,m.time,m.bytes,sum(s.iterations for s in fits),ok),'\t'))
      flush(io)
      ok || error("nonconvergence in regression benchmark")
    end
  end
end
