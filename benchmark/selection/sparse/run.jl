include("io.jl")
using Profile
engine=ARGS[1];rep=parse(Int,ARGS[2]);mode=ARGS[3]
if engine=="baseline"
  @eval module Baseline
    using LinearAlgebra, Statistics, ProximalBase, ProximalSolvers
    using CoordinateDescent: erfcinv
    import StatsAPI
    include("baseline_slope.jl")
  end
elseif engine=="SLOPE.jl"
  @eval import SLOPE as SJ
end
function fit(c)
  if engine=="SLOPE.jl"
    f=SJ.slope(c.X,c.y;α=c.factors,λ=c.lambda,fit_intercept=c.intercept,
      centering=c.intercept ? :mean : :none,scaling=c.standardize_x ? :sd : :none,
      tol=1e-8,max_it=100000,max_clusters=size(c.X,2)+1,dev_change_tol=0.0,dev_ratio_tol=1.0)
    length(f.coefficients)==length(c.factors) || error("truncated path")
    hcat((vec(Matrix(b)) for b in f.coefficients)...),c.intercept ? [only(b) for b in f.intercepts] : zeros(length(c.factors))
  else
    backend=engine=="baseline" ? Baseline : CD
    f=backend.SLOPEPath(c.X,c.y,c.factors;lambda=c.lambda,intercept=c.intercept,
      standardize_x=c.standardize_x,tol=1e-7,max_iter=100000)
    all(f.converged) || error("$engine convergence")
    hcat(f.βpath...),f.β0path
  end
end
ids=mode=="memory" ? [parse(Int,ARGS[4])] : parse.(Int,split(get(ENV,"CASES","1,2,3,4,5,6,7,8,9,10,11"),','))
engine=="baseline" && filter!(!=(11),ids)
path=joinpath(OUT,"$(mode)-$(engine)-$(rep)$(mode=="memory" ? "-$(only(ids))" : "").csv")
mkpath(OUT)
open(joinpath(OUT,"environment-$engine-$rep.txt"),"w") do io
  println(io,"Julia=$VERSION node=$(gethostname()) threads=$(Threads.nthreads()) BLAS=$(BLAS.get_config())")
  println(io,"CoordinateDescent=$(pathof(CD)) ProximalBase=$(pathof(PB)) ProximalSolvers=$(pathof(CD.ProximalSolvers))")
  engine=="SLOPE.jl" && println(io,"SLOPE=$(pkgversion(SJ))")
end
open(path,"w") do io
  println(io,"rep,case,engine,inner,seconds,bytes,relative_residual,design_bytes,workspace_bytes")
  for id in ids
    c=loadcase(id)
    fit(c) # compile before calibrating timed batch size
    warm=@elapsed fit(c)
    GC.gc();ncalls=mode=="memory" ? 1 : clamp(ceil(Int,0.1/max(warm,1e-6)),1,20)
    B=zeros(0,0);b0=Float64[]
    for inner in 1:(mode=="memory" ? 1 : 3)
      r=@timed begin
        result=nothing
        for _ in 1:ncalls;result=fit(c);end
        result
      end
      B,b0=r.value
      cert,_=certificate(c,B,b0)
      # Do not mask failures: record coefficients and diagnostics before failing.
      writedlm(joinpath(OUT,"coef-$rep-$id-$engine.csv"),vcat(transpose(b0),B),',')
      designbytes=0;workspacebytes=0
      if engine=="native"
        Z,v,_,_,_=CD._slope_data(c.X,c.y,c.intercept,c.standardize_x)
        designbytes=Base.summarysize(Z);workspacebytes=Base.summarysize(CD.ProximalSolvers._slope_workspace(Z,v))
      end
      println(io,join((rep,id,engine,inner,r.time/ncalls,r.bytes/ncalls,cert,designbytes,workspacebytes),','));flush(io)
      cert<=1e-7 || error("$engine case$id common certificate $cert")
    end
    if mode=="timing" && engine=="native" && id in (4,10)
      Profile.clear();started=time();calls=0
      Profile.@profile while calls<1 || time()-started<0.25;fit(c);calls+=1;end
      open(joinpath(OUT,"profile-$rep-$id.txt"),"w") do pi
        println(pi,"rep=$rep case=$id calls=$calls")
        Profile.print(IOContext(pi,:displaysize=>(10000,240));format=:flat,sortedby=:count,mincount=2)
      end
    end
    println("DONE $engine $mode rep=$rep case=$id");flush(stdout)
    GC.gc()
  end
end
