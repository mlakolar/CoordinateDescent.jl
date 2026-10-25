include("solvers.jl")
using DelimitedFiles
function loadbin(file)
    open(file) do io
        read!(io,Vector{Float64}(undef,filesize(file)÷8))
    end
end
# Recover candidate dual entries at nearly-zero residuals using active slope
# equalities. Subsequent projection/contraction makes the lower bound valid
# even if this recovery is approximate or the optimum is degenerate.
function recoverdual(X,y,b,a,lambda)
    n,p=size(X);r=y.-a.-X*b;u=.5 .- (r.<0)
    Z=findall(abs.(r).<=1e-5);S=findall(abs.(b).>1e-6)
    if !isempty(Z)
        A=hcat(ones(n),X[:,S])'
        rhs=vcat(0.,n*lambda.*sign.(b[S])) .- A*u .+ A[:,Z]*u[Z]
        u[Z]=A[:,Z]\rhs
    end
    u
end
function main()
 case=ARGS[1];seed=parse(Int,ARGS[2]);out=ARGS[3]
 n=Dict("small"=>1000,"tall"=>10000,"tall100k"=>100000)[case];p=100
 base=joinpath(out,"$case-$seed");X=reshape(loadbin(base*"-X.bin"),n,p);y=loadbin(base*"-y.bin");ls=loadbin(base*"-lambda.bin");w=ones(p)
 # Read the isolated R timing observations; startup and binary I/O were excluded.
 rtimes=readdlm(base*"-Rtimes.csv",',',Any;header=true)[1]
 open(base*"-paths.csv","w") do io
  println(io,"case,seed,rep,method,n,p,tau,status,levels,total_s,setup_s,optimize_s,extract_s,bytes,gc_s,iterations,host")
  open(base*"-solutions.csv","w") do so
   println(so,"case,seed,rep,method,k,lambda,primal,dual,gap,sumdual,smooth_kkt,nnz")
   function record(method,rep,r,wall,bytes,gctime)
    println(io,join((case,seed,rep,method,n,p,.5,r.status,length(r.answers),wall,r.setup,r.optimize,r.extract,bytes,gctime,r.iters,gethostname()),','));flush(io)
    for (k,s) in enumerate(r.answers)
     c=certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u)
     sk=method in ("conquer","newton-0.2") ? first(smoothkkt(X,y,s.b,s.a,s.lambda,w,PB.ConvolutionCheck(.5,.2))) : NaN
     println(so,join((case,seed,rep,method,k,s.lambda,c.primal,c.dual,c.gap,c.sumdual,sk,count(abs.(s.b).>1e-7)),','));flush(so)
    end
   end
   for row in eachrow(rtimes)
    rep=Int(row[3]);name=replace(string(row[4]),'"'=>"");coef=reshape(loadbin(base*"-$name-$rep.bin"),p+1,8)
    answers=Any[]
    for k in eachindex(ls)
     b=coef[2:end,k];a=coef[1,k]
     u=name=="conquer" ? last(smoothkkt(X,y,b,a,ls[k],w,PB.ConvolutionCheck(.5,.2))) : recoverdual(X,y,b,a,ls[k])
     push!(answers,(;b,a,u,lambda=ls[k]))
    end
    r=(;answers,status="complete",setup=NaN,optimize=NaN,extract=NaN,iters=0)
    record(name,rep,r,Float64(row[5]),0.,0.)
   end
   for name in ("lp-dual-dual-warm","gurobi-primal-primal-warm","newton-0.2")
    tx=X[1:60,1:12];ty=y[1:60]
    if name=="newton-0.2"
      smoothpath(tx,ty,[.2,.1],.5,ones(12),:newton;h=.2)
    else
      lppath(tx,ty,[.2,.1],.5,ones(12),name=="lp-dual-dual-warm" ? :dual : :primal,name=="lp-dual-dual-warm" ? :dual : :primal,true;engine=name=="lp-dual-dual-warm" ? :highs : :gurobi)
    end
   end
   rng=Xoshiro(seed+91000)
   for rep in 1:2,name in shuffle(rng,["lp-dual-dual-warm","gurobi-primal-primal-warm","newton-0.2"])
    GC.gc()
    t=@timed if name=="newton-0.2"
      smoothpath(X,y,ls,.5,w,:newton;h=.2)
    else
      lppath(X,y,ls,.5,w,name=="lp-dual-dual-warm" ? :dual : :primal,name=="lp-dual-dual-warm" ? :dual : :primal,true;limit=30.,engine=name=="lp-dual-dual-warm" ? :highs : :gurobi)
    end
    record(name,rep,t.value,t.time,t.bytes,t.gctime)
    println("DONE $case $seed $rep $name $(t.value.status) $(t.time)s");flush(stdout)
   end
  end
 end
end
main()
