include("solvers.jl")
include("null_basis.jl")
include("split_common.jl")
function readbinary(file)
 open(file) do io;read!(io,Vector{Float64}(undef,filesize(file)÷8));end
end
function runmethod(name,X,y,ls;limit=30.)
 w=ones(size(X,2))
 name=="lp-dual-nullbasis-warm" && return nullpath(X,y,ls,.5,w;limit)
 name=="lp-dual-dual-warm" && return lppath(X,y,ls,.5,w,:dual,:dual,true;limit)
 splitpath(X,y,ls,.5,w,name=="lp-split-primal-warm" ? :primal : :dual;limit)
end
function main()
 case=ARGS[1];seed=parse(Int,ARGS[2]);out=ARGS[3];mkpath(out)
 names=["lp-dual-nullbasis-warm","lp-dual-dual-warm"]
 case=="ultrawide" && append!(names,["lp-split-primal-warm","lp-split-dual-warm"])
 tx,ty=data("small",1100)
 for name in names;runmethod(name,tx[1:60,1:12],ty[1:60],[.2,.1];limit=10.);end
 if case=="tall100k"
  base=joinpath("rbench-results","$case-$seed")
  X=reshape(readbinary(base*"-X.bin"),100000,100);y=readbinary(base*"-y.bin");ls=readbinary(base*"-lambda.bin")
 else
  X,y=data(case,seed);n,p=size(X);lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/n
  ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)))
 end
 n,p=size(X);w=ones(p)
 open(joinpath(out,"$case-$seed-paths.csv"),"w") do io
  println(io,"case,seed,rep,method,n,p,tau,status,levels,total_s,setup_s,optimize_s,extract_s,bytes,gc_s,iterations,host,first_lp_pivots")
  open(joinpath(out,"$case-$seed-solutions.csv"),"w") do so
   println(so,"case,seed,rep,method,k,lambda,primal,dual,gap,sumdual,smooth_kkt,nnz")
   for rep in 1:2,name in shuffle(Xoshiro(seed+rep),names)
    GC.gc();t=@timed runmethod(name,X,y,ls);r=t.value
    firstp=name=="lp-dual-nullbasis-warm" && !isempty(r.answers) ? first(r.answers).iterations : NaN
    println(io,join((case,seed,rep,name,n,p,.5,r.status,length(r.answers),t.time,r.setup,r.optimize,r.extract,t.bytes,t.gctime,r.iters,gethostname(),firstp),','));flush(io)
    for (k,s) in enumerate(r.answers)
     c=certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u)
     println(so,join((case,seed,rep,name,k,s.lambda,c.primal,c.dual,c.gap,c.sumdual,NaN,count(abs.(s.b).>1e-7)),','));flush(so)
    end
    println("DONE $case $seed $rep $name $(r.status) $(t.time)s first_pivots=$firstp");flush(stdout)
   end
  end
 end
end
main()
