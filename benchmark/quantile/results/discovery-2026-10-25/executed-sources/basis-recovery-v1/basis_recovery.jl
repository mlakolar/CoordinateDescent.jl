include("basis_common.jl")
include("initialized_split.jl")
using DelimitedFiles
function recoverrun(name,X,y,ls;limit=30.)
 if startswith(name,"lp-split-nullbasis")
  return initializedsplitpath(X,y,ls,.5,ones(size(X,2)),endswith(name,"primal") ? :primal : :dual;limit)
 end
 runmethod(name,X,y,ls;limit)
end
function main()
 case=ARGS[1];seed=parse(Int,ARGS[2]);out="basis-recovery-results";mkpath(out)
 plan=readdlm("basis-recovery-plan.tsv",'\t',Any)
 cells=[(Int(r[3]),string(r[4])) for r in eachrow(plan) if string(r[1])==case && Int(r[2])==seed]
 tx,ty=data("small",1200)
 for name in unique(last.(cells));recoverrun(name,tx[1:60,1:12],ty[1:60],[.2,.1];limit=10.);end
 if case=="tall100k"
  base=joinpath("rbench-results","$case-$seed")
  X=reshape(readbinary(base*"-X.bin"),100000,100);y=readbinary(base*"-y.bin");ls=readbinary(base*"-lambda.bin")
 else
  X,y=data(case,seed);lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/size(X,1)
  ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)))
 end
 n,p=size(X);w=ones(p)
 open(joinpath(out,"$case-$seed-paths.csv"),"w") do io
  println(io,"case,seed,rep,method,n,p,tau,status,levels,total_s,setup_s,optimize_s,extract_s,bytes,gc_s,iterations,host,first_lp_pivots")
  open(joinpath(out,"$case-$seed-solutions.csv"),"w") do so
   println(so,"case,seed,rep,method,k,lambda,primal,dual,gap,sumdual,smooth_kkt,nnz")
   for (rep,name) in shuffle(Xoshiro(seed+120000),cells)
    GC.gc();t=@timed recoverrun(name,X,y,ls);r=t.value
    firstp=occursin("nullbasis",name) && !isempty(r.answers) ? first(r.answers).iterations : NaN
    println(io,join((case,seed,rep,name,n,p,.5,r.status,length(r.answers),t.time,r.setup,r.optimize,r.extract,t.bytes,t.gctime,r.iters,gethostname(),firstp),','));flush(io)
    for (k,s) in enumerate(r.answers)
     c=certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u)
     println(so,join((case,seed,rep,name,k,s.lambda,c.primal,c.dual,c.gap,c.sumdual,NaN,count(abs.(s.b).>1e-7)),','));flush(so)
    end
    println("RECOVER $case $seed $rep $name $(r.status) $(t.time)s first_pivots=$firstp");flush(stdout)
   end
  end
 end
end
main()
