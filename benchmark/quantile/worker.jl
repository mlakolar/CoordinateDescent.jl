include("solvers.jl")
using Printf, Profile

function data(case,seed)
    configs=Dict("small"=>(1000,100,0.,false),"wide"=>(1000,1000,.5,false),
      "correlated"=>(1000,100,.9,false),"tall"=>(10000,100,.5,false),
      "large"=>(10000,1000,.5,false),"ultrawide"=>(1000,10000,0.,false),
      "tall100k"=>(100000,100,0.,false),"sparse100k"=>(100000,10000,0.,true))
    n,p,rho,sp=configs[case]; rng=Xoshiro(seed)
    X=sp ? sprandn(rng,n,p,.001) : randn(rng,n,p)
    if rho>0
        for j in 2:p
            @views X[:,j] .= rho.*X[:,j-1] .+ sqrt(1-rho^2).*X[:,j]
        end
    end
    for j in 1:p
        @views X[:,j] ./= sqrt(sum(abs2,X[:,j])/n)
    end
    beta=zeros(p); beta[1:10]=range(1.,.3,length=10)
    # Heteroscedastic t3 errors in correlated and wide cases.
    e=randn(rng,n)
    if case in ("wide","correlated","large")
        e ./= sqrt.((randn(rng,n).^2 .+ randn(rng,n).^2 .+ randn(rng,n).^2)./3)
        e .*= .5 .+ .5.*abs.(X[:,1])
    end
    X,.7 .+ X*beta .+ e
end
function runmethod(name,X,y,ls,tau,w;limit=30.)
    parts=split(name,'-')
    if parts[1]=="lp"
        lppath(X,y,ls,tau,w,Symbol(parts[2]),Symbol(parts[3]),parts[4]=="warm";limit)
    elseif parts[1]=="admm"
        ladmmpath(X,y,ls,tau,w;limit)
    else
        smoothpath(X,y,ls,tau,w,Symbol(parts[1]);h=parse(Float64,parts[2]),limit,
           warm=length(parts)<3 || parts[3]!="cold")
    end
end
function main()
    case=ARGS[1]; seed=parse(Int,ARGS[2]); reps=parse(Int,ARGS[3]); phase=ARGS[4]
    out=ARGS[5]; mkpath(out)
    methods=["lp-$f-$s-$w" for f in ("primal","dual") for s in ("primal","dual","ipm") for w in ("cold","warm")]
    append!(methods,["apg-0.2","apg-0.05","newton-0.2","newton-0.05","admm"])
    phase=="main" && (methods=["lp-primal-dual-cold","lp-primal-dual-warm","lp-primal-primal-warm",
       "lp-dual-dual-cold","lp-dual-dual-warm","lp-primal-ipm-cold","newton-0.2","newton-0.05","apg-0.2","admm"])
    phase=="scale" && (methods=["lp-primal-dual-warm","lp-dual-dual-warm","newton-0.2","apg-0.2","admm"])
    # Compile methods on the SAME data type before collecting measurements.
    tx,ty=data("small",100); tx=tx[1:50,1:12]; ty=ty[1:50]
    startswith(case,"sparse") && (tx=SparseArrays.sparse(tx))
    for name in methods
        runmethod(name,tx,ty,[.2,.05],.5,ones(12);limit=2.)
    end
    X,y=data(case,seed); n,p=size(X); w=ones(p)
    tau=case=="correlated" ? .2 : .5
    lm=maximum(abs.(X'*(tau .- (y .< quantile(y,tau)))))/n
    ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)))
    rng=Xoshiro(90000+seed)
    pathfile=joinpath(out,"$case-$seed-paths.csv"); solfile=joinpath(out,"$case-$seed-solutions.csv")
    open(pathfile,"w") do io
      println(io,"case,seed,rep,method,n,p,tau,status,levels,total_s,setup_s,optimize_s,extract_s,bytes,gc_s,iterations,host")
      open(solfile,"w") do so
       println(so,"case,seed,rep,method,k,lambda,primal,dual,gap,sumdual,smooth_kkt,nnz")
       for rep in 1:reps, name in shuffle(rng,methods)
        GC.gc()
        try
            timed=@timed runmethod(name,X,y,ls,tau,w;limit=phase=="pilot" ? 20. : 30.)
            r=timed.value
            println(io,join((case,seed,rep,name,n,p,tau,r.status,length(r.answers),timed.time,r.setup,r.optimize,r.extract,timed.bytes,timed.gctime,r.iters,gethostname()),',')); flush(io)
            for (k,s) in enumerate(r.answers)
                c=certificate(X,y,s.b,s.a,s.lambda,tau,w,s.u)
                sk=startswith(name,"newton") || startswith(name,"apg") ? first(smoothkkt(X,y,s.b,s.a,s.lambda,w,PB.ConvolutionCheck(tau,parse(Float64,split(name,'-')[2])))) : NaN
                println(so,join((case,seed,rep,name,k,s.lambda,c.primal,c.dual,c.gap,c.sumdual,sk,count(abs.(s.b).>1e-7)),','));flush(so)
            end
            println("DONE $case seed=$seed rep=$rep $name $(r.status) $(round(timed.time,digits=3))s"); flush(stdout)
        catch err
            println(io,join((case,seed,rep,name,n,p,tau,"ERROR",0,NaN,NaN,NaN,NaN,0,0,0,gethostname()),','));flush(io)
            showerror(stderr,err,catch_backtrace());println(stderr);flush(stderr)
        end
       end
      end
    end
    if phase=="pilot"
        open(joinpath(out,"$case-$seed-profile.txt"),"w") do io
         for name in ("lp-primal-dual-warm","lp-dual-dual-warm","newton-0.2","apg-0.2","admm")
            Profile.clear(); @profile runmethod(name,X,y,ls,tau,w;limit=2.)
            println(io,"METHOD $name");Profile.print(io;C=true,format=:flat,sortedby=:count)
         end
        end
    end
end
main()
