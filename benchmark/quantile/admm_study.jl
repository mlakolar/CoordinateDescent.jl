include("solvers.jl")
include("block_admm.jl")
using Test
function data(case,seed)
    n,p=Dict("small"=>(1000,100),"wide"=>(1000,1000),"tall"=>(10000,100),"ultrawide"=>(1000,10000))[case]
    rng=Xoshiro(seed);X=randn(rng,n,p)
    for j in 2:p
        @views X[:,j] .= 0.5 .* X[:,j-1] .+ sqrt(.75) .* X[:,j]
    end
    X,.7 .+ X[:,1:10]*collect(range(1.,.3,length=10)) .+ randn(rng,n)
end
function runmethod(name,X,y,ls)
    w=ones(size(X,2))
    name=="block-admm" && return blockadmmpath(X,y,ls,.5,w;limit=30.)
    name=="linear-admm" && return ladmmpath(X,y,ls,.5,w;limit=30.)
    lppath(X,y,ls,.5,w,:dual,:dual,true;limit=30.)
end
function main()
    case=ARGS[1];seed=parse(Int,ARGS[2]);out=ARGS[3];mkpath(out)
    if case=="test"
        @testset "factorized ADMM certificates" begin
            rng=Xoshiro(615)
            for (n,p) in ((60,12),(30,60)), tau in (.2,.5,.8)
                X=randn(rng,n,p);y=.7 .+ X[:,1:3]*ones(3) .+ randn(rng,n)
                ls=[.2,.1];r=blockadmmpath(X,y,ls,tau,ones(p);limit=30.)
                @test r.status=="complete"
                for s in r.answers
                    c=certificate(X,y,s.b,s.a,s.lambda,tau,ones(p),s.u)
                    @test -1e-7<=c.gap<=1e-5*(1+abs(c.primal))
                end
            end
        end
        return
    end
    tx,ty=data("small",100)
    for name in ("block-admm","linear-admm","lp-dual-dual-warm");runmethod(name,tx[1:60,1:12],ty[1:60],[.2,.1]);end
    X,y=data(case,seed);n,p=size(X);w=ones(p);lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/n
    ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)));rng=Xoshiro(18000+seed)
    open(joinpath(out,"$case-$seed-paths.csv"),"w") do io
      println(io,"case,seed,rep,method,n,p,tau,status,levels,total_s,setup_s,optimize_s,extract_s,bytes,gc_s,iterations,host")
      open(joinpath(out,"$case-$seed-solutions.csv"),"w") do so
       println(so,"case,seed,rep,method,k,lambda,primal,dual,gap,sumdual,smooth_kkt,nnz")
       for rep in 1:2, name in shuffle(rng,["block-admm","linear-admm","lp-dual-dual-warm"])
        GC.gc();t=@timed runmethod(name,X,y,ls);r=t.value
        println(io,join((case,seed,rep,name,n,p,.5,r.status,length(r.answers),t.time,r.setup,r.optimize,r.extract,t.bytes,t.gctime,r.iters,gethostname()),','));flush(io)
        for (k,s) in enumerate(r.answers)
            c=certificate(X,y,s.b,s.a,s.lambda,.5,w,s.u)
            println(so,join((case,seed,rep,name,k,s.lambda,c.primal,c.dual,c.gap,c.sumdual,NaN,count(abs.(s.b).>1e-7)),','));flush(so)
        end
        println("DONE $case $seed $rep $name $(r.status) $(t.time)s");flush(stdout)
       end
      end
    end
end
main()
