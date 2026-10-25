# An explicitly oracle-assisted stopping ablation: the known optimal loss is
# supplied by the benchmark, not available to a general-purpose user solver.
ENV["CD81_LIBRARY"]="1"
include(joinpath(@__DIR__,"julia_worker.jl"))
function matched_rank(d)
    c=cache(d);began=time()
    function stop(iter,state)
        B=PA.default_solution(iter,state)
        value=c.f(B)
        abs(value-d.scalars[5])/max(1,abs(value))<=1e-8 || time()-began>CAP
    end
    B,k=PA.ForwardBackward(;maxit=20000,stop)(;x0=zeros(d.p,d.q),f=c.f,
               g=PO.IndBallRank(d.r),gamma=1/c.g.L,adaptive=false)
    B,k,time()-began<CAP
end
matched_rank(problem("inputs/small-1"))
for rep in 1:5, name in ("medium","wide")
    d=problem("inputs/$name-$rep");out="results/rank-matched/$name-$rep/julia";mkpath(out)
    matched_rank(d)
    open(joinpath(out,"rank-core.csv"),"w") do f
        println(f,join(FIELDS,','))
        for rd in 1:3
            t=@elapsed B,k,ok=matched_rank(d)
            obj,gap,res=certificate(d,"rank",B,0)
            status=gap<=1e-6 && res==0 ? "passed" : "accuracy_failed"
            @assert status=="passed"
            println(f,join(("rank_matched","julia",name,rep,"rank","pa_rank_oracle","fit",
                0,rd,t,k,obj,gap,res,status,ok,Sys.maxrss()/1024,0,"known_optimal_loss_stopping_1e-8"),','))
            rd==1 && open(g->write(g,B),joinpath(out,"pa_rank_oracle-0.bin"),"w")
        end
    end
    Profile.clear();began=time();calls=0
    @profile while time()-began<.25 || calls==0
        matched_rank(d);calls+=1
    end
    open(joinpath(out,"profile-pa_rank_oracle.txt"),"w") do f
        println(f,"case=",name," rep=",rep," calls=",calls," samples=",length(Profile.fetch()))
        Profile.print(f;format=:flat,sortedby=:count,mincount=2)
    end
end
