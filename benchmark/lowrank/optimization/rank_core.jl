ENV["CD81_LIBRARY"]="1"
include(joinpath(@__DIR__,"julia_worker.jl"))
function coefficient_only(d)
    z=CD._lr_data(d.X,d.Y,true);f=CD._lr_svd(z)
    CD._lr_rank(d.r,min(f.xrank,d.q))
    CD._lr_rrcoef(z,f,d.r)
end
coefficient_only(problem("inputs/small-1"))
for rep in 1:5, name in ("small","medium","responses","wide","ill","deficient")
    d=problem("inputs/$name-$rep");out="results/rank-core/$name-$rep/julia";mkpath(out)
    coefficient_only(d)
    open(joinpath(out,"rank-core.csv"),"w") do f
        println(f,join(FIELDS,','))
        for rd in 1:3
            t=@elapsed B=coefficient_only(d)
            obj,gap,res=certificate(d,"rank",B,0)
            status=gap<=1e-6 && res==0 ? "passed" : "accuracy_failed"
            @assert status=="passed"
            println(f,join(("rank_core","julia",name,rep,"rank","cd_rank_core","coefficient_only",
                0,rd,t,0,obj,gap,res,status,true,Sys.maxrss()/1024,0,"excludes_result_diagnostics"),','))
            if rd==1
                open(g->write(g,B),joinpath(out,"cd_rank_core-0.bin"),"w")
            end
        end
    end
    if name in ("medium","responses","wide")
        Profile.clear();began=time();calls=0
        @profile while time()-began<.25 || calls==0
            coefficient_only(d);calls+=1
        end
        open(joinpath(out,"profile-cd_rank_core.txt"),"w") do f
            println(f,"case=",name," rep=",rep," calls=",calls," samples=",length(Profile.fetch()))
            Profile.print(f;format=:flat,sortedby=:count,mincount=2)
        end
    end
end
