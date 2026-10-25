include("solvers.jl")
using Profile
function runprofile(name,X,y,ls)
    w=ones(size(X,2))
    if name=="highs-dual-dual"
        lppath(X,y,ls,.5,w,:dual,:dual,true;limit=2.)
    elseif name=="highs-primal-dual"
        lppath(X,y,ls,.5,w,:primal,:dual,true;limit=2.)
    elseif name=="gurobi-primal-dual"
        lppath(X,y,ls,.5,w,:primal,:dual,true;limit=2.,engine=:gurobi)
    elseif name=="gurobi-primal-primal"
        lppath(X,y,ls,.5,w,:primal,:primal,true;limit=2.,engine=:gurobi)
    elseif name=="newton"
        smoothpath(X,y,ls,.5,w,:newton;h=.2,limit=2.)
    elseif name=="apg"
        smoothpath(X,y,ls,.5,w,:apg;h=.2,limit=2.)
    else
        ladmmpath(X,y,ls,.5,w;limit=2.)
    end
end
function main()
    out=ARGS[1];mkpath(out)
    for seed in (501,502), p in (100,1000)
        rng=Xoshiro(seed); X=randn(rng,1000,p); y=.7 .+ X[:,1:10]*ones(10) .+ randn(rng,1000)
        lm=maximum(abs.(X'*(.5 .- (y .< median(y)))))/1000
        ls=collect(exp.(range(log(lm*1.05),log(lm*.1),length=8)))
        for name in ("highs-dual-dual","highs-primal-dual","gurobi-primal-dual","gurobi-primal-primal","newton","apg","admm")
            # Warm every dispatch path before sampling. Short methods are run
            # repeatedly so a profile is not built from just a few snapshots.
            runprofile(name,X[1:60,1:12],y[1:60],[.2,.05])
            GC.gc();Profile.clear();count=0;start=time()
            @profile while time()-start<1.5
                runprofile(name,X,y,ls);count+=1
            end
            open(joinpath(out,"p$p-seed$seed-$name.txt"),"w") do raw
                println(raw,"name=$name n=1000 p=$p seed=$seed fits=$count elapsed=$(time()-start) host=$(gethostname())")
                io=IOContext(raw,:displaysize=>(2000,240))
                Profile.print(io;C=true,format=:flat,sortedby=:count)
            end
        end
    end
end
main()
