ENV["CD81_LIBRARY"]="1"
include(joinpath(@__DIR__,"julia_worker.jl"))
rep=parse(Int,ARGS[1]);d=problem("inputs/medium-$rep");out="results/adaptive/medium-$rep/julia";mkpath(out)
w=readarr("inputs/medium-$rep/adaptive_weights.bin",d.p)
λmax=parse(Float64,strip(read("inputs/medium-$rep/adaptive_max.txt",String)))
z=CD._lr_data(d.X,d.Y,true);H=z.X'z.X/d.n;C=z.X'z.Y/d.n
function fit(frac)
    CD._lr_sparse(z,d.r,frac*λmax,w,d.B0;max_iter=1000,tol=1e-8,inner_max_iter=10000,inner_tol=1e-9)
end
function factors(frac,Bexpected)
    S=svd(d.B0);V=S.V[:,1:d.r];A=d.B0*V;B=A*V';λ=frac*λmax
    for k in 1:1000
        old=B;inner=CD._lr_group!(A,H,C*V,λ,w;max_iter=10000,tol=1e-9)
        S=svd(C'A);V=S.U*S.Vt;B=A*V'
        if inner && norm(B-old)<=1e-8*max(1,norm(B)) &&
            CD._lr_group_kkt(A,H*A-C*V,λ,w)<=1e-9*max(1,norm(C*V))
            break
        end
    end
    @assert norm(B-Bexpected)<=1e-7*max(1,norm(Bexpected))
    A,V
end
fit(.1);baseline=Sys.maxrss()/1024
open(joinpath(out,"adaptive.csv"),"w") do f
    println(f,join(FIELDS,','))
    for rd in 1:3, frac in (.5,.1,.02)
        t=@elapsed sol=fit(frac)
        A,V=factors(frac,sol.B);λ=frac*λmax;E=z.X*A*V'-z.Y
        G=H*A-C*V;GV=E'z.X*A/d.n;tangent=GV-V*((V'GV+GV'V)/2)
        err=max(CD._lr_group_kkt(A,G,λ,w)/max(1,norm(C*V)),
                norm(tangent)/max(1,norm(z.Y'z.X*A/d.n)))
        obj=sum(abs2,E)/(2d.n)+λ*sum(w[j]*norm(view(A,j,:)) for j in 1:d.p)
        status=err<=1e-6 ? "passed" : "accuracy_failed"
        println(f,join(("adaptive","julia","medium",rep,"sparse_adaptive","cd_sparse_adaptive",
            "fit",frac,rd,t,sol.iterations,obj,NaN,err,status,sol.converged,Sys.maxrss()/1024,baseline,""),','))
        flush(f)
        if rd==1
            for (tag,x) in (("bin",sol.B),("A",A),("V",V))
                open(g->write(g,x),joinpath(out,"cd_sparse_adaptive-$frac.$tag"),"w")
            end
        end
    end
end
Profile.clear();began=time();calls=0
@profile while time()-began<.25 || calls==0
    fit(.1);global calls+=1
end
open(joinpath(out,"profile-cd_sparse_adaptive.txt"),"w") do f
    println(f,"rep=",rep," calls=",calls," samples=",length(Profile.fetch()))
    Profile.print(f;format=:flat,sortedby=:count,mincount=2)
end
