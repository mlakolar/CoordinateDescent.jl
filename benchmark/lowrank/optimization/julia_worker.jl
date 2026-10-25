using LinearAlgebra, Statistics, DelimitedFiles, Profile, Printf
using CoordinateDescent, ProximalBase, ProximalSolvers
import ProximalAlgorithms as PA
import ProximalOperators as PO
import ProximalCore as PC
const CD=CoordinateDescent
const PB=ProximalBase
BLAS.set_num_threads(1)
const CAP=30.0
const RSS_BASELINE=Ref(0.0)
const FACTORS=Ref{Any}(nothing)
const FIELDS=split("phase,language,case,rep,family,solver,mode,frac,round,seconds,iterations,objective,gap,residual,status,native_converged,rss_peak_kb,rss_baseline_kb,detail",',')

readarr(path,dims...) = open(f->reshape(read!(f,Vector{Float64}(undef,prod(dims))),dims...),path)
function problem(path)
    n,p,q,r=vec(readdlm(joinpath(path,"dims.txt"),Int))
    scalars=vec(readdlm(joinpath(path,"scalars.txt"),Float64))
    name,rep=rsplit(basename(path),'-';limit=2)
    (;n,p,q,r,case=name,rep=parse(Int,rep),scalars,
      X=readarr(joinpath(path,"X.bin"),n,p),Y=readarr(joinpath(path,"Y.bin"),n,q),
      B0=readarr(joinpath(path,"B0.bin"),p,q),A0=readarr(joinpath(path,"A0.bin"),p,r),
      V0=readarr(joinpath(path,"V0.bin"),q,r),layer=readarr(joinpath(path,"layer.bin"),n,q),
      u0=vec(readarr(joinpath(path,"u0.bin"),p)),v0=vec(readarr(joinpath(path,"v0.bin"),q)))
end
prepare(d) = (z=CD._lr_data(d.X,d.Y,true);(;d=z,g=CD._lr_gradient(z)))
svt(B,t) = (S=svd(B); (S.U .* max.(S.S .- t,0)')*S.Vt)
function ncert(X,Y,B,λ,L)
    n=size(X,1); R=Y-X*B; G=-X'R/n
    P=sum(abs2,R)/(2n)+λ*sum(svdvals(B))
    Θ=R/n .* min(1,λ/max(opnorm(G),floatmin(Float64)))
    D=dot(Y,Θ)-n*sum(abs2,Θ)/2
    pg=L*norm(B-svt(B-G/L,λ/L))/max(1,norm(X'Y/n))
    P,max(0,P-D)/max(1,abs(P),abs(D)),pg
end
function scert(X,Y,B,λ,r;factors=nothing)
    if factors===nothing
        S=svd(B); A=S.U[:,1:r].*S.S[1:r]';V=S.V[:,1:r]
    else
        A,V=factors
        norm(A*V'-B)<=1e-7*max(1,norm(B)) || error("factor reconstruction mismatch")
    end
    E=X*A*V'-Y;G=X'E*V/size(X,1)
    err=CD._lr_group_kkt(A,G,λ,ones(size(A,1)))/max(1,norm(X'Y*V/size(X,1)))
    GV=E'X*A/size(X,1); tangent=GV-V*((V'GV+GV'V)/2)
    err=max(err,norm(tangent)/max(1,norm(Y'X*A/size(X,1))))
    sum(abs2,E)/(2size(X,1))+λ*sum(norm(view(A,j,:)) for j in axes(A,1)),NaN,err
end
function lasso_kkt(x,g,λ)
    maximum(abs(x[j])<=1e-12 ? max(0,abs(g[j])-λ) : abs(g[j]+λ*sign(x[j])) for j in eachindex(x))
end
function ccert(X,Y,B,λ)
    S=svd(B);S.S[1]<=1e-14 && return (sum(abs2,Y)/(2size(X,1)),NaN,0.0)
    u=S.U[:,1];v=S.V[:,1]*S.S[1];E=X*(u*v')-Y
    gu=X'E*v/size(X,1);gv=E'X*u/size(X,1)
    a=lasso_kkt(u,gu,λ*sum(abs,v))/max(1,norm(X'Y*v/size(X,1)))
    b=lasso_kkt(v,gv,λ*sum(abs,u))/max(1,norm(Y'X*u/size(X,1)))
    sum(abs2,E)/(2size(X,1))+λ*sum(abs,u)*sum(abs,v),NaN,max(a,b)
end
function certificate(d,fam,B,frac;factors=nothing)
    X=d.X.-mean(d.X;dims=1);Y=d.Y.-mean(d.Y;dims=1)
    if fam=="nuclear"
        ncert(X,Y,B,frac*d.scalars[1],d.scalars[4])
    elseif fam=="rank"
        obj=sum(abs2,Y-X*B)/(2d.n)
        obj,abs(obj-d.scalars[5])/max(1,abs(obj)),max(0,rank(B)-d.r)
    elseif fam=="sparse"
        scert(X,Y,B,frac*d.scalars[2],d.r;factors)
    else
        ccert(X,d.layer,B,frac*d.scalars[3])
    end
end

struct GramLoss{D,G} <: PB.AbstractFunction
    d::D
    g::G
end
(f::GramLoss)(B)=sum(abs2,f.d.Y-f.d.X*B)/(2size(f.d.X,1))
PB.issmooth(::Type{<:GramLoss})=true
PB.isconvex(::Type{<:GramLoss})=true
PB.gradient!(G,f::GramLoss,B)=(G .= CD._lr_grad(f.d,f.g,B);G)
function PB.value_and_gradient!(G,f::GramLoss,B)
    PB.gradient!(G,f,B);f(B)
end
function PA.value_and_gradient(f::GramLoss,B)
    G=similar(B);v=PB.value_and_gradient!(G,f,B);v,G
end
function cache(d)
    z=prepare(d)
    (;d=z.d,g=z.g,f=GramLoss(z.d,z.g),
       ls=PB.Scaled(PB.LeastSquaresLoss(z.d.Y,z.d.X),1/d.n))
end
function nuclear(solver,d,frac;start=zeros(d.p,d.q),cached=nothing,trace=nothing)
    if solver=="cd_nuclear" && cached===nothing
        sol=CD.nuclear_regression(d.X,d.Y,frac*d.scalars[1];max_iter=20000,tol=1e-8)
        return sol.B,sol.iterations,sol.converged,""
    end
    c=cached===nothing ? cache(d) : cached
    λ=frac*d.scalars[1];began=time()
    if solver=="cd_nuclear"
        sol=CD._lr_nuclear(c.d,c.g,λ,start;max_iter=20000,tol=1e-8)
        return sol.B,sol.iterations,sol.converged,""
    elseif solver in ("ps_fista","ps_admm")
        method=solver=="ps_fista" ? ProximalSolvers.AccProxGradDescent() :
            ProximalSolvers.ADMMSolver(;ρ=c.g.L,abstol=1e-9,reltol=1e-8)
        f=solver=="ps_fista" ? c.f : c.ls
        opt=ProximalSolvers.ProximalOptions(;maxiter=20000,ftol=1e-30,xtol=1e-30,
             gradtol=5e-9,allow_f_increases=true,time_limit=CAP)
        sol=ProximalSolvers.solve!(copy(start),f,PB.NuclearNorm(λ),method,opt)
        return sol.minimizer,sol.iterations,ProximalSolvers.converged(sol),"native_stopping"
    else
        alg=solver=="pa_fista" ? PA.FastForwardBackward :
            solver=="pa_panoc" ? PA.PANOC : PA.ForwardBackward
        k=Ref(0)
        function stop(iter,state)
            k[]+=1
            B=PA.default_solution(iter,state)
            if k[]%10==0
                G=CD._lr_grad(c.d,c.g,B)
                res=c.g.L*norm(B-svt(B-G/c.g.L,λ/c.g.L))/max(1,norm(c.g.C))
                if trace!==nothing
                    obj,gap,pg=ncert(c.d.X,c.d.Y,B,λ,c.g.L)
                    push!(trace,(k[],time()-began,obj,gap,pg))
                end
                res<=1e-8 && return true
            end
            time()-began>CAP
        end
        B,it=alg(;maxit=20000,stop)(;x0=copy(start),f=c.f,g=PO.NuclearNorm(λ),
                    gamma=1/c.g.L,adaptive=false)
        return B,it,time()-began<CAP,time()-began>=CAP ? "timeout" : ""
    end
end
function group_fista!(A,H,C,λ;tol=1e-9,maxiter=10000,L=opnorm(H))
    Z=copy(A);prev=copy(A);t=1.0
    for k in 1:maxiter
        W=Z-(H*Z-C)/L
        for j in axes(W,1)
            z=norm(view(W,j,:));W[j,:].*=max(0,1-λ/(L*max(z,floatmin(Float64))))
        end
        if k%10==0 && CD._lr_group_kkt(W,H*W-C,λ,ones(size(W,1)))<=tol*max(1,norm(C))
            A.=W;return true
        end
        tn=(1+sqrt(1+4t*t))/2
        if dot(Z-W,W-prev)>0;tn=1;t=1;end
        Z.=W+((t-1)/tn)*(W-prev);prev.=W;t=tn;A.=W
    end
    false
end
function sparse(d,frac;proximal=false)
    if !proximal
        z=CD._lr_data(d.X,d.Y,true)
        sol=CD._lr_sparse(z,d.r,frac*d.scalars[2],ones(d.p),d.B0;
             max_iter=1000,tol=1e-8,inner_max_iter=10000,inner_tol=1e-9)
        return sol.B,sol.iterations,sol.converged,""
    end
    z=CD._lr_data(d.X,d.Y,true);H=z.X'z.X/d.n;C=z.X'z.Y/d.n
    A=copy(d.A0);V=copy(d.V0);B=A*V';λ=frac*d.scalars[2];began=time();L=opnorm(H)
    for k in 1:1000
        old=copy(B)
        ok=group_fista!(A,H,C*V,λ;L)
        S=svd(C'A);V=S.U*S.Vt;B=A*V'
        if ok && norm(B-old)<=1e-8*max(1,norm(B)) &&
            CD._lr_group_kkt(A,H*A-C*V,λ,ones(d.p))<=1e-9*max(1,norm(C*V))
            FACTORS[]=(A,V);return B,k,true,""
        end
        if time()-began>CAP;FACTORS[]=(A,V);return B,k,false,"timeout";end
    end
    FACTORS[]=(A,V);B,1000,false,"iteration_limit"
end
# Recover factors for the existing API, which returns B only. This exact replay
# of its frozen outer loop is outside the fit timer and is needed only to
# certify a rank-deficient factorization without choosing new null directions.
function sparse_factors(d,frac,Bexpected)
    z=CD._lr_data(d.X,d.Y,true);F=svd(d.B0);V=F.V[:,1:d.r];A=d.B0*V
    H=z.X'z.X/d.n;C=z.X'z.Y/d.n;λ=frac*d.scalars[2];B=A*V'
    for k in 1:1000
        old=B
        inner=CD._lr_group!(A,H,C*V,λ,ones(d.p);max_iter=10000,tol=1e-9)
        S=svd(C'A);V=S.U*S.Vt;B=A*V'
        if inner && norm(B-old)<=1e-8*max(1,norm(B)) &&
            CD._lr_group_kkt(A,H*A-C*V,λ,ones(d.p))<=1e-9*max(1,norm(C*V))
            break
        end
    end
    norm(B-Bexpected)<=1e-7*max(1,norm(Bexpected)) || error("native factor replay mismatch")
    A,V
end
function lasso_fista!(u,H,z,λ;tol=1e-9,maxiter=10000,L=opnorm(H))
    L==0 && return true
    y=copy(u);prev=copy(u);t=1.0
    for k in 1:maxiter
        w=sign.(y-(H*y-z)/L).*max.(abs.(y-(H*y-z)/L).-λ/L,0)
        if k%10==0 && lasso_kkt(w,H*w-z,λ)<=tol*max(1,norm(z))
            u.=w;return true
        end
        tn=(1+sqrt(1+4t*t))/2
        if dot(y-w,w-prev)>0;tn=1;t=1;end
        y.=w+((t-1)/tn)*(w-prev);prev.=w;t=tn;u.=w
    end
    false
end
function component(d,frac;proximal=false)
    z=CD._lr_data(d.X,d.layer,false);z=merge(z,(;X=d.X.-mean(d.X;dims=1)))
    λ=frac*d.scalars[3]
    if !proximal
        B,ok,it,df=CD._lr_unit(z,λ,ones(d.p),ones(d.q),d.u0,d.v0;
                   max_iter=1000,tol=1e-8,inner_max_iter=10000,inner_tol=1e-9)
        return B,it,ok,""
    end
    H=z.X'z.X/d.n;C=z.X'z.Y/d.n;u=copy(d.u0);v=copy(d.v0)
    B=zeros(d.p,d.q);began=time();Lh=opnorm(H)
    for k in 1:1000
        old=copy(B)
        ok=lasso_fista!(u,H*sum(abs2,v),C*v,λ;L=Lh*sum(abs2,v))
        du=sum(abs,u);du==0 && return (zero(B),k,true,"")
        u./=du
        h=dot(u,H*u);v.=sign.(C'u).*max.(abs.(C'u).-λ,0)./max(h,floatmin(Float64))
        dv=sum(abs,v);dv==0 && return (zero(B),k,true,"")
        v./=dv;B=(u*v').*dv
        ok && norm(B-old)<=1e-8*max(1,norm(B)) && return (B,k,true,"")
        time()-began>CAP && return (B,k,false,"timeout")
    end
    B,1000,false,"iteration_limit"
end
function runfit(s,d,frac;mode="fit")
    if s=="cd_rank"
        sol=CD.reduced_rank(d.X,d.Y,d.r);return sol.B,0,true,""
    elseif s=="pa_rank"
        c=cache(d);began=time()
        stop=(it,state)->PA.default_stopping_criterion(1e-8,it,state)||time()-began>CAP
        B,k=PA.ForwardBackward(;maxit=20000,stop)(;x0=zeros(d.p,d.q),f=c.f,
                   g=PO.IndBallRank(d.r),gamma=1/c.g.L,adaptive=false)
        return B,k,time()-began<CAP,""
    elseif s in ("cd_sparse","block_fista")
        return sparse(d,frac;proximal=s=="block_fista")
    elseif s in ("cd_component","component_fista")
        return component(d,frac;proximal=s=="component_fista")
    elseif mode=="path"
        c=cache(d);B=zeros(d.p,d.q);its=0;ok=true;detail=""
        for level in exp.(range(log(.5),log(.02);length=8))
            B,k,flag,detail=nuclear(s,d,level;start=B,cached=c)
            its+=k;ok &= flag
        end
        return B,its,ok,detail
    end
    nuclear(s,d,frac)
end
family(s)=occursin("rank",s) ? "rank" : s in ("cd_sparse","block_fista") ? "sparse" :
          occursin("component",s) ? "component" : "nuclear"
function row!(io,d,out,s,frac,rd,phase;mode="fit")
    c=mode=="kernel" ? cache(d) : nothing
    FACTORS[]=nothing
    GC.gc(false)
    t=@elapsed B,it,ok,detail= c===nothing ? runfit(s,d,frac;mode) : nuclear(s,d,frac;cached=c)
    f=family(s);level=mode=="path" ? .02 : frac
    if s=="cd_sparse"
        FACTORS[]=sparse_factors(d,frac,B)
    end
    obj,gap,res=certificate(d,f,B,level;factors=FACTORS[])
    status=res<=1e-6 && (isnan(gap)||gap<=1e-6) ? "passed" : "accuracy_failed"
    detail=="timeout" && (status="timeout")
    values=(phase,"julia",d.case,d.rep,f,s,mode,level,rd,t,it,obj,gap,res,status,ok,Sys.maxrss()/1024,RSS_BASELINE[],detail)
    println(io,join(values,','));flush(io)
    rd==1 && mode=="fit" && open(fh->write(fh,B),joinpath(out,"$s-$(frac).bin"),"w")
    if rd==1 && mode=="fit" && f=="sparse" && FACTORS[]!==nothing
        A,V=FACTORS[]
        open(fh->write(fh,A),joinpath(out,"$s-$(frac).A"),"w")
        open(fh->write(fh,V),joinpath(out,"$s-$(frac).V"),"w")
    end
    println(s," ",d.case," rep",d.rep," ",level," ",round(t;digits=4),"s ",status," gap=",gap," residual=",res)
    flush(stdout)
end
function main()
    input,out=ARGS[1:2];phase=length(ARGS)>=3 ? ARGS[3] : "main";mkpath(out)
    d=problem(input)
    solvers=["cd_rank","cd_nuclear","ps_fista","ps_admm","pa_fista","pa_panoc",
             "cd_sparse","block_fista","cd_component","component_fista"]
    d.case in ("medium","wide") && push!(solvers,"pa_rank")
    d.case in ("medium","ill") && push!(solvers,"pa_pg")
    memory=length(ARGS)>=4 ? ARGS[4] : ""
    !isempty(memory) && (solvers=[memory])
    warm=problem(joinpath(dirname(input),"small-1"))
    warmwide=merge(warm,(;n=30,X=warm.X[1:30,:],Y=warm.Y[1:30,:],layer=warm.layer[1:30,:]))
    for s in solvers
        for w in (warm,warmwide)
            try
                runfit(s,w,.1)
                if family(s)=="nuclear"
                    c=cache(w);nuclear(s,w,.1;cached=c)
                    runfit(s,w,.1;mode="path")
                end
            catch err
                showerror(stdout,err,catch_backtrace());println()
            end
        end
    end
    RSS_BASELINE[]=Sys.maxrss()/1024
    println("Julia ",VERSION," BLAS ",BLAS.get_config()," threads=",BLAS.get_num_threads())
    io=open(joinpath(out,isempty(memory) ? "julia.csv" : "memory.csv"),"w")
    println(io,join(FIELDS,','))
    if !isempty(memory)
        row!(io,d,out,memory,.1,1,phase;mode="memory");close(io);return
    end
    if phase!="pilot"
        open(joinpath(out,"setup.csv"),"w") do f
            println(f,"round,setup_seconds,first_loss_prox_seconds,cached_loss_prox_seconds")
            for rd in 1:3
                pre=@elapsed c=cache(d)
                x=zeros(d.p,d.q);y=similar(x)
                first=@elapsed PB.prox!(y,c.ls,x,1/c.g.L)
                cached=@elapsed PB.prox!(y,c.ls,x,1/c.g.L)
                println(f,join((rd,pre,first,cached),','))
            end
        end
    end
    for rd in 1:(phase=="pilot" ? 1 : 3)
        for s in circshift(solvers,mod(d.rep+rd,length(solvers)))
            for frac in (family(s)=="rank" ? (0.0,) : (.5,.1,.02))
                try
                    row!(io,d,out,s,frac,rd,phase)
                catch err
                    detail=replace(sprint(showerror,err),','=>';','\n'=>' ')
                    println(io,join((phase,"julia",d.case,d.rep,family(s),s,"fit",frac,rd,
                       NaN,-1,NaN,NaN,NaN,"error",false,Sys.maxrss()/1024,0,detail),','))
                    flush(io);println(stderr,s,": ",detail)
                end
            end
        end
    end
    if phase!="pilot"
        for s in filter(s->family(s)=="nuclear" && s!="pa_pg",solvers), mode in ("kernel","path"),rd in 1:3
            try row!(io,d,out,s,.1,rd,phase;mode) catch err;println(stderr,s," ",mode,": ",sprint(showerror,err));end
        end
        if d.case in ("medium","responses","wide")
            for s in solvers
                s in ("pa_rank","pa_pg") && continue
                Profile.clear();began=time();calls=0
                try
                    @profile while time()-began<.25 || calls==0
                        runfit(s,d,.1);calls+=1
                    end
                    open(joinpath(out,"profile-$s.txt"),"w") do f
                        println(f,"case=",d.case," rep=",d.rep," calls=",calls," samples=",length(Profile.fetch()))
                        Profile.print(f;format=:flat,sortedby=:count,mincount=2)
                    end
                catch err;println(stderr,"profile ",s,": ",sprint(showerror,err));end
            end
            for s in ("pa_fista","pa_panoc")
                tr=Tuple[];nuclear(s,d,.1;trace=tr)
                open(joinpath(out,"trace-$s.csv"),"w") do f
                    println(f,"iteration,seconds,objective,gap,residual")
                    foreach(x->println(f,join(x,',')),tr)
                end
            end
        end
    end
    close(io)
end
get(ENV,"CD81_LIBRARY","0")=="1" || main()
