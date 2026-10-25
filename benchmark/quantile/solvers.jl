# Experimental issue #78 solvers. This is a benchmark API, not a package API.
using JuMP, HiGHS, LinearAlgebra, SparseArrays, Statistics, Random
import CoordinateDescent as CD
import ProximalBase as PB
const MOI = JuMP.MOI
BLAS.set_num_threads(1)
checkloss(r, tau) = max(tau*r, (tau-1)*r)
soft(x, t) = sign(x)*max(abs(x)-t, 0)
penalty(b, w) = sum(w .* abs.(b))
objective(X,y,b,a,lambda,tau,w) = mean(checkloss.(y .- a .- X*b,tau)) + lambda*penalty(b,w)

# A feasible dual lower bound, including the unpenalized intercept. Projection
# onto the box intersected with sum(u)=0, followed by a radial contraction.
function certificate(X,y,b,a,lambda,tau,w,u)
    lo, hi = minimum(u)-tau, maximum(u)-(tau-1)
    for _ in 1:65
        c=(lo+hi)/2
        sum(clamp.(u .- c,tau-1,tau)) > 0 ? (lo=c) : (hi=c)
    end
    v=clamp.(u .- (lo+hi)/2,tau-1,tau)
    z=X'*v / length(y)
    ratio=maximum(abs.(z) ./ (lambda .* w); init=0.0)
    v ./= max(1.0, ratio*(1+1e-12))
    primal=objective(X,y,b,a,lambda,tau,w)
    dual=dot(y,v)/length(y)
    (; primal, dual, gap=primal-dual, sumdual=abs(sum(v))/length(y))
end

function lpmodel(X,y,lambda,tau,w,form,strategy; limit=30.0)
    n,p=size(X)
    m=direct_model(HiGHS.Optimizer())
    set_silent(m)
    set_optimizer_attribute(m,"threads",1)
    set_optimizer_attribute(m,"parallel","off")
    set_optimizer_attribute(m,"time_limit",limit)
    set_optimizer_attribute(m,"primal_feasibility_tolerance",1e-7)
    set_optimizer_attribute(m,"dual_feasibility_tolerance",1e-7)
    if strategy==:ipm
        set_optimizer_attribute(m,"solver","ipm")
    else
        set_optimizer_attribute(m,"solver","simplex")
        set_optimizer_attribute(m,"simplex_strategy",strategy==:primal ? 4 : 1)
    end
    if form==:primal
        @variable(m,b[1:p])
        @variable(m,a)
        @variable(m,t[1:p]>=0)
        @variable(m,rp[1:n]>=0)
        @variable(m,rn[1:n]>=0)
        @constraint(m,eq,X*b .+ a .+ rp .- rn .== y)
        @constraint(m,b .<= t)
        @constraint(m,-b .<= t)
        @objective(m,Min,(tau*sum(rp)+(1-tau)*sum(rn))/n+lambda*sum(w.*t))
        return (;m,b,a,t,eq,form)
    else
        @variable(m,tau-1 <= u[1:n] <= tau)
        @constraint(m,eq,sum(u)==0)
        @constraint(m,upper,X'*u .<= n*lambda.*w)
        @constraint(m,lower,-X'*u .<= n*lambda.*w)
        @objective(m,Max,dot(y,u)/n)
        return (;m,u,eq,upper,lower,form)
    end
end
function lpupdate!(s,lambda,w,n)
    if s.form==:primal
        set_objective_coefficient(s.m,s.t,lambda.*w)
    else
        set_normalized_rhs.(s.upper,n*lambda.*w)
        set_normalized_rhs.(s.lower,n*lambda.*w)
    end
end
function lpget(s,n)
    if s.form==:primal
        return value.(s.b),value(s.a),n.*dual.(s.eq)
    end
    # JuMP maximization duals have the opposite economic sign.
    b=n.*(dual.(s.lower).-dual.(s.upper))
    a=-n*dual(s.eq)
    return b,a,value.(s.u)
end

function lppath(X,y,lambdas,tau,w,form,strategy,warm; limit=30.0)
    n,p=size(X); start=time(); setup=0.0; optimize=0.0; extract=0.0; iters=0
    answers=Any[]; s=nothing; status="complete"
    for lambda in lambdas
        time()-start>=limit && (status="budget"; break)
        t=time()
        if s===nothing || !warm
            s=lpmodel(X,y,lambda,tau,w,form,strategy; limit=max(.01,limit-(time()-start)))
        else
            lpupdate!(s,lambda,w,n)
        end
        setup+=time()-t
        set_optimizer_attribute(s.m,"time_limit",max(.01,limit-(time()-start)))
        t=time(); optimize!(s.m); optimize+=time()-t
        ts=termination_status(s.m)
        if ts != MOI.OPTIMAL
            status=string(ts); break
        end
        iters+=MOI.get(backend(s.m), MOI.SimplexIterations())
        t=time(); b,a,u=lpget(s,n); extract+=time()-t
        push!(answers,(;b,a,u,lambda))
    end
    (;answers,status,setup,optimize,extract,iters,total=time()-start)
end

function smoothfg(X,y,b,a,loss)
    eta=X*b .+ a
    vals=PB.value_and_deriv.(Ref(loss),eta,y)
    d=last.(vals)
    mean(first.(vals)),X'*d/length(y),mean(d),-d
end
function smoothkkt(X,y,b,a,lambda,w,loss)
    _,g,g0,u=smoothfg(X,y,b,a,loss)
    r=maximum(abs.(b .- soft.(b .- g,lambda.*w)); init=0.0)
    max(r,abs(g0)),u
end
function apgsolve(X,y,lambda,tau,w,loss,b,a; limit=30.0,tol=1e-6,maxiter=20000)
    start=time(); z=copy(b); z0=a; momentum=1.0; L=1.0; iterations=0
    for k in 1:maxiter
        iterations=k
        f,g,g0,_=smoothfg(X,y,z,z0,loss)
        bn=similar(b); an=a
        for _ in 1:60
            bn .= soft.(z .- g./L,lambda.*w./L); an=z0-g0/L
            fn,_,_,_=smoothfg(X,y,bn,an,loss)
            db=bn-z; da=an-z0
            fn <= f+dot(g,db)+g0*da+L*(sum(abs2,db)+da^2)/2+1e-13 && break
            L*=2
        end
        tm=(1+sqrt(1+4momentum^2))/2
        # Adaptive restart, with the intercept included in the same geometry.
        if dot(z-bn,bn-b)+(z0-an)*(an-a)>0
            tm=1.0; momentum=1.0
        end
        z .= bn .+ ((momentum-1)/tm).*(bn-b); z0=an+(momentum-1)/tm*(an-a)
        b=bn; a=an; momentum=tm; L=max(L*.9,1e-12)
        if k%5==0
            residual,_=smoothkkt(X,y,b,a,lambda,w,loss)
            residual<=tol && return b,a,k,true
            time()-start>limit && return b,a,k,false
        end
    end
    b,a,iterations,false
end
function smoothpath(X,y,lambdas,tau,w,method; h=.2,limit=30.0,warm=true)
    start=time(); answers=Any[]; iters=0; b=zeros(size(X,2)); a=quantile(y,tau)
    loss=PB.ConvolutionCheck(tau,h)
    if method==:newton && warm
        # Use the production proximal Newton path, including model setup.
        path=CD.GLMLassoPath(X,y,lambdas,loss,CD.CDOptions(opt_tol=1e-8,max_iter=10000);
            intercept=true,standardize_x=false,screening=true,devmax=1,
            newton_tol=1e-10,newton_max_iter=100,rng=Xoshiro(719))
        for k in eachindex(path.λpath)
            b=Vector(path.βpath[k]); a=path.β0path[k]
            _,u=smoothkkt(X,y,b,a,path.λpath[k],w,loss)
            push!(answers,(;b,a,u,lambda=path.λpath[k]))
        end
        iters=sum(path.iterations)
        status=all(path.converged) ? "complete" : "not_converged"
    else
        status="complete"
        for lambda in lambdas
            time()-start>limit && (status="budget";break)
            !warm && (b=zeros(size(X,2));a=quantile(y,tau))
            b,a,k,ok=apgsolve(X,y,lambda,tau,w,loss,b,a;limit=max(.01,limit-(time()-start)))
            iters+=k
            _,u=smoothkkt(X,y,b,a,lambda,w,loss)
            push!(answers,(;b=copy(b),a,u,lambda))
            !ok && (status="not_converged";break)
        end
    end
    total=time()-start
    (;answers,status,setup=0.0,optimize=total,extract=0.0,iters,total)
end

# Two-block linearized ADMM for X*b+a+r=y. The beta block uses the
# majorizing proximal step, so it avoids a p-by-p factorization. rho is fixed;
# a conservative spectral estimate uses a power iteration plus a safety margin.
function ladmmpath(X,y,lambdas,tau,w;limit=30.0,tol=1e-5,rho=1.0)
    start=time(); n,p=size(X); b=zeros(p); a=quantile(y,tau); r=y.-a; v=zeros(n)
    q=ones(p+1); q/=norm(q); L=0.0
    for _ in 1:40
        z=X*view(q,1:p) .+ q[end]
        q=vcat(X'*z,sum(z)); L=norm(q); q/=L
    end
    L*=1.1
    answers=Any[]; iters=0; status="complete"
    for lambda in lambdas
        ok=false
        for k in 1:30000
            iters+=1
            z=X*b .+ a .+ r .- y .+ v
            bn=soft.(b .- X'*z./L,n*lambda.*w./(rho*L)); an=a-sum(z)/L
            eta=X*bn .+ an; oldr=copy(r)
            z=y.-eta.-v
            r .= z .- clamp.(z,(tau-1)/rho,tau/rho)
            primal=eta.+r.-y
            v .+= primal
            b=bn; a=an
            if k%20==0
                c=certificate(X,y,b,a,lambda,tau,w,-rho.*v)
                if c.gap<=tol*(1+abs(c.primal)) && norm(primal,Inf)<=1e-4
                    ok=true;break
                end
                time()-start>limit && break
            end
        end
        push!(answers,(;b=copy(b),a,u=-rho.*copy(v),lambda))
        !ok && (status="not_converged";break)
    end
    total=time()-start
    (;answers,status,setup=0.0,optimize=total,extract=0.0,iters,total)
end
