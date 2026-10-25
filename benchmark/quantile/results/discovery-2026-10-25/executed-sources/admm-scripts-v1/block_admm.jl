# Factorized two-block ADMM: jointly prox the residual and split coefficients.
# Included after solvers.jl; benchmark-only, with a bounded dense factor size.
function blockadmmpath(X,y,lambdas,tau,w;limit=30.0,tol=1e-5,rho=1.0)
    start=time(); n,p=size(X)
    min(n,p+1)<=2000 || error("factor dimension above the study's2000 limit")
    scale=sqrt(n); B=hcat(X,ones(n))/scale; yy=y/scale; d=p+1; z=zeros(d);z[end]=quantile(y,tau)
    theta=copy(z); r=yy.-B*z; v=zeros(n); u=zeros(d)
    setupstart=time()
    small=d<=n
    gram=small ? B'*B : B*B'
    for j in axes(gram,1);gram[j,j]+=1.;end
    fact=cholesky!(Symmetric(gram))
    setup=time()-setupstart; answers=Any[]; iters=0; status="complete"
    weights=vcat(w,0.)
    for lambda in lambdas
        ok=false
        for k in 1:30000
            iters+=1
            rhs=B'*(yy.-r.-v) .+ z .- u
            theta=small ? fact\rhs : rhs .- B'*(fact\(B*rhs))
            eta=B*theta
            q=yy.-eta.-v
            r=q .- clamp.(q,(tau-1)/(rho*scale),tau/(rho*scale))
            z=soft.(theta.+u,lambda.*weights./rho)
            pr=eta.+r.-yy; pz=theta.-z
            v .+= pr;u .+= pz
            if k%20==0
                c=certificate(X,y,z[1:p],z[end],lambda,tau,w,-rho*scale.*v)
                if c.gap<=tol*(1+abs(c.primal)) && max(scale*norm(pr,Inf),norm(pz,Inf))<=1e-4
                    ok=true;break
                end
                time()-start>limit && break
            end
        end
        push!(answers,(;b=copy(z[1:p]),a=z[end],u=-rho*scale.*copy(v),lambda))
        !ok && (status="not_converged";break)
    end
    total=time()-start
    (;answers,status,setup,optimize=total-setup,extract=0.,iters,total)
end
