using HighDimensionalModels, ProximalBase, Random, LinearAlgebra, Statistics
BLAS.set_num_threads(1)
case,shard,maxrep,stage=ARGS
shard=parse(Int,shard);maxrep=parse(Int,maxrep)
family=startswith(case,"gaussian") ? :gaussian : :binomial
kind=last(split(case,'-'))
n,p,rho,s=kind=="correlated" ? (100,500,.9,15) : kind=="benign" ? (500,100,.3,5) : (200,500,.3,5)
opt=CDOptions(;opt_tol=1e-8,gaptol=0)
profiles=[("base",60,.001),("dense",120,.001),("deep",120,1e-5),("deeper",180,1e-7)]
outdir=joinpath("outputs",stage,case);mkpath(outdir)
function data(rep)
 offset=kind=="correlated" ? 10000 : kind=="benign" ? 20000 : kind=="random" ? 30000 : 0
 rg=Xoshiro((stage=="canary" ? 91800000 : 91700000)+offset+rep)
 X=randn(rg,n,p)
 for j=2:p;X[:,j] .= rho.*X[:,j-1] .+ sqrt(1-rho^2).*X[:,j];end
 X .-= mean(X;dims=1);X ./=sqrt.(mean(abs2,X;dims=1))
 ids=kind=="random" ? randperm(rg,p)[1:s] : collect(1:s)
 b=zeros(p);b[ids] .= (family==:gaussian ? .5 : .7).*(-1.).^(0:s-1)
 eta=.2 .+ X*b
 mu=family==:gaussian ? eta : 1 ./ (1 .+exp.(-eta))
 y=family==:gaussian ? mu+randn(rg,n) : Float64.(rand(rg,n).<mu)
 null=[j for j in 1:p if b[j]==0][1:5]
 X,y,b,mu,vcat(ids[1:5],null)
end
function runrep(rep)
 dest=joinpath(outdir,string(rep));isfile(dest*".complete") && return
 X,y,truth,mutrue,targets=data(rep)
 htrue=family==:gaussian ? ones(n) : mutrue.*(1 .-mutrue)
 t=time_ns()
 base=debiased_lasso(X,y;family,targets,options=opt,threads=false,nodewise_lambda=:ldpe,ldpe_unpenalized=4,ldpe_kappa=.5,rng=Xoshiro(950000+rep))
 initial_seconds=(time_ns()-t)/1e9
 bh=base.initial;b0=base.intercept
 etah=X*bh .+b0
 muh=family==:gaussian ? etah : 1 ./ (1 .+exp.(-etah))
 hh=family==:gaussian ? ones(n) : max.(muh.*(1 .-muh),eps(Float64))
 open(dest*".csv","w") do io
 println(io,"rep,profile,weights,initial,target,truth,estimate,se,true_variance_se,robust_se,pvalue,covered,rejected,stochastic,linear_bias,nonlinear_remainder,identity_error,lambda,upper_lambda,lower_lambda,endpoint,bias_factor,noise_ratio,bias_target_met,converged,projection_seconds,initial_seconds,score_orthogonality,initial_l1,initial_intercept_error")
 for (wlabel,h) in (family==:gaussian ? [("estimated",hh)] : [("estimated",hh),("oracle",htrue)])
  center=vec(X'*h)/sum(h);Ac=X .-center';W=sqrt.(h).*Ac
  for (profile,nlevels,floor) in vcat(profiles,n>p ? [("exact",0,0.)] : Tuple{String,Int,Float64}[])
   tp=time_ns()
   P=profile=="exact" ? nothing : (profile=="base" && wlabel=="estimated" ? base.projection : debiasing_matrix(W;lambda=:ldpe,intercept=false,targets,threads=false,options=opt,ldpe_unpenalized=4,ldpe_kappa=.5,ldpe_nlambda=nlevels,ldpe_minratio=floor,rng=Xoshiro(960000+rep)))
   S=profile=="exact" ? Ac*((W'*W)\Matrix{Float64}(I,p,p))[:,targets]*n : Ac*Matrix(P.theta)'
   pt=(time_ns()-tp)/1e9
   all(isfinite,S) || error("nonfinite score")
   scale=vec(sqrt.(mean(abs2,W;dims=1)));Z=W./scale'
   for (ilabel,b,intercept,mu) in [("estimated",bh,b0,muh),("oracle",truth,.2,mutrue)]
    delta=X*(b-truth).+(intercept-.2)
    resid=y-mu
    stochastic=vec(S'*(y-mutrue))/n
    linear=(b-truth)[targets]-vec(S'*(h.*delta))/n
    nonlinear=vec(S'*(mutrue-mu+h.*delta))/n
    est=b[targets]+vec(S'*resid)/n
    se=vec(sqrt.(sum(abs2,S.*sqrt.(h);dims=1)))/n
    tvse=vec(sqrt.(sum(abs2,S.*sqrt.(htrue);dims=1)))/n
    infl=S.*resid;infl .-=mean(infl;dims=1)
    rse=vec(sqrt.(sum(abs2,infl;dims=1)))/n
    err=est-truth[targets]-stochastic-linear-nonlinear
    maximum(abs,err)<1e-8*(1+maximum(abs,est)) || error("decomposition identity failed")
    for a in eachindex(targets)
     j=targets[a];pv=2*HighDimensionalModels._pnorm(-abs(est[a]/se[a]))
     upper=NaN;lower=NaN;endpoint="none"
     if P!==nothing
      # Reconstruct the exact conditional grid endpoints after restricted projection.
      cross=Z'*Z[:,j]/n;others=sort([k for k in 1:p if k!=j];by=k->(-abs(cross[k]),k));ids=others[1:4]
      z=Z[:,j]-Z[:,ids]*(Z[:,ids]\Z[:,j]);conditional=Z'*z/n
      upper=maximum(abs(conditional[k]) for k in 1:p if k!=j && !(k in ids));lower=upper*floor
      endpoint=isapprox(P.lambda[a],upper;rtol=1e-7) ? "upper" : isapprox(P.lambda[a],lower;rtol=1e-7) ? "lower" : "interior"
     end
     orth=abs(dot(S[:,a],h)) / max(norm(S[:,a])*norm(h),eps())
     values=(rep,profile,wlabel,ilabel,j,truth[j],est[a],se[a],tvse[a],rse[a],pv,abs(est[a]-truth[j])<=1.959963984540054*se[a],pv<.05,stochastic[a],linear[a],nonlinear[a],err[a],P===nothing ? 0. : P.lambda[a],upper,lower,endpoint,P===nothing ? 0. : P.bias_factor[a],P===nothing ? 1. : P.noise_factor[a]/P.reference_noise[a],P===nothing ? true : P.bias_target_met[a],base.converged && (P===nothing || all(P.converged)),pt,initial_seconds,orth,norm(b-truth,1),intercept-.2)
     println(io,join(values,','))
    end
   end
  end
 end
 end
 write(dest*".complete","ok\n")
end
for rep=shard:8:maxrep
 try
  runrep(rep)
 catch e
  open(joinpath(outdir,string(rep)*".error"),"w") do io;println(io,sprint(showerror,e,catch_backtrace()));end
 end
end
println("STUDY_DONE ",case," shard=",shard)
