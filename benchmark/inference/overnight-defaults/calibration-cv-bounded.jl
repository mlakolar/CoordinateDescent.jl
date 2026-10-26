# Julia solver counterpart of native RCAL CV, with independently evaluated KKT/objective.
using HighDimensionalModels,ProximalBase,LinearAlgebra,Random,Statistics,DelimitedFiles
BLAS.set_num_threads(1)
include("calibration-loss.jl")
caseid,rs=ARGS;rep=parse(Int,rs);root=joinpath("outputs","calibration-cv-engine",caseid,rs);outroot=joinpath("outputs","calibration-cv-bounded",caseid,rs);mkpath(outroot)
isfile(joinpath(root,"j.csv")) || error("Reference did not reach calibration stage")
readv(nm)=vec(readdlm(joinpath(root,nm*".csv"),',',Float64))
j=Int(only(readv("j")));weights=readv("weight");offset=readv("offset");levels=readv("levels");foldid=Int.(readv("foldid"))
stem=joinpath("outputs","development",caseid,rs);liu=startswith(caseid,"liu-")
X=readdlm(joinpath(stem,liu ? "cal-X.csv" : "X.csv"),',',Float64);y=vec(readdlm(joinpath(stem,liu ? "cal-y.csv" : "y.csv"),',',Float64));N=X[:,setdiff(1:size(X,2),j)];M=hcat(ones(length(y)),N)
opt=CDOptions(;opt_tol=1e-10,gaptol=0,max_iter=2500)
W=weights./mean(weights);n,p=size(N);K=maximum(foldid);errors=fill(NaN,length(levels),K)
mask=readdlm(joinpath("outputs","calibration-recession",caseid,rs,"mask.csv"),',',Any;header=true)[1]
unsafe(a,k)=any(r->r[1]=="actual_cv" && Int(r[2])==k && Int(r[3])==a && lowercase(string(r[4]))=="true",eachrow(mask))
# Explicit training KKT checks reject small-change false convergence.
kkt_error(M,yy,ww,oo,b,lam)=begin
 eta=oo+M*b;wn=ww./mean(ww);g=vec(mean(M.*(wn.*((1 .-yy).-yy.*exp.(-eta)));dims=1));nz=abs.(b[2:end]).>1e-9
 maximum(vcat(abs(g[1]),abs.(g[2:end][nz].+lam.*sign.(b[2:end][nz])),max.(abs.(g[2:end][.!nz]).-lam,0.)))
end
fold_checks=Vector{Any}()
t=time_ns()
try
 if isfile(joinpath(outroot,"julia-errors.csv"))
  global errors=readdlm(joinpath(outroot,"julia-errors.csv"),',',Float64)
 else
 for k=1:K
  train=findall(!=(k),foldid);test=findall(==(k),foldid)
  allowed=filter(a->!unsafe(a,k),eachindex(levels));isempty(allowed) && continue
  f=GLMLassoPath(N[train,:],y[train],levels[allowed],CalibrationLoss(),opt;weights=W[train],offset=offset[train],intercept=true,standardize_x=false,newton_tol=1e-10,newton_max_iter=100,devmax=1.,rng=Xoshiro(1920+rep+k))
  for aidx=1:length(f.λpath)
   a=allowed[aidx];b=vcat(f.β0path[aidx],Vector(f.βpath[aidx]))
   ke=kkt_error(M[train,:],y[train],W[train],offset[train],b,levels[a])
   push!(fold_checks,(k,a,f.converged[aidx],ke))
   f.converged[aidx] && isfinite(ke) && ke<=2e-5 || continue
   eta=offset[test]+M[test,:]*b
   errors[a,k]=mean(W[test].*map((e,yy)->CalibrationLoss()(e,yy),eta,y[test]))
  end
 end
 open(joinpath(outroot,"julia-fold-checks.csv"),"w") do io
  println(io,"fold,level,converged,kkt")
  for r in fold_checks;println(io,join(r,','));end
 end
 writedlm(joinpath(outroot,"julia-errors.csv"),errors,',')
 end
 finite=filter(a->!unsafe(a,0),findall(vec(all(isfinite,errors;dims=2))));isempty(finite) && error("No fully converged CV level")
 means=vec(mean(errors;dims=2));best=finite[argmin(means[finite])];mc=std(errors[best,:])/sqrt(K)
 eligible=filter(a->means[a]<=means[best]+mc,finite);oneSE=eligible[argmax(levels[eligible])]
 open(joinpath(outroot,"julia-cv-checks.csv"),"w") do io
  println(io,"case,rep,policy,index,reference_index,lambda,converged,kkt,reference_kkt,objective_error,coefficient_error,predictor_error,cv_max_loss_error,seconds,passes")
  for (pol,a) in [("best",best),("oneSE",oneSE)]
   f=glm_lasso(N,y,levels[a],CalibrationLoss(),opt;weights=W,offset,intercept=true,standardize_x=false,newton_tol=1e-10,newton_max_iter=100,rng=Xoshiro(1930+rep))
   b=vcat(f.intercept,Vector(f.x));eta=offset+M*b;grad=vec(mean(M.*(W.*((1 .-y).-y.*exp.(-eta)));dims=1));nz=abs.(b[2:end]).>1e-9
   kkt=maximum(vcat(abs(grad[1]),abs.(grad[2:end][nz].+levels[a].*sign.(b[2:end][nz])),max.(abs.(grad[2:end][.!nz]).-levels[a],0.)))
   obj=mean(W.*map((e,yy)->CalibrationLoss()(e,yy),eta,y))+levels[a]*sum(abs,b[2:end])
   # Save the full numerical solution before optional reference parsing.
   writedlm(joinpath(outroot,"julia-"*pol*".csv"),b,',')
   open(joinpath(outroot,"julia-"*pol*".meta.csv"),"w") do m
    println(m,"index,lambda,converged,kkt,objective,seconds,passes")
    println(m,join((a,levels[a],f.converged,kkt,obj,(time_ns()-t)/1e9,f.iterations),','))
   end
   meta=joinpath(outroot,"rcal-"*pol*".meta.csv");ri=rk=oe=ce=pe=cve=NaN
   if isfile(meta)
    mm=readdlm(meta,',',Any;header=true)[1];ri=mm[1,1];rk=mm[1,3];bref=vec(readdlm(joinpath(outroot,"rcal-"*pol*".csv"),',',Float64))
    oe=abs(obj-Float64(mm[1,4]));ce=maximum(abs,b-bref);pe=maximum(abs,M*(b-bref))
    raw=readdlm(joinpath(outroot,"rcal-errors.csv"),',',Any);er=map(x->x=="NA" ? NaN : Float64(x),raw);common=isfinite.(er).&isfinite.(errors);cve=maximum(abs,er[common]-errors[common])
   end
   println(io,join((caseid,rep,pol,a,ri,levels[a],f.converged,kkt,rk,oe,ce,pe,cve,(time_ns()-t)/1e9,f.iterations),','))
   writedlm(joinpath(outroot,"julia-"*pol*".csv"),b,',')
  end
 end
catch e
 write(joinpath(outroot,"julia.error"),sprint(showerror,e,catch_backtrace()))
end
