# Julia solver counterpart of native RCAL CV, with independently evaluated KKT/objective.
using HighDimensionalModels,ProximalBase,LinearAlgebra,Random,Statistics,DelimitedFiles
BLAS.set_num_threads(1)
include("calibration-loss.jl")
caseid,rs=ARGS;rep=parse(Int,rs);root=joinpath("outputs","calibration-cv-engine",caseid,rs)
isfile(joinpath(root,"j.csv")) || error("Reference did not reach calibration stage")
readv(nm)=vec(readdlm(joinpath(root,nm*".csv"),',',Float64))
j=Int(only(readv("j")));weights=readv("weight");offset=readv("offset");levels=readv("levels");foldid=Int.(readv("foldid"))
stem=joinpath("outputs","development",caseid,rs);liu=startswith(caseid,"liu-")
X=readdlm(joinpath(stem,liu ? "cal-X.csv" : "X.csv"),',',Float64);y=vec(readdlm(joinpath(stem,liu ? "cal-y.csv" : "y.csv"),',',Float64));N=X[:,setdiff(1:size(X,2),j)];M=hcat(ones(length(y)),N)
opt=CDOptions(;opt_tol=1e-10,gaptol=0,max_iter=100000)
W=weights./mean(weights);n,p=size(N);K=maximum(foldid);errors=fill(NaN,length(levels),K)
t=time_ns()
try
 for k=1:K
  train=findall(!=(k),foldid);test=findall(==(k),foldid)
  f=GLMLassoPath(N[train,:],y[train],levels,CalibrationLoss(),opt;weights=W[train],offset=offset[train],intercept=true,standardize_x=false,newton_tol=1e-10,newton_max_iter=500,devmax=1.,rng=Xoshiro(1920+rep+k))
  for a=1:length(f.λpath)
   f.converged[a] || continue
   b=Vector(f.βpath[a]);eta=offset[test].+f.β0path[a].+N[test,:]*b
   errors[a,k]=mean(W[test].*map((e,yy)->CalibrationLoss()(e,yy),eta,y[test]))
  end
 end
 writedlm(joinpath(root,"julia-errors.csv"),errors,',')
 finite=findall(vec(all(isfinite,errors;dims=2)));isempty(finite) && error("No fully converged CV level")
 means=vec(mean(errors;dims=2));best=finite[argmin(means[finite])];mc=std(errors[best,:])/sqrt(K)
 eligible=filter(a->means[a]<=means[best]+mc,finite);oneSE=eligible[argmax(levels[eligible])]
 open(joinpath(root,"julia-cv-checks.csv"),"w") do io
  println(io,"case,rep,policy,index,reference_index,lambda,converged,kkt,reference_kkt,objective_error,coefficient_error,predictor_error,cv_max_loss_error,seconds,passes")
  for (pol,a) in [("best",best),("oneSE",oneSE)]
   f=glm_lasso(N,y,levels[a],CalibrationLoss(),opt;weights=W,offset,intercept=true,standardize_x=false,newton_tol=1e-10,newton_max_iter=500,rng=Xoshiro(1930+rep))
   b=vcat(f.intercept,Vector(f.x));eta=offset+M*b;grad=vec(mean(M.*(W.*((1 .-y).-y.*exp.(-eta)));dims=1));nz=abs.(b[2:end]).>1e-9
   kkt=maximum(vcat(abs(grad[1]),abs.(grad[2:end][nz].+levels[a].*sign.(b[2:end][nz])),max.(abs.(grad[2:end][.!nz]).-levels[a],0.)))
   obj=mean(W.*map((e,yy)->CalibrationLoss()(e,yy),eta,y))+levels[a]*sum(abs,b[2:end])
   meta=joinpath(root,"rcal-"*pol*".meta.csv");ri=rk=oe=ce=pe=cve=NaN
   if isfile(meta)
    mm=readdlm(meta,',',Any;header=true)[1];ri=mm[1,1];rk=mm[1,3];bref=readv("rcal-"*pol)
    oe=abs(obj-Float64(mm[1,4]));ce=maximum(abs,b-bref);pe=maximum(abs,M*(b-bref))
    er=readdlm(joinpath(root,"rcal-errors.csv"),',',Float64);common=isfinite.(er).&isfinite.(errors);cve=maximum(abs,er[common]-errors[common])
   end
   println(io,join((caseid,rep,pol,a,ri,levels[a],f.converged,kkt,rk,oe,ce,pe,cve,(time_ns()-t)/1e9,f.iterations),','))
   writedlm(joinpath(root,"julia-"*pol*".csv"),b,',')
  end
 end
catch e
 write(joinpath(root,"julia.error"),sprint(showerror,e,catch_backtrace()))
end
