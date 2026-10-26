using HighDimensionalModels,ProximalBase,LinearAlgebra,Random,Statistics,DelimitedFiles
BLAS.set_num_threads(1)
include("calibration-loss.jl")
caseid,rs=ARGS;rep=parse(Int,rs);stem=joinpath("outputs","development",caseid,rs)
liu=startswith(caseid,"liu-")
X=readdlm(joinpath(stem,liu ? "cal-X.csv" : "X.csv"),',',Float64);y=vec(readdlm(joinpath(stem,liu ? "cal-y.csv" : "y.csv"),',',Float64))
n,p=size(X);N=X[:,2:end];M=hcat(ones(n),N)
root=joinpath("outputs","calibration-engine",caseid,rs)
W=vec(readdlm(joinpath(root,"weights.csv"),',',Float64));offs=vec(readdlm(joinpath(root,"offset.csv"),',',Float64))
opt=CDOptions(;opt_tol=1e-10,gaptol=0,max_iter=100000)
# Scalar derivative and curvature contract checked independently on several arguments.
loss=CalibrationLoss()
for eta in [-3.,-.5,0.,2.,5.], response in [0.,1.]
 h=1e-5;d=(loss(eta+h,response)-loss(eta-h,response))/(2h)
 d2=(ProximalBase.deriv(loss,eta+h,response)-ProximalBase.deriv(loss,eta-h,response))/(2h)
 @assert isapprox(d,ProximalBase.deriv(loss,eta,response);atol=1e-8,rtol=1e-8)
 @assert isapprox(d2,ProximalBase.deriv2(loss,eta,response);atol=1e-8,rtol=1e-8)
end
open(joinpath(root,"julia-checks.csv"),"w") do io
 println(io,"case,rep,multiplier,converged,kkt,reference_kkt,objective,reference_objective,max_coefficient_error,max_predictor_error,seconds,passes,parity_pass")
 for (mult,tag) in [(2.,"2"),(.7,"0.7")]
  isfile(joinpath(root,"rcal-"*tag*".error")) && continue
  try
   lambda=mult*sqrt(log(p)/n)
   meta=readdlm(joinpath(root,"rcal-"*tag*".meta.csv"),',',Any;header=true)[1]
   bref=vec(readdlm(joinpath(root,"rcal-"*tag*".csv"),',',Float64))
   t=time_ns()
   f=glm_lasso(N,y,lambda,CalibrationLoss(),opt;intercept=true,standardize_x=false,weights=W,offset=offs,newton_tol=1e-10,newton_max_iter=500,rng=Xoshiro(1020+rep))
   elapsed=(time_ns()-t)/1e9;b=vcat(f.intercept,Vector(f.x));eta=offs+M*b
   grad=vec(mean(M.*(W.*((1 .-y).-y.*exp.(-eta)));dims=1));nz=abs.(b[2:end]).>1e-9
   kkt=maximum(vcat(abs(grad[1]),abs.(grad[2:end][nz].+lambda.*sign.(b[2:end][nz])),max.(abs.(grad[2:end][.!nz]).-lambda,0.)))
   obj=mean(W.*(y.*exp.(-eta).+(1 .-y).*eta))+lambda*sum(abs,b[2:end])
   ce=maximum(abs,b-bref);pe=maximum(abs,M*(b-bref));oe=abs(obj-Float64(meta[1,4]))
   # Different algorithms: check original objective/KKT, predictor and coefficients.
   pass=f.converged && kkt<=2e-5 && oe<=1e-7 && ce<=2e-3 && pe<=2e-3
   println(io,join((caseid,rep,mult,f.converged,kkt,meta[1,3],obj,meta[1,4],ce,pe,elapsed,f.iterations,pass),','))
   writedlm(joinpath(root,"julia-"*tag*".csv"),b,',')
  catch e
   write(joinpath(root,"julia-"*tag*".error"),sprint(showerror,e,catch_backtrace()))
  end
 end
end
