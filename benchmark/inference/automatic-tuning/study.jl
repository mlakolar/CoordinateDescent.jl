# All simulation and inference runs under Slurm on Discovery.
using CoordinateDescent, Random, LinearAlgebra, Statistics, DelimitedFiles
BLAS.set_num_threads(1)
stage, case, shard = ARGS
shard=parse(Int,shard)
label=startswith(case,"gaussian") ? :gaussian : :binomial
stress=occursin("correlated",case); benign=occursin("benign",case)
n,p,rho,s = stress ? (100,500,.9,15) : benign ? (400,200,.3,5) : (200,500,.3,5)
seedbase = stage=="development" ? 91200000 : 91300000
out=joinpath("study",stage*"-"*case*"-"*string(shard)*".csv")
mkpath("study")
function data(rep)
  rg=Xoshiro(seedbase + (stress ? 10000 : benign ? 20000 : 0) + rep)
  X=randn(rg,n,p)
  for j=2:p
    X[:,j] .= rho .* X[:,j-1] .+ sqrt(1-rho^2).*X[:,j]
  end
  X .-= mean(X;dims=1); X ./= sqrt.(mean(abs2,X;dims=1))
  beta=zeros(p); beta[1:s] .= (label==:gaussian ? .5 : .7).*(-1.).^(0:s-1)
  eta=.2 .+ X*beta
  y=label==:gaussian ? eta+randn(rg,n) : Float64.(rand(rg,n).<1 ./ (1 .+ exp.(-eta)))
  X,y,beta
end
function row(io,rep,method,r,truth,elapsed)
  ci=confint(r); active=truth.!=0; null=.!active
  coverage=(ci[:,1].<=truth).&(truth.<=ci[:,2]); rej=r.pvalues.<.05
  len=ci[:,2]-ci[:,1]
  open(out*".coefficients","a") do co
    for j in eachindex(truth)
      println(co,join((rep,method,j,truth[j],r.beta[j],r.se[j],r.pvalues[j]),','))
    end
  end
  println(io,join((rep,method,r.converged,mean(coverage[active]),mean(coverage[null]),mean(rej[null]),mean(rej[active]),mean(len[active]),mean(len[null]),mean((r.beta[active]-truth[active]).*sign.(truth[active])),mean((r.beta[active]-truth[active])./r.se[active].*sign.(truth[active])),r.sigma,r.lambda,first(r.projection.lambda),elapsed),',')); flush(io)
end
function runrep(io,rep)
  X,y,beta=data(rep); targets=vcat(1:5,s+1:s+5)
  t=time_ns()
  f=debiased_lasso(X,y;family=label,targets,threads=false,rng=Xoshiro(920000+rep))
  row(io,rep,"automatic",f,beta[targets],(time_ns()-t)/1e9)
  if stage=="development"
    # Compare the old universal projection at the identical CV-selected initializer.
    g=debiased_lasso(X,y,f.lambda;family=label,initial=f.initial,initial_intercept=f.intercept,
      nodewise_lambda=:universal,targets,threads=false,rng=Xoshiro(920000+rep))
    row(io,rep,"cv_initial_universal_projection",g,beta[targets],0.)
  end
  if label==:gaussian
    known=debiased_lasso(X,y,f.lambda;initial=f.initial,sigma=1.,projection=f.projection)
    row(io,rep,"known_sigma",known,beta[targets],0.)
    oracle=debiased_lasso(X,y;initial=beta,sigma=1.,projection=f.projection)
    row(io,rep,"oracle_initial_and_sigma",oracle,beta[targets],0.)
  end
end
# Independent small warm-up.
W=randn(Xoshiro(42),60,8); b=label==:gaussian ? W[:,1]+randn(Xoshiro(43),60) : Float64.(W[:,1].>0)
debiased_lasso(W,b;family=label,threads=false,rng=Xoshiro(44))
open(out,"w") do io
 println(io,"rep,method,converged,coverage_active,coverage_null,type1,power,length_active,length_null,signed_bias_active,signed_z_bias_active,sigma,lambda,node_lambda,fit_seconds")
 for rep=shard:8:100
   try
     runrep(io,rep)
   catch e
     open(out*".errors","a") do err
       println(err,"rep=",rep," ",sprint(showerror,e))
     end
   end
 end
end
println("STUDY_COMPLETE ",stage," ",case," shard=",shard)
