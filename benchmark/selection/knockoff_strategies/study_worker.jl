# Generate and fit one immutable scenario/shard under Slurm.
using LinearAlgebra, Random, TOML, SHA, DelimitedFiles
import HighDimensionalModels as H
import Knockoffs as KJ
BLAS.set_num_threads(1)
root=ENV["HDM_KNOCKOFF_STUDY"];sc=parse(Int,ARGS[1]);shard=parse(Int,ARGS[2]);count=parse(Int,ARGS[3])
n,p=sc in (9,10) ? (180,300) : (300,100)
rho=sc in (1,2,11,14) ? 0. : sc in (5,6,13,16) ? .8 : .4
supplied_rho=sc==14 ? .4 : sc==15 ? 0. : sc==16 ? .4 : rho
same=sc in (2,4,6,8,10);k=sc in (11,12,13) ? 0 : 20;amp=sc in (7,8) ? 3. : 4.5
S=[supplied_rho^abs(i-j) for i in 1:p,j in 1:p]
truth=[rho^abs(i-j) for i in 1:p,j in 1:p]
strategies=(:equicorrelated,:mvr,:maxent)
label="scenario-$sc-shard-$shard";out=joinpath(root,label)
ispath(out) && error("immutable study shard already exists");mkpath(out)
plans=Any[]
open(joinpath(out,"construction-status.csv"),"w") do io
 println(io,"strategy,success,error")
 for strategy in strategies
  try
   plan=H.knockoff_diagonal(S;strategy);push!(plans,plan)
   println(io,join((strategy,true,""),','))
  catch e
   push!(plans,nothing);println(io,join((strategy,false,replace(sprint(showerror,e),','=>';','\n'=>' ')),','))
  end
 end
end
# Failed diagonals remain explicit and are never replaced by another strategy.
writedlm(joinpath(out,"supplied-covariance.csv"),S,',')
writedlm(joinpath(out,"true-covariance.csv"),truth,',')
d=hcat((plan===nothing ? fill(NaN,p) : plan.s for plan in plans)...)
writedlm(joinpath(out,"diagonals.csv"),d,',')
open(joinpath(out,"inputs.bin"),"w") do inputs
 open(joinpath(out,"fits.bin"),"w") do fits
  open(joinpath(out,"R-inputs.bin"),"w") do ri
   open(joinpath(out,"fit-status.csv"),"w") do status
    println(status,"seed,strategy,success,converged,iterations,api_match,error")
    for dataset in 1:count
     seed=106_600_000_000+1_000_000sc+10_000shard+dataset;rng=Xoshiro(seed)
     X=randn(rng,n,p)*cholesky(Symmetric(truth)).U
     beta=zeros(p);beta[1:k].=(amp/sqrt(n)).*(same ? ones(k) : ifelse.(isodd.(1:k),1.,-1.))
     y=X*beta+randn(rng,n);Z=randn(rng,n,p)
     write(inputs,Int64[n,p,k,sc,shard,dataset,seed]);write(inputs,X);write(inputs,y);write(inputs,Z);write(inputs,beta)
     B=fill(NaN,2p,3)
     for (j,strategy) in enumerate(strategies)
      ok=false;conv=false;iterations=-1;api=false;err=""
      try
       plans[j]===nothing && error("construction failed; see construction-status.csv")
       K=H.gaussian_knockoffs(X,zeros(p),S;s=plans[j].s,noise=Z)
       A=hcat(X,K)
       fit=H.lasso(A,y,.04;intercept=false,standardize_x=false,order=:cyclic,opt_tol=1e-9,max_iter=100000)
       B[:,j]=Vector(fit.x);conv=fit.converged;iterations=fit.iterations;ok=true
       if dataset==1
        filter=H.knockoff_filter(X,y,.04;mu=zeros(p),Sigma=S,noise=Z,strategy,order=:cyclic,opt_tol=1e-9,max_iter=100000)
        W=abs.(B[1:p,j])-abs.(B[p+1:2p,j]);t=H.knockoff_threshold(W)
        selected=isfinite(t) ? findall(>=(t),W) : Int[]
        api=filter.Xk==K && filter.W==W && filter.selected==selected
        api || error("public filter and explicit native pipeline differ")
       else
        api=true
       end
       if dataset<=3
        write(ri,Int64[n,p,k,sc,shard,dataset,seed,j]);write(ri,A);write(ri,y)
       end
      catch e
       err=replace(sprint(showerror,e),','=>';','\n'=>' ');ok=false
      end
      println(status,join((seed,strategy,ok,conv,iterations,api,err),','));flush(status)
     end
     write(fits,Int64[seed,p,3]);write(fits,B);flush(fits);flush(inputs);flush(ri)
    end
   end
  end
 end
end
metadata=Dict("julia"=>string(VERSION),"cpu"=>Sys.cpu_info()[1].model,
 "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"Knockoffs"=>string(pkgversion(KJ)),
 "scenario"=>sc,"shard"=>shard,"count"=>count,"population_rho"=>rho,"supplied_rho"=>supplied_rho,
 "lambda"=>.04,"q"=>.1,"lasso_order"=>"cyclic","opt_tol"=>1e-9,"max_iter"=>100000,
 "input_sha256"=>bytes2hex(sha256(read(joinpath(out,"inputs.bin")))),
 "knockoffs_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoffs.jl")))),
 "construction_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoff_construction.jl")))),
 "backend_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(KJ)),"utilities.jl")))))
open(io->TOML.print(io,metadata),joinpath(out,"metadata.toml"),"w")
println("STUDY_WORKER_DONE $sc $shard $count")
