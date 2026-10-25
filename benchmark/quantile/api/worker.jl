using HighDimensionalModels, HiGHS, JuMP, Gurobi, Random, LinearAlgebra, Statistics, DelimitedFiles, Profile
const CD=HighDimensionalModels
BLAS.set_num_threads(1)
const GRB=Gurobi.Env(output_flag=0)
function loadbin(file)
 open(file) do io;read!(io,Vector{Float64}(undef,filesize(file)÷8));end
end
function gurobipath(X,y,ls,τ;limit=60.)
 n,p=size(X);start=time_ns();m=direct_model(Gurobi.Optimizer(GRB));set_silent(m)
 for (a,b) in [("Threads",1),("Method",0),("FeasibilityTol",1e-8),("OptimalityTol",1e-8)];set_optimizer_attribute(m,a,b);end
 @variable(m,b[1:p]);@variable(m,a);@variable(m,t[1:p]>=0);@variable(m,rp[1:n]>=0);@variable(m,rn[1:n]>=0)
 @constraint(m,eq,X*b .+ a .+ rp .- rn .== y);@constraint(m,b .<= t);@constraint(m,-b .<= t)
 @objective(m,Min,(τ*sum(rp)+(1-τ)*sum(rn))/n+first(ls)*sum(t))
 ss=QuantileLassoSolution[]
 for λ in ls
  remaining=limit-(time_ns()-start)/1e9;remaining<=0 && break
  set_objective_coefficient(m,t,fill(λ,p));set_optimizer_attribute(m,"TimeLimit",remaining);optimize!(m)
  termination_status(m)==JuMP.MOI.OPTIMAL || break
  push!(ss,CD._qsolution(X,y,value.(b),value(a),n.*dual.(eq),λ,ones(p),τ,true,:exact,0.,:OPTIMAL,
                        JuMP.MOI.get(backend(m),JuMP.MOI.SimplexIterations()),true,2e-5))
 end
 ss
end
function runmethod(name,X,y,ls,τ)
 name=="api-exact" && return QuantileLassoPath(X,y,ls;tau=τ,standardize_x=false,tol=2e-5,time_limit=60.).solutions
 name=="api-smooth" && return QuantileLassoPath(X,y,ls;tau=τ,standardize_x=false,tol=2e-5,method=:smooth,bandwidth=.2,rng=Xoshiro(20261025)).solutions
 gurobipath(X,y,ls,τ)
end
function rdiagnostics(X,y,B,U,ls,w,τ,h)
 [CD._qsolution(X,y,B[2:end,k],B[1,k],U[:,k],ls[k],w,τ,true,h==0 ? :exact : :smooth,h,:R_returned,0,false,2e-5) for k in eachindex(ls)]
end
function main()
 case=ARGS[1];seed=parse(Int,ARGS[2]);base="inputs/$case-$seed"
 n,p,τ,L=vec(readdlm(base*"-meta.csv",',',Float64));n=Int(n);p=Int(p);L=Int(L)
 X=reshape(loadbin(base*"-X.bin"),n,p);y=loadbin(base*"-y.bin");ls=loadbin(base*"-ls.bin");w=ones(p)
 for method in ["api-exact","api-smooth","gurobi-primal"]
  runmethod(method,X[1:60,1:12],y[1:60],[.2,.1],τ)
 end
 # Warm the external-output diagnostic dispatch too; its first compilation
 # must not be charged to the first R method in a shuffled block.
 for h in (0.,.2)
  rdiagnostics(X[1:60,1:12],y[1:60],zeros(13,2),zeros(60,2),[.2,.1],ones(12),τ,h)
 end
 # Validate the extracted quantreg workspace before ranking any timing.
 pathrows=Any[];solrows=Any[];saved=Dict{Tuple{String,Int},Any}()
 function record(method,rep,ss,fit_s,diag_s,bytes,available,status)
  full=length(ss)==L;qualified=full && all(s.converged for s in ss)
  push!(pathrows,(case,seed,rep,method,n,p,τ,L,length(ss),fit_s,diag_s,fit_s+diag_s,bytes,available,qualified,status,gethostname()))
  for (k,s) in enumerate(ss)
   push!(solrows,(case,seed,rep,method,k,s.λ,s.objective,s.check_objective,s.dual_objective,s.duality_gap,s.certificate_gap,s.kkt_residual,s.dual_feasibility,s.complementarity,s.converged,s.status==:R_returned ? "not_reported" : s.native_converged,s.status))
  end
  saved[(method,rep)]=ss
 end
 methods=["api-exact","api-smooth","gurobi-primal","quantreg-native-dual","rqpen-br","rqpen-fn","rqpen-huber","conquer-dual"]
 for rep in 1:2,method in shuffle(Xoshiro(seed+rep+90000),methods)
  GC.gc()
  if method in ["api-exact","api-smooth","gurobi-primal"]
   try
    t=@timed runmethod(method,X,y,ls,τ)
    record(method,rep,t.value,t.time,0.,t.bytes,true,length(t.value)==L ? "returned" : "incomplete")
    println("DONE $case $seed $rep $method $(t.time)s levels=$(length(t.value))");flush(stdout)
   catch e
    showerror(stderr,e,catch_backtrace());println(stderr)
    record(method,rep,QuantileLassoSolution[],NaN,NaN,NaN,true,"ERROR")
   end
  else
   logfile="logs/$case-$seed-$rep-$method.log"
   envarg="R_LIBS_USER=$(ENV["R_LIBS_USER"]),OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1"
   cmd=`timeout 75 apptainer exec --cleanenv --env $envarg /apps/containers/r/4.5/r-4.5.3.sif Rscript competitors.R fit $case $seed $method $rep`
   exitcode=0
   open(logfile,"w") do io
    proc=run(pipeline(ignorestatus(cmd),stdout=io,stderr=io));exitcode=proc.exitcode
   end
   open("results/$case-$seed-R-exits.tsv","a") do io;println(io,"$rep\t$method\t$exitcode");end
   file="results/$case-$seed-$rep-$method"
   if !isfile(file*"-seconds.txt")
    record(method,rep,QuantileLassoSolution[],NaN,NaN,NaN,false,"R_failed_or_timeout");continue
   end
   B=reshape(loadbin(file*"-B.bin"),p+1,L);fit_s=parse(Float64,strip(read(file*"-seconds.txt",String)))
   if isfile(file*"-U.bin")
    U=reshape(loadbin(file*"-U.bin"),n,L);h=method=="conquer-dual" ? .2 : 0.
    t=@timed rdiagnostics(X,y,B,U,ls,w,τ,h)
    record(method,rep,t.value,fit_s,t.time,NaN,true,"returned")
   else
    record(method,rep,QuantileLassoSolution[],fit_s,NaN,NaN,false,"coefficients_only")
   end
  end
 end
 # Coefficient-only fits get an exact-objective comparison from an independent
 # feasible bound on this identical input; this is not a returned raw dual.
 bounds=Dict{Float64,Float64}()
 for ss in values(saved),s in ss
  bound=s.objective-s.certificate_gap
  bounds[s.λ]=max(get(bounds,s.λ,-Inf),bound)
 end
 open("results/$case-$seed-coefficients.csv","w") do io
  println(io,"case,seed,rep,method,k,lambda,check_objective,exact_lower_bound,check_gap")
  for rep in 1:2,method in ["rqpen-br","rqpen-fn","rqpen-huber"]
   file="results/$case-$seed-$rep-$method-B.bin";isfile(file)||continue
   B=reshape(loadbin(file),p+1,L)
   for k in 1:L
    r=y.-B[1,k].-X*B[2:end,k];v=mean(CD._qcheck.(r,τ))+ls[k]*sum(abs,B[2:end,k])
    # Use an exact lower bound, including when a smooth solution supplies u:
    # subtract its conjugate contribution, since it is a different objective.
    exactbound=maximum((dot(y,s.certificate_dual)/n for ss in values(saved) for s in ss if s.λ==ls[k]);init=0.)
    println(io,join((case,seed,rep,method,k,ls[k],v,exactbound,v-exactbound),','))
   end
  end
 end
 for (kind,header,rows) in [("paths","case,seed,rep,method,n,p,tau,requested_levels,levels,fit_s,diagnostics_s,total_s,julia_bytes,raw_dual_available,qualified,status,host",pathrows),
                            ("solutions","case,seed,rep,method,k,lambda,objective,check_objective,dual_objective,raw_gap,certificate_gap,kkt,dual_feasibility,complementarity,converged,native_converged,status",solrows)]
  open("results/$case-$seed-$kind.csv","w") do io
   println(io,header);for row in rows;println(io,join(row,','));end
  end
 end
 if case in ("small","wide") && seed in (1201,1202)
  for method in ["api-exact","api-smooth"]
   Profile.clear();start=time();fits=0
   while time()-start<1.5
    @profile runmethod(method,X,y,ls,τ)
    fits+=1
   end
   open("results/$case-$seed-$method-profile.txt","w") do io
    println(io,"method=$method fits=$fits elapsed=$(time()-start) host=$(gethostname())")
    Profile.print(io;C=true,format=:flat,sortedby=:count)
   end
  end
 end
end
main()
