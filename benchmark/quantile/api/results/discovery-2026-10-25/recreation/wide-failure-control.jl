using CoordinateDescent,HiGHS,LinearAlgebra,Random,DelimitedFiles
const CD=CoordinateDescent
BLAS.set_num_threads(1)
loadbin(f)=open(io->read!(io,Vector{Float64}(undef,filesize(f)÷8)),f)
function diagnostics(X,y,B,U,ls,τ,h=.2)
 [CD._qsolution(X,y,B[2:end,k],B[1,k],U[:,k],ls[k],ones(size(X,2)),τ,true,h==0 ? :exact : :smooth,h,:R_returned,0,false,2e-5) for k in eachindex(ls)]
end
function apifit(X,y,ls,τ,tight)
 QuantileLassoPath(X,y,ls;tau=τ,method=:smooth,bandwidth=.2,standardize_x=false,
   options=CDOptions(opt_tol=tight ? 1e-11 : 1e-8,max_iter=tight ? 20000 : 10000),
   newton_tol=tight ? 1e-14 : 1e-10,newton_max_iter=tight ? 200 : 100,
   rng=Xoshiro(20261025)).solutions
end
function main()
 mkpath("wide-failure-control");seed=1202
 open("wide-failure-control/paths.csv","w") do io
  println(io,"case,seed,rep,method,fit_s,diagnostics_s,total_s,max_kkt,max_dual_feasibility,max_absolute_raw_gap,max_certificate_gap,qualified_2e5,qualified_2e6,native_converged")
  for case in ["wide"]
   base="inputs/$case-$seed";meta=vec(readdlm(base*"-meta.csv",',',Float64));n,p=Int(meta[1]),Int(meta[2]);τ=meta[3]
   X=reshape(loadbin(base*"-X.bin"),n,p);y=loadbin(base*"-y.bin");ls=loadbin(base*"-ls.bin")
   for tight in [false,true];apifit(X[1:60,1:12],y[1:60],[.2,.1],τ,tight);end
   for h in (0.,.2);diagnostics(X[1:60,1:12],y[1:60],zeros(13,2),zeros(60,2),[.2,.1],τ,h);end
   configs=["api-default","api-tight"]
   for rep in 1:2,label in shuffle(Xoshiro(seed+rep),configs)
    GC.gc();diag=0.;fit_s=NaN;ss=QuantileLassoSolution[]
    if startswith(label,"api")
     t=@timed apifit(X,y,ls,τ,label=="api-tight");ss=t.value;fit_s=t.time
    else
     eps=label=="quantreg-precomputed" ? "1e-8" : split(label,'-';limit=2)[2]
     envarg="R_LIBS_USER=$(ENV["R_LIBS_USER"]),OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1"
     log="wide-failure-control/$case-$rep-$label.log"
     cmd=`timeout 75 apptainer exec --cleanenv --env $envarg /apps/containers/r/4.5/r-4.5.3.sif Rscript accuracy.R $case $seed $eps $rep $label`
     proc=open(f->run(pipeline(ignorestatus(cmd),stdout=f,stderr=f)),log,"w")
     file="wide-failure-control/$case-$seed-$rep-$label"
     if proc.exitcode==0 && isfile(file*"-seconds.txt")
      B=reshape(loadbin(file*"-B.bin"),p+1,length(ls));U=reshape(loadbin(file*"-U.bin"),n,length(ls))
      fit_s=parse(Float64,strip(read(file*"-seconds.txt",String)));t=@timed diagnostics(X,y,B,U,ls,τ,label=="quantreg-precomputed" ? 0. : .2);ss=t.value;diag=t.time
     end
    end
    gate(tol)=length(ss)==length(ls) && all(s.kkt_residual<=tol && abs(s.duality_gap)<=tol*(1+abs(s.objective)) for s in ss)
    println(io,join((case,seed,rep,label,fit_s,diag,fit_s+diag,
      isempty(ss) ? NaN : maximum(s.kkt_residual for s in ss),isempty(ss) ? NaN : maximum(s.dual_feasibility for s in ss),
      isempty(ss) ? NaN : maximum(abs(s.duality_gap) for s in ss),isempty(ss) ? NaN : maximum(s.certificate_gap for s in ss),gate(2e-5),gate(2e-6),
      startswith(label,"api") ? all(s.native_converged for s in ss) : "not_reported"),','));flush(io)
    println("ACCURACY $case $rep $label $fit_s");flush(stdout)
   end
  end
 end
end
main()
