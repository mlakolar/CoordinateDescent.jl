using HighDimensionalModels, ProximalSolvers, HiGHS, LinearAlgebra, Random, DelimitedFiles, Profile
const HDM=HighDimensionalModels
BLAS.set_num_threads(1)
loadbin(f)=open(io->read!(io,Vector{Float64}(undef,filesize(f)÷8)),f)
function fitapi(mode,X,y,ls,τ)
 target=mode=="certified-tight" ? 2e-7 : 2e-6
 QuantileLassoPath(X,y,ls;tau=τ,method=:smooth,bandwidth=.2,standardize_x=false,tol=target,
  options=CDOptions(opt_tol=1e-8,max_iter=10000,gaptol=mode=="legacy" ? 0. : nothing),rng=Xoshiro(20261025))
end
function diagnostics(X,y,B,U,ls,τ)
 [HDM._qsolution(X,y,B[2:end,k],B[1,k],U[:,k],ls[k],ones(size(X,2)),τ,true,:smooth,.2,:R_returned,0,false,2e-6) for k in eachindex(ls)]
end
function main()
 case=ARGS[1];seed=parse(Int,ARGS[2]);base="inputs/$case-$seed"
 n,p,τ,L=vec(readdlm(base*"-meta.csv",',',Float64));n=Int(n);p=Int(p);L=Int(L)
 X=reshape(loadbin(base*"-X.bin"),n,p);y=loadbin(base*"-y.bin");ls=loadbin(base*"-ls.bin")
 configs=["certified-tight","conquer-1e-8"]
 for mode in configs[1:1];fitapi(mode,X[1:60,1:12],y[1:60],[.2,.1],τ);end
 diagnostics(X[1:60,1:12],y[1:60],zeros(13,2),zeros(60,2),[.2,.1],τ)
 open("results/$case-$seed-paths.csv","w") do io
  open("results/$case-$seed-solutions.csv","w") do so
   println(io,"case,seed,rep,method,levels,fit_s,diagnostics_s,total_s,max_kkt,max_abs_rawgap,max_certificate_gap,qualified_2e6,qualified_2e7,all_native,status,host")
   println(so,"case,seed,rep,method,k,lambda,objective,raw_gap,certificate_gap,kkt,dual_feasibility,converged,native_converged")
   for rep in 1:2,mode in shuffle(Xoshiro(seed+rep+90000),configs)
    GC.gc();ss=QuantileLassoSolution[];fit_s=NaN;diag_s=0.;status="returned";path=nothing
    try
     if startswith(mode,"conquer")
      eps=split(mode,'-';limit=2)[2];out="results/$case-$seed-$rep-$(parse(Float64,eps))"
      envarg="R_LIBS_USER=$(ENV["R_LIBS_USER"]),OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1"
      cmd=`timeout 90 apptainer exec --cleanenv --env $envarg /apps/containers/r/4.5/r-4.5.3.sif Rscript reference.R $case $seed $eps $rep`
      proc=open(f->run(pipeline(ignorestatus(cmd),stdout=f,stderr=f)),"logs/$case-$seed-$rep-$mode.log","w")
      # R's numeric filename convention is 1e-06, not Julia's 1.0e-6.
      out="results/$case-$seed-$rep-$(eps=="1e-6" ? "1e-06" : eps=="1e-7" ? "1e-07" : "1e-08")"
      if proc.exitcode==0 && isfile(out*"-seconds.txt")
       B=reshape(loadbin(out*"-B.bin"),p+1,L);U=reshape(loadbin(out*"-U.bin"),n,L)
       fit_s=parse(Float64,strip(read(out*"-seconds.txt",String)))
       t=@timed diagnostics(X,y,B,U,ls,τ);ss=t.value;diag_s=t.time
      else
       status="R_failed_or_timeout_$(proc.exitcode)"
      end
     else
      t=@timed fitapi(mode,X,y,ls,τ);path=t.value;ss=path.solutions;fit_s=t.time
     end
    catch e
     showerror(stderr,e,catch_backtrace());println(stderr);status="ERROR"
    end
    gate(tol)=length(ss)==L && all(s.kkt_residual<=tol && abs(s.duality_gap)<=tol*(1+abs(s.objective)) for s in ss)
    mx(f)=isempty(ss) ? NaN : maximum(f(s) for s in ss)
    println(io,join((case,seed,rep,mode,length(ss),fit_s,diag_s,fit_s+diag_s,mx(s->s.kkt_residual),mx(s->abs(s.duality_gap)),mx(s->s.certificate_gap),gate(2e-6),gate(2e-7),startswith(mode,"conquer") ? "not_reported" : all(s.native_converged for s in ss),status,gethostname()),','));flush(io)
    for (k,s) in enumerate(ss)
     println(so,join((case,seed,rep,mode,k,s.λ,s.objective,s.duality_gap,s.certificate_gap,s.kkt_residual,s.dual_feasibility,s.converged,startswith(mode,"conquer") ? "not_reported" : s.native_converged),','))
    end
    flush(so);println("DONE $case $seed $rep $mode $fit_s qualified=$(gate(2e-6))");flush(stdout)
    if case=="wide" && seed==1202 && rep==1 && mode=="certified" && path!==nothing
     for (kind,values) in [("raw-dual",dual_scores(path)),("coefficients",coef(path)),("certificate",reduce(hcat,[s.certificate_dual for s in ss]))]
      open(f->write(f,values),"results/anchor-$kind.bin","w")
     end
    end
   end
  end
 end
 if false
  Profile.clear();start=time();fits=0
  while time()-start<1.5;@profile fitapi("certified",X,y,ls,τ);fits+=1;end
  open("results/$case-$seed-profile.txt","w") do io
   println(io,"method=certified fits=$fits elapsed=$(time()-start) host=$(gethostname())")
   Profile.print(io;C=true,format=:flat,sortedby=:count)
  end
 end
end
main()
