using LinearAlgebra,Random,Statistics,Profile,Serialization,TOML,SHA,DelimitedFiles
import HighDimensionalModels as H
import Knockoffs as KJ
BLAS.set_num_threads(1)
variant,case_str,rep_str=ARGS;case=parse(Int,case_str);rep=parse(Int,rep_str)
root=ENV["HDM_KNOCKOFF_RUNTIME"];out=joinpath(root,"$variant-case-$case-rep-$rep")
ispath(out) && error("immutable runtime worker exists");mkpath(out)
specs=[(600,200,0.),(600,200,.4),(400,400,.8),(200,600,.4),(600,200,.4)]
n,p,rho=specs[case];fixed=case==5;S0=[rho^abs(i-j) for i in 1:p,j in 1:p]
strategies=variant=="baseline" ? (:equicorrelated,) : (:equicorrelated,:mvr,:maxent)
opts=(;niter=1000,tol=1e-10,λmin=1e-10,robust=false,verbose=false)
fields(z)=(z.selected,z.W,z.threshold,z.Xk,z.q,z.lambda,z.method)
function profile_call(call,path)
 Profile.clear();Profile.@profile begin
  deadline=time()+.12;calls=0
  while calls==0 || (calls<10000 && time()<deadline);call();calls+=1;end
 end
 data=Profile.fetch()
 buffer=IOBuffer();println(buffer,"raw_stack_words=",length(data));Profile.print(buffer;format=:flat,sortedby=:count)
 write(path*"-cpu.txt",String(take!(buffer)))
 Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=.005 call()
 allocations=Profile.Allocs.fetch().allocs
 sample_rate=.005
 if isempty(allocations)
  Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=1.0 call()
  allocations=Profile.Allocs.fetch().allocs;sample_rate=1.0
 end
 buffer=IOBuffer();println(buffer,"sampled_allocations=",length(allocations));println(buffer,"sample_rate=",sample_rate)
 for a in sort!(collect(allocations);by=x->x.size,rev=true)[1:min(10,length(allocations))]
  println(buffer,"bytes=",a.size," type=",a.type," stack=",a.stacktrace)
 end
 write(path*"-alloc.txt",String(take!(buffer)))
 length(data)>0 || error("empty CPU profile")
 # Small deterministic diagonal calls can allocate too little for sparse
 # sampling. Record an explicit full-rate allocation witness separately.
 witness=@allocated call()
 witness>0 || error("missing allocation witness")
 witness
end
open(joinpath(out,"timings.csv"),"w") do timings
 open(joinpath(out,"checks.csv"),"w") do checks
  println(timings,"variant,case,rep,input,seed,strategy,payload,round,seconds,bytes,gctime")
  println(checks,"input,strategy,payload,round,valid,residual,profile_allocation_witness")
  for input in 1:3
   seed=106_900_000_000+1_000_000case+10_000rep+input;rng=Xoshiro(seed)
   X=randn(rng,n,p)*cholesky(Symmetric(S0)).U;y=X[:,1:20]*fill(4.5/sqrt(n),20)+randn(rng,n);Z=randn(rng,n,p)
   S=fixed ? X'X : S0;mu=zeros(p)
   open(joinpath(out,"input-$input.bin"),"w") do io
    write(io,Int64[n,p,seed]);write(io,X);write(io,y);write(io,Z);write(io,S)
   end
   F=fixed ? X./sqrt.(sum(abs2,X;dims=1)) : nothing
   if fixed
    open(joinpath(out,"unitnorm-$input.bin"),"w") do io;write(io,F);end
   end
   for index in circshift(collect(eachindex(strategies)),mod(rep+input,length(strategies)))
    strategy=strategies[index]
    diagonal=variant=="baseline" ? H.equicorrelated_s(S) : H.knockoff_diagonal(S;strategy).s
    # Anonymous closures avoid local-method capture replacement across cases.
    factory=variant=="baseline" ? (() -> H.equicorrelated_s(S)) : (() -> H.knockoff_diagonal(S;strategy))
    sampling=fixed ? (() -> H.fixed_knockoffs(X;s=diagonal)) : (() -> H.gaussian_knockoffs(X,mu,S;s=diagonal,noise=Z))
    construct=if fixed
     variant=="baseline" ? (() -> H.fixed_knockoffs(X)) : (() -> H.fixed_knockoffs(X;strategy))
    else
     variant=="baseline" ? (() -> H.gaussian_knockoffs(X,mu,S;noise=Z)) : (() -> H.gaussian_knockoffs(X,mu,S;strategy,noise=Z))
    end
    rng_construct=fixed ? nothing : variant=="baseline" ? (() -> H.gaussian_knockoffs(X,mu,S;rng=Xoshiro(seed+7000))) :
      (() -> H.gaussian_knockoffs(X,mu,S;strategy,rng=Xoshiro(seed+7000)))
    filter=if fixed
     variant=="baseline" ? (() -> H.knockoff_filter(X,y,.04;method=:fixed,order=:cyclic,opt_tol=1e-9,max_iter=100000)) :
      (() -> H.knockoff_filter(X,y,.04;method=:fixed,strategy,order=:cyclic,opt_tol=1e-9,max_iter=100000))
    else
     variant=="baseline" ? (() -> H.knockoff_filter(X,y,.04;mu,Sigma=S,rng=Xoshiro(seed+7000),order=:cyclic,opt_tol=1e-9,max_iter=100000)) :
      (() -> H.knockoff_filter(X,y,.04;mu,Sigma=S,strategy,rng=Xoshiro(seed+7000),order=:cyclic,opt_tol=1e-9,max_iter=100000))
    end
    payloads=Any[("diagonal",factory),("sampling",sampling),("construct",construct),("filter",filter)]
    if fixed && strategy===:equicorrelated
     unit_construct=variant=="baseline" ? (() -> H.fixed_knockoffs(F)) : (() -> H.fixed_knockoffs(F;strategy))
     push!(payloads,("construct-unitnorm",unit_construct))
    end
    fixed || push!(payloads,("construct-rng",rng_construct))
    if variant!="baseline" && strategy!==:equicorrelated
     direct=() -> KJ.solve_s(Symmetric(copy(S)),strategy;opts...)
     push!(payloads,("raw-optimizer",direct))
    end
    for (payload,call) in circshift(payloads,mod(rep+input+index,length(payloads)))
     warm=call()
     if payload=="filter"
      serialize(joinpath(out,"fields-$input-$strategy.bin"),fields(warm))
     elseif payload=="construct"
      serialize(joinpath(out,"matrix-$input-$strategy.bin"),warm)
     elseif payload=="diagonal"
      d=variant=="baseline" ? warm : warm.s
      writedlm(joinpath(out,"diagonal-$input-$strategy.csv"),d,',')
     end
     for round in 1:3
      GC.gc();result=@timed call();value=result.value
      valid=true;res=NaN
      if payload=="filter"
       valid=isequal(fields(value),fields(warm));K=value.Xk;A=hcat(X,K)
       fit=H.lasso(A,y,.04;order=:cyclic,opt_tol=1e-9,max_iter=100000)
       b=Vector(fit.x);g=A'*(A*b-y)/n;v=b-g;prox=sign.(v).*max.(abs.(v).-.04,0)
       res=norm(b-prox,Inf)/max(1,norm(A'y/n,Inf));valid &= res<=1e-7
       W=abs.(b[1:p])-abs.(b[p+1:2p]);valid &= W==value.W
      elseif payload=="construct-unitnorm"
       valid=value==warm && isapprox(value'value,F'F;rtol=1e-9,atol=1e-9)
       valid &= isapprox(F'value,F'F-Diagonal(H.equicorrelated_s(F'F));rtol=1e-9,atol=1e-9)
      elseif payload in ("sampling","construct","construct-rng")
       valid=value==warm
       if fixed
        valid &= isapprox(value'value,X'X;rtol=1e-9,atol=1e-9) && isapprox(X'value,X'X-Diagonal(diagonal);rtol=1e-9,atol=1e-9)
       end
      end
      println(timings,join((variant,case,rep,input,seed,strategy,payload,round,result.time,result.bytes,result.gctime),','));flush(timings)
      println(checks,join((input,strategy,payload,round,valid,res,0),','));flush(checks)
      valid || error("runtime correctness failed $strategy $payload")
     end
     witness=profile_call(call,joinpath(out,"profile-$input-$strategy-$payload"))
     println(checks,join((input,strategy,payload,0,true,NaN,witness),','));flush(checks)
    end
   end
  end
 end
end
metadata=Dict("variant"=>variant,"case"=>case,"rep"=>rep,"julia"=>string(VERSION),"cpu"=>Sys.cpu_info()[1].model,
 "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"Knockoffs"=>string(pkgversion(KJ)),
 "knockoffs_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoffs.jl")))),
 "backend_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(KJ)),"utilities.jl")))),
 "process_peak_rss_bytes"=>Sys.maxrss())
open(io->TOML.print(io,metadata),joinpath(out,"metadata.toml"),"w")
println("RUNTIME_WORKER_DONE $variant $case $rep")
