include("adapters.jl")
include("instrument.jl")
using StatsAPI
folder,method,out=ARGS;mkpath(out)
a=vec(readdlm(joinpath(folder,"dims.txt")));n,p,q=Int.(a[1:3]);lm=a[4]
readarr(name,dims...)=open(f->reshape(read!(f,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,name*".bin"))
X,Y=readarr("X",n,p),readarr("Y",n,q)
input_hashes=Dict(k=>bytes2hex(sha256(read(joinpath(folder,k*".bin")))) for k in ("X","Y"))
rss()=parse(Int,match(r"VmRSS:\s+(\d+) kB",read("/proc/self/status",String)).captures[1])*1024
ext=Base.get_extension(H,:HighDimensionalModelsProximalAlgorithmsExt)
source=joinpath(dirname(pathof(H)),"..","ext","HighDimensionalModelsProximalAlgorithmsExt.jl")
md=Dict{String,Any}("worker_sha256"=>bytes2hex(sha256(read(@__FILE__))),"adapters_sha256"=>bytes2hex(sha256(read(joinpath(@__DIR__,"adapters.jl")))),"method"=>method,"input_hashes"=>input_hashes,"extension_sha256"=>bytes2hex(sha256(read(source))),"lowrank_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"lowrank.jl")))),"cpu"=>Sys.cpu_info()[1].model,"julia"=>string(VERSION),"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"blas"=>string(BLAS.get_config()))
d=H._lr_data(X,Y,true);g=H._lr_gradient(d);cached=(;d,g)
config=method in ("baseline","candidate") ? "native" : method
fractions=[.5,.1,.02];path_fractions=exp.(range(log(.5),log(.02);length=8));records=Any[];outcomes=Any[];failures=Any[]
function call_fit(mode,frac;stats=nothing)
 controlled(X,Y,frac*lm,config;cached=mode=="cached" ? cached : nothing,stats)
end
function call_path(mode;stats=nothing)
 if method in ("baseline","candidate","apg") && mode=="public"
  solver=method=="apg" ? :apg : :panoc
  path=LowRankPath(X,Y;solver,λpath=path_fractions.*lm,tol=1e-8,max_iter=20000)
  return [(;B=s.B,iterations=s.iterations,converged=s.converged,solution=s) for s in path.solutions]
 end
 pathdata=mode=="cached" ? cached : (d=H._lr_data(X,Y,true);(;d,g=H._lr_gradient(d)))
 state=zeros(p,q);fits=Any[]
 for frac in path_fractions
  z=controlled(X,Y,frac*lm,config;cached=pathdata,start=state,stats);push!(fits,z)
  state=z.solution!==nothing && z.solution.rank==0 ? zeros(p,q) : copy(z.B)
 end
 fits
end
fingerprint(fits)=bytes2hex(sha256(vcat([reinterpret(UInt8,vec(z.B)) for z in fits]...)))
function store!(fits,mode,kind,fracs,binary)
 for (index,z) in enumerate(fits)
  offset=position(binary);write(binary,z.B)
  s=z.solution;intercept=s===nothing ? d.my-z.B'*d.mx : s.intercept;write(binary,intercept)
  residual_error=s===nothing ? NaN : norm(s.residuals-(d.Y-d.X*z.B))/max(1.,norm(d.Y))
  push!(records,Dict("mode"=>mode,"kind"=>kind,"fraction"=>fracs[index],"lambda"=>fracs[index]*lm,"offset_bytes"=>offset,"iterations"=>z.iterations,"converged"=>z.converged,"has_full_result"=>s!==nothing,"objective"=>s===nothing ? NaN : s.objective,"rank"=>s===nothing ? -1 : s.rank,"df"=>s===nothing ? -1 : s.df,"rows"=>s===nothing ? Int[] : s.activerows,"native_residual_error"=>residual_error))
 end
end
workloads=Any[]
for mode in ("public","cached")
 for frac in fractions
  push!(workloads,(mode=mode,kind="cold",fraction=frac,fracs=[frac],call=()->[call_fit(mode,frac)]))
 end
 push!(workloads,(mode=mode,kind="path",fraction=0.,fracs=path_fractions,call=()->call_path(mode)))
end
references=Dict{String,Any}()
open(joinpath(out,"fits.bin"),"w") do binary
 open(joinpath(out,"timings.csv"),"w") do io
  println(io,"mode,kind,fraction,round,seconds,bytes,iterations,converged,coefficient_sha256")
  for w in workloads
   key=w.mode*"/"*w.kind*"/"*string(w.fraction)
   try
    warm=w.call()
    if !all(z->z.converged,warm)
     store!(warm,w.mode,w.kind,w.fracs,binary);push!(failures,Dict("workload"=>key,"error"=>"warmup nonconverged; repeats/profiles skipped"));continue
    end
    if !haskey(md,"warmed_current_rss");md["warmed_current_rss"]=rss();md["warmed_peak_rss"]=Sys.maxrss();end
    hashes=String[];lastfits=nothing;its=Any[]
    for rd in 1:3
     GC.gc();tm=@timed w.call();fits=tm.value;lastfits=fits
     h=fingerprint(fits);push!(hashes,h);push!(its,[z.iterations for z in fits])
     println(io,join((w.mode,w.kind,w.fraction,rd,tm.time,tm.bytes,sum(z.iterations for z in fits),all(z->z.converged,fits),h),','));flush(io)
    end
    @assert length(unique(hashes))==1 && all(z->z.converged,lastfits)
    references[key]=(hash=hashes[1],iterations=its[1]);store!(lastfits,w.mode,w.kind,w.fracs,binary)
    push!(outcomes,Dict("workload"=>key,"status"=>"complete","hash"=>hashes[1]))
   catch e
    push!(failures,Dict("workload"=>key,"error"=>sprint(showerror,e,catch_backtrace())))
   end
  end
 end
end
md["timing_peak_rss"]=Sys.maxrss();md["timing_current_rss"]=rss()
open(f->TOML.print(f,Dict("records"=>records)),joinpath(out,"fits.toml"),"w")
# Profiles use public middle/path calls, bounded repeated sampling outside timers.
for (label,call) in (("middle",()->[call_fit("public",.1)]),("path",()->call_path("public")))
 key=label=="middle" ? "public/cold/0.1" : "public/path/0.0"
 haskey(references,key) || continue
 Profile.clear();repeats=0;started=time()
 Profile.@profile while repeats<30 && (repeats==0 || time()-started<.2)
  call();repeats+=1
 end
 open(joinpath(out,"$label-cpu.txt"),"w") do io
  println(io,"profile_calls=",repeats," raw_stack_words=",length(Profile.fetch()));Profile.print(io;format=:flat,sortedby=:count)
 end
end
Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=.001 call_fit("public",.1)
allocs=Profile.Allocs.fetch().allocs
open(joinpath(out,"alloc.txt"),"w") do io
 println(io,"sampled_allocations=",length(allocs))
 for a in sort(collect(allocs);by=a->a.size,rev=true)[1:min(20,length(allocs))];println(io,"bytes=",a.size," type=",a.type," stack=",a.stacktrace);end
end
md["profile_peak_rss"]=Sys.maxrss()
# Cached component probes are distinct workloads, not additive attribution.
open(joinpath(out,"components.csv"),"w") do io
 println(io,"component,round,seconds,bytes")
 loss=ext.RegressionLoss(d,g);penalty=PB.NuclearNorm(.1lm);B=call_fit("cached",.1).B
 reuse_final=method in ("candidate","finalreuse","guarded_final")
 diagnostic_call=()->begin
  if reuse_final
   residual=d.Y-d.X*B;singular=svdvals(B);r=Base.count(>(min(size(B)...)*eps(Float64)*first(singular)),singular)
   H._lr_solution(d,B,.1lm,0,true,sum(abs2,residual)/(2n)+.1lm*sum(singular),:nuclear;coefficient_rank=r,residual)
  else
   H._lr_solution(d,B,.1lm,0,true,loss(B)+penalty(B),:nuclear)
  end
 end
 components=[("cached_complete_result",diagnostic_call),("centering",()->H._lr_data(X,Y,true)),("gradient_setup",()->H._lr_gradient(d)),("returned_point_value_gradient",()->PA.value_and_gradient(loss,B))]
 for (name,call) in components
  call()
  for rd in 1:3;GC.gc();tm=@timed call();println(io,join((name,rd,tm.time,tm.bytes),','));end
 end
end
# No timing or profile occurs after this exact native-source instrumentation.
counts=Any[]
if method in ("baseline","candidate")
 original_hash,replay_hash=instrument_native!(ext);md["counter_original_sha256"]=original_hash;md["counter_replay_sha256"]=replay_hash
 for w in workloads
  key=w.mode*"/"*w.kind*"/"*string(w.fraction);haskey(references,key) || continue
  empty!(ext._HDM_STUDY_COUNTS);fits=Base.invokelatest(w.call)
  @assert fingerprint(fits)==references[key].hash && [z.iterations for z in fits]==references[key].iterations
  push!(counts,Dict("workload"=>key,"counts"=>copy(ext._HDM_STUDY_COUNTS),"matched_timed_B_and_iterations"=>true))
 end
elseif method!="apg"
 for w in workloads
  key=w.mode*"/"*w.kind*"/"*string(w.fraction);haskey(references,key) || continue
  stats=Dict{String,Int}()
  fits=w.kind=="path" ? call_path(w.mode;stats) : [call_fit(w.mode,w.fraction;stats)]
  @assert fingerprint(fits)==references[key].hash && [z.iterations for z in fits]==references[key].iterations
  push!(counts,Dict("workload"=>key,"counts"=>stats,"matched_timed_B_and_iterations"=>true))
 end
end
md["whole_worker_peak_rss"]=Sys.maxrss();md["outcomes"]=outcomes;md["failures"]=failures
open(f->TOML.print(f,Dict("records"=>records)),joinpath(out,"fits.toml"),"w")
open(f->TOML.print(f,Dict("records"=>counts)),joinpath(out,"counts.toml"),"w")
open(f->TOML.print(f,md),joinpath(out,"metadata.toml"),"w")
