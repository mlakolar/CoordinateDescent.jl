using HighDimensionalModels, LinearAlgebra, Statistics, Profile, TOML, SHA
BLAS.set_num_threads(1)
folder,out,variant=ARGS;mkpath(out)
n,p,q=parse.(Int,split(read(joinpath(folder,"dims.txt"),String)))
readarr(key,dims...)=open(io->reshape(read!(io,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,"$key.bin"))
X,Y=readarr("X",n,p),readarr("Y",n,q);prepared=TOML.parsefile(joinpath(folder,"prepared.toml"))
case,repstr=rsplit(basename(folder),'-';limit=2);rep=parse(Int,repstr)
configs=[(solver,frac,false) for solver in (:cd,:fista) for frac in (.5,.1,.02)]
case=="medium" && append!(configs,[(solver,.1,true) for solver in (:cd,:fista)])
name(c)=string(c[1],'-',c[2],'-',c[3] ? "adaptive" : "unit")
lambda(c)=c[2]*prepared[c[3] ? "lambda_adaptive" : "lambda_unit"]
function fit(c)
  sparse_reduced_rank(X,Y,3,lambda(c);block_solver=c[1],adaptive=c[3],
    max_iter=1000,tol=1e-8,inner_max_iter=10000,inner_tol=1e-9)
end
rss()=parse(Int,match(r"VmRSS:\s+(\d+) kB",read("/proc/self/status",String)).captures[1])*1024
for c in configs;fit(c);end
metadata=Dict("case"=>case,"rep"=>rep,"variant"=>variant,"julia"=>string(VERSION),
  "cpu"=>Sys.cpu_info()[1].model,"blas"=>string(BLAS.get_config()),"blas_threads"=>BLAS.get_num_threads(),"julia_threads"=>Threads.nthreads(),
  "warmed_current_rss"=>rss(),"warmed_peak_rss"=>Sys.maxrss(),
  "X_sha256"=>bytes2hex(sha256(read(joinpath(folder,"X.bin")))),"Y_sha256"=>bytes2hex(sha256(read(joinpath(folder,"Y.bin")))),
  "source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(HighDimensionalModels)),"lowrank.jl")))))
saved=Dict{String,Any}()
open(joinpath(out,"timings.csv"),"w") do io
  println(io,"variant,case,rep,config,round,seconds,bytes,iterations,converged,objective,B_sha256")
  for rd in 1:3,c in circshift(configs,mod(rep+rd,length(configs)))
    GC.gc();t=@timed fit(c);s=t.value;tag=name(c)
    digest=bytes2hex(sha256(reinterpret(UInt8,vec(s.B))))
    println(io,join((variant,case,rep,tag,rd,t.time,t.bytes,s.iterations,s.converged,s.objective,digest),','));flush(io)
    if rd==1
      saved[tag]=s
      open(f->write(f,s.B),joinpath(out,"$tag-B.bin"),"w")
      open(f->write(f,s.residuals),joinpath(out,"$tag-E.bin"),"w")
      open(f->write(f,s.intercept),joinpath(out,"$tag-intercept.bin"),"w")
      open(f->TOML.print(f,Dict("rank"=>s.rank,"df"=>s.df,"support"=>s.activerows,"objective"=>s.objective,"iterations"=>s.iterations,"converged"=>s.converged)),joinpath(out,"$tag.toml"),"w")
    end
  end
end
metadata["current_after_timing"]=rss();metadata["peak_after_timing"]=Sys.maxrss()
if case in ("wide","responses","ill")
  for c in configs
    calls=0;began=time();Profile.clear()
    Profile.@profile while calls==0 || (calls<20 && time()-began<.25)
      fit(c);calls+=1
    end
    open(joinpath(out,"cpu-$(name(c)).txt"),"w") do io
      println(io,"calls=",calls," raw_stack_words=",length(Profile.fetch()));Profile.print(io;format=:flat,sortedby=:count)
    end
  end
  Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=.05 fit((:fista,.1,false))
  a=Profile.Allocs.fetch().allocs
  open(joinpath(out,"alloc-fista.txt"),"w") do io
    println(io,"sampled_allocations=",length(a))
    for z in sort(collect(a);by=x->x.size,rev=true)[1:min(30,length(a))];println(io,"bytes=",z.size," type=",z.type," stack=",z.stacktrace);end
  end
end
metadata["whole_worker_peak_before_replay"]=Sys.maxrss()
include("probe.jl");install_probe!()
for c in configs
  tag=name(c);foreach(k->(HDM102_COUNTS[k]=0),keys(HDM102_COUNTS))
  HDM102_STATE[]=nothing
  s=Base.invokelatest(fit,c);state=HDM102_STATE[];state===nothing && error("missing actual factors")
  reference=saved[tag]
  norm(s.B-reference.B)<=1e-12*max(1,norm(reference.B)) && s.iterations==reference.iterations || error("diagnostic replay differs from timed fit")
  for (key,z) in (("A",state.A),("V",state.V),("weights",state.w));open(f->write(f,z),joinpath(out,"$tag-$key.bin"),"w");end
  open(f->TOML.print(f,Dict("replay_coefficient_error"=>norm(s.B-reference.B)/max(1,norm(reference.B)),"iterations"=>s.iterations,
    "L"=>state.L,"cd_sweeps"=>HDM102_COUNTS[:cd_sweeps],"fista_steps"=>HDM102_COUNTS[:fista_steps],"kkt_calls"=>HDM102_COUNTS[:kkt_calls])),joinpath(out,"$tag-probe.toml"),"w")
end
metadata["whole_worker_peak_with_replay"]=Sys.maxrss()
open(io->TOML.print(io,metadata),joinpath(out,"metadata.toml"),"w")
