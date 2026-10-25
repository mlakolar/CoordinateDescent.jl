using HighDimensionalModels, LinearAlgebra, Statistics, TOML, SHA, Profile
BLAS.set_num_threads(1)
input,out,variant=ARGS;mkpath(out)
m=TOML.parsefile(joinpath(input,"meta.toml"));n=m["n"]
readvector(name,count)=open(s->read!(s,Vector{Float64}(undef,count)),joinpath(input,"$name.bin"))
y=readvector("y",n);x=m["irregular"] ? readvector("x",n) : nothing
levels=m["path"]==0 ? [m["lo"]/n] : exp.(range(log(m["hi"]),log(m["lo"]);length=m["path"]))./n
fit()=m["path"]==0 ? trend_filter(y,only(levels);order=m["order"],x) : trend_filter_path(y,levels;order=m["order"],x)
solutions(z)=z isa GeneralizedLassoPath ? z.solutions : [z]
# Warm the actual wrapper, type and output branch on a separate small input.
for k in 0:2, grid in (nothing,cumsum(fill(1.1,64)))
  z=sin.(1:64)
  trend_filter(z,1e-3;order=k,x=grid)
  trend_filter_path(z,[1e-2,1e-3];order=k,x=grid)
end
fit() # Warm this wrapper and the actual problem size before all three rounds.
rss()=parse(Int,match(r"VmRSS:\s+(\d+) kB",read("/proc/self/status",String)).captures[1])*1024
meta=Dict("variant"=>variant,"case"=>m["case"],"rep"=>m["rep"],"julia"=>string(VERSION),
  "cpu"=>Sys.cpu_info()[1].model,"blas"=>string(BLAS.get_config()),"blas_threads"=>BLAS.get_num_threads(),
  "julia_threads"=>Threads.nthreads(),
  "warmed_current_rss"=>rss(),"warmed_peak_rss"=>Sys.maxrss(),
  "y_sha256"=>bytes2hex(sha256(read(joinpath(input,"y.bin")))),
  "x_sha256"=>bytes2hex(sha256(read(joinpath(input,"x.bin")))),
  "source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(HighDimensionalModels)),"structured.jl")))))
open(joinpath(out,"timings.csv"),"w") do io
  println(io,"variant,case,rep,round,seconds,bytes,iterations,converged,primal_residual,dual_residual")
  for round in 1:3
    GC.gc();t=@timed fit();s=solutions(t.value)
    println(io,join((variant,m["case"],m["rep"],round,t.time,t.bytes,sum(z.iterations for z in s),
      all(z.converged for z in s),maximum(z.primal_residual for z in s),maximum(z.dual_residual for z in s)),','));flush(io)
    if round==1
      open(sio->foreach(z->write(sio,z.x),s),joinpath(out,"coefficients.bin"),"w")
      open(sio->TOML.print(sio,Dict("objectives"=>[z.objective for z in s],"df"=>[z.df for z in s],
        "iterations"=>[z.iterations for z in s],"converged"=>[z.converged for z in s],
        "primal_residual"=>[z.primal_residual for z in s],"dual_residual"=>[z.dual_residual for z in s])),joinpath(out,"fit.toml"),"w")
    end
  end
end
meta["peak_after_timing"]=Sys.maxrss();meta["current_after_timing"]=rss()
profile_delay = n >= 1_000_000 || (m["path"] > 0 && m["order"] == 2) ? 0.005 : 0.001
Profile.init(n=6_000_000,delay=profile_delay)
meta["cpu_profile_delay"] = profile_delay
Profile.clear();Profile.@profile fit()
open(joinpath(out,"cpu.txt"),"w") do io
  println(io,"raw_stack_words=",length(Profile.fetch()))
  Profile.print(io;format=:flat,sortedby=:count)
end
Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=.01 fit()
allocs=Profile.Allocs.fetch().allocs
open(joinpath(out,"alloc.txt"),"w") do io
  println(io,"sampled_allocations=",length(allocs))
  for a in sort(collect(allocs);by=z->z.size,rev=true)[1:min(30,length(allocs))]
    println(io,"bytes=",a.size," type=",a.type," stack=",a.stacktrace)
  end
end
meta["whole_worker_peak_rss"]=Sys.maxrss()
open(s->TOML.print(s,meta),joinpath(out,"metadata.toml"),"w")
