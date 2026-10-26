include("adapters.jl")
case,out=ARGS;mkpath(out);folder=joinpath("inputs/pilot-v1",case)
a=vec(readdlm(joinpath(folder,"dims.txt")));n,p,q=Int.(a[1:3]);lm=a[4];L=a[5]
readarr(name,dims...)=open(f->reshape(read!(f,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,name*".bin"))
X,Y=readarr("X",n,p),readarr("Y",n,q);λ=.1lm
configs=["native","apg","pb95","po95","pb1_boundary","po1_boundary","pb99_safe","shared","gram","quadtrait","finalreuse","combined","coeffonly","cadence1","cadence5","cadence20"]
md=Dict{String,Any}("case"=>case,"rep"=>0,"cpu"=>Sys.cpu_info()[1].model,"julia"=>string(VERSION),"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),"blas"=>string(BLAS.get_config()),"X_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(X)))),"Y_sha256"=>bytes2hex(sha256(reinterpret(UInt8,vec(Y)))))
rows=Any[];failures=Any[];hashes=Dict{String,String}();nativeB=Ref{Any}(nothing)
open(joinpath(out,"timings.csv"),"w") do io
 println(io,"case,config,mode,round,seconds,bytes,iterations,converged")
 for config in configs
  try
   # Public setup and common cached setup are distinct workloads.
   d=H._lr_data(X,Y,true);g=H._lr_gradient(d);cached=(;d,g)
   for mode in ("public","cached")
    cachearg=mode=="public" ? nothing : cached
    fit()=controlled(X,Y,λ,config;cached=cachearg)
    fit();z=nothing
    for rd in 1:3
     GC.gc();tm=@timed fit();z=tm.value
     println(io,join((case,config,mode,rd,tm.time,tm.bytes,z.iterations,z.converged),','));flush(io)
    end
    z.converged || error("native nonconvergence")
    open(f->write(f,z.B),joinpath(out,"$config-$mode.bin"),"w")
    if mode=="public"
     hashes[config]=bytes2hex(sha256(reinterpret(UInt8,vec(z.B))))
     config=="native" && (nativeB[]=copy(z.B))
     if config=="pb95";@assert isapprox(z.B,nativeB[];atol=1e-11,rtol=1e-10);end
    end
    push!(rows,Dict("config"=>config,"mode"=>mode,"iterations"=>z.iterations,"converged"=>z.converged,"rank"=>z.solution===nothing ? -1 : z.solution.rank,"df"=>z.solution===nothing ? -1 : z.solution.df))
   end
   fit()=controlled(X,Y,λ,config)
   Profile.clear();Profile.@profile fit()
   open(joinpath(out,"$config-cpu.txt"),"w") do f
    println(f,"raw_stack_words=",length(Profile.fetch()));Profile.print(f;format=:flat,sortedby=:count)
   end
   Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=.01 fit()
   allocs=Profile.Allocs.fetch().allocs
   open(joinpath(out,"$config-alloc.txt"),"w") do f
    println(f,"sampled_allocations=",length(allocs))
    for a in sort(collect(allocs);by=a->a.size,rev=true)[1:min(15,length(allocs))];println(f,"bytes=",a.size," type=",a.type," stack=",a.stacktrace);end
   end
   if !(config in ("native","apg"))
    counters=Dict{String,Int}();z=controlled(X,Y,λ,config;stats=counters)
    @assert bytes2hex(sha256(reinterpret(UInt8,vec(z.B))))==hashes[config]
    open(f->TOML.print(f,counters),joinpath(out,"$config-counts.toml"),"w")
   end
  catch e
   push!(failures,Dict("config"=>config,"error"=>sprint(showerror,e,catch_backtrace())))
  end
 end
end
# Short path calibration only for the baseline, existing APG and combined proposal.
levels=exp.(range(log(.5lm),log(.02lm);length=8))
for config in ("native","apg","combined")
 try
  pathfit()=begin
   d=H._lr_data(X,Y,true);g=H._lr_gradient(d);cached=(;d,g);start=zeros(p,q);fits=Any[]
   for λ in levels
    z=controlled(X,Y,λ,config;cached,start);push!(fits,z);start=z.B
   end
   fits
  end
  pathfit();z=nothing
  open(joinpath(out,"$config-path-timings.csv"),"w") do io
   println(io,"round,seconds,bytes")
   for rd in 1:3;GC.gc();tm=@timed pathfit();z=tm.value;println(io,join((rd,tm.time,tm.bytes),','));flush(io);end
  end
  @assert all(s->s.converged,z)
  for (j,s) in enumerate(z);open(f->write(f,s.B),joinpath(out,"$config-path-$j.bin"),"w");end
 catch e
  push!(failures,Dict("config"=>config*"-path","error"=>sprint(showerror,e,catch_backtrace())))
 end
end
md["outcomes"]=rows;md["failures"]=failures;md["whole_worker_peak_rss"]=Sys.maxrss()
open(f->TOML.print(f,md),joinpath(out,"metadata.toml"),"w")
