include("adapters.jl")
case,config,out=ARGS;mkpath(out);folder=joinpath("inputs/pilot-v1",case)
a=vec(readdlm(joinpath(folder,"dims.txt")));n,p,q=Int.(a[1:3]);λ=.1a[4]
readarr(name,dims...)=open(f->reshape(read!(f,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,name*".bin"))
X,Y=readarr("X",n,p),readarr("Y",n,q)
fit()=controlled(X,Y,λ,config)
fit();result=Ref{Any}(nothing)
open(joinpath(out,"timings.csv"),"w") do io
 println(io,"case,config,mode,round,seconds,bytes,iterations,converged")
 for rd in 1:3
  GC.gc();tm=@timed fit();z=tm.value;result[]=z
  println(io,join((case,config,"public",rd,tm.time,tm.bytes,z.iterations,z.converged),','));flush(io)
 end
end
z=result[]
@assert z.converged
old=open(f->reshape(read!(f,Vector{Float64}(undef,p*q)),p,q),joinpath("results/pilot-v1",case,"$config-public.bin"))
@assert z.B==old
open(f->write(f,z.B),joinpath(out,"B.bin"),"w")
Profile.clear();Profile.@profile for _ in 1:10;fit();end
open(joinpath(out,"cpu.txt"),"w") do io
 println(io,"raw_stack_words=",length(Profile.fetch()));Profile.print(io;format=:flat,sortedby=:count)
end
Profile.Allocs.clear();Profile.Allocs.@profile sample_rate=.01 fit()
allocs=Profile.Allocs.fetch().allocs
open(joinpath(out,"alloc.txt"),"w") do io
 println(io,"sampled_allocations=",length(allocs))
 for a in sort(collect(allocs);by=a->a.size,rev=true)[1:min(15,length(allocs))];println(io,"bytes=",a.size," type=",a.type," stack=",a.stacktrace);end
end
open(f->TOML.print(f,Dict("case"=>case,"config"=>config,"matched_original_B"=>true,"cpu"=>Sys.cpu_info()[1].model,"blas_threads"=>BLAS.get_num_threads(),"julia_threads"=>Threads.nthreads(),"whole_worker_peak_rss"=>Sys.maxrss())),joinpath(out,"metadata.toml"),"w")
