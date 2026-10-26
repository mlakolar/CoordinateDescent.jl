include("adapters.jl")
case,out=ARGS;mkpath(out);folder=joinpath("inputs/pilot-v1",case)
a=vec(readdlm(joinpath(folder,"dims.txt")));n,p,q=Int.(a[1:3]);λ=.1a[4]
readarr(name,dims...)=open(f->reshape(read!(f,Vector{Float64}(undef,prod(dims))),dims...),joinpath(folder,name*".bin"))
X,Y=readarr("X",n,p),readarr("Y",n,q)
d=H._lr_data(X,Y,true);g=H._lr_gradient(d);cached=(;d,g)
configs=["native","apg","pb95","po95","pb1_boundary","po1_boundary","pb99_safe","shared","gram","quadtrait","finalreuse","combined","coeffonly","cadence1","cadence5","cadence20"]
matched=String[]
open(joinpath(out,"timings.csv"),"w") do io
 println(io,"case,config,mode,round,seconds,bytes,iterations,converged")
 for config in configs
  fit = () -> controlled(X,Y,λ,config;cached=cached)
  fit();z=nothing
  for rd in 1:3
   GC.gc();tm=@timed fit();z=tm.value
   println(io,join((case,config,"cached",rd,tm.time,tm.bytes,z.iterations,z.converged),','));flush(io)
  end
  @assert z.converged
  old=open(f->reshape(read!(f,Vector{Float64}(undef,p*q)),p,q),joinpath("results/pilot-v1",case,"$config-cached.bin"))
  @assert z.B==old
  push!(matched,config)
  open(f->write(f,z.B),joinpath(out,"$config.bin"),"w")
 end
end
# A direct setup allocation witness verifies that cached calls exclude data/gradient setup.
# Anonymous closure bindings avoid Julia's repeated local named-method overwrite.
controlled(X,Y,λ,"pb95");controlled(X,Y,λ,"pb95";cached=cached)
GC.gc();public_probe=@timed controlled(X,Y,λ,"pb95")
GC.gc();cached_probe=@timed controlled(X,Y,λ,"pb95";cached=cached)
@assert public_probe.value.B==cached_probe.value.B
@assert public_probe.bytes>cached_probe.bytes
open(f->TOML.print(f,Dict("case"=>case,"matched_original_B"=>matched,"public_setup_witness_allocated"=>public_probe.bytes,"cached_setup_witness_allocated"=>cached_probe.bytes,"cpu"=>Sys.cpu_info()[1].model,"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads())),joinpath(out,"metadata.toml"),"w")
