using HighDimensionalModels, ProximalSolvers, ProximalBase, HiGHS, Test, LinearAlgebra, Random, DelimitedFiles
BLAS.set_num_threads(1)
include("ProximalOPT.jl/test/convolution_gap.jl")
include("HighDimensionalModels.jl/test/quantile.jl")
loadbin(f)=open(io->read!(io,Vector{Float64}(undef,filesize(f)÷8)),f)
base="../inputs/wide-1202"
meta=vec(readdlm(base*"-meta.csv",',',Float64));n,p=Int(meta[1]),Int(meta[2])
X=reshape(loadbin(base*"-X.bin"),n,p);y=loadbin(base*"-y.bin");ls=loadbin(base*"-ls.bin")
open("pilot-wide.csv","w") do io
 println(io,"method,seconds,levels,max_kkt,max_abs_rawgap,max_certificate_gap,all_converged,all_native")
 for mode in ("legacy","certified")
  t=@timed QuantileLassoPath(X,y,ls;method=:smooth,bandwidth=.2,tol=2e-6,standardize_x=false,
      options=CDOptions(opt_tol=1e-8,max_iter=10000,gaptol=mode=="legacy" ? 0. : nothing),rng=Xoshiro(20261025))
  ss=t.value.solutions
  println(io,join((mode,t.time,length(ss),maximum(s.kkt_residual for s in ss),maximum(abs(s.duality_gap) for s in ss),maximum(s.certificate_gap for s in ss),all(s.converged for s in ss),all(s.native_converged for s in ss)),','));flush(io)
  mode=="certified" && @test t.value.complete && all(s.converged for s in ss)
 end
end
