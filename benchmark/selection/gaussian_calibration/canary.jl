using HighDimensionalModels, Test, LinearAlgebra, SHA, TOML
BLAS.set_num_threads(1)
root=dirname(dirname(pathof(HighDimensionalModels)))
for file in ["selection.jl","slope_sparse.jl","selection_golden.jl","slope_calibration.jl"]
  include(joinpath(root,"test",file))
end
metadata=Dict("julia"=>string(VERSION),"cpu"=>Sys.cpu_info()[1].model,
  "julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
  "source_sha256"=>bytes2hex(sha256(read(joinpath(root,"src","slope.jl")))),
  "export_sha256"=>bytes2hex(sha256(read(joinpath(root,"src","HighDimensionalModels.jl")))))
open(io->TOML.print(io,metadata),joinpath("results","canary-"*string(VERSION)*".toml"),"w")
