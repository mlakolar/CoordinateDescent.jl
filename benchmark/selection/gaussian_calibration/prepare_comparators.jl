using Pkg
# Only the isolated comparator environment is mutable. Production source and the
# passed release/LTS environments remain frozen for the correctness receipts.
Pkg.activate("env-comparison-1.10")
Pkg.offline(true)
Pkg.add(PackageSpec(name="SLOPE",version="1.3.1");preserve=Pkg.PRESERVE_ALL)
Pkg.status(;mode=Pkg.PKGMODE_MANIFEST)
using HighDimensionalModels, SLOPE, LinearAlgebra, SHA, TOML
BLAS.set_num_threads(1)
@assert pkgversion(SLOPE)==v"1.3.1"
source=joinpath(dirname(pathof(SLOPE)),"regweights.jl")
metadata=Dict("julia"=>string(VERSION),"SLOPE"=>string(pkgversion(SLOPE)),
  "regweights_source_sha256"=>bytes2hex(sha256(read(source))),
  "candidate_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(HighDimensionalModels)),"slope.jl")))),
  "cpu"=>Sys.cpu_info()[1].model,"blas_threads"=>BLAS.get_num_threads(),"julia_threads"=>Threads.nthreads())
mkpath("results/comparator-preparation-v1")
open(io->TOML.print(io,metadata),"results/comparator-preparation-v1/metadata.toml","w")
