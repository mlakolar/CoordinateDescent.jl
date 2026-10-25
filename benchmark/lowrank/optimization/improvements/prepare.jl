using Pkg
root = pwd()
version = ARGS[1]
path = joinpath(root, "env-" * version)
Pkg.activate(path)
Pkg.develop(PackageSpec(path = joinpath(root, "package")))
Pkg.add(PackageSpec(name = "ProximalAlgorithms", version = "0.7.0"))
Pkg.resolve()
Pkg.precompile()
Pkg.status()
using LinearAlgebra
BLAS.set_num_threads(1)
println("Julia ", VERSION, "; BLAS threads ", BLAS.get_num_threads())
include(joinpath(root, "package", "test", "runtests.jl"))
