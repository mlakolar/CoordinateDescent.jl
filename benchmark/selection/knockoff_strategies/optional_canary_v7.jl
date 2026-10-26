using Pkg, TOML, SHA, LinearAlgebra, Test
BLAS.set_num_threads(1)
root=pwd();version=ARGS[1];env=joinpath(root,"env-optional7-"*version)
ispath(env) && error("immutable optional environment already exists");mkpath(env)
core=TOML.parsefile(joinpath(root,"env-candidate2-"*version,"Manifest.toml"))
paths=Dict("HighDimensionalModels"=>joinpath(root,"candidate2"))
for name in ("ProximalBase","ProximalSolvers");paths[name]=only(core["deps"][name])["path"];end
project=Dict("deps"=>Dict("HighDimensionalModels"=>"c00ed354-b9be-54b8-8a23-c5ac771bc43a",
 "Knockoffs"=>"878bf26d-0c49-448a-9df5-b057c815d613","Test"=>"8dfed614-e22c-5e08-85e1-65c5234f0b40"),
 "compat"=>Dict("Knockoffs"=>"=3.1.4"),"sources"=>Dict(name=>Dict("path"=>path) for (name,path) in paths))
for name in ("ProximalBase","ProximalSolvers")
 project["deps"][name]=only(core["deps"][name])["uuid"]
end
open(io->TOML.print(io,project),joinpath(env,"Project.toml"),"w")
# Fresh minimal release environment: no old stdlib manifest, no unrelated test
# dependencies. Registry dependencies may resolve for this Julia version;
# all scientific benchmarks use the separate pinned 1.10 comparison environment.
Pkg.activate(env);Pkg.offline(false)
Pkg.develop([PackageSpec(path=paths[name]) for name in sort!(collect(keys(paths)))])
using HighDimensionalModels
import Knockoffs
const H=HighDimensionalModels
@test dirname(pathof(H))==joinpath(root,"candidate2","src")
@test pkgversion(Knockoffs)==v"3.1.4"
@testset "optional extension release/LTS" begin
 include(joinpath(root,"candidate2","test","knockoff_extension.jl"))
end
metadata=Dict("julia"=>string(VERSION),"mode"=>"optional7","julia_threads"=>Threads.nthreads(),
 "blas_threads"=>BLAS.get_num_threads(),"cpu"=>Sys.cpu_info()[1].model,"Knockoffs"=>string(pkgversion(Knockoffs)),
 "source_paths"=>paths,"backend_utilities_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(Knockoffs)),"utilities.jl")))),
 "knockoffs_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoffs.jl")))),
 "construction_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoff_construction.jl")))))
open(io->TOML.print(io,metadata),joinpath(root,"optional-v7-results","optional-$version.toml"),"w")
