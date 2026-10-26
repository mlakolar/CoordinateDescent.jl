using Pkg, TOML, SHA, LinearAlgebra, Test
BLAS.set_num_threads(1)
root=pwd(); version=ARGS[1]; env=joinpath(root,"env-optional4-"*version)
source=joinpath(dirname(dirname(root)),"20261025-cd82-compare","env")
ispath(env) && error("immutable optional environment already exists")
cp(source,env)
# This isolated historical comparison environment already pins Knockoffs3.1.4
# and Roots2.3.0. Do not relax the passed core environment's unrelated test deps.
project=TOML.parsefile(joinpath(env,"Project.toml"))
project["deps"]["HighDimensionalModels"]=pop!(project["deps"],"CoordinateDescent")
core=TOML.parsefile(joinpath(root,"env-candidate2-"*version,"Manifest.toml"))
paths=Dict("HighDimensionalModels"=>joinpath(root,"candidate2"))
for name in ("ProximalBase","ProximalSolvers")
  paths[name]=only(core["deps"][name])["path"]
end
project["sources"]=Dict(name=>Dict("path"=>path) for (name,path) in paths)
open(io->TOML.print(io,project),joinpath(env,"Project.toml"),"w")
manifest=TOML.parsefile(joinpath(env,"Manifest.toml"))
manifest["deps"]["HighDimensionalModels"]=pop!(manifest["deps"],"CoordinateDescent")
# The historical manifest was resolved on Julia1.10. Replace only its
# stdlib entries with the already validated matching-version core entries.
for (name, entries) in manifest["deps"]
  entry=only(entries)
  if !haskey(entry,"git-tree-sha1") && !haskey(entry,"path") && haskey(core["deps"],name)
    manifest["deps"][name]=core["deps"][name]
  end
end
manifest["julia_version"]=core["julia_version"]
open(io->TOML.print(io,manifest),joinpath(env,"Manifest.toml"),"w")
Pkg.activate(env);Pkg.offline(true)
Pkg.develop([PackageSpec(path=paths[name]) for name in sort!(collect(keys(paths)))];preserve=Pkg.PRESERVE_ALL)
using HighDimensionalModels
import Knockoffs
const H=HighDimensionalModels
@test dirname(pathof(H))==joinpath(root,"candidate2","src")
@test pkgversion(Knockoffs)==v"3.1.4"
@testset "optional extension release/LTS" begin
  include(joinpath(root,"candidate2","test","knockoff_extension.jl"))
end
metadata=Dict("julia"=>string(VERSION),"mode"=>"optional4","julia_threads"=>Threads.nthreads(),
  "blas_threads"=>BLAS.get_num_threads(),"cpu"=>Sys.cpu_info()[1].model,
  "Knockoffs"=>string(pkgversion(Knockoffs)),"source_paths"=>paths,
  "backend_utilities_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(Knockoffs)),"utilities.jl")))),
  "knockoffs_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoffs.jl")))),
  "construction_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoff_construction.jl")))))
open(io->TOML.print(io,metadata),joinpath(root,"optional-v4-results","optional-$version.toml"),"w")
