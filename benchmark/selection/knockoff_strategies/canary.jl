using Pkg, TOML, SHA, LinearAlgebra, Test
BLAS.set_num_threads(1)
root=pwd();version=ARGS[1];mode=ARGS[2];env=joinpath(root,"env-candidate2-"*version)
if mode=="core"
  source=joinpath(dirname(root),"issue99","env-candidate3-"*version)
  ispath(env) && error("immutable candidate environment already exists")
  cp(source,env)
  # Julia1.13 source declarations must agree with the manifest/develop target.
  project=TOML.parsefile(joinpath(env,"Project.toml"))
  if haskey(project,"sources")
    project["sources"]["HighDimensionalModels"]=Dict("path"=>joinpath(root,"candidate2"))
    open(io->TOML.print(io,project),joinpath(env,"Project.toml"),"w")
  end
  Pkg.activate(env);Pkg.offline(true)
  Pkg.develop(PackageSpec(path=joinpath(root,"candidate2"));preserve=Pkg.PRESERVE_ALL)
else
  # Separate optional environment; preserve the passed core environment.
  optional=joinpath(root,"env-optional2-"*version)
  ispath(optional) && error("immutable optional environment already exists")
  cp(env,optional);Pkg.activate(optional);Pkg.offline(true)
  Pkg.add(PackageSpec(name="Knockoffs",version="3.1.4");preserve=Pkg.PRESERVE_ALL)
end
using HighDimensionalModels
const H=HighDimensionalModels
@test startswith(dirname(pathof(H)),joinpath(root,"candidate2"))
if mode=="core"
  @testset "core release/LTS selection and construction" begin
    for file in ("selection.jl","slope_sparse.jl","selection_golden.jl","knockoff_construction.jl")
      include(joinpath(root,"candidate2","test",file))
    end
  end
else
  @testset "optional extension release/LTS" begin
    include(joinpath(root,"candidate2","test","knockoff_extension.jl"))
  end
end
metadata=Dict("julia"=>string(VERSION),"mode"=>mode,"julia_threads"=>Threads.nthreads(),
  "blas_threads"=>BLAS.get_num_threads(),"cpu"=>Sys.cpu_info()[1].model,
  "knockoffs_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoffs.jl")))),
  "construction_source_sha256"=>bytes2hex(sha256(read(joinpath(dirname(pathof(H)),"knockoff_construction.jl")))))
if mode=="optional"
  backend=Base.get_extension(H,:HighDimensionalModelsKnockoffsExt).Knockoffs
  @test pkgversion(backend)==v"3.1.4"
  metadata["Knockoffs"]=string(pkgversion(backend))
  metadata["backend_utilities_sha256"]=bytes2hex(sha256(read(joinpath(dirname(pathof(backend)),"utilities.jl"))))
end
open(io->TOML.print(io,metadata),joinpath(root,"results","$mode-$version.toml"),"w")
