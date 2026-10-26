using Pkg,TOML
root=pwd();env=joinpath(root,"env-runtime-baseline-1.10")
ispath(env) && error("immutable baseline comparison environment exists")
cp(joinpath(root,"env-optional3-1.10"),env)
p=TOML.parsefile(joinpath(env,"Project.toml"));p["sources"]["HighDimensionalModels"]=Dict("path"=>joinpath(dirname(root),"issue99","baseline"))
open(io->TOML.print(io,p),joinpath(env,"Project.toml"),"w")
Pkg.activate(env);Pkg.offline(true)
Pkg.develop(PackageSpec(path=joinpath(dirname(root),"issue99","baseline"));preserve=Pkg.PRESERVE_ALL)
using HighDimensionalModels
import Knockoffs
@assert dirname(pathof(HighDimensionalModels))==joinpath(dirname(root),"issue99","baseline","src")
println("RUNTIME_BASELINE_ENV_READY")
