using Pkg,TOML
root=ENV["RUN"];fixed=joinpath(root,"test-source")
mkpath(fixed)
for name in ("src","test","ext","Project.toml")
  cp(joinpath(root,"source",name),joinpath(fixed,name);force=true)
end
d=TOML.parsefile(joinpath(fixed,"Project.toml"))
d["sources"]["ProximalBase"]=Dict("path"=>joinpath(root,"pb-source"))
d["sources"]["ProximalSolvers"]=Dict("path"=>joinpath(root,"ps-source"))
open(io->TOML.print(io,d),joinpath(fixed,"Project.toml"),"w")
Pkg.activate(joinpath(root,"testenv"))
Pkg.develop([PackageSpec(path=joinpath(root,"pb-source")),
             PackageSpec(path=joinpath(root,"ps-source")),PackageSpec(path=fixed)])
Pkg.test("CoordinateDescent";julia_args=["--depwarn=error"])
