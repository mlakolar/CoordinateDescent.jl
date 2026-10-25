using Pkg, TOML
# Test the exact runtime dependency sources, including in Pkg.test's sandbox.
const root=ENV["RUN"]
const prior="/home1/mkolar/juliaPkg/runs/20261025-cd82-selection/depot/packages"
const fixed=joinpath(root,get(ENV,"SPARSE_TEST_LABEL","test-source"))
mkpath(fixed)
for name in ("src","test","ext","Project.toml")
  cp(joinpath(root,"source",name),joinpath(fixed,name);force=true)
end
project=joinpath(fixed,"Project.toml")
d=TOML.parsefile(project)
d["sources"]["ProximalBase"]=Dict("path"=>joinpath(prior,"ProximalBase/86sFe"))
d["sources"]["ProximalSolvers"]=Dict("path"=>get(ENV,"SPARSE_PS_SOURCE",joinpath(prior,"ProximalSolvers/4JP0K")))
open(io->TOML.print(io,d),project,"w")
Pkg.activate(joinpath(root,get(ENV,"SPARSE_TEST_ENV","testenv-fixed")))
Pkg.develop([PackageSpec(path=d["sources"]["ProximalBase"]["path"]),
             PackageSpec(path=d["sources"]["ProximalSolvers"]["path"]),
             PackageSpec(path=fixed)])
Pkg.test("CoordinateDescent";julia_args=["--depwarn=error"])
