# Keep optional Knockoffs' Roots2 requirement separate from unrelated test deps.
using Pkg, TOML
include(joinpath(@__DIR__, "master_dependencies.jl"))
package_root=dirname(@__DIR__)
project=TOML.parsefile(joinpath(package_root,"Project.toml"))
environment=joinpath(package_root,".master-knockoff-env","v$(VERSION.major).$(VERSION.minor)")
mkpath(environment)
deps=Dict("HighDimensionalModels"=>project["uuid"],
          "Knockoffs"=>project["weakdeps"]["Knockoffs"],
          "Test"=>project["extras"]["Test"])
open(joinpath(environment,"Project.toml"),"w") do io
  TOML.print(io,Dict("deps"=>deps,"compat"=>Dict("Knockoffs"=>"=3.1.4"),
    "sources"=>Dict("HighDimensionalModels"=>Dict("path"=>package_root))))
end
instantiate_master(environment;workspace=dirname(package_root),owner=package_root)
run(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$environment $(joinpath(package_root,"test","knockoff_extension.jl"))`)
