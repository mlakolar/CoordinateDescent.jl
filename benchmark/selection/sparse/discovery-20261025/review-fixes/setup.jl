using Pkg, TOML
const root=ENV["FIX_ROOT"]
const hdm=joinpath(root,"HighDimensionalModels.jl")
const environment=joinpath(root,"env-$(VERSION.major).$(VERSION.minor)")
d=TOML.parsefile(joinpath(hdm,"Project.toml"))
deps=merge(copy(d["deps"]),d["extras"])
deps[d["name"]]=d["uuid"]
deps["Statistics"]="10745b16-79ce-11e8-11f9-7d13ad32a3b2"
compat=Dict(k=>v for (k,v) in d["compat"] if k=="julia" || haskey(deps,k))
mkpath(environment)
open(io->TOML.print(io,Dict("deps"=>deps,"compat"=>compat)),joinpath(environment,"Project.toml"),"w")
Pkg.offline(true)
Pkg.activate(environment)
Pkg.develop([PackageSpec(path=joinpath(root,p)) for p in
 ("ProximalBase.jl","ProximalOPT.jl","KernelSmoothing.jl","HighDimensionalModels.jl")])
Pkg.instantiate(;allow_autoprecomp=false)
println("REPAIR_ENV_READY version=",VERSION)
