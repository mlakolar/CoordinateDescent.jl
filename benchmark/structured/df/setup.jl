using Pkg, TOML
root = @__DIR__
for source in ("baseline", "candidate")
    envdir = joinpath(root,"env-$(source)-$(VERSION.major).$(VERSION.minor)")
    project = TOML.parsefile(joinpath(root,source,"Project.toml"))
    deps = merge(project["deps"], project["extras"], project["weakdeps"])
    deps["HighDimensionalModels"] = project["uuid"]
    mkpath(envdir)
    open(joinpath(envdir,"Project.toml"),"w") do io
        TOML.print(io,Dict("deps"=>deps))
    end
    Pkg.activate(envdir)
    Pkg.develop([PackageSpec(path=joinpath(root,p)) for p in
        ("ProximalBase.jl","ProximalOPT.jl","KernelSmoothing.jl",source)])
    Pkg.instantiate()
    Pkg.status()
end
