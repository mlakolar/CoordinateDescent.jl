# Test the optional extension against registered releases within package compat.
using Pkg, TOML

package_root = dirname(@__DIR__)
project = TOML.parsefile(joinpath(package_root, "Project.toml"))
mktempdir() do environment
    # Develop explicitly: Julia 1.10 does not honor Project.toml [sources].
    open(joinpath(environment, "Project.toml"), "w") do io
        TOML.print(io, Dict(
            "deps" => Dict("Knockoffs" => project["weakdeps"]["Knockoffs"],
                           "Test" => project["extras"]["Test"]),
            "compat" => Dict("Knockoffs" => project["compat"]["Knockoffs"],
                             "Test" => project["compat"]["Test"])))
    end
    Pkg.activate(environment)
    Pkg.develop(path=package_root)
    Pkg.add([Pkg.PackageSpec(name="Knockoffs", uuid=project["weakdeps"]["Knockoffs"]),
             Pkg.PackageSpec(name="Test", uuid=project["extras"]["Test"])])
    Pkg.resolve()
    dependencies = Pkg.dependencies()
    for info in values(dependencies)
        info.name == "HighDimensionalModels" && continue
        (info.is_tracking_path || info.is_tracking_repo) &&
            error("extension tests require registered dependencies: $(info.name)")
    end
    println("Resolved Knockoffs version: ", only(info.version for info in values(dependencies) if info.name == "Knockoffs"))
    Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
    run(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --project=$environment $(joinpath(package_root, "test", "knockoff_extension.jl"))`)
end
