# Examples/docs use this checkout and registered runtime dependencies. Only
# KernelSmoothing, an integration dependency that is not yet in General, is
# fetched by URL at an immutable commit. It is not needed by Pkg.test().
using Pkg, TOML
package_root = dirname(@__DIR__)
environment = isempty(ARGS) ? package_root : abspath(ARGS[1])
Pkg.activate(environment)
if environment == package_root
    Pkg.instantiate()
else
    project_path = joinpath(environment, "Project.toml")
    original = read(project_path, String)
    project = TOML.parse(original)
    # Resolve the integration helper before introducing the unregistered owner.
    # Julia 1.10 ignores [sources], so do not rely on it for this bootstrap.
    delete!(project["deps"], "HighDimensionalModels")
    delete!(get(project, "compat", Dict()), "HighDimensionalModels")
    kernel_source = get(get(project, "sources", Dict()), "KernelSmoothing", nothing)
    delete!(project, "sources")
    try
        open(project_path, "w") do io
            TOML.print(io, project)
        end
        if haskey(project["deps"], "KernelSmoothing")
            kernel_source === nothing && error("missing pinned KernelSmoothing source")
            occursin(r"^[0-9a-f]{40}$", kernel_source["rev"]) ||
                error("KernelSmoothing source must use a full commit SHA")
            Pkg.add(PackageSpec(url=kernel_source["url"], rev=kernel_source["rev"]))
        end
        Pkg.develop(PackageSpec(path=package_root))
        Pkg.instantiate()
    finally
        write(project_path, original)
    end
    Pkg.resolve()
end
for info in values(Pkg.dependencies())
    info.name == "HighDimensionalModels" && continue
    if info.name == "KernelSmoothing"
        info.is_tracking_repo && !info.is_tracking_path ||
            error("KernelSmoothing must use the pinned repository commit")
    else
        (info.is_tracking_path || info.is_tracking_repo) &&
            error("release environment requires registered dependencies: $(info.name)")
    end
end
Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
