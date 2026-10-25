# Run from a package's scripts/instantiate.jl, or pass an environment directory.
using Pkg, TOML

# Temporary companion for PR #84 until ProximalOPT #90 reaches master.
# Only a newly created checkout is changed; existing checkouts remain untouched.
const SLOPE_COMPANION_REV = "3bee2f91604fe08d34a07d45bcc638e3b4face3a"

const WORKSPACE_PACKAGES = Dict(
    "ProximalBase" => ("ProximalBase.jl", "ProximalBase.jl"),
    "ProximalSolvers" => ("ProximalOPT.jl", "ProximalSolvers.jl"),
    "HighDimensionalModels" => ("HighDimensionalModels.jl", "HighDimensionalModels.jl"),
    "CovarianceSelection" => ("CovarianceSelection.jl", "CovarianceSelection.jl"),
    "DiffPrecTest" => ("DiffPrecTest.jl", "DiffPrecTest.jl"),
    "KernelSmoothing" => ("KernelSmoothing.jl", "KernelSmoothing.jl"),
    "FunctionalDataAnalysis" => ("FunctionalDataAnalysis.jl", "FunctionalDataAnalysis.jl"),
    "FunctionalGraphicalModels" => ("FunctionalGraphicalModels.jl", "FunctionalGraphicalModels.jl"),
)

function instantiate_master(environment::AbstractString;
                            workspace::Union{Nothing,AbstractString}=nothing,
                            owner::AbstractString=environment)
    environment = realpath(environment)
    owner = realpath(owner)
    workspace = workspace === nothing ? dirname(owner) : realpath(workspace)
    owner_project = TOML.parsefile(joinpath(owner, "Project.toml"))
    owner_name = get(owner_project, "name", "")
    specs = Dict{String,Pkg.PackageSpec}()
    visiting = Set{String}()
    function visit(project_path)
        project = TOML.parsefile(project_path)
        deps = merge(get(project, "deps", Dict()), get(project, "extras", Dict()))
        for name in sort!(collect(keys(deps)))
            haskey(WORKSPACE_PACKAGES, name) || continue
            name in visiting && continue
            push!(visiting, name)
            folder, repository = WORKSPACE_PACKAGES[name]
            is_owner = name == owner_name
            checkout = is_owner ? owner : joinpath(workspace, folder)
            if !isdir(checkout)
                url = "https://github.com/mlakolar/$repository"
                run(`git clone --branch master --single-branch $url $checkout`)
                if name == "ProximalSolvers" && !isfile(joinpath(checkout, "src", "solvers", "slope.jl"))
                    run(`git -C $checkout fetch origin $SLOPE_COMPANION_REV`)
                    run(`git -C $checkout checkout --detach $SLOPE_COMPANION_REV`)
                end
            end
            branch = strip(read(`git -C $checkout branch --show-current`, String))
            companion = name == "ProximalSolvers" &&
                strip(read(`git -C $checkout rev-parse HEAD`, String)) == SLOPE_COMPANION_REV
            (is_owner || branch == "master" || companion) || error("$name must use its master checkout: $checkout is on $branch")
            if name == "ProximalSolvers" && !isfile(joinpath(checkout, "src", "solvers", "slope.jl"))
                error("SLOPE requires companion PR #90 ($SLOPE_COMPANION_REV) until it reaches master; existing checkout left unchanged: $checkout")
            end
            dependency = TOML.parsefile(joinpath(checkout, "Project.toml"))
            dependency["name"] == name || error("wrong package name at $checkout")
            dependency["uuid"] == deps[name] || error("wrong UUID for $name at $checkout")
            specs[name] = Pkg.PackageSpec(; path=checkout)
            visit(joinpath(checkout, "Project.toml"))
        end
    end
    project_path = joinpath(environment, "Project.toml")
    original = read(project_path, String)
    visit(project_path)
    Pkg.activate(environment)
    try
        if isempty(specs)
            Pkg.resolve()
        else
            Pkg.develop([specs[name] for name in sort!(collect(keys(specs)))])
        end
        Pkg.instantiate()
    finally
        # Transitive master dependencies must not become stale direct dependencies.
        write(project_path, original)
    end
    # Keep extra-only master paths in the manifest for Julia 1.10 test sandboxes.
    # Resolving again after restoring Project.toml would prune those entries.
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: julia instantiate_master.jl PACKAGE_DIRECTORY [WORKSPACE_DIRECTORY]")
    instantiate_master(ARGS[1]; workspace=length(ARGS) > 1 ? ARGS[2] : dirname(realpath(ARGS[1])))
end
