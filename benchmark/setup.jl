using Pkg, TOML

const PACKAGE = "HighDimensionalModels"

# The manifest of an environment directory, by the names Pkg looks for
function manifest_path(environment)
  v = "v$(VERSION.major).$(VERSION.minor)"
  for name in ("JuliaManifest-$v.toml", "Manifest-$v.toml", "JuliaManifest.toml", "Manifest.toml")
    path = joinpath(environment, name)
    isfile(path) && return path
  end
  error("no manifest in $environment")
end

# Every manifest entry except the measured package, by name
function other_entries(path)
  manifest = TOML.parsefile(path)
  haskey(manifest, "deps") || error("$path is not a manifest of format 2")
  Dict(name => entries for (name, entries) in manifest["deps"] if name != PACKAGE)
end

function setup_environment(args)
  length(args) in (2, 3) ||
    error("usage: julia benchmark/setup.jl PACKAGE_ROOT ENV_DIR [REFERENCE_ENV]")
  source, environment = realpath(args[1]), abspath(args[2])
  reference = length(args) == 3 ? manifest_path(realpath(args[3])) : nothing
  project = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))
  package = TOML.parsefile(joinpath(source, "Project.toml"))
  get(package, "name", "") == PACKAGE && get(package, "uuid", "") == project["deps"][PACKAGE] ||
    error("PACKAGE_ROOT must be a $PACKAGE checkout")
  target = joinpath(environment, "Project.toml")
  ispath(target) && error("Refusing an existing ENV_DIR/Project.toml")
  if reference !== nothing
    julia = get(TOML.parsefile(reference), "julia_version", "")
    julia == string(VERSION) ||
      error("REFERENCE_ENV was resolved by Julia $julia; use one made with Julia $VERSION")
  end
  # The explicit develop call resolves the checkout on Julia 1.10, which ignores [sources].
  pop!(project, "sources", nothing)
  mkpath(environment)
  open(target, "w") do io
    TOML.print(io, project)
  end
  Pkg.activate(environment)
  if reference === nothing
    Pkg.develop(path = source)
  else
    # Start from the reference resolution and change only the measured package.
    cp(reference, joinpath(environment, "Manifest.toml"))
    Pkg.develop(path = source; preserve = Pkg.PRESERVE_ALL)
  end
  Pkg.instantiate()
  if reference !== nothing
    expected, actual = other_entries(reference), other_entries(manifest_path(environment))
    changed = sort!([name for name in union(keys(expected), keys(actual))
                     if get(expected, name, nothing) != get(actual, name, nothing)])
    isempty(changed) ||
      error("manifest entries differ from REFERENCE_ENV: $(join(changed, ", "))")
  end
  info = only(p for p in values(Pkg.dependencies()) if p.name == PACKAGE)
  realpath(info.source) == source || error("$PACKAGE resolves to $(info.source), not $source")
  Pkg.status(mode = Pkg.PKGMODE_MANIFEST)
end

if abspath(PROGRAM_FILE) == @__FILE__
  setup_environment(ARGS)
end
