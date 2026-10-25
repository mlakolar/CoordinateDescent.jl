using Documenter, Literate, HighDimensionalModels

# The examples become pages, run when the documentation is built. The first
# comment line of an example is its title; a section marker "# --- title ---"
# and a printed header println("== title") become headings, and the line that
# says how to run the script is dropped.
const EXAMPLES = joinpath(@__DIR__, "..", "examples")
const GENERATED = joinpath(@__DIR__, "src", "examples")
const TITLE = r"\A# (.+)\n"
const SECTION = r"^# -+ (.+?) -*$"m
const PRINTED = r"^println\(\"(?:\\n)?== (.+)\"\)$"m
const RUNLINE = r"^#\s+julia --project=examples .*\n"m
function preprocess(s)
    s = replace(s, TITLE => s"# # \1\n", RUNLINE => "")
    s = replace(s, SECTION => m -> "# ## " * uppercasefirst(match(SECTION, m)[1]) * "\n#")
    replace(s, PRINTED => m -> "# ## " * match(PRINTED, m)[1] * "\n#")
end
rm(GENERATED; force = true, recursive = true)
mkpath(GENERATED)
example_pages = Pair{String, String}[]
for file in sort(filter(f -> occursin(r"^\d\d_.*\.jl$", f), readdir(EXAMPLES)))
    title = match(TITLE, read(joinpath(EXAMPLES, file), String))[1]
    Literate.markdown(joinpath(EXAMPLES, file), GENERATED; documenter = true, credit = false, preprocess)
    push!(example_pages, title => joinpath("examples", replace(file, r"\.jl$" => ".md")))
end

makedocs(
    sitename = "HighDimensionalModels.jl",
    authors = "Mladen Kolar",
    modules = [HighDimensionalModels],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mlakolar.github.io/HighDimensionalModels.jl/stable/",
        mathengine = Documenter.KaTeX(),
        size_threshold = 400_000,
    ),
    pages = [
        "Home" => "index.md",
        "The lasso and its tuning-free variants" => "lasso.md",
        "Paths, screening and solver options" => "paths.md",
        "The group lasso" => "group.md",
        "Variable selection with error control" => "selection.md",
        "Fused regression, trend filtering and graphs" => "structured.md",
        "Writing a loss or a penalty" => "extending.md",
        "Varying-coefficient models" => "varying.md",
        "Examples" => example_pages,
        "API reference" => [
            "Estimators" => "api/estimators.md",
            "Solutions" => "api/solutions.md",
            "Coordinate descent, losses and options" => "api/coordinate_descent.md",
            "Varying-coefficient models" => "api/varying.md",
        ],
    ],
    checkdocs = :exports,
)

deploydocs(repo = "github.com/mlakolar/HighDimensionalModels.jl.git", devbranch = "master", push_preview = false)
