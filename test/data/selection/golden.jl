# Runs test/selection_golden.jl against the fixtures in ARGS[1]; see README.md.
ENV["CD_SELECTION_GOLDEN"] = ARGS[1]
include(joinpath(@__DIR__, "..", "..", "selection_golden.jl"))
