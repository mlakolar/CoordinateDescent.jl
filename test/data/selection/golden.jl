ENV["CD_SELECTION_GOLDEN"] = ARGS[1]
include(joinpath(@__DIR__, "..", "..", "test", "selection_golden.jl"))
