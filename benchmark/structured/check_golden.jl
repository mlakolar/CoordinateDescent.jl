# Same independent golden tests as CI, against fixtures generated in this run.
ENV["CD_STRUCTURED_GOLDEN"] = abspath(ARGS[1])
include(joinpath(@__DIR__, "..", "..", "test", "structured_golden.jl"))
