# Same independent golden tests as CI, against the fixtures in ARGS[1].
ENV["CD_STRUCTURED_GOLDEN"] = abspath(ARGS[1])
include(joinpath(@__DIR__, "..", "..", "structured_golden.jl"))
