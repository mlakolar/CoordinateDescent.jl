module SolverRegressions
using Test
for name in ("convolution_gap","glm","gap_stopping","duality_gap","newton_nonconvex","generic_newton")
 println("SOLVER REGRESSION ",name);flush(stdout)
 include("ProximalOPT.jl/test/$name.jl")
end
end
module ModelRegressions
using Test, HighDimensionalModels
for name in ("quantile","glm","glm_penalties")
 println("MODEL REGRESSION ",name);flush(stdout)
 include("HighDimensionalModels.jl/test/$name.jl")
end
end
