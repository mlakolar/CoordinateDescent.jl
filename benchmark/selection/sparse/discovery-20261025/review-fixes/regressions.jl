using Test, LinearAlgebra, HighDimensionalModels, ProximalBase, ProximalSolvers
BLAS.set_num_threads(1)
const root=ENV["FIX_ROOT"]
@testset "Review regressions, replication $(ARGS[1]), Julia $VERSION" begin
  for name in ("selection","slope_sparse","selection_golden")
    include(joinpath(root,"HighDimensionalModels.jl","test",name*".jl"))
  end
  include(joinpath(root,"ProximalOPT.jl","test","apg_numerical_failure.jl"))
  include(joinpath(root,"ProximalOPT.jl","test","slope.jl"))
end
println("REPAIR_REGRESSIONS_PASS replication=",ARGS[1]," version=",VERSION)
