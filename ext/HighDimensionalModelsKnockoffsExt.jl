module HighDimensionalModelsKnockoffsExt

using HighDimensionalModels, Knockoffs, LinearAlgebra
const H = HighDimensionalModels

for strategy in (:mvr, :maxent)
  @eval function H._knockoff_optimize(R::Matrix{Float64}, ::Val{$(QuoteNode(strategy))}, options::NamedTuple)
    # Do not call the external sampler. The native caller validates the actual
    # mapped-back diagonal and original objective before constructing a draw.
    s = Knockoffs.solve_s(Symmetric(R), $(QuoteNode(strategy)); options...)
    s, (backend=:Knockoffs, backend_version=string(pkgversion(Knockoffs)),
        backend_status=:unavailable, backend_iterations=nothing)
  end
end

end
