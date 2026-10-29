######################################################################
# Two-penalty tuning
######################################################################

# Internal opt-in protocol for the package's estimators:
# _grid_problem(f, data...; rng, kwargs...) owns preprocessing and mutable
#   workspaces. It exposes λmax::NTuple{2,T} (marginal zero thresholds) and
#   names::NTuple{2,Symbol}. Each fold constructs a separate problem.
# _grid_fit(problem, λ1, λ2, warm) returns (solution, warm′). Solutions have
#   converged::Bool and implement coef/predict; they may have several fitted
#   components. The layer deep-copies warm before every fit and branch.
# _grid_folds(f, data...; nfolds, folds, rng) returns (folds, K). The fold
#   representation belongs to f: rows, per-task folds, or target-only folds.
# _grid_train(f, data..., folds, k) returns the raw training data as a tuple.
# _grid_score(f, solution, data..., folds, k) returns (loss, weight), with
#   loss/weight the fold's mean validation error. Weights must be positive,
#   finite and independent of the penalty pair. Task weighting belongs here.
# No coefficient layout, fold layout or validation population is assumed by
# the layer. In particular, prediction for multi-component solutions and
# target-only validation never require changes to these constructors.

"""
    PenaltyGrid(f, data...; λ1 = nothing, λ2 = nothing, nλ = (10, 10),
                λminratio = (0.01, 0.01), kwargs...)

Fit an estimator implementing the internal two-penalty protocol, currently
[`sparse_nuclear_regression`](@ref). `solutions[i,j]` uses levels `λ1[i]`
and `λ2[j]`, whose meanings are recorded in `names` (`(:rank, :sparse)`).
Explicit axes are nonempty, decreasing, finite and nonnegative; zeros and
repeated levels are allowed. Automatic axes descend logarithmically from
the respective marginal zero thresholds times `1 + √eps`.

Fits proceed down columns, starting each column from the head of the
previous one. Warm states and returned solutions are independent.
`coef` and `predict` return matrices of results over the grid.
"""
struct PenaltyGrid{T<:AbstractFloat,S}
  solutions::Matrix{S}
  λ1::Vector{T}
  λ2::Vector{T}
  names::NTuple{2,Symbol}
end

function _grid_axis(::Type{T}, axis, λmax, nλ, ratio) where {T}
  if axis === nothing
    nλ isa Integer && nλ > 0 || throw(ArgumentError("nλ entries must be positive integers"))
    ratio isa Real && isfinite(ratio) && 0 < ratio <= 1 ||
      throw(ArgumentError("λminratio entries must be in (0, 1]"))
    λs = T[λmax * (1 + sqrt(eps(T))) * ratio^(k/max(1, nλ-1)) for k in 0:nλ-1]
  else
    values = collect(axis)
    all(x -> x isa Real && isfinite(x) && x >= 0, values) ||
      throw(ArgumentError("penalty axes must be finite and nonnegative"))
    λs = T.(values)
  end
  isempty(λs) && throw(ArgumentError("penalty axes must be nonempty"))
  all(x -> isfinite(x) && x >= 0, λs) || throw(ArgumentError("penalty axes must be finite and nonnegative"))
  issorted(λs; rev = true) || throw(ArgumentError("penalty axes must be decreasing"))
  λs
end

function _grid_solutions(problem, λ1, λ2)
  head, warmhead = _grid_fit(problem, first(λ1), first(λ2), nothing)
  solutions = Matrix{typeof(head)}(undef, length(λ1), length(λ2))
  solutions[1, 1] = head
  for j in eachindex(λ2)
    if j > 1
      head, warmhead = _grid_fit(problem, first(λ1), λ2[j], deepcopy(warmhead))
      solutions[1, j] = head
    end
    warm = deepcopy(warmhead)
    for i in 2:length(λ1)
      solution, warm = _grid_fit(problem, λ1[i], λ2[j], deepcopy(warm))
      solutions[i, j] = solution
    end
  end
  solutions
end

function PenaltyGrid(f, data...; λ1 = nothing, λ2 = nothing, nλ = (10, 10),
                     λminratio = (0.01, 0.01), rng::AbstractRNG = default_rng(), kwargs...)
  length(nλ) == 2 || throw(ArgumentError("nλ must have two entries"))
  length(λminratio) == 2 || throw(ArgumentError("λminratio must have two entries"))
  problem = _grid_problem(f, data...; rng, kwargs...)
  T = float(promote_type(map(typeof, problem.λmax)...))
  a = _grid_axis(T, λ1, problem.λmax[1], nλ[1], λminratio[1])
  b = _grid_axis(T, λ2, problem.λmax[2], nλ[2], λminratio[2])
  PenaltyGrid(_grid_solutions(problem, a, b), a, b, problem.names)
end

"""
    CVPenaltyGrid(f, data...; nfolds = 5, folds = nothing,
                  rng = Random.default_rng(), threads = true, kwargs...)

K-fold validation of [`PenaltyGrid`](@ref). Every training fold refits
preprocessing on its own observations, using the numerical axes of the
full-data grid. Each fold has its own seeded generator and workspaces.
For [`sparse_nuclear_regression`](@ref), `cvm` is mean squared prediction
error per response, weighted by held-out row counts. `cvse` is
`sqrt(∑ₖ wₖ (errorₖ - cvm)^2 / (∑ₖ wₖ (K-1)))`.

A pair is eligible only if the full-data fit and every fold fit converge;
otherwise `cvm` and `cvse` are `NaN`. An empty eligible set raises an error.
`index` selects the minimum `cvm`, breaking ties in column-major order;
`λmin = (grid.λ1[index[1]], grid.λ2[index[2]])`. There is no two-dimensional
one-standard-error rule. `coef` and `predict` use this full-data solution.
"""
struct CVPenaltyGrid{T<:AbstractFloat,G,F}
  grid::G
  cvm::Matrix{T}
  cvse::Matrix{T}
  index::CartesianIndex{2}
  λmin::NTuple{2,T}
  folds::F
end

function CVPenaltyGrid(f, data...; nfolds::Integer = 5, folds = nothing,
                       rng::AbstractRNG = default_rng(), threads::Bool = true, kwargs...)
  fl, K = _grid_folds(f, data...; nfolds, folds, rng)
  seeds = rand(rng, UInt64, K + 1)
  grid = PenaltyGrid(f, data...; rng = Xoshiro(seeds[end]), kwargs...)
  T = eltype(grid.λ1)
  errors = fill(T(NaN), length(grid.λ1), length(grid.λ2), K)
  weights = zeros(T, K)
  function fold!(k)
    training = _grid_train(f, data..., fl, k)
    opts = merge((; kwargs...), (; λ1 = grid.λ1, λ2 = grid.λ2))
    fitted = PenaltyGrid(f, training...; rng = Xoshiro(seeds[k]), opts...)
    for j in CartesianIndices(grid.solutions)
      loss, weight = _grid_score(f, fitted.solutions[j], data..., fl, k)
      wk = T(weight)
      isfinite(wk) && wk > 0 || throw(ArgumentError("validation weights must be finite and positive"))
      if j == first(CartesianIndices(grid.solutions))
        weights[k] = wk
      else
        wk == weights[k] || throw(ArgumentError("validation weights must not depend on penalty levels"))
      end
      if fitted.solutions[j].converged && grid.solutions[j].converged
        errors[j, k] = loss / wk
      end
    end
  end
  _runfolds(fold!, K, threads)
  cvm = fill(T(NaN), size(grid.solutions))
  cvse = similar(cvm)
  fill!(cvse, T(NaN))
  total = sum(weights)
  for j in CartesianIndices(cvm)
    e = [errors[j, k] for k in 1:K]
    all(isfinite, e) || continue
    cvm[j] = dot(weights, e) / total
    cvse[j] = sqrt(sum(weights .* (e .- cvm[j]).^2) / (total*(K-1)))
  end
  eligible = findall(isfinite, cvm)
  isempty(eligible) && throw(ErrorException("no penalty pair converged on the full data and every training fold"))
  index = eligible[argmin(cvm[eligible])]
  CVPenaltyGrid(grid, cvm, cvse, index, (grid.λ1[index[1]], grid.λ2[index[2]]), fl)
end

function _grid_folds(::typeof(sparse_nuclear_regression), X, Y; nfolds, folds, rng)
  size(X, 1) > 0 || throw(ArgumentError("X must have nonempty dimensions"))
  fl, K = _foldlabels(rng, size(X, 1), nfolds, folds)
  all(>(0), fl) || throw(ArgumentError("fold labels must be positive"))
  fl, K
end
function _grid_train(::typeof(sparse_nuclear_regression), X, Y, folds, k)
  tr = findall(!=(k), folds)
  (X[tr, :], Y[tr, :])
end
function _grid_score(::typeof(sparse_nuclear_regression), solution, X, Y, folds, k)
  te = findall(==(k), folds)
  (sum(abs2, Y[te, :] - predict(solution, X[te, :])) / size(Y, 2), length(te))
end

StatsAPI.coef(p::PenaltyGrid) = coef.(p.solutions)
StatsAPI.predict(p::PenaltyGrid, X) = map(s -> predict(s, X), p.solutions)
StatsAPI.coef(cv::CVPenaltyGrid) = coef(cv.grid.solutions[cv.index])
StatsAPI.predict(cv::CVPenaltyGrid, X) = predict(cv.grid.solutions[cv.index], X)
StatsAPI.fit(::Type{PenaltyGrid}, f, data...; kwargs...) = PenaltyGrid(f, data...; kwargs...)
StatsAPI.fit(::Type{CVPenaltyGrid}, f, data...; kwargs...) = CVPenaltyGrid(f, data...; kwargs...)
