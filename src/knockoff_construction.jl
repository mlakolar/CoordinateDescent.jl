"""
    KnockoffDiagonal

An owned knockoff diagonal `s` in covariance/Gram units, its `strategy`, and
`diagnostics`. It owns no covariance or sampling-factor cache. Pass `s=plan.s`
to use this vector explicitly with a constructor; that use is recorded as
`:user`. Optimizer iterations/status are unavailable from the optional backend;
the independent stationarity check is recorded separately.
"""
struct KnockoffDiagonal{T}
  s::Vector{T}
  strategy::Symbol
  diagnostics::NamedTuple
end

function _knockoff_strategy(strategy, s, options)
  options isa NamedTuple || throw(ArgumentError("construction_options must be a NamedTuple"))
  actual = strategy === nothing ? (s === nothing ? :equicorrelated : :user) : strategy
  actual in (:equicorrelated, :mvr, :maxent, :user) ||
    throw(ArgumentError("strategy must be :equicorrelated, :mvr, :maxent or :user"))
  if actual === :user
    s isa AbstractVector || throw(ArgumentError("strategy=:user requires an s vector"))
    isempty(options) || throw(ArgumentError("user diagonals do not accept construction_options"))
  else
    s === nothing || throw(ArgumentError("supplied s conflicts with strategy=$actual; use :user"))
    allowed = actual === :equicorrelated ? (:safety,) : (:niter, :tol, :λmin, :robust, :verbose, :optimality_tol)
    all(k -> k in allowed, keys(options)) || throw(ArgumentError("unknown construction option for $actual"))
  end
  actual
end

function _knockoff_optimize(R, strategy::Val, options::NamedTuple)
  throw(ArgumentError("MVR/maxent strategies require loading the optional Knockoffs package with `using Knockoffs`"))
end

function _knockoff_optimizer_options(options)
  niter = get(options, :niter, 1000)
  niter isa Integer && 0 < niter <= typemax(Int) || throw(ArgumentError("niter must be a positive Int"))
  tol, floor = Float64(get(options, :tol, 1e-10)), Float64(get(options, :λmin, 1e-10))
  isfinite(tol) && tol > 0 || throw(ArgumentError("optimizer tol must be finite and positive"))
  isfinite(floor) && floor >= 0 || throw(ArgumentError("λmin must be finite and nonnegative"))
  robust, verbose = get(options, :robust, false), get(options, :verbose, false)
  robust isa Bool && verbose isa Bool || throw(ArgumentError("robust and verbose must be Bool"))
  (; niter=Int(niter), tol, λmin=floor, robust, verbose)
end

function _knockoff_stationarity(R, a, strategy)
  all(isfinite, a) && all(>(0), a) || error("optimizer returned a nonpositive or nonfinite diagonal")
  M = Symmetric(2R - Diagonal(a))
  F = cholesky(M; check=false)
  issuccess(F) || error("optimizer returned a diagonal without a finite interior objective")
  Q = F \ Matrix{Float64}(I, length(a), length(a))
  left, right, objective = if strategy === :mvr
    (vec(sum(abs2, Q; dims=1)), inv.(a).^2, tr(Q) + sum(inv, a))
  else
    (diag(Q), inv.(a), -2sum(log, diag(F.U)) - sum(log, a))
  end
  all(isfinite, left) && all(isfinite, right) && isfinite(objective) ||
    error("optimizer objective/certificate is nonfinite")
  residual = maximum(abs.(left .- right) ./ max.(1.0, left, right))
  (; objective, stationarity_relative=residual, stationarity_absolute=norm(left .- right, Inf))
end

function _knockoff_choose_diagonal(S, strategy, s, options)
  actual = _knockoff_strategy(strategy, s, options)
  if actual === :equicorrelated
    safety = get(options, :safety, 1e-8)
    d = equicorrelated_s(S; safety)
    # Retain existing arithmetic and promotion, including Float32 input.
    return KnockoffDiagonal(d, actual, (backend=:native, backend_status=:not_applicable,
      options=(; safety), certificate=:scale_relative_feasibility))
  elseif actual === :user
    return KnockoffDiagonal(Vector{eltype(S)}(s), actual,
      (backend=:user, backend_status=:not_applicable, options=(;), certificate=:scale_relative_feasibility))
  end
  T = eltype(S)
  T in (Float32, Float64) || throw(ArgumentError("optimized strategies support Float32/Float64 covariance inputs"))
  opts = _knockoff_optimizer_options(options)
  threshold = Float64(get(options, :optimality_tol, max(1e-6, 100eps(T))))
  isfinite(threshold) && 0 < threshold < 1 || throw(ArgumentError("optimality_tol must be finite and in (0, 1)"))
  diagonal = Float64.(diag(S))
  scales = sqrt.(diagonal)
  R = Float64.(S) ./ (scales * scales')
  raw, backend = _knockoff_optimize(copy(R), Val(actual), opts)
  raw isa AbstractVector && length(raw) == size(S, 1) || error("optimizer returned an invalid diagonal dimension")
  all(isfinite, raw) && all(>(0), raw) || error("optimizer returned a nonpositive/nonfinite diagonal")
  d = T.(raw .* diagonal)
  # Certify the actual returned, rounded vector against the ORIGINAL objective,
  # not the backend's λmin perturbation or a change-in-iterate stopping rule.
  certificate = _knockoff_stationarity(R, Float64.(d) ./ diagonal, actual)
  certificate.stationarity_relative <= threshold ||
    error("$actual diagonal failed original-objective stationarity: $(certificate.stationarity_relative) > $threshold")
  KnockoffDiagonal(d, actual, merge(backend, certificate,
    (options=opts, certificate=:original_objective_stationarity, optimality_tol=threshold,
     stationarity_passed=true, optimizer_precision=:Float64, returned_precision=Symbol(string(T)))))
end

function _knockoff_construction_info(plan)
  (strategy=plan.strategy, s=copy(plan.s),
   diagnostics=merge(plan.diagnostics, (feasibility=:validated,)))
end

"""
    knockoff_diagonal(Sigma; strategy=nothing, s=nothing, construction_options=(;))

Construct and validate an owned diagonal. Missing `strategy` selects
`:equicorrelated`, or `:user` when `s` is supplied. Explicit `:user` requires s;
other strategies forbid it. `:mvr` and `:maxent` use the optional Knockoffs
extension for diagonal optimization only; sampling remains native. Options
are `niter`, `tol`, `λmin`, `robust`, `verbose`, `optimality_tol`; equicorrelation
accepts only `safety`. They are separate from lasso `CDOptions`.

Optimized diagonals use Float64 correlation-scale optimization, return input
precision in covariance units, and must pass original-objective stationarity
and the native scale-relative joint/conditional PSD checks. Failure raises an
error without substituting another strategy. The backend does not expose
actual iterations or a convergence flag; diagnostics label this as unavailable.
Any feasible diagonal still needs the model/statistic assumptions for FDR.
No estimated covariance, fitted statistic, or sampling-factor cache is added.
"""
function knockoff_diagonal(Sigma::AbstractMatrix{<:Real}; strategy=nothing, s=nothing,
                           construction_options=(;))
  _knockoff_strategy(strategy, s, construction_options)
  S = _knockoff_covariance(Sigma)
  plan = _knockoff_choose_diagonal(S, strategy, s, construction_options)
  _knockoff_parameters_from_diagonal(S, plan.s)
  KnockoffDiagonal(plan.s, plan.strategy, merge(plan.diagnostics, (feasibility=:validated,)))
end
