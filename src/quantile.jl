# Quantile regression keeps the optimizer's observation-level scores intact.
# A separate feasible vector is used for lower bounds, never for inference.

"""
    QuantileLassoSolution

A quantile-lasso fit on the original columns of `X`. `β`, `intercept`,
`residuals`, `λ` and the effective `penalty_factors` describe the fitted problem.
`dual` contains the raw observation-level scores; see [`dual_scores`](@ref).
`certificate_dual` is a separate feasible vector used only
for lower bounds. No rounding or projection is applied to `dual` or `β`.

`objective` is the fitted (exact or smoothed) penalized objective;
`check_objective` always uses the check loss. `duality_gap` uses the raw dual,
whereas `certificate_gap` uses the feasible certificate. `kkt_residual`,
`dual_feasibility` and `complementarity` check the returned primal–dual pair.
`converged` uses those numerical checks at the requested tolerance;
`native_converged` separately records the backend's stopping flag. `status`
retains its termination reason. Duals can be nonunique at zero residuals.
"""
struct QuantileLassoSolution
  β::Vector{Float64}
  intercept::Float64
  residuals::Vector{Float64}
  dual::Vector{Float64}
  certificate_dual::Vector{Float64}
  penalty_factors::Vector{Float64}
  λ::Float64
  tau::Float64
  method::Symbol
  bandwidth::Float64
  hasintercept::Bool
  objective::Float64
  check_objective::Float64
  dual_objective::Float64
  duality_gap::Float64
  certificate_gap::Float64
  kkt_residual::Float64
  dual_feasibility::Float64
  complementarity::Float64
  status::Symbol
  iterations::Int
  converged::Bool
  native_converged::Bool
end

"""
    QuantileLassoPath(X, y, λpath; tau=0.5, method=:exact, max_hat_s=Inf, kwargs...)
    QuantileLassoPath(X, y; nλ=100, λminratio=0.01, kwargs...)

Fit `mean(rho_tau(y - intercept - X*β - offset)) + λ*sum(w .* abs.(β))`
along a nonincreasing, finite, nonnegative grid. Every saved level retains
both coefficients and raw dual scores, including at zero residuals.
`path[k]` is a [`QuantileLassoSolution`](@ref); `path.λpath` contains the
returned levels and `requested_λpath` the requested grid. `complete` is false
when a limit stops the path. There is no coefficient or dual interpolation:
`coef(path, λ)` requires a saved level.

`max_hat_s=Inf` limits the number of nonzero slopes. It must be a nonnegative
real number other than NaN. After saving the first fit with
`count(!iszero, β) > max_hat_s`, later levels are skipped. The intercept does
not count; nonzero slopes with zero penalty factors do. The exceeding fit is
retained, as in `LassoPath` and `GLMLassoPath`; QR.jl dropped it. If levels are
skipped, `complete=false` and `status=:max_hat_s`. Exceeding the limit only at
the last requested level leaves `complete=true` and the usual status.

`method=:exact` uses HiGHS dual simplex with an analytic quantile starting
basis and reuses that basis/model along the path. Load `HiGHS` to activate the
optional extension. `initial_basis=:default` disables the analytic start;
`simplex=:primal` selects primal simplex. `time_limit` limits the entire exact
path between levels and passes the remaining time to HiGHS. Preparation and
diagnostics can overrun that limit, and automatic-grid preparation precedes
the path timer. A stopped path retains available
primal–dual results and the termination status; it never fills missing levels.
The support limit is checked only after an `OPTIMAL` level with a saved pair.
Non-`OPTIMAL` termination or a missing pair takes precedence. The path time budget
is checked before each level; a support stop after an optimal level takes
precedence over a budget exhausted during that level.

`method=:smooth` uses the existing proximal Newton path with the Epanechnikov
convolution loss at explicit `bandwidth` (default 0.2). Its raw dual scores are
the smoothed loss scores, not exact check-loss duals. `time_limit` is currently
supported only by the exact backend. `options`, `newton_tol`,
`newton_max_iter`, `screening` and `rng` control proximal Newton.

The intercept is unpenalized (`intercept=true`). `penalty_factors` are finite
nonnegative slope weights, one by default. With `standardize_x=true` (the
default), they are multiplied by column population standard deviations,
centered when fitting an intercept. Coefficients and duals remain on the
original data scale. Zero factors and λ=0 are supported on explicit grids;
the automatic grid requires positive effective factors. `offset` is fixed and
unpenalized. Observation weights and formula fits are not implemented here.
Inputs are converted to Float64; sparse designs remain sparse.

The automatic grid starts at the intercept-only score maximum, inflated for
roundoff, and ends at `λminratio` times that maximum. This is a numerical path,
not pivotal calibration or cross-validation; use [`quantile_lambda`](@ref) or
[`CVQuantileLassoPath`](@ref) for those choices. `tol=2e-5` is the external KKT and
relative raw-gap tolerance. Backends supporting convolution certificates also
use it during smoothed fitting, honoring a smaller positive `options.gaptol`.
An explicit `options.gaptol=0` keeps the legacy objective-change rule; older
backends retain their original stopping and are checked afterward.

For positive λ and factors, the certificate projects scores onto their box
and intercept constraint, then contracts to meet the slope constraints. With
zero λ or any zero factor it uses the feasible zero vector, whose bound may be
loose; numerical convergence still checks the raw pair. No inferential
validity is asserted merely from solver convergence.
"""
struct QuantileLassoPath
  λpath::Vector{Float64}
  solutions::Vector{QuantileLassoSolution}
  requested_λpath::Vector{Float64}
  complete::Bool
  status::Symbol
  nobs::Int
  nvars::Int
  hasintercept::Bool
  method::Symbol
end

"""
    quantile_lasso(X, y, λ; kwargs...)

Fit one level of [`QuantileLassoPath`](@ref), returning a
[`QuantileLassoSolution`](@ref). A backend limit with no available primal–dual
pair throws an error; an available but inaccurate pair has `converged=false`.
"""
function quantile_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real; kwargs...)
  path = QuantileLassoPath(X, y, [λ]; kwargs...)
  isempty(path.solutions) && error("quantile solver returned no primal–dual pair ($(path.status))")
  only(path.solutions)
end

"""
    dual_scores(sol::QuantileLassoSolution)
    dual_scores(path::QuantileLassoPath)
    dual_scores(r::QuantileRefit)

Raw observation scores `u`. At a valid solution, `tau-1 ≤ uᵢ ≤ tau`.
A stopped fit can retain an infeasible iterate; check `converged` and its
numerical diagnostics before using it. At an exact solution,
`uᵢ=tau` for a positive residual and `tau-1` for a negative residual; scores
at zero residuals come from the optimizer, not from a residual-sign rule.
With an intercept, `sum(u)=0`; slope stationarity is
`X'u/n ∈ λ*w .* ∂abs(β)`. For a smooth fit these are convolution-loss scores.
A whole path returns an n-by-level matrix; `dual_scores(path[k])` returns one vector.
The returned vectors are owned by the saved fits, not by a live optimizer.

QR.jl's historical `getXi` convention is `-dual_scores(sol)/nobs(sol)`, the
dual of `y - a - X*β == rp - rn` with the mean check-loss objective. Apply the
same conversion to a path; transpose it for QR.jl's level-by-observation layout.
This changes only sign and normalization, and always uses the raw scores.
"""
dual_scores(sol::QuantileLassoSolution) = sol.dual
dual_scores(path::QuantileLassoPath) = isempty(path.solutions) ? zeros(path.nobs, 0) : reduce(hcat, dual_scores.(path.solutions))
Base.length(path::QuantileLassoPath) = length(path.solutions)
Base.getindex(path::QuantileLassoPath, k::Integer) = path.solutions[k]

_qcheck(r, τ) = max(τ*r, (τ-1)*r)
function _qconjugate(u, τ, h)
  h == 0 && return 0.0
  q = clamp(u - (τ-0.5), -0.5, 0.5)
  s = 2sin(asin(2q)/3)
  -3h*(1-s*s)^2/16
end

function _qcertificate(X, u, λ, w, τ, intercept)
  v = clamp.(u, τ-1, τ)
  if λ == 0 || any(iszero, w)
    fill!(v, 0)
    return v
  end
  if intercept
    lo, hi = minimum(u)-τ, maximum(u)-(τ-1)
    for _ in 1:60
      c = (lo+hi)/2
      sum(x -> clamp(x-c, τ-1, τ), u) > 0 ? (lo=c) : (hi=c)
    end
    v .= clamp.(u .- (lo+hi)/2, τ-1, τ)
  end
  ratio = maximum(abs.(X'v) ./ (length(v)*λ .* w); init=0.0)
  v ./= max(1.0, ratio*(1+1e-12))
  v
end

function _qsolution(X, y, β, a, u, λ, w, τ, intercept, method, h,
                    status, iterations, native_converged, tol)
  n = length(y); r = y .- a .- X*β
  loss = h == 0 ? nothing : ProximalBase.ConvolutionCheck(τ, h)
  check = sum(x -> _qcheck(x, τ), r)/n + λ*dot(w, abs.(β))
  obj = h == 0 ? check : sum(x -> loss(0.0, x), r)/n + λ*dot(w, abs.(β))
  # ScalarLoss's callable interface is prediction first, response second.
  g = X'u / n
  feasible = max(maximum(max.(τ-1 .- u, u .- τ); init=0.0),
                 intercept ? abs(sum(u))/n : 0.0,
                 maximum(abs.(g) .- λ.*w; init=0.0))
  # A continuous proximal KKT residual: interior-point solvers return tiny
  # nonzero coefficients for inactive slopes. Do not round those coefficients
  # or test their sign as though they were materially active.
  stationarity = maximum((abs(β[j] - sign(β[j]+g[j])*max(abs(β[j]+g[j])-λ*w[j],0.0))
                         for j in eachindex(β)); init=0.0)
  comp = maximum((abs((h == 0 ? _qcheck(r[i], τ) : loss(0.0, r[i])) +
                     _qconjugate(u[i], τ, h)-r[i]*u[i]) for i in 1:n); init=0.0)
  rawdual = dot(y, u)/n - sum(x -> _qconjugate(x, τ, h), u)/n
  v = _qcertificate(X, u, λ, w, τ, intercept)
  bound = dot(y, v)/n - sum(x -> _qconjugate(x, τ, h), v)/n
  kkt = max(feasible, stationarity, comp)
  gap = obj-rawdual
  ok = isfinite(obj) && isfinite(kkt) && kkt <= tol && abs(gap) <= tol*(1+abs(obj))
  QuantileLassoSolution(copy(β), a, r, copy(u), v, copy(w), λ, τ, method, h,
                        intercept, obj, check, rawdual, gap, obj-bound, kkt,
                        feasible, comp, status, iterations, ok, native_converged)
end

function _qdata(X, y, penalty_factors, standardize_x, intercept, offset)
  n, p = size(X)
  n > 0 && p > 0 || throw(ArgumentError("X must have observations and columns"))
  Xf, yf = _qrefit_data(X,y,offset)
  w = penalty_factors === nothing ? ones(p) : Float64.(penalty_factors)
  length(w) == p || throw(DimensionMismatch("penalty_factors must have one entry per column"))
  all(x -> isfinite(x) && x >= 0, w) || throw(ArgumentError("penalty_factors must be finite and nonnegative"))
  standardize_x && (w .*= _loadings(Xf, intercept))
  Xf, yf, w
end

function _qrefit_data(X, y, offset)
  n = size(X,1)
  n > 0 || throw(ArgumentError("X must have observations"))
  length(y) == n || throw(DimensionMismatch("X and y have different observation counts"))
  _check_finite(X, "X"); _check_finite(y, "y")
  Xf = eltype(X) == Float64 ? X : Float64.(X)
  yf = collect(Float64,y)
  if offset !== nothing
    length(offset) == n || throw(DimensionMismatch("offset and y have different lengths"))
    _check_finite(offset, "offset"); yf .-= offset
  end
  Xf, yf
end

# Overridden by the optional HiGHS extension with typed methods.
function _quantile_exact_path(args...; kwargs...)
  throw(ArgumentError("exact quantile regression requires the optional HiGHS backend; run `using HiGHS`"))
end

function _quantile_exact_refit(args...; kwargs...)
  throw(ArgumentError("exact quantile regression requires the optional HiGHS backend; run `using HiGHS`"))
end

function _quantile_smooth_path(Xf, yf, ls, w, tau, intercept, bandwidth;
                               tol, max_hat_s=Inf, options, newton_tol, newton_max_iter, screening, rng)
  # Use the requested observation-score accuracy for Newton's certified
  # stopping too. The backend's feasible certificate and its raw-score guard
  # are distinct; the saved scores always remain the loss gradient.
  loss = ProximalBase.ConvolutionCheck(Float64(tau), Float64(bandwidth))
  # Older solver releases keep their original stopping behavior and still
  # receive the independent post-fit checks below.
  supported = isdefined(ProximalSolvers, :_hasconjugate) && ProximalSolvers._hasconjugate(loss)
  gt = options.gaptol === nothing ? Float64(tol) : min(options.gaptol, Float64(tol))
  smooth_options = supported ? CDOptions(options; gaptol=gt) : options
  path = GLMLassoPath(Xf, yf, ls, loss, smooth_options;
                     intercept, standardize_x=false, penalty=NormL1(1.0, w), devmax=1,
                     max_hat_s, newton_tol, newton_max_iter, screening, rng)
  ss = QuantileLassoSolution[]
  loss = path.loss
  for k in eachindex(path.λpath)
    β, a = Vector(path.βpath[k]), path.β0path[k]
    η = Xf*β .+ a
    u = [-last(ProximalBase.value_and_deriv(loss, η[i], yf[i])) for i in eachindex(yf)]
    push!(ss, _qsolution(Xf, yf, β, a, u, ls[k], w, Float64(tau), intercept, :smooth,
                         Float64(bandwidth), path.converged[k] ? :optimal : :not_converged,
                         path.iterations[k], path.converged[k], Float64(tol)))
  end
  ss, length(ss) < length(ls) ? :max_hat_s : all(s -> s.converged, ss) ? :optimal : :not_converged
end

function QuantileLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λpath::AbstractVector{<:Real};
                          tau::Real=0.5, method::Symbol=:exact, bandwidth::Real=0.2,
                          penalty_factors=nothing, standardize_x::Bool=true, intercept::Bool=true,
                          offset=nothing, tol::Real=2e-5, time_limit::Real=Inf, max_hat_s=Inf,
                          initial_basis::Symbol=:quantile, simplex::Symbol=:dual,
                          options::CDOptions=CDOptions(opt_tol=1e-8, max_iter=10000),
                          newton_tol::Real=1e-10, newton_max_iter::Integer=100,
                          screening::Bool=true, rng::AbstractRNG=default_rng())
  start = time_ns()
  isfinite(tau) && 0 < tau < 1 || throw(ArgumentError("tau must lie strictly between zero and one"))
  isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be positive and finite"))
  time_limit > 0 || throw(ArgumentError("time_limit must be positive"))
  max_hat_s isa Real && !isnan(max_hat_s) && max_hat_s >= 0 ||
    throw(ArgumentError("max_hat_s must be a nonnegative real number other than NaN"))
  method in (:exact, :smooth) || throw(ArgumentError("method must be :exact or :smooth"))
  initial_basis in (:quantile, :default) || throw(ArgumentError("invalid initial_basis"))
  simplex in (:dual, :primal) || throw(ArgumentError("simplex must be :dual or :primal"))
  method == :smooth && (!isfinite(bandwidth) || bandwidth <= 0) && throw(ArgumentError("bandwidth must be positive and finite"))
  method == :smooth && isfinite(time_limit) && throw(ArgumentError("time_limit is supported only by the exact backend"))
  ls = Float64.(λpath)
  !isempty(ls) && all(x -> isfinite(x) && x >= 0, ls) && issorted(ls; rev=true) ||
    throw(ArgumentError("λpath must be nonempty, finite, nonnegative and nonincreasing"))
  Xf, yf, w = _qdata(X, y, penalty_factors, standardize_x, intercept, offset)
  sols, status = if method == :exact
    _quantile_exact_path(Xf, yf, ls, w, Float64(tau), intercept;
                         tol=Float64(tol), time_limit=Float64(time_limit), max_hat_s, initial_basis, simplex, start)
  else
    _quantile_smooth_path(Xf,yf,ls,w,Float64(tau),intercept,Float64(bandwidth);
                          tol=Float64(tol),max_hat_s,options,newton_tol,newton_max_iter,screening,rng)
  end
  QuantileLassoPath(ls[1:length(sols)], sols, ls, length(sols)==length(ls), status,
                    length(y), size(X,2), intercept, method)
end

function QuantileLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                          nλ::Integer=100, λminratio::Real=0.01, tau::Real=0.5,
                          penalty_factors=nothing, standardize_x::Bool=true, intercept::Bool=true,
                          offset=nothing, kwargs...)
  nλ >= 2 || throw(ArgumentError("nλ must be at least two"))
  isfinite(λminratio) && 0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0,1]"))
  isfinite(tau) && 0 < tau < 1 || throw(ArgumentError("tau must be in (0,1)"))
  Xf, yf, w = _qdata(X, y, penalty_factors, standardize_x, intercept, offset)
  all(>(0), w) || throw(ArgumentError("automatic grids need positive effective penalty factors; supply λpath explicitly"))
  a = intercept ? sort(yf)[clamp(ceil(Int, length(yf)*tau),1,length(yf))] : 0.0
  u = Float64.(tau .- (yf .< a))
  # At an empirical quantile, choose the tied scores to satisfy sum(u)=0.
  if intercept
    tied = findall(==(a), yf)
    isempty(tied) || (u[tied] .-= sum(u)/length(tied))
  end
  lm = maximum(abs.(Xf'u) ./ (length(yf).*w)) * (1+sqrt(eps(Float64)))
  ls = lm == 0 ? zeros(nλ) : exp.(range(log(lm), log(lm*λminratio); length=nλ))
  QuantileLassoPath(X, y, ls; tau, penalty_factors, standardize_x, intercept, offset, kwargs...)
end

StatsAPI.coef(sol::QuantileLassoSolution) = sol.hasintercept ? [sol.intercept; sol.β] : copy(sol.β)
StatsAPI.coef(path::QuantileLassoPath) = isempty(path.solutions) ? zeros(path.nvars+path.hasintercept, 0) : reduce(hcat, coef.(path.solutions))
function StatsAPI.coef(path::QuantileLassoPath, λ::Real)
  k = findfirst(==(Float64(λ)), path.λpath)
  k === nothing && throw(ArgumentError("quantile paths do not interpolate; refit at the requested λ"))
  coef(path[k])
end
function StatsAPI.predict(sol::QuantileLassoSolution, X::AbstractMatrix{<:Real}; offset=nothing)
  _checkcolumns(X, length(sol.β))
  _addoffset!(Vector(X*sol.β) .+ sol.intercept, offset)
end
function StatsAPI.predict(path::QuantileLassoPath, X::AbstractMatrix{<:Real}; offset=nothing)
  _checkcolumns(X, path.nvars)
  isempty(path.solutions) && return zeros(size(X,1),0)
  reduce(hcat, [predict(s, X; offset) for s in path.solutions])
end
StatsAPI.residuals(sol::QuantileLassoSolution) = sol.residuals
StatsAPI.nobs(sol::QuantileLassoSolution) = length(sol.residuals)
StatsAPI.nobs(path::QuantileLassoPath) = path.nobs
StatsAPI.dof(sol::QuantileLassoSolution) = count(!iszero, sol.β)+sol.hasintercept
StatsAPI.fit(::Type{QuantileLassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) = QuantileLassoPath(X,y,args...;kwargs...)

StatsAPI.fit(::Type{QuantileLassoSolution}, X::AbstractMatrix, y::AbstractVector, λ::Real; kwargs...) = quantile_lasso(X,y,λ;kwargs...)

"""
    QuantileRefit

An unpenalized quantile fit on the sorted column indices in `support`.
`β` has one entry per original column and is zero off that support;
`intercept` is the fitted intercept. `fit` is the restricted
[`QuantileLassoSolution`](@ref) at λ=0, including its raw dual scores,
objectives, numerical diagnostics and native termination status.

`coef` puts the intercept first when fitted. `predict` takes the full design
and an optional offset. `residuals`, `nobs` and `dual_scores` use `fit`.
`dof` counts `length(support) + fit.hasintercept`, including selected columns
whose refitted coefficient is zero. This result provides computation, not
post-selection inference.
"""
struct QuantileRefit
  support::Vector{Int}
  β::Vector{Float64}
  intercept::Float64
  fit::QuantileLassoSolution
end

function _qrefit_support(support, p)
  (eltype(support) <: Bool || any(x -> x isa Bool, support)) && throw(ArgumentError("Bool supports are not column indices; use findall"))
  all(j -> 1 <= j <= p, support) || throw(ArgumentError("support indices must lie in 1:p"))
  S = sort!(collect(Int,support))
  allunique(S) || throw(ArgumentError("support indices must be unique"))
  S
end

function _qrefit_taus(tau)
  taus = tau isa Real ? [tau] : tau isa AbstractVector{<:Real} ? tau :
    throw(ArgumentError("tau must be a scalar or a vector of real quantiles"))
  !isempty(taus) && all(τ -> isfinite(τ) && 0 < τ < 1, taus) ||
    throw(ArgumentError("tau must be nonempty and lie strictly between zero and one"))
  collect(Float64,taus)
end

function _qrefit_controls(method, bandwidth, tol, time_limit, initial_basis, simplex)
  isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be positive and finite"))
  time_limit > 0 || throw(ArgumentError("time_limit must be positive"))
  method in (:exact, :smooth) || throw(ArgumentError("method must be :exact or :smooth"))
  initial_basis in (:quantile, :default) || throw(ArgumentError("invalid initial_basis"))
  simplex in (:dual, :primal) || throw(ArgumentError("simplex must be :dual or :primal"))
  method == :smooth && (!isfinite(bandwidth) || bandwidth <= 0) && throw(ArgumentError("bandwidth must be positive and finite"))
  method == :smooth && isfinite(time_limit) && throw(ArgumentError("time_limit is supported only by the exact backend"))
  nothing
end

function _qrefit_check(X, y, sol, offset)
  n,p = size(X)
  length(y) == length(sol.residuals) == n && length(sol.β) == p ||
    throw(ArgumentError("X, y and the source fit have different dimensions"))
  offset === nothing || length(offset) == n || throw(ArgumentError("offset and y have different lengths"))
  nothing
end

function _qrefit_check(X, z, sol)
  r = z .- sol.intercept .- X*sol.β
  maximum(abs,r .- sol.residuals) <= sqrt(eps(Float64))*(1+maximum(abs,z)) ||
    throw(ArgumentError("X, y or offset do not reproduce the source fit's residuals"))
  nothing
end

function _qrefit(X, y, S, taus, intercept;
                 method, bandwidth, tol, time_limit, initial_basis, simplex, start,
                 options, newton_tol, newton_max_iter, screening, rng,
                 exact_refit=_quantile_exact_refit)
  A = X[:,S]; w = ones(length(S))
  fits = if method == :exact
    exact_refit(A,y,taus,intercept;tol=Float64(tol),time_limit=Float64(time_limit),initial_basis,simplex,start)
  else
    ss = QuantileLassoSolution[]
    for τ in taus
      sols, _ = _quantile_smooth_path(A,y,[0.0],w,τ,intercept,Float64(bandwidth);
                                      tol=Float64(tol),options,newton_tol,newton_max_iter,screening,rng)
      isempty(sols) && error("quantile refit returned no primal–dual pair")
      push!(ss,only(sols))
    end
    ss
  end
  [begin
     β = zeros(size(X,2)); β[S] = sol.β
     QuantileRefit(copy(S),β,sol.intercept,sol)
   end for sol in fits]
end

"""
    quantile_refit(X, y, support; tau=0.5, method=:exact, intercept=true,
                   offset=nothing, bandwidth=0.2, tol=2e-5, time_limit=Inf,
                   initial_basis=:quantile, simplex=:dual, options=CDOptions(...),
                   newton_tol=1e-10, newton_max_iter=100, screening=true,
                   rng=Random.default_rng())
    quantile_refit(X, y, sol::QuantileLassoSolution; tau=sol.tau, offset=nothing, kwargs...)

Refit without a penalty on a vector of integer column indices, possibly empty.
The support is copied, sorted and checked for duplicates and indices outside
`1:size(X,2)`. Use `findall` to convert a Bool mask. A scalar `tau` returns a
[`QuantileRefit`](@ref); a nonempty vector returns one refit per quantile in
its supplied order, including repeats. Quantiles must lie in (0,1).
Inputs are converted to Float64; sparse designs remain sparse.

The solution form takes the exact nonzero support of `sol.β` and inherits its
method, bandwidth and intercept setting. Dimensions are checked, then
`y - offset - sol.intercept - X*sol.β` must reproduce `sol.residuals` within
`sqrt(eps(Float64))*(1 + maximum(abs, y - offset))` in maximum norm.
This detects incompatible responses, offsets or scaling but does not establish
that `X` is the original design: a change with `ΔX*sol.β == 0` passes.

Columns outside the support are fixed at zero, including zero-loading columns
with zero fitted coefficients. List them explicitly to keep them. No support
threshold, penalty factors, standardization or support limit is accepted.
`tol`, `options`, Newton and screening controls have the same meaning as for
[`QuantileLassoPath`](@ref).

Exact refits require `using HiGHS`. One model is reused across quantiles by
changing only score bounds. Every requested refit is attempted, including after
a non-optimal native termination with an available primal–dual pair. Inspect
`r.fit.status`, `native_converged` and `converged` separately. A missing pair
throws an error. `time_limit` is one deadline for the entire call; an expired
deadline before a requested solve throws an error. Preparation and diagnostics
can overrun it. Finite limits are rejected for `method=:smooth`, which fits
each quantile independently without warm starts.

Diagnostics use only the restricted design. At λ=0 the feasible certificate
is zero and its gap can be loose; convergence checks the raw pair. Scores and
coefficients can be nonunique in degenerate problems; the optimizer's values
are retained. These refits do not provide post-selection inference.
"""
function quantile_refit(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, support::AbstractVector{<:Integer};
                        tau=0.5, method::Symbol=:exact, intercept::Bool=true, offset=nothing,
                        bandwidth::Real=0.2, tol::Real=2e-5, time_limit::Real=Inf,
                        initial_basis::Symbol=:quantile, simplex::Symbol=:dual,
                        options::CDOptions=CDOptions(opt_tol=1e-8,max_iter=10000),
                        newton_tol::Real=1e-10, newton_max_iter::Integer=100,
                        screening::Bool=true, rng::AbstractRNG=default_rng())
  start = time_ns()
  taus = _qrefit_taus(tau)
  _qrefit_controls(method,bandwidth,tol,time_limit,initial_basis,simplex)
  S = _qrefit_support(support,size(X,2))
  Xf, yf = _qrefit_data(X,y,offset)
  refits = _qrefit(Xf,yf,S,taus,intercept;method,bandwidth,tol,time_limit,initial_basis,simplex,start,
                    options,newton_tol,newton_max_iter,screening,rng)
  tau isa Real ? only(refits) : refits
end

function quantile_refit(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, sol::QuantileLassoSolution;
                        tau=sol.tau, offset=nothing, tol::Real=2e-5, time_limit::Real=Inf,
                        initial_basis::Symbol=:quantile, simplex::Symbol=:dual,
                        options::CDOptions=CDOptions(opt_tol=1e-8,max_iter=10000),
                        newton_tol::Real=1e-10, newton_max_iter::Integer=100,
                        screening::Bool=true, rng::AbstractRNG=default_rng())
  start = time_ns()
  taus = _qrefit_taus(tau)
  _qrefit_controls(sol.method,sol.bandwidth,tol,time_limit,initial_basis,simplex)
  _qrefit_check(X,y,sol,offset)
  Xf, yf = _qrefit_data(X,y,offset)
  _qrefit_check(Xf,yf,sol)
  refits = _qrefit(Xf,yf,findall(!iszero,sol.β),taus,sol.hasintercept;
                    method=sol.method,bandwidth=sol.bandwidth,tol,time_limit,initial_basis,simplex,start,
                    options,newton_tol,newton_max_iter,screening,rng)
  tau isa Real ? only(refits) : refits
end

"""
    refit_quantile_path(path::QuantileLassoPath, X, y; tau=nothing,
                        offset=nothing, kwargs...)

Refit each distinct exact-nonzero support of the saved fits once. Return a
`Dict{Vector{Int},QuantileRefit}`, indexed by
`refits[findall(!iszero, path[k].β)]`. An empty path returns an empty dictionary;
a truncated path uses only its saved fits.

The method and intercept setting come from `path`. Each support uses the
quantile and bandwidth of its first saved fit, unless a scalar `tau` is supplied.
Each such source fit passes the dimension and residual consistency checks of
[`quantile_refit`](@ref), with the same limits on what they establish.
Solver controls and the interruption contract are those of `quantile_refit`.
For exact refits, `time_limit` is one deadline across all supports. Models are
not reused across different supports. No inference is provided.
"""
function refit_quantile_path(path::QuantileLassoPath, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
                             tau=nothing, offset=nothing, tol::Real=2e-5, time_limit::Real=Inf,
                             initial_basis::Symbol=:quantile, simplex::Symbol=:dual,
                             options::CDOptions=CDOptions(opt_tol=1e-8,max_iter=10000),
                             newton_tol::Real=1e-10, newton_max_iter::Integer=100,
                             screening::Bool=true, rng::AbstractRNG=default_rng())
  _qrefit_path(path,X,y;tau,offset,tol,time_limit,initial_basis,simplex,
                options,newton_tol,newton_max_iter,screening,rng)
end

function _qrefit_path(path, X, y; tau=nothing, offset=nothing, tol=2e-5, time_limit=Inf,
                       initial_basis=:quantile, simplex=:dual,
                       options=CDOptions(opt_tol=1e-8,max_iter=10000),
                       newton_tol=1e-10, newton_max_iter=100, screening=true, rng=default_rng(),
                       exact_refit=_quantile_exact_refit)
  start = time_ns()
  tau === nothing || tau isa Real || throw(ArgumentError("tau must be a scalar for path refits"))
  tau === nothing || _qrefit_taus(tau)
  _qrefit_controls(path.method,isempty(path.solutions) ? 0.2 : path[1].bandwidth,
                    tol,time_limit,initial_basis,simplex)
  size(X) == (path.nobs,path.nvars) && length(y) == path.nobs ||
    throw(ArgumentError("X, y and the source path have different dimensions"))
  offset === nothing || length(offset) == length(y) || throw(ArgumentError("offset and y have different lengths"))
  refits = Dict{Vector{Int},QuantileRefit}()
  isempty(path.solutions) && return refits
  Xf, yf = _qrefit_data(X,y,offset)
  for sol in path.solutions
    S = findall(!iszero,sol.β)
    haskey(refits,S) && continue
    _qrefit_check(X,y,sol,offset)
    _qrefit_check(Xf,yf,sol)
    taus = _qrefit_taus(tau === nothing ? sol.tau : tau)
    _qrefit_controls(path.method,sol.bandwidth,tol,time_limit,initial_basis,simplex)
    refits[S] = only(_qrefit(Xf,yf,S,taus,path.hasintercept;
                             method=path.method,bandwidth=sol.bandwidth,tol,time_limit,initial_basis,simplex,start,
                             options,newton_tol,newton_max_iter,screening,rng,exact_refit))
  end
  refits
end

StatsAPI.coef(r::QuantileRefit) = r.fit.hasintercept ? [r.intercept; r.β] : copy(r.β)
function StatsAPI.predict(r::QuantileRefit, X::AbstractMatrix{<:Real}; offset=nothing)
  _checkcolumns(X,length(r.β))
  _addoffset!(Vector(X*r.β) .+ r.intercept,offset)
end
StatsAPI.residuals(r::QuantileRefit) = residuals(r.fit)
StatsAPI.nobs(r::QuantileRefit) = nobs(r.fit)
StatsAPI.dof(r::QuantileRefit) = length(r.support)+r.fit.hasintercept
dual_scores(r::QuantileRefit) = dual_scores(r.fit)
