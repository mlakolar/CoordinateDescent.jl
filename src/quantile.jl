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
    QuantileLassoPath(X, y, λpath; tau=0.5, method=:exact, kwargs...)
    QuantileLassoPath(X, y; nλ=100, λminratio=0.01, kwargs...)

Fit `mean(rho_tau(y - intercept - X*β - offset)) + λ*sum(w .* abs.(β))`
along a nonincreasing, finite, nonnegative grid. Every saved level retains
both coefficients and raw dual scores, including at zero residuals.
`path[k]` is a [`QuantileLassoSolution`](@ref); `path.λpath` contains the
returned levels and `requested_λpath` the requested grid. `complete` is false
when a limit stops the path. There is no coefficient or dual interpolation:
`coef(path, λ)` requires a saved level.

`method=:exact` uses HiGHS dual simplex with an analytic quantile starting
basis and reuses that basis/model along the path. Load `HiGHS` to activate the
optional extension. `initial_basis=:default` disables the analytic start;
`simplex=:primal` selects primal simplex. `time_limit` limits the entire exact
path between levels and passes the remaining time to HiGHS. Preparation and
diagnostics can overrun that limit, and automatic-grid preparation precedes
the path timer. A stopped path retains available
primal–dual results and the termination status; it never fills missing levels.

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
not pivotal calibration or cross-validation. `tol=2e-5` is the external KKT and
relative raw-gap tolerance, distinct from backend stopping rules.

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
  length(y) == n || throw(DimensionMismatch("X and y have different observation counts"))
  _check_finite(X, "X"); _check_finite(y, "y")
  Xf = eltype(X) == Float64 ? X : Float64.(X)
  yf = Float64.(y)
  if offset !== nothing
    length(offset) == n || throw(DimensionMismatch("offset and y have different lengths"))
    _check_finite(offset, "offset"); yf .-= offset
  end
  w = penalty_factors === nothing ? ones(p) : Float64.(penalty_factors)
  length(w) == p || throw(DimensionMismatch("penalty_factors must have one entry per column"))
  all(x -> isfinite(x) && x >= 0, w) || throw(ArgumentError("penalty_factors must be finite and nonnegative"))
  standardize_x && (w .*= _loadings(Xf, intercept))
  Xf, yf, w
end

# Overridden by the optional HiGHS extension with a typed method.
function _quantile_exact_path(args...; kwargs...)
  throw(ArgumentError("exact quantile regression requires the optional HiGHS backend; run `using HiGHS`"))
end

function QuantileLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λpath::AbstractVector{<:Real};
                          tau::Real=0.5, method::Symbol=:exact, bandwidth::Real=0.2,
                          penalty_factors=nothing, standardize_x::Bool=true, intercept::Bool=true,
                          offset=nothing, tol::Real=2e-5, time_limit::Real=Inf,
                          initial_basis::Symbol=:quantile, simplex::Symbol=:dual,
                          options::CDOptions=CDOptions(opt_tol=1e-8, max_iter=10000),
                          newton_tol::Real=1e-10, newton_max_iter::Integer=100,
                          screening::Bool=true, rng::AbstractRNG=default_rng())
  start = time_ns()
  isfinite(tau) && 0 < tau < 1 || throw(ArgumentError("tau must lie strictly between zero and one"))
  isfinite(tol) && tol > 0 || throw(ArgumentError("tol must be positive and finite"))
  time_limit > 0 || throw(ArgumentError("time_limit must be positive"))
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
                         tol=Float64(tol), time_limit=Float64(time_limit), initial_basis, simplex, start)
  else
    path = GLMLassoPath(Xf, yf, ls, ProximalBase.ConvolutionCheck(Float64(tau), Float64(bandwidth)), options;
                       intercept, standardize_x=false, penalty=NormL1(1.0, w), devmax=1,
                       newton_tol, newton_max_iter, screening, rng)
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
    ss, all(s -> s.converged, ss) ? :optimal : :not_converged
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
