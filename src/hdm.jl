######################################################################
#
#   Inference after selection, as in R's hdm
#
######################################################################

# the standard normal quantile and distribution functions
_qnorm(t::T) where {T<:AbstractFloat} = -sqrt(T(2)) * erfcinv(2t)
_pnorm(z::T) where {T<:AbstractFloat} = erfc(-z / sqrt(T(2))) / 2

"""
    theory_lambda(n, p; c = 1.1, γ = 0.1 / log(n))

The penalty level `λ0 = c Φ⁻¹(1 - γ/(2p)) / √n` of Belloni, Chen, Chernozhukov
and Hansen (2012) for the lasso with data-driven loadings,

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λ0 ∑ⱼ Γⱼ |βⱼ|,   Γⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² εᵢ² / n),

the scale of [`feasible_lasso!`](@ref). It is R's hdm penalty
`2c √n Φ⁻¹(1 - γ/(2p))` for `‖y - Xβ‖² + λ ∑ⱼ Γⱼ |βⱼ|`, divided by `2n`, with
hdm's defaults for `c` and `γ`. `Φ⁻¹` is the standard normal quantile.
"""
function theory_lambda(n::Integer, p::Integer; c::Real = 1.1, γ::Real = 0.1 / log(n))
  n >= 2 && p >= 1 || throw(ArgumentError("theory_lambda needs n ≥ 2 and p ≥ 1"))
  0 < γ < 1 || throw(ArgumentError("γ must be in (0, 1), but it is $γ."))
  c > 0 || throw(ArgumentError("c must be positive, but it is $c."))
  c * _qnorm(1 - float(γ) / (2p)) / sqrt(n)
end

"""
    feasible_lasso!(x, X, y, options = IterLassoOptions(); λ0 = :theory, c = 1.1, γ = 0.1 / log(n), intercept = false)

[`feasible_lasso!`](@ref) at the penalty level of [`theory_lambda`](@ref)`(n, p; c, γ)`.
R's hdm uses `c = 0.5` instead when it does not refit by least squares
(`rlasso(…, post = FALSE)`); see [`rlasso`](@ref) for hdm's procedure.
"""
function feasible_lasso!(x::SparseIterate{T}, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                        options::IterLassoOptions = IterLassoOptions(); λ0::Symbol = :theory, c::Real = 1.1,
                        γ::Real = 0.1 / log(size(X, 1)), kwargs...) where {T<:AbstractFloat}
  λ0 === :theory || throw(ArgumentError("λ0 must be a number or :theory, but it is :$λ0."))
  feasible_lasso!(x, X, y, T(theory_lambda(size(X, 1), size(X, 2); c, γ)), options; kwargs...)
end

# least squares of y on the columns S of X, centered at xbar (and y at ȳ), by a
# pivoted QR; returns the coefficients, residuals, and the solve's rank decision
function _ols(X::AbstractMatrix{T}, y::AbstractVector{T}, S::Vector{Int}, xbar::Vector{T}, ȳ::T) where {T}
  Z = X[:, S] .- transpose(xbar[S])
  if isempty(S)
    b, rnk = T[], 0
  else
    F = qr(Z, ColumnNorm())
    if eltype(F) <: LinearAlgebra.BlasFloat
      # The same ldiv! used by pivoted-QR backslash also returns its rank.
      # rank(F) uses a different, diagonal-based threshold.
      rhs = zeros(eltype(F), max(size(Z)...), 1)
      rhs[1:size(X, 1), 1] .= y .- ȳ
      _, rnk = LinearAlgebra.ldiv!(F, rhs, min(size(F)...) * eps(real(eltype(F))))
      b = rhs[1:length(S), 1]
    else
      # Julia's generic QRPivoted backslash assumes full rank.
      b, rnk = F \ (y .- ȳ), min(size(F)...)
    end
  end
  b, (y .- ȳ) .- Z * b, rnk
end

"""
    RLassoSolution

Result returned by [`rlasso`](@ref). Its positional constructor is internal;
obtain a solution by calling `rlasso`.

# Fields
- `x::SparseVector`: the coefficients, refitted by least squares on the
  selected columns when `post` is true (as hdm's `rlasso` returns them), the
  lasso coefficients otherwise
- `intercept`: the intercept, zero without one
- `selected::Vector{Int}`: the selected columns, equal to
  `findall(!iszero, lasso)`
- `λ0`, `loadings::Vector`: the nominal penalty level on the scale of
  [`feasible_lasso!`](@ref), and the loadings updated from the final residuals
  (unchanged on an empty-selection return; hdm's `lambda` is `2n λ0 Γ`)
- `residuals::Vector`, `σ`: the residuals of the final coefficients and their
  standard deviation
- `iterations::Int`: the lasso fits of the procedure
- `post::Bool`
- `hasintercept::Bool`: whether the procedure fitted an intercept
- `lasso::SparseVector`, `lasso_intercept`: a copy of the penalized coefficients
  of the final coordinate-descent solve, and their intercept (zero without one)
- `lasso_λ0`, `lasso_loadings::Vector`: the penalty level and a copy of the
  loadings actually used by that solve, before the residual-based update;
  `lasso_λ0 == λ0/2` if the final solve was the first one and `post` is true,
  and `lasso_λ0 == λ0` otherwise
- `converged::Bool`: the outer standard-deviation stopping criterion was met;
  an empty-selection return also counts as converged
- `solver_converged::Bool`: every coordinate-descent solve converged,
  including the solve preceding an empty-selection return
- `rank::Int`: the numerical rank of the selected columns after centering
  when fitting an intercept, zero for no selected columns. For BLAS floating
  types this is the rank returned by the pivoted-QR least-squares solve used
  by `_ols`: incremental estimates of the smallest/largest singular values
  of leading R blocks, with `rcond = min(n, length(selected)) * eps(T)`.
  A block is rejected when `smax * rcond > smin`. This is not the diagonal
  threshold used by `rank(::QRPivoted)`. Other floating types follow Julia's
  generic pivoted-QR solve, which assumes full rank on a successful solve.
- `tuning::NamedTuple`: `(c, γ, max_iter, tol, options)` as used by the fit.
  `max_iter` limits the outer loadings loop; `options.max_iter` limits the
  coordinate-descent passes of each solve

[`coef`](@ref), [`predict`](@ref), [`residuals`](@ref), [`nobs`](@ref),
[`dof`](@ref) and [`deviance`](@ref) apply to it.
"""
struct RLassoSolution{T}
  x::SparseVector{T, Int}
  intercept::T
  selected::Vector{Int}
  λ0::T
  loadings::Vector{T}
  residuals::Vector{T}
  σ::T
  iterations::Int
  post::Bool
  hasintercept::Bool
  lasso::SparseVector{T, Int}
  lasso_intercept::T
  lasso_λ0::T
  lasso_loadings::Vector{T}
  converged::Bool
  solver_converged::Bool
  rank::Int
  tuning::NamedTuple
end

"""
    rlasso(X, y, options = CDOptions(); post = true, intercept = true, c = post ? 1.1 : 0.5,
           γ = 0.1 / log(n), max_iter = 15, tol = 1e-5, rng = default_rng())

The lasso with data-driven penalty loadings as R's hdm computes it
(`rlasso` with its defaults: heteroskedastic and X-independent penalty), the
procedure of Belloni, Chen, Chernozhukov and Hansen (2012):

1. Start from the residuals of least squares on the five columns most
   correlated with `y`, and compute the loadings `Γⱼ` from them (see
   [`feasible_lasso!`](@ref)).
2. Solve the lasso `‖y - β₀ - Xβ‖² / (2n) + λ0 ∑ⱼ Γⱼ |βⱼ|` with
   `λ0 =` [`theory_lambda`](@ref)`(n, p; c, γ)`, at half the level in the first
   step when `post` is true, as hdm does.
3. With `post`, refit least squares on the selected columns; update the
   loadings from the residuals; repeat 2–3 until the standard deviation of the
   residuals changes by less than `tol`, or `max_iter` times.

The coefficients returned are the refitted ones when `post` is true. With
`intercept`, the columns and `y` are centered and `β₀ = ȳ - x̄'β`. Each lasso
is solved by coordinate descent with `options`, warm started from the previous
one. Random coordinate orders are drawn from `rng::AbstractRNG`; a private
generator leaves the default RNG unchanged. The default preserves the stream
of calls seeded with `Random.seed!`.

Returns an [`RLassoSolution`](@ref), retaining both the final penalized solve
(`lasso`, `lasso_intercept`, `lasso_λ0`, `lasso_loadings`) and the returned
coefficients/residuals, together with outer/solver convergence, selected-design
rank, and effective tuning. The outer `max_iter` limits loadings iterations;
`options.max_iter` limits coordinate-descent passes per solve. See also
[`post_lasso`](@ref) and [`double_selection`](@ref).
"""
function rlasso(X::AbstractMatrix{T}, y::AbstractVector{T}, options::CDOptions = CDOptions();
                post::Bool = true, intercept::Bool = true, c::Real = post ? 1.1 : 0.5,
                γ::Real = 0.1 / log(size(X, 1)), max_iter::Union{Nothing, Integer} = nothing, tol::Real = 1e-5,
                rng::AbstractRNG = default_rng(), kwargs...) where {T<:AbstractFloat}
  max_iter, kwargs = _legacy_keyword(max_iter, kwargs, Val(:maxIter), Val(:max_iter), 15)
  max_iter::Integer
  _check_keywords(CDOptions, (), kwargs)
  n, p = size(X)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  _check_finite(X, "X"); _check_finite(y, "y")
  max_iter >= 1 || throw(ArgumentError("max_iter must be positive, but it is $max_iter."))
  xbar = intercept ? _column_means(X) : zeros(T, p)
  ȳ = intercept ? mean(y) : zero(T)
  λ0 = T(theory_lambda(n, p; c, γ))
  tuning = (; c, γ, max_iter, tol, options)

  # hdm's init_values: least squares on the five columns most correlated with y
  sd = vec(std(X; dims = 1))
  cors = [sd[j] > 0 ? abs(cor(view(X, :, j), y)) : zero(T) for j in 1:p]
  S0 = partialsortperm(cors, 1:min(5, p); rev = true)
  _, e, _ = _ols(X, y, collect(S0), xbar, ȳ)
  Γ = _get_loadings!(Vector{T}(undef, p), X, e, xbar)

  m = CDLeastSquaresLoss(y, X; intercept)
  solver = _solver(options; rng)
  x = SparseIterate(T, p)
  s0 = std(y)
  b, S, iterations = T[], Int[], 0
  lasso_λ0, lasso_loadings = λ0, T[]
  converged, solver_converged, selected_rank = false, true, 0
  for it in 1:max_iter
    iterations = it
    lasso_λ0 = it == 1 && post ? λ0 / 2 : λ0
    lasso_loadings = copy(Γ)
    result = solve!(x, m, NormL1(lasso_λ0, copy(Γ)), solver)
    solver_converged &= result.converged
    S = sort!([k for k in activecoordinates(x) if !iszero(x[k])])
    if isempty(S)          # nothing selected: the intercept only, as hdm returns it
      e = y .- ȳ
      return RLassoSolution{T}(spzeros(T, p), ȳ, Int[], λ0, Γ, e, std(e), it, post, intercept,
                               spzeros(T, p), ȳ, lasso_λ0, lasso_loadings, true, solver_converged, 0, tuning)
    end
    if post
      b, e, selected_rank = _ols(X, y, S, xbar, ȳ)
    else
      b, e = T[x[k] for k in S], copy(ProximalSolvers.residuals(m))
    end
    s1 = std(e)
    _get_loadings!(Γ, X, e, xbar)
    if abs(s0 - s1) < tol
      converged = true
      break
    end
    s0 = s1
  end
  β = sparsevec(S, b, p)
  lasso = sparsevec(S, T[x[k] for k in S], p)
  if !post
    _, _, selected_rank = _ols(X, y, S, xbar, ȳ)
  end
  RLassoSolution{T}(β, ȳ - dot(xbar, β), S, λ0, Γ, e, std(e), iterations, post, intercept,
                   lasso, intercept ? ȳ - dot(xbar, lasso) : zero(T), lasso_λ0, lasso_loadings,
                   converged, solver_converged, selected_rank, tuning)
end

function rlasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, options::CDOptions = CDOptions();
                rng::AbstractRNG = default_rng(), kwargs...)
  T, Xf, yf = _float_data(X, y)
  rlasso(Xf, yf, options; rng, kwargs...)
end

"""
    PostLassoResult

Result returned by [`post_lasso`](@ref): the least-squares coefficients `x`
(a `SparseVector`, zero off the selected columns) and `intercept`, their
heteroskedasticity-robust (HC1) standard errors `se` and `se_intercept`, the
`selected` columns, the `residuals`, and `vcov`, the HC1 covariance of
`[β₀; β_S]` (of `β_S` without an intercept), and `hasintercept`.

The functions of StatsAPI apply to it, with `[β₀; β]` (`β` without an
intercept) as the coefficients: [`coef`](@ref), `stderror`, `vcov` (a sparse
matrix, zero off the selected columns), `confint` (normal intervals, zero
width off the selected columns), [`predict`](@ref), [`residuals`](@ref),
[`nobs`](@ref), [`dof`](@ref) and [`deviance`](@ref).
"""
struct PostLassoResult{T}
  x::SparseVector{T, Int}
  intercept::T
  se::SparseVector{T, Int}
  se_intercept::T
  selected::Vector{Int}
  residuals::Vector{T}
  vcov::Matrix{T}
  hasintercept::Bool
end

_support(sol::LassoSolution) = sort!([k for k in activecoordinates(sol.x) if !iszero(sol.x[k])])
_support(sol::RLassoSolution) = sol.selected
_support(S::AbstractVector{<:Integer}) = sort!(collect(Int, S))

"""
    post_lasso(X, y, sol; intercept = true)

Least squares of `y` on the columns that the lasso `sol` selected (a
[`LassoSolution`](@ref), an [`RLassoSolution`](@ref), or a vector of column
indices), with an intercept unless `intercept` is false, and the
heteroskedasticity-robust standard errors HC1,
`n/(n - k) ⋅ (Z'Z)⁻¹ Z' Diagonal(e²) Z (Z'Z)⁻¹` for the design `Z = [1 X_S]`
with `k` columns. Returns a [`PostLassoResult`](@ref).
"""
function post_lasso(X::AbstractMatrix{T}, y::AbstractVector{T}, sol; intercept::Bool = true) where {T<:AbstractFloat}
  n, p = size(X)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  S = _support(sol)
  Z = intercept ? [ones(T, n) X[:, S]] : X[:, S]
  k = size(Z, 2)
  k < n || throw(ArgumentError("least squares on $k columns needs more than $n observations"))
  F = qr(Z)
  rank(F.R) == k || throw(ArgumentError("the selected columns are collinear"))
  coef = F \ y
  e = y .- Z * coef
  Rinv = inv(UpperTriangular(F.R))
  B = Matrix(F.Q)' .* transpose(e)              # Q'Diagonal(e)
  V = (T(n) / (n - k)) .* (Rinv * (B * B') * Rinv')
  V = (V + V') / 2
  i0 = intercept ? 1 : 0
  se = sqrt.(max.(diag(V), zero(T)))
  PostLassoResult{T}(sparsevec(S, coef[i0+1:end], p), intercept ? coef[1] : zero(T),
                     sparsevec(S, se[i0+1:end], p), intercept ? se[1] : zero(T), S, e, V, intercept)
end

post_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, sol; kwargs...) =
  ((T, Xf, yf) = _float_data(X, y); post_lasso(Xf, yf, sol; kwargs...))

"""
    TreatmentEffect

Result returned by [`double_selection`](@ref) for the prespecified treatment
coefficient in original units. The positional constructor is internal.

`estimate`, `se`, `t`, `pvalue`, and `ci` give the estimate, standard error,
normal statistic, two-sided normal p-value, and interval at the requested
level. `selected` is the union of the nuisance supports and, for double
selection, the forced controls. `method`, `n`, `vcov_type`, and `post` record
the procedure. `refit` equals `post` for partialling and is `nothing` for double
selection. `selected_y`, `selected_d`, and `forced` record the separate supports
and the controls forced into double selection (empty for partialling).

`rank` is the rank of the intercept and selected-control basis (1 for
partialling); `dof_residual` is the final regression's residual degrees of
freedom. `target_variation` is the fraction of centered treatment variation
remaining after projection. `max_leverage` and the observation-level `influence`
refer to the final regression. `nuisance_y` and `nuisance_d` retain the
[`RLassoSolution`](@ref)s, including their outer and solver convergence flags.
Non-convergence is reported, without rejecting the effect estimate.

`coef`, `stderror`, `vcov` and `confint` (at any `level`) give the estimate as
a model with one coefficient; `nobs` is `n`.
"""
struct TreatmentEffect{T}
  estimate::T
  se::T
  t::T
  pvalue::T
  ci::Tuple{T, T}
  selected::Vector{Int}
  method::Symbol
  n::Int
  vcov_type::Symbol
  post::Bool
  refit::Union{Nothing, Bool}
  selected_y::Vector{Int}
  selected_d::Vector{Int}
  forced::Vector{Int}
  rank::Int
  dof_residual::Int
  target_variation::T
  max_leverage::T
  influence::Vector{T}
  nuisance_y::RLassoSolution{T}
  nuisance_d::RLassoSolution{T}
end

function Base.show(io::IO, r::TreatmentEffect)
  print(io, "TreatmentEffect (", r.method, ", ", r.vcov_type, "): ", round(r.estimate; sigdigits = 4), " (se ",
        round(r.se; sigdigits = 3), ", p = ", round(r.pvalue; sigdigits = 3), "), ",
        length(r.selected), " controls selected")
  failed = String[]
  r.nuisance_y.converged && r.nuisance_y.solver_converged || push!(failed, "y")
  r.nuisance_d.converged && r.nuisance_d.solver_converged || push!(failed, "d")
  isempty(failed) || print(io, "; nuisance non-convergence: ", join(failed, ", "))
end

_effect_tau(::Type{T}) where {T} = max(T(1e-7), sqrt(eps(T)))

function _effect_types(method, vcov_type)
  method isa Symbol && method in (:doubleselection, :partialling) ||
    throw(ArgumentError("method must be :doubleselection or :partialling, but it is $method."))
  vcov_type isa Symbol && vcov_type in (:hdm, :homoskedastic, :HC0, :HC1, :HC2, :HC3) ||
    throw(ArgumentError("vcov must be :hdm, :homoskedastic, :HC0, :HC1, :HC2 or :HC3, but it is $vcov_type."))
  nothing
end

function _effect_finite(a, name)
  i = findfirst(x -> !isfinite(x), a)
  i === nothing || throw(ArgumentError("$name[$i] must be finite, but it is $(a[i])."))
  nothing
end

function _effect_data(y, d, X, method, vcov_type, level)
  n, p = size(X)
  length(y) == length(d) == n || throw(DimensionMismatch(
    "length(y)=$(length(y)), length(d)=$(length(d)), size(X,1)=$n must agree"))
  _effect_types(method, vcov_type)
  n >= 3 || throw(ArgumentError("n must be at least 3, but it is $n."))
  p >= 1 || throw(ArgumentError("p must be positive, but it is $p."))
  0 < level < 1 || throw(ArgumentError("level must be in (0,1), but it is $level."))
  _effect_finite(y, "y"); _effect_finite(d, "d"); _effect_finite(X, "X")
  T = float(promote_type(eltype(y), eltype(d), eltype(X)))
  yf, df, Xf = convert(Vector{T}, y), convert(Vector{T}, d), convert(Matrix{T}, X)
  for (name, a) in (("d", df), ("y", yf))
    variation = norm(a .- mean(a))
    variation > 0 && !all(==(first(a)), a) || throw(ArgumentError(
      "centered $name norm=$variation; $name must be nonconstant."))
  end
  yf, df, Xf
end

function _effect_fit(fit::RLassoSolution, response::Vector{T}, X, name) where {T}
  n, p = size(X)
  length(fit.residuals) == n || throw(DimensionMismatch(
    "$name residual length=$(length(fit.residuals)), expected n=$n"))
  for field in (:x, :lasso, :loadings, :lasso_loadings)
    length(getfield(fit, field)) == p || throw(DimensionMismatch(
      "$name.$field length=$(length(getfield(fit, field))), expected p=$p"))
  end
  fit.hasintercept || throw(ArgumentError("$name.hasintercept must be true, but it is $(fit.hasintercept)."))
  eltype(fit.x) === T || throw(ArgumentError("$name eltype=$(eltype(fit.x)), expected $T."))
  S = fit.selected
  issorted(S) && allunique(S) && all(j -> 1 <= j <= p, S) && S == findall(!iszero, fit.lasso) ||
    throw(ArgumentError("$name.selected=$S must be sorted, unique, in 1:$p, and equal the lasso support."))
  error = norm(fit.residuals .- (response .- fit.intercept .- X * fit.x), Inf)
  bound = sqrt(eps(T)) * (norm(response, Inf) + abs(fit.intercept) + maximum(abs.(X) * abs.(fit.x)))
  isfinite(error) && isfinite(bound) && error <= bound || throw(ArgumentError(
    "$name residual consistency error=$error exceeds bound=$bound."))
  nothing
end

# Pure final stage. Original-response norms are supplied by the public path to
# make the relative degeneracy rules refer to the data's original units.
function _effect_final(v::AbstractVector{T}, w::AbstractVector{T}, h_W::AbstractVector{T},
                       k::Int, s::Int, vcov_type, method;
                       target_norm::Real = norm(v), outcome_norm::Real = norm(w)) where {T<:AbstractFloat}
  n = length(v)
  n >= 3 || throw(ArgumentError("n must be at least 3, but it is $n."))
  length(w) == length(h_W) == n || throw(DimensionMismatch(
    "final-stage lengths v=$n, w=$(length(w)), h_W=$(length(h_W)) must agree"))
  _effect_types(method, vcov_type)
  τ = _effect_tau(T)
  nv = norm(v)
  nv > τ * target_norm || throw(ArgumentError(
    "residualized target norm=$nv must exceed tau*centered d norm=$(τ * target_norm)."))
  k < n || throw(ArgumentError("k=$k must be less than n=$n."))
  if vcov_type === :hdm && method === :doubleselection
    n - s - 1 >= 1 || throw(ArgumentError("n-s-1=$(n-s-1) must be at least 1 (n=$n, s=$s)."))
  end
  vv = dot(v, v)
  α = dot(v, w) / vv
  e = w .- α .* v
  ne = norm(e)
  ne > τ * outcome_norm || throw(ArgumentError(
    "residual norm=$ne must exceed tau*centered y norm=$(τ * outcome_norm)."))
  influence = (T(n) / vv) .* v .* e
  leverage = h_W .+ abs2.(v) ./ vv
  if vcov_type in (:HC2, :HC3)
    maximum(leverage) <= 1 - τ || throw(ArgumentError(
      "max leverage=$(maximum(leverage)) must be at most 1-tau=$(1-τ)."))
  end
  variance = if vcov_type === :homoskedastic || (vcov_type === :hdm && method === :partialling)
    sum(abs2, e) / (n - (vcov_type === :hdm ? 2 : k)) / vv
  else
    meat = abs2.(v) .* abs2.(e)
    if vcov_type in (:HC2, :HC3)
      meat ./= (1 .- leverage) .^ (vcov_type === :HC2 ? 1 : 2)
    end
    hc0 = sum(meat) / vv^2
    vcov_type === :HC1 ? hc0 * n / (n - k) :
      vcov_type === :hdm ? hc0 * n / (n - s - 1) : hc0
  end
  (; estimate = α, residuals = e, influence, leverage, variance, jacobian = -vv / n)
end

function _effect_layout(y::Vector{T}, d, X, fit_y, fit_d, method, forced) where {T}
  n = length(y)
  S = sort!(union(fit_y.selected, fit_d.selected, method === :doubleselection ? Int.(forced) : Int[]))
  if method === :partialling
    return (; v = fit_d.residuals .- mean(fit_d.residuals),
              w = fit_y.residuals .- mean(fit_y.residuals),
              h_W = fill(inv(T(n)), n), k = 2, s = 0, rank = 1, selected = S)
  end
  τ = _effect_tau(T)
  Z = X[:, S] .- mean(X[:, S]; dims = 1)
  norms = [norm(view(Z, :, j)) for j in eachindex(S)]
  keep = [j for j in eachindex(S) if norms[j] > τ * norm(view(X, :, S[j]))]
  Q = if isempty(keep)
    zeros(T, n, 0)
  else
    F = qr(Z[:, keep] ./ transpose(norms[keep]), ColumnNorm())
    r = count(>(τ), abs.(diag(F.R)))
    Matrix(F.Q)[:, 1:r]
  end
  dc, yc = d .- mean(d), y .- mean(y)
  r = size(Q, 2)
  (; v = dc .- Q * (Q' * dc), w = yc .- Q * (Q' * yc),
     h_W = inv(T(n)) .+ vec(sum(abs2, Q; dims = 2)),
     k = r + 2, s = length(S), rank = r + 1, selected = S)
end

function _effect_from_fits(y::Vector{T}, d, X, fit_y, fit_d, method, vcov_type, forced, level) where {T}
  _effect_fit(fit_y, y, X, "fit_y")
  _effect_fit(fit_d, d, X, "fit_d")
  fit_y.post == fit_d.post || throw(ArgumentError(
    "fit_y.post=$(fit_y.post) and fit_d.post=$(fit_d.post) must agree."))
  layout = _effect_layout(y, d, X, fit_y, fit_d, method, forced)
  result = _effect_final(layout.v, layout.w, layout.h_W, layout.k, layout.s, vcov_type, method;
                         target_norm = norm(d .- mean(d)), outcome_norm = norm(y .- mean(y)))
  α, se = result.estimate, sqrt(result.variance)
  t = α / se
  z = _qnorm(1 - (1 - T(level)) / 2)
  TreatmentEffect{T}(α, se, t, 2 * _pnorm(-abs(t)), (α - z * se, α + z * se), layout.selected, method,
    length(y), vcov_type, fit_y.post, method === :partialling ? fit_y.post : nothing,
    copy(fit_y.selected), copy(fit_d.selected), method === :doubleselection ? sort!(unique(Int.(forced))) : Int[],
    layout.rank, length(y) - layout.k, dot(layout.v, layout.v) / sum(abs2, d .- mean(d)),
    maximum(result.leverage), result.influence, fit_y, fit_d)
end

"""
    double_selection(y, d, X; method = :doubleselection, vcov = :hdm, post = true,
                     forced = Int[], level = 0.95, options = CDOptions(), rng = default_rng(), kwargs...)
    double_selection(y, d, X, fit_y::RLassoSolution, fit_d::RLassoSolution;
                     method = :doubleselection, vcov = :hdm, forced = Int[], level = 0.95)

Inference for the prespecified coefficient `α` in `y = μ + αd + Xβ + ε`, in
original units, with an intercept throughout. This procedure does not justify
choosing the target after inspecting estimated effects. `kwargs` (`c`, `γ`,
`max_iter`, `tol`) go to [`rlasso`](@ref); `options` controls its coordinate
descent and `rng` supplies both fits' random orders. Double selection fits d
then y; partialling fits y then d. `intercept=false`, weights, and offsets are
unsupported.

Double selection uses the sorted union of both supports and `forced`. Its
control basis W contains an intercept and centered selected controls. With
`τ = max(1e-7, sqrt(eps(T)))`, a centered column of norm at most τ times its raw
norm is aliased with the intercept. Remaining columns are scaled to unit norm;
pivoted QR retains the `r` directions with `abs(R[j,j]) > τ`. Projection gives
`v = M_W d`, `w = M_W y`, base leverages `h_W`, `k = r+2`, and nominal control
count `s`. This rule differs from R's dqrdc2 and from the nuisance QR solve.
Partialling centers the native nuisance residuals to obtain v and w, with
`h_W = 1/n`, `k = 2`, and `s = 0`; `forced` is ignored for this method.

In both layouts, `α̂ = dot(v,w)/dot(v,v)`, `e = w - α̂v`,
`φᵢ = n*vᵢ*eᵢ/dot(v,v)`, and `hᵢ = h_Wᵢ + vᵢ²/dot(v,v)`. The variance is:

| `vcov` | Variance |
|:--|:--|
| `:homoskedastic` | `sum(e²)/(n-k)/dot(v,v)` |
| `:HC0` | `sum(v² .* e²)/dot(v,v)^2` |
| `:HC1` | `HC0*n/(n-k)` |
| `:HC2` | `sum(v² .* e² ./ (1-h))/dot(v,v)^2` |
| `:HC3` | `sum(v² .* e² ./ (1-h).^2)/dot(v,v)^2` |
| `:hdm`, partialling | homoskedastic variance with k=2 |
| `:hdm`, double selection | `HC0*n/(n-s-1)`, with nominal s including aliases |

The default preserves hdm 0.3.2's variance conventions. Intervals are
`α̂ ± z*se`; t statistics and two-sided p-values use the standard normal.

Inputs must be finite, with n≥3, p≥1, 0<level<1, and nonconstant y and d.
All variances require k<n; hdm double selection also requires n-s-1≥1.
Reject `norm(v) ≤ τ*norm(d-mean(d))`, `norm(e) ≤ τ*norm(y-mean(y))`, and, for
HC2/HC3, `maximum(h) > 1-τ`. Invalid quantities raise `ArgumentError` with
their values; incompatible dimensions raise `DimensionMismatch`.

The precomputed method consumes no RNG and does not mutate either fit. It
checks dimensions, promoted element type, intercepts, matching post flags,
sorted unique lasso supports, and residual consistency with y/d and X to a
budget `sqrt(eps(T)) * (norm(response,Inf) + abs(b0) + maximum(abs.(X)*abs.(x)))`
for `norm(residuals - (response - b0 - X*x), Inf)`. These checks establish consistency, not
provenance: they cannot establish which data or algorithm produced a record.
Nuisance non-convergence is retained in the result and displayed by `show`.
"""
function double_selection(y::AbstractVector{<:Real}, d::AbstractVector{<:Real}, X::AbstractMatrix{<:Real};
                         method = :doubleselection, vcov = :hdm, level::Real = 0.95,
                         forced::AbstractVector{<:Integer} = Int[], post::Bool = true,
                         options::CDOptions = CDOptions(), rng::AbstractRNG = default_rng(),
                         intercept = true, kwargs...)
  intercept === true || throw(ArgumentError("intercept must be true, but it is $intercept."))
  y, d, X = _effect_data(y, d, X, method, vcov, level)
  if method === :doubleselection
    fit_d = rlasso(X, d, options; post, rng, kwargs...)
    fit_y = rlasso(X, y, options; post, rng, kwargs...)
  else
    fit_y = rlasso(X, y, options; post, rng, kwargs...)
    fit_d = rlasso(X, d, options; post, rng, kwargs...)
  end
  _effect_from_fits(y, d, X, fit_y, fit_d, method, vcov, forced, level)
end

function double_selection(y::AbstractVector{<:Real}, d::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
                          fit_y::RLassoSolution, fit_d::RLassoSolution;
                          method = :doubleselection, vcov = :hdm,
                          forced::AbstractVector{<:Integer} = Int[], level::Real = 0.95)
  y, d, X = _effect_data(y, d, X, method, vcov, level)
  _effect_from_fits(y, d, X, fit_y, fit_d, method, vcov, forced, level)
end
