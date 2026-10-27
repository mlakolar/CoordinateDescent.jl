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

Result returned by [`double_selection`](@ref): the `estimate` of the effect of
the treatment, its standard error `se`, the `t` statistic and the two-sided
`pvalue` against zero (normal approximation), the confidence interval `ci` at
the requested level, the `selected` controls, the `method` and the number of
observations `n`. `coef`, `stderror`, `vcov` and `confint` (at any `level`)
give the estimate as a model with one coefficient, and `nobs` is `n`.
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
end

function Base.show(io::IO, r::TreatmentEffect)
  print(io, "TreatmentEffect (", r.method, "): ", round(r.estimate; sigdigits = 4), " (se ",
        round(r.se; sigdigits = 3), ", p = ", round(r.pvalue; sigdigits = 3), "), ",
        length(r.selected), " controls selected")
end

"""
    double_selection(y, d, X; method = :doubleselection, level = 0.95, forced = Int[], post = true, kwargs...)

Inference on the effect `α` of a treatment `d` in `y = αd + Xβ + ε` with many
controls `X`, after selecting them by [`rlasso`](@ref) (`kwargs` go to it), as
R's hdm `rlassoEffect` computes it:

- `method = :doubleselection` (Belloni, Chernozhukov and Hansen, 2014): select
  the controls that predict `y` and those that predict `d`, then regress `y` on
  `d`, the union of both (and the `forced` controls) and an intercept. The
  standard error is the robust one of hdm,
  `√(mean(v² ξ²) / (n mean(v²)²))`, with `ξ` the residuals of that regression
  scaled by `√(n/(n - s - 1))` for `s` selected controls, and `v` the residuals
  of `d` on the selected controls and an intercept.
- `method = :partialling`: regress the residuals of `rlasso` of `y` on `X` on
  those of `d` on `X`, with an intercept; the standard error is that of least
  squares, as in hdm.

The confidence interval is `estimate ± z se` with the normal quantile `z` at
`level`. Returns a [`TreatmentEffect`](@ref).
"""
function double_selection(y::AbstractVector{T}, d::AbstractVector{T}, X::AbstractMatrix{T};
                         method::Symbol = :doubleselection, level::Real = 0.95,
                         forced::AbstractVector{<:Integer} = Int[], post::Bool = true,
                         kwargs...) where {T<:AbstractFloat}
  n = length(y)
  length(d) == n && size(X, 1) == n || throw(DimensionMismatch("y, d and X must have the same number of observations"))
  0 < level < 1 || throw(ArgumentError("level must be in (0, 1), but it is $level."))
  if method === :doubleselection
    Sd = rlasso(X, d; post, kwargs...).selected
    Sy = rlasso(X, y; post, kwargs...).selected
    S = sort!(union(Sd, Sy, Int.(forced)))
    Z = [ones(T, n) d X[:, S]]
    coef = qr(Z, ColumnNorm()) \ y
    α = coef[2]
    ξ = (y .- Z * coef) .* sqrt(T(n) / (n - length(S) - 1))
    v = isempty(S) ? d .- mean(d) : (W = [ones(T, n) X[:, S]]; d .- W * (qr(W, ColumnNorm()) \ d))
    mv2 = mean(abs2, v)
    se = sqrt(mean(abs2.(v) .* abs2.(ξ)) / mv2^2 / n)
  elseif method === :partialling
    ry, rd = rlasso(X, y; post, kwargs...), rlasso(X, d; post, kwargs...)
    S = sort!(union(ry.selected, rd.selected))
    u, w = ry.residuals, rd.residuals .- mean(rd.residuals)
    α = dot(w, u .- mean(u)) / dot(w, w)
    res = (u .- mean(u)) .- α .* w
    se = sqrt(sum(abs2, res) / (n - 2) / dot(w, w))
  else
    throw(ArgumentError("method must be :doubleselection or :partialling, but it is :$method."))
  end
  t = α / se
  z = _qnorm(1 - (1 - T(level)) / 2)
  TreatmentEffect{T}(α, se, t, 2 * _pnorm(-abs(t)), (α - z * se, α + z * se), S, method, n)
end

function double_selection(y::AbstractVector{<:Real}, d::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}; kwargs...)
  T = float(promote_type(eltype(y), eltype(d), eltype(X)))
  double_selection(convert(Vector{T}, y), convert(Vector{T}, d), convert(Matrix{T}, X); kwargs...)
end
