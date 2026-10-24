######################################################################
#
#   Penalties other than the ℓ1 norm in lasso and LassoPath, and the
#   adaptive lasso
#
######################################################################

# The penalty of lasso and LassoPath at level one: nothing for the lasso (NormL1
# with the loadings of standardizeX), the elastic net Scaled(ElasticNet(α, 1 - α), 1)
# for alpha < 1, or the template penalty with its level set to one and its
# parameters converted to T
function _template(::Type{T}, penalty, alpha::Real) where {T}
  0 < alpha <= 1 || throw(ArgumentError("alpha must be in (0, 1], but it is $alpha."))
  if penalty === nothing
    return alpha == 1 ? nothing : Scaled(ElasticNet(T(alpha), T(1 - alpha)), one(T))
  end
  alpha == 1 || throw(ArgumentError("give the elastic net by alpha or by penalty, not both."))
  _atlevelone(T, penalty)
end

_atlevelone(::Type{T}, g::NormL1) where {T} = NormL1(one(T), g.w)
_atlevelone(::Type{T}, g::NormL1Nonneg) where {T} = NormL1Nonneg(one(T), g.w)
_atlevelone(::Type{T}, g::ElasticNet) where {T} = ElasticNet(one(T), T(g.λ2), g.w)
_atlevelone(::Type{T}, g::Scaled{<:ElasticNet}) where {T} = Scaled(ElasticNet(T(g.f.λ1), T(g.f.λ2), g.f.w), one(T))
_atlevelone(::Type{T}, g::MCP) where {T} = MCP(one(T), T(g.a))
_atlevelone(::Type{T}, g::SCAD) where {T} = SCAD(one(T), T(g.a))
_atlevelone(::Type, g::ProximalBase.BoxConstrained) = throw(ArgumentError("give bounds on the coefficients by lower and upper, not by Constrained in penalty."))
_atlevelone(::Type, g) = throw(ArgumentError("penalty must be a NormL1, NormL1Nonneg, ElasticNet, Scaled(ElasticNet(α, 1 - α), c), MCP or SCAD, whose level λ replaces, but it is a $(typeof(g))."))

# the penalty with level λ, as the paths of ProximalSolvers set it: the scale
# of a scaled elastic net, whose two terms then move together, and λ1 of an
# ElasticNet, whose λ2 is fixed
_atlevel(g::NormL1, λ) = NormL1(λ, g.w)
_atlevel(g::NormL1Nonneg, λ) = NormL1Nonneg(λ, g.w)
_atlevel(g::ElasticNet, λ) = ElasticNet(λ, g.λ2, g.w)
_atlevel(g::Scaled{<:ElasticNet}, λ) = Scaled(g.f, λ)
_atlevel(g::MCP, λ) = MCP(λ, g.a)
_atlevel(g::SCAD, λ) = SCAD(λ, g.a)
_atlevel(c::ProximalBase.BoxConstrained, λ) = ProximalBase.BoxConstrained(_atlevel(c.f, λ), c.box)

# The bounds lower ≤ β ≤ upper of lasso, LassoPath and glmLasso (numbers, or
# vectors with a bound per coefficient) as a box on the coefficients that the
# penalty applies to: s .* β when the columns are divided by the scales s, β
# for s = nothing. Returns nothing without bounds. As in glmnet, zero must be
# feasible, lower ≤ 0 ≤ upper, so that paths start from zero at λmax.
function _box(::Type{T}, lower, upper, p::Int, s = nothing) where {T}
  all(==(-Inf), lower) && all(==(Inf), upper) && return nothing
  l, u = _bound(T, lower, p, "lower"), _bound(T, upper, p, "upper")
  all(<=(0), l) || throw(ArgumentError("the lower bounds must be nonpositive, so that zero is feasible, as in glmnet."))
  all(>=(0), u) || throw(ArgumentError("the upper bounds must be nonnegative, so that zero is feasible, as in glmnet."))
  s === nothing ? IndBox(l, u) : IndBox(l .* s, u .* s)
end

function _bound(::Type{T}, b::Real, p::Int, name) where {T}
  isnan(b) && throw(ArgumentError("$name is NaN."))
  T(b)
end

function _bound(::Type{T}, b::AbstractVector{<:Real}, p::Int, name) where {T}
  length(b) == p || throw(DimensionMismatch("$name has length $(length(b)), but X has $p columns."))
  any(isnan, b) && throw(ArgumentError("$name contains NaN."))
  convert(Vector{T}, b)
end

# the penalty g restricted to the box (g itself for nothing)
_constrain(g, ::Nothing) = g
_constrain(g, box::IndBox) = Constrained(g, box)

# the ℓ1 penalties, whose standardization is computed as loadings
_isl1(template) = template === nothing || template isa Union{NormL1, NormL1Nonneg}

# the ℓ1 penalty of the template at level λ, its weights multiplied by the
# loadings ω (nothing for none)
_l1penalty(::Nothing, λ, ω) = ω === nothing ? NormL1(λ) : NormL1(λ, ω)
_l1penalty(g::NormL1, λ, ω) = NormL1(λ, _combine(g.w, ω))
_l1penalty(g::NormL1Nonneg, λ, ω) = NormL1Nonneg(λ, _combine(g.w, ω))
_combine(w, ::Nothing) = w
_combine(::Nothing, ω) = ω
_combine(w::AbstractArray, ω::AbstractVector{T}) where {T} = T.(vec(w)) .* ω
_combine(::Nothing, ::Nothing) = nothing

# the scales of the columns standardized for a penalty other than the ℓ1 norm:
# the loadings of standardizeX, with one for a column of zero spread, which
# then has no effect on the fit
function _scales(X::AbstractMatrix, intercept::Bool, w)
  s = _loadings(X, intercept, w)
  s[iszero.(s)] .= 1
  s
end

# the columns of X divided by the scales s, a copy (X itself for nothing)
_scaleColumns(X::AbstractMatrix, ::Nothing) = X
_scaleColumns(X::AbstractMatrix, s::Vector) = X ./ transpose(s)
_scaleColumns(X::SparseMatrixCSC, s::Vector) = X * Diagonal(inv.(s))

# the coefficients of the columns of the data from those of the columns divided
# by s
function _unscale!(x::SparseIterate, s::Vector)
  for k in activecoordinates(x)
    x[k] /= s[k]
  end
  x
end
_unscale(β::SparseVector, s::Vector) = SparseVector(length(β), copy(β.nzind), β.nzval ./ s[β.nzind])

# Minimizes f + g for lasso and LassoPath, g a penalty with a level: as
# _solve!, which, when options.warmStart is false, follows a path of
# options.numSteps levels from λmax for the ℓ1 norm, but for any penalty, and
# always for a
# nonconvex penalty (MCP, SCAD), whose local solution depends on the path, as in
# ncvreg; the levels of a nonconvex path are solved to the full tolerance
function _solveLevels!(x, m, g, options::CDOptions; rng::AbstractRNG = default_rng())
  convex = isconvex(g)
  path = options.numSteps > 0 && !(convex && options.warmStart)
  (path && !(g isa NormL1)) || return _solve!(x, m, g, options; rng)
  s = _solver(options; rng)
  fill!(x, zero(eltype(x)))
  solve!(x, m, g, s; continuation = options.numSteps, continuation_tol = convex ? sqrt(s.tol) : s.tol)
end

"""
    adaptiveLasso(X, y, λ, options=CDOptions(); init=:lasso, initλ=λ, γ=1, intercept=false, standardizeX=false,
                  kwargs...)

The adaptive lasso of Zou (2006),

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λ ∑ⱼ |βⱼ| / |β̃ⱼ|^γ,

whose loadings come from a first estimate `β̃`: large coefficients are
penalized less, so that they are less biased, and small ones more, so that
fewer variables that do not matter are selected. `β̃` is

- `init = :lasso`: the [`lasso`](@ref) at `initλ`, by default `λ`, with
  `intercept`, `standardizeX` and the keywords `kwargs`;
- `init = :ridge`: ridge regression at `initλ`,
  `‖y - β₀ - Xβ‖² / (2n) + initλ ‖β‖² / 2`, on the standardized columns with
  `standardizeX` (glmnet's `alpha = 0`);
- `init = β̃`, a vector: for example least squares, `X \\ y` when `n > p`, or the
  coefficients of a lasso whose level was chosen by cross-validation.

A coefficient with `β̃ⱼ = 0` has an infinite loading and stays at zero. The
second fit has no standardization, as with `γ = 1` the loadings already make it
invariant to the scales of the columns. The keywords `kwargs` (`weights`,
`offset`, `gram`, `rng` and the fields of [`CDOptions`](@ref)) apply to both
fits. Returns the [`LassoSolution`](@ref) of the second fit, whose `penalty` is
`NormL1(λ, ω)` with the loadings `ωⱼ = 1/|β̃ⱼ|^γ`.

Reference: H. Zou (2006). The adaptive lasso and its oracle properties.
*Journal of the American Statistical Association* 101(476), 1418–1429.
"""
function adaptiveLasso(X::AbstractMatrix{T}, y::AbstractVector{T}, λ::T, options::CDOptions=CDOptions();
                       init::Union{Symbol, AbstractVector{<:Real}}=:lasso, initλ::Real=λ, γ::Real=1,
                       intercept::Bool=false, standardizeX::Bool=false, kwargs...) where {T<:AbstractFloat}
  λ > 0 || throw(ArgumentError("λ must be positive, but it is $λ."))
  γ > 0 || throw(ArgumentError("γ must be positive, but it is $γ."))
  p = size(X, 2)
  β̃ = if init isa AbstractVector
    length(init) == p || throw(DimensionMismatch("init has length $(length(init)), but X has $p columns."))
    init
  elseif init === :lasso
    lasso(X, y, T(initλ), options; intercept, standardizeX, kwargs...).x
  elseif init === :ridge
    lasso(X, y, zero(T), options; intercept, standardizeX, penalty = ElasticNet(zero(T), T(initλ)), kwargs...).x
  else
    throw(ArgumentError("init must be :lasso, :ridge or a vector of $p coefficients, but it is :$init."))
  end
  ω = T[iszero(β̃[j]) ? T(Inf) : inv(abs(T(β̃[j])))^T(γ) for j in 1:p]
  lasso(X, y, λ, ω, options; intercept, kwargs...)
end

function adaptiveLasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
                       options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  adaptiveLasso(Xf, yf, T(λ), options; kwargs...)
end
