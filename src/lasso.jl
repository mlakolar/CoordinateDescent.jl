######################################################################
#
#   Lasso Solution
#
######################################################################

"""
    LassoSolution

Result returned by [`lasso`](@ref), [`sqrtLasso`](@ref),
[`scaledLasso`](@ref) and [`feasibleLasso!`](@ref).

# Fields
- `x::SparseIterate`: the estimated coefficients
- `residuals::Vector`: `y - intercept - X * x`
- `penalty`: the penalty at which `x` was computed, a `NormL1` for the ℓ1
  estimators, or the penalty of [`lasso`](@ref)'s `alpha` or `penalty` at the
  level `λ`
- `σ`: an estimate of the noise level (see the documentation of each solver),
  or `nothing`
- `iterations::Int`: passes of coordinate descent over the coordinates, summed
  over all the lasso fits the solver made
- `converged::Bool`: whether coordinate descent met its stopping rule within
  `maxIter` passes and, for [`scaledLasso!`](@ref) and [`feasibleLasso!`](@ref),
  the outer iteration met its stopping rule within its `maxIter` iterations
- `intercept`: the unpenalized intercept, zero unless the solver was called
  with `intercept = true`
- `hasintercept::Bool`: whether the solver fitted an intercept
- `weights`: the observation weights of the fit, `nothing` for none
- `scales`: `nothing` when `penalty` applies to `x`, or the scales `s` of the
  columns when it applies to `s .* x`, the coefficients of the standardized
  columns `X[:, j] / s[j]` (`standardizeX` with a penalty other than the ℓ1
  norm)

The functions of StatsAPI apply to it: [`coef`](@ref), [`predict`](@ref),
[`residuals`](@ref), [`nobs`](@ref), [`dof`](@ref) and [`deviance`](@ref).
"""
struct LassoSolution{T, S}
  x::SparseIterate{T}
  residuals::Vector{T}
  penalty::S
  σ::Union{T, Nothing}
  iterations::Int
  converged::Bool
  intercept::T
  hasintercept::Bool
  weights::Union{Nothing, Vector{T}}
  scales::Union{Nothing, Vector{T}}

  LassoSolution{T, S}(x::SparseIterate{T}, residuals::AbstractVector{T}, penalty::S, σ::T,
                      iterations::Integer, converged::Bool, intercept::T, hasintercept::Bool,
                      weights::Union{Nothing, Vector{T}} = nothing,
                      scales::Union{Nothing, Vector{T}} = nothing) where {T, S<:AbstractFunction} =
    new(x, residuals, penalty, σ, iterations, converged, intercept, hasintercept, weights, scales)
end

# the penalty loadings of standardized columns: the standard deviations
# sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n) of the columns, centered at their means x̄ⱼ with an
# intercept and at zero otherwise
function _loadings(X::AbstractMatrix{T}, intercept::Bool) where {T}
  n, p = size(X)
  xbar = intercept ? _column_means(X) : zeros(T, p)
  ω = Vector{T}(undef, p)
  @inbounds for k in 1:p
    c = xbar[k]
    a = zero(T)
    @simd for i in 1:n
      a += (X[i, k] - c) * (X[i, k] - c)
    end
    ω[k] = sqrt(a / n)
  end
  ω
end

# the same over the stored entries of a sparse X: the n - nnz entries of a
# column that are not stored contribute x̄ⱼ² each
function _loadings(X::SparseMatrixCSC{T}, intercept::Bool) where {T}
  n, p = size(X)
  xbar = intercept ? _column_means(X) : zeros(T, p)
  vals = nonzeros(X)
  ω = Vector{T}(undef, p)
  for k in 1:p
    c = xbar[k]
    r = nzrange(X, k)
    a = (n - length(r)) * c * c
    for j in r
      a += (vals[j] - c) * (vals[j] - c)
    end
    ω[k] = sqrt(a / n)
  end
  ω
end

# the loadings of standardized columns with observation weights w: the weighted
# standard deviations sqrt(∑ᵢ wᵢ (Xᵢⱼ - x̄ⱼ)² / ∑ᵢ wᵢ), centered at the weighted
# means x̄ⱼ with an intercept and at zero otherwise
_loadings(X::AbstractMatrix, intercept::Bool, ::Nothing) = _loadings(X, intercept)

function _loadings(X::AbstractMatrix{T}, intercept::Bool, w::AbstractVector{T}) where {T}
  n, p = size(X)
  W = sum(w)
  xbar = intercept ? _column_means(X, w) : zeros(T, p)
  ω = Vector{T}(undef, p)
  if issparse(X)
    rows, vals = rowvals(X), nonzeros(X)
    for k in 1:p
      c, a, wnz = xbar[k], zero(T), zero(T)
      for j in nzrange(X, k)
        a += w[rows[j]] * (vals[j] - c) * (vals[j] - c)
        wnz += w[rows[j]]
      end
      ω[k] = sqrt(max(a + (W - wnz) * c * c, zero(T)) / W)
    end
  else
    @inbounds for k in 1:p
      c, a = xbar[k], zero(T)
      @simd for i in 1:n
        a += w[i] * (X[i, k] - c) * (X[i, k] - c)
      end
      ω[k] = sqrt(a / W)
    end
  end
  ω
end

# whether to solve on the Gram matrix: gram is true, false or :auto, which picks
# it for a path on dense data with at least as many rows as columns and not too
# many columns; at one λ, forming the Gram matrix costs more than the few passes
# it makes cheaper (benchmark/gram_bench.jl)
const _GRAM_PMAX = 2000

function _usegram(gram, X::AbstractMatrix, path::Bool)
  gram isa Bool && return gram
  gram === :auto || throw(ArgumentError("gram must be true, false or :auto, but it is $gram."))
  n, p = size(X)
  path && !issparse(X) && n >= p && p <= _GRAM_PMAX
end

# A = (X - 1x̄')'W(X - 1x̄')/∑ᵢ wᵢ, the Gram matrix of the centered columns with
# observation weights w (W = I and ∑ᵢ wᵢ = n without them), formed from blocks
# of rows so that the centered data are never copied whole, and from X itself
# without centering and weights
function _gram(X::AbstractMatrix{T}, xbar::Vector{T}, w::Union{Nothing, AbstractVector{T}} = nothing) where {T}
  n, p = size(X)
  W = w === nothing ? T(n) : sum(w)
  A = zeros(T, p, p)
  blas = T <: LinearAlgebra.BlasFloat
  if issparse(X)
    # X'WX - W x̄x̄', as X'w = W x̄ for the weighted means x̄
    A .= Matrix(transpose(X) * (w === nothing ? X : w .* X))
    any(!iszero, xbar) && (A .-= W .* (xbar .* transpose(xbar)))
  elseif blas && X isa StridedMatrix && all(iszero, xbar) && w === nothing
    BLAS.syrk!('U', 'T', one(T), X, zero(T), A)
  else
    rows = max(1, min(n, 2^20 ÷ max(p, 1)))
    B = Matrix{T}(undef, rows, p)
    for i0 in 1:rows:n
      r = i0:min(n, i0 + rows - 1)
      Bv = view(B, 1:length(r), :)
      Bv .= view(X, r, :) .- transpose(xbar)
      w === nothing || (Bv .*= sqrt.(view(w, r)))
      blas ? BLAS.syrk!('U', 'T', one(T), Bv, one(T), A) : mul!(A, transpose(Bv), Bv, one(T), one(T))
    end
  end
  if blas && !issparse(X)       # syrk! fills the upper triangle
    for j in 1:p, i in j+1:p
      A[i, j] = A[j, i]
    end
  end
  A ./= W
end

# The least squares loss ∑ᵢ wᵢ (yᵢ - oᵢ - β₀ - Xᵢ'β)² / (2∑ᵢ wᵢ), with observation
# weights w (wᵢ = 1 and ∑ᵢ wᵢ = n for nothing) and an offset o (zero for
# nothing), or with squareroot its square root √(∑ᵢ wᵢ (yᵢ - oᵢ - β₀ - Xᵢ'β)²), as
# coordinate descent takes it. On the residuals a coordinate step costs O(n).
# With gram, on A = (X - 1x̄')'W(X - 1x̄')/∑w and b = -X'W(y - o - ȳ)/∑w (x̄ and ȳ
# the weighted means), a step costs O(p) after O(np²) to form A: the quadratic
# β'Aβ/2 + b'β + c, whose constant c = ∑ᵢ wᵢ (yᵢ - oᵢ - ȳ)²/(2∑w) makes it the same
# loss with the same stopping rule, or √(β'Aβ + 2b'β + 2c), the square root
# divided by √∑w. Returns the model, the factor by which the penalty is divided
# (√∑w for the square root on the Gram matrix, 1 otherwise), and for the Gram
# form the column means and the mean of y - o, which give the intercept and the
# residuals (nothing otherwise).
function _lsmodel(X::AbstractMatrix{T}, y::AbstractVector{T}, intercept::Bool, gram::Bool;
                  squareroot::Bool = false, w = nothing, o = nothing) where {T}
  if !gram
    m = if w === nothing && o === nothing
      squareroot ? CDSqrtLassoLoss(y, X; intercept) : CDLeastSquaresLoss(y, X; intercept)
    elseif squareroot
      cdmodel(SqrtLeastSquaresLoss(_minus(y, o), X; weights = w), nothing; intercept)
    else
      W = w === nothing ? T(size(X, 1)) : sum(w)
      cdmodel((1 / W) * GLMLoss(ProximalBase.Squared(), y, X; weights = w, offset = o), nothing; intercept)
    end
    return m, one(T), nothing
  end
  _check_finite(y, "y")
  _check_finite(X, "X")
  n, p = size(X)
  W = w === nothing ? T(n) : sum(w)
  yo = _minus(y, o)
  xbar = intercept ? _column_means(X, w) : zeros(T, p)
  ybar = intercept ? (w === nothing ? mean(yo) : dot(w, yo) / W) : zero(T)
  yc = yo .- ybar
  wyc = w === nothing ? yc : w .* yc
  A = _gram(X, xbar, w)
  b = (transpose(X) * wyc) ./ (-W)        # (X - 1x̄')'W(y - o - ȳ) = X'W(y - o - ȳ)
  c = dot(yc, wyc) / (2W)
  m = squareroot ? cdmodel(SqrtQuadraticFunction(A, b, 2c; check = false), nothing) :
                   cdmodel(QuadraticFunction(A, b, c), nothing)
  m, squareroot ? sqrt(W) : one(T), (xbar, ybar)
end

# the intercept β₀ = ȳ - x̄'β of the Gram form, from its column means and mean of y
function _gramintercept((xbar, ybar), x::SparseIterate{T}) where {T}
  v = ybar
  for k in activecoordinates(x)
    v -= xbar[k] * x[k]
  end
  v
end

# the intercept of a fit by the model m, which gd describes for the Gram form
_intercept(m, x, ::Nothing) = ProximalSolvers.intercept(m, x)
_intercept(m, x, gd) = _gramintercept(gd, x)

# the solution of a lasso fit by the model m at the penalty g, with weights w and
# offset o; the Gram form (gd, see _lsmodel) has no residuals, which are computed
# from X and y. σ is the sample standard deviation of the residuals, or with
# weights their weighted root mean square. With the column scales s, X holds the
# columns divided by s and x their coefficients, which are mapped back to those
# of the columns of the data
function _lassoSolution(x::SparseIterate{T}, m, g, r::CDResults, intercept::Bool, X, y, gd;
                        w = nothing, o = nothing, scales = nothing) where {T}
  β0 = _intercept(m, x, gd)
  res = gd === nothing ? ProximalSolvers.residuals(m) : _minus(y, o) .- β0 .- Vector(X * _sparsevec(x))
  σ = w === nothing ? std(res) : _getSigma(w, res)
  scales === nothing || _unscale!(x, scales)
  LassoSolution{T, typeof(g)}(x, res, g, σ, r.passes, r.converged, β0, intercept, w, scales)
end


######################################################################
#
#   Lasso Interface
#
######################################################################

"""
    lasso(X, y, λ, options=CDOptions(); intercept=false, standardizeX=false, gram=:auto, weights=nothing,
          offset=nothing, alpha=1, penalty=nothing, rng=Random.default_rng(), kwargs...)
    lasso(X, y, λ, ω, options=CDOptions(); intercept=false, gram=:auto, weights=nothing, offset=nothing,
          rng=Random.default_rng(), kwargs...)

Solve the lasso

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λ ∑ⱼ ωⱼ |βⱼ|

by coordinate descent. The intercept `β₀` is not penalized, and it is zero
unless `intercept` is true. In the first form the loadings are `ωⱼ = 1`, or
with `standardizeX` the standard deviations `sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)` of the
columns, with `x̄ⱼ` the column mean when `intercept` is true and zero
otherwise; the coefficients are on the scale of the data either way. Returns a
[`LassoSolution`](@ref) whose `σ` is the sample standard deviation of the
residuals.

The keyword arguments `kwargs` are fields of [`CDOptions`](@ref) that replace
those of `options`, as in `lasso(X, y, λ; maxIter=10_000, optTol=1e-8)`, and
the random order of coordinate descent is drawn from `rng`.

`X` can be any matrix and `y` any vector. A `Matrix`, or a view of a range of
its rows or of some of its columns, is read in place. A `SparseMatrixCSC` stays
sparse; coordinate descent still reads its columns entry by entry, so for now
it is slower than the dense matrix. Any other matrix, such as a transpose or a
view of selected rows, is copied once into a `Matrix` (a `SparseMatrixCSC` if
it is sparse). The same holds for every solver of this package.

With `gram = true`, coordinate descent works on the Gram matrix of the
(centered) columns, `A = X'X / n`, instead of the residuals. Each coordinate
step then costs `O(p)` instead of `O(n)`, after `O(np²)` operations and `p²`
numbers of memory to form `A`. The loss, the stopping rule and so the
solution are the same, to the tolerance. Forming `A` pays off over the many
passes of a path when `n ≥ p`, but rarely at one `λ`. The default,
`gram = :auto`, uses it for [`LassoPath`](@ref) on a dense `X` with `n ≥ p`
and `p ≤ $(_GRAM_PMAX)`, and never at one `λ`.

With observation `weights` `w` (nonnegative, with a positive sum) and an
`offset` `o`, the loss is

    ∑ᵢ wᵢ (yᵢ - oᵢ - β₀ - Xᵢ'β)² / (2 ∑ᵢ wᵢ).

Dividing by `∑ᵢ wᵢ` follows glmnet: `λ` means the same with and without weights,
unit weights give the unweighted fit, integer weights the fit of replicated
rows, and zero weights leave observations out. The intercept, the loadings of
`standardizeX` (`sqrt(∑ᵢ wᵢ (Xᵢⱼ - x̄ⱼ)² / ∑ᵢ wᵢ)`) and the Gram matrix are
weighted as well. The residuals are `y - o - β₀ - Xβ`, and with weights `σ` is
their weighted root mean square, `sqrt(∑ᵢ wᵢ rᵢ² / ∑ᵢ wᵢ)`.

# Other penalties

`alpha < 1` gives glmnet's elastic net, `λ (α ∑ⱼ |βⱼ| + (1 - α) ‖β‖² / 2)`,
for `0 < α ≤ 1`. `penalty` gives another penalty of ProximalBase whose level
`λ` replaces: `NormL1(1, w)` and `NormL1Nonneg(1, w)` (the lasso and the
nonnegative lasso with weights `w`), `ElasticNet(1, λ2, w)` (`λ ∑ⱼ wⱼ|βⱼ| +
λ2 ‖β‖²/2`, with `λ2` fixed), `Scaled(ElasticNet(α, 1 - α, w), 1)` (the elastic
net with weights on its ℓ1 part), and the nonconvex `MCP(1, a)` and
`SCAD(1, a)`, whose concavity `a` is kept. The level of the penalty given is
not used: `MCP(1, 3)` means "MCP with `a = 3`".

With `standardizeX`, the penalty applies to the coefficients of the standardized
columns `X[:, j] / sⱼ`, `sⱼ` their standard deviations, as in glmnet and ncvreg;
the coefficients returned are those of the columns of `X`. For the ℓ1 norm this
is the same as the loadings `ωⱼ = sⱼ`, which is how it is computed. For the
other penalties `X` is divided by `s` once (a copy), and `s` is stored as the
solution's `scales`.

A nonconvex penalty has local solutions, and the one found depends on the
starting point. As in ncvreg, it is computed along `options.numSteps` levels
from `λmax`, the smallest level with a zero solution, down to `λ`, each solved
to the tolerance and warm started from the previous one; `numSteps = 0` solves
at `λ` from zero. Use [`LassoPath`](@ref) for the solutions at many levels. For
the steps of coordinate descent to be exact, MCP needs `a > 1` and SCAD
`a > 2` on standardized columns. With `options.warmStart` false, a convex
penalty other than the ℓ1 norm is also solved along such a path.
"""
function lasso(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  options::CDOptions=CDOptions();
  intercept::Bool=false,
  standardizeX::Bool=false,
  gram::Union{Bool, Symbol}=:auto,
  weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
  offset::Union{Nothing, AbstractVector{<:Real}}=nothing,
  alpha::Real=1,
  penalty::Union{Nothing, AbstractFunction}=nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...) where {T<:AbstractFloat}

  options = _options(options, kwargs)
  X, y = _design(X), _response(y)
  w, o = _weights(T, weights, size(X, 1)), _offset(T, offset, size(X, 1))
  template = _template(T, penalty, alpha)
  x = SparseIterate(T, size(X, 2))
  if _isl1(template)
    m, _, gd = _lsmodel(X, y, intercept, _usegram(gram, X, false); w, o)
    g = _l1penalty(template, λ, standardizeX ? _loadings(X, intercept, w) : nothing)
    return _lassoSolution(x, m, g, _solveLevels!(x, m, g, options; rng), intercept, X, y, gd; w, o)
  end
  s = standardizeX ? _scales(X, intercept, w) : nothing
  Xs = _scaleColumns(X, s)
  m, _, gd = _lsmodel(Xs, y, intercept, _usegram(gram, X, false); w, o)
  g = _atlevel(template, λ)
  _lassoSolution(x, m, g, _solveLevels!(x, m, g, options; rng), intercept, Xs, y, gd; w, o, scales = s)
end

function lasso(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions();
  intercept::Bool=false,
  gram::Union{Bool, Symbol}=:auto,
  weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
  offset::Union{Nothing, AbstractVector{<:Real}}=nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...) where {T<:AbstractFloat}

  options = _options(options, kwargs)
  X, y = _design(X), _response(y)
  w, o = _weights(T, weights, size(X, 1)), _offset(T, offset, size(X, 1))
  x = SparseIterate(T, size(X, 2))
  m, _, gd = _lsmodel(X, y, intercept, _usegram(gram, X, false); w, o)
  g = NormL1(λ, ω)
  _lassoSolution(x, m, g, _solve!(x, m, g, options; rng), intercept, X, y, gd; w, o)
end

function lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
  options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  lasso(Xf, yf, T(λ), options; kwargs...)
end

function lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
  ω::AbstractArray{<:Real}, options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  lasso(Xf, yf, T(λ), convert(Array{T}, ω), options; kwargs...)
end

######################################################################
#
#   Sqrt-Lasso Interface
#
######################################################################

"""
    sqrtLasso(X, y, λ, options=CDOptions(); intercept=false, standardizeX=true, gram=:auto, weights=nothing,
              offset=nothing, rng=Random.default_rng(), kwargs...)
    sqrtLasso(X, y, λ, ω, options=CDOptions(); intercept=false, gram=:auto, weights=nothing, offset=nothing,
              rng=Random.default_rng(), kwargs...)

Solve the square-root lasso of Belloni, Chernozhukov and Wang (2011)

    minimize  ‖y - β₀ - Xβ‖ + λ ∑ⱼ ωⱼ |βⱼ|

by coordinate descent, with an unpenalized intercept `β₀` that is zero unless
`intercept` is true. In the first form the loadings are the standard deviations
`ωⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)` of the columns when `standardizeX` is true
(centered only with an intercept) and `ωⱼ = 1` otherwise. The optimal `λ`
does not depend on the noise level. Returns a [`LassoSolution`](@ref) whose
`σ` is the sample standard deviation of the residuals. The keyword arguments
`kwargs`, `gram` and `rng` are as for [`lasso`](@ref); on the Gram matrix the
loss is the square root of a quadratic (as in [`CDSqrtQuadraticLoss`](@ref)).
With observation `weights` `w` and an `offset` `o`, the loss is the weighted
norm `sqrt(∑ᵢ wᵢ (yᵢ - oᵢ - β₀ - Xᵢ'β)²)`, the loadings are weighted as in
[`lasso`](@ref), and `σ` is the weighted root mean square of the residuals.
"""
function sqrtLasso(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  options::CDOptions=CDOptions();
  intercept::Bool=false,
  standardizeX::Bool=true,
  gram::Union{Bool, Symbol}=:auto,
  weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
  offset::Union{Nothing, AbstractVector{<:Real}}=nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...) where {T<:AbstractFloat}

  options = _options(options, kwargs)
  X, y = _design(X), _response(y)
  w, o = _weights(T, weights, size(X, 1)), _offset(T, offset, size(X, 1))
  ω = standardizeX ? _loadings(X, intercept, w) : nothing
  _sqrtLasso(X, y, λ, ω, options, intercept, gram, rng, w, o)
end

# the square-root lasso with loadings ω (nothing for none), weights w and offset
# o; on the Gram matrix the loss is divided by √∑w, and so is the penalty level
function _sqrtLasso(X::AbstractMatrix{T}, y::AbstractVector{T}, λ::T, ω, options::CDOptions,
                    intercept::Bool, gram, rng::AbstractRNG, w, o) where {T}
  x = SparseIterate(T, size(X, 2))
  m, s, gd = _lsmodel(X, y, intercept, _usegram(gram, X, false); squareroot = true, w, o)
  penalty(l) = ω === nothing ? NormL1(l) : NormL1(l, ω)
  r = _solve!(x, m, penalty(λ / s), options; rng)
  _lassoSolution(x, m, penalty(λ), r, intercept, X, y, gd; w, o)
end

function sqrtLasso(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::Array{T},
  options::CDOptions=CDOptions();
  intercept::Bool=false,
  gram::Union{Bool, Symbol}=:auto,
  weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
  offset::Union{Nothing, AbstractVector{<:Real}}=nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...) where {T<:AbstractFloat}

  options = _options(options, kwargs)
  X, y = _design(X), _response(y)
  w, o = _weights(T, weights, size(X, 1)), _offset(T, offset, size(X, 1))
  _sqrtLasso(X, y, λ, ω, options, intercept, gram, rng, w, o)
end

function sqrtLasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
  options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  sqrtLasso(Xf, yf, T(λ), options; kwargs...)
end

function sqrtLasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
  ω::AbstractArray{<:Real}, options::CDOptions=CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  sqrtLasso(Xf, yf, T(λ), convert(Array{T}, ω), options; kwargs...)
end


######################################################################
#
#   Scaled Lasso Interface
#
######################################################################

"""
    scaledLasso!(x, X, y, λ, ω, options=IterLassoOptions(); intercept=false, weights=nothing, offset=nothing,
                 rng=Random.default_rng(), kwargs...)
    scaledLasso(X, y, λ, ω, options=IterLassoOptions(); intercept=false, weights=nothing, offset=nothing,
                rng=Random.default_rng(), kwargs...)
    scaledLasso!(x, f::CDSqrtQuadraticLoss, λ, ω, options=CDOptions(); rng=Random.default_rng(), kwargs...)

Scaled lasso of Sun and Zhang (2012), which estimates the coefficients and the
noise level jointly:

    minimize  ‖y - β₀ - Xβ‖² / (2nσ) + σ/2 + λ ∑ⱼ ωⱼ |βⱼ|   over β and σ > 0.

The intercept `β₀` is not penalized, and it is zero unless `intercept` is true.
The minimum over `σ` is at `σ̂ = ‖y - β₀ - Xβ‖ / √n`, so `β̂` solves the
square-root lasso of Belloni, Chernozhukov and Wang (2011),

    minimize  ‖y - β₀ - Xβ‖ / √n + λ ∑ⱼ ωⱼ |βⱼ|,

which one run of coordinate descent solves (on [`CDSqrtLassoLoss`](@ref), with
the penalty level `√n λ`), with the options `options.optionsCD`; the other
fields of [`IterLassoOptions`](@ref) are not used. Coordinate descent starts
from and overwrites `x` (unless `options.optionsCD.warmStart` is false).
Returns a [`LassoSolution`](@ref) whose `σ` is `σ̂`, and whose penalty is that of
the lasso that `β̂` solves, `NormL1(λσ̂, ω)`.

The third form solves the same problem from sufficient statistics: with the
loss [`CDSqrtQuadraticLoss`](@ref)`(A, b, c)`, `A = X'X/n`, `b = -X'y/n` and
`c = y'y/n`, it needs neither `X` nor `y`, and a coordinate update costs `O(p)`
instead of `O(n)`. With `A = S`, `b = -S[:, j]`, `c = S[j, j]` and `ω[j] = Inf`
it is the scaled lasso of variable `j` on the others from their covariance
matrix `S`, as in the precision matrix estimator of Sun and Zhang (2013); for
all the variables, check `S` once and build the losses with `check = false`.
Its solution has no residuals (an empty vector) and no intercept.

The keyword arguments `kwargs` are fields of the options that replace those of
`options` (of [`IterLassoOptions`](@ref) in the first two forms, of
[`CDOptions`](@ref) in the third), and the random order of coordinate descent
is drawn from `rng`. With observation `weights` `w` and an `offset` `o`, the
loss is `∑ᵢ wᵢ rᵢ² / (2σ ∑ᵢ wᵢ) + σ/2` for the residuals `r = y - o - β₀ - Xβ`,
so that `σ̂ = sqrt(∑ᵢ wᵢ rᵢ² / ∑ᵢ wᵢ)`.

Before version 0.6.1, `scaledLasso!` alternated lasso fits with updates of `σ̂`
until the relative change of `σ̂` was below `options.optTol` (by default 1e-2),
which stopped short of the solution and solved a lasso at every iteration.
"""
function scaledLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::AbstractVector{T},
  options::IterLassoOptions=IterLassoOptions();
  intercept::Bool=false,
  weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
  offset::Union{Nothing, AbstractVector{<:Real}}=nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...
  )  where {T<:AbstractFloat}

  options = _options(options, kwargs)
  X, y = _design(X), _response(y)
  n = size(X, 1)
  w, o = _weights(T, weights, n), _offset(T, offset, n)
  W = w === nothing ? T(n) : sum(w)
  m, _, _ = _lsmodel(X, y, intercept, false; squareroot = true, w, o)
  r = _solve!(x, m, NormL1(sqrt(W) * λ, ω), options.optionsCD; rng)
  res = ProximalSolvers.residuals(m)
  σ = w === nothing ? norm(res) / sqrt(T(n)) : _getSigma(w, res)
  g = NormL1(λ * σ, ω)
  LassoSolution{T, typeof(g)}(x, res, g, σ, r.passes, r.converged, ProximalSolvers.intercept(m, x), intercept, w)
end

# reweight! with the options of the iterative procedures, warm starting each
# lasso fit from the previous one; with maxIter = 0 there is no fit
function _reweight!(x, m, g, update, options::IterLassoOptions; rng::AbstractRNG = default_rng())
  if options.maxIter == 0
    ProximalSolvers.initialize!(m, x)
    return ReweightResults(x, g, g, 0, false, NaN, 0, true)
  end
  reweight!(x, m, g, update, _solver(options.optionsCD; rng); maxiter = options.maxIter, tol = options.optTol)
end

scaledLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{<:Real},
  y::AbstractVector{<:Real},
  λ::Real,
  ω::AbstractVector{<:Real},
  options::IterLassoOptions=IterLassoOptions();
  kwargs...
  ) where {T<:AbstractFloat} =
  scaledLasso!(x, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y), T(λ),
               convert(AbstractVector{T}, ω), options; kwargs...)

function scaledLasso!(
  x::SparseIterate{T},
  f::SqrtQuadModel{T},
  λ::T,
  ω::AbstractVector{T},
  options::CDOptions=CDOptions();
  rng::AbstractRNG=default_rng(),
  kwargs...) where {T<:AbstractFloat}

  r = _solve!(x, f, NormL1(λ, ω), _options(options, kwargs); rng)
  σ = ProximalSolvers.loss_value(f, x)
  g = NormL1(λ * σ, ω)
  LassoSolution{T, typeof(g)}(x, T[], g, σ, r.passes, r.converged, zero(T), false)
end

scaledLasso!(x::SparseIterate{T}, f::SqrtQuadModel{T}, λ::Real, ω::AbstractVector{<:Real},
             options::CDOptions=CDOptions(); kwargs...) where {T<:AbstractFloat} =
  scaledLasso!(x, f, T(λ), convert(AbstractVector{T}, ω), options; kwargs...)

"""
    scaledLasso(X, y, λ, ω, options=IterLassoOptions())

Same as [`scaledLasso!`](@ref), starting from a zero iterate.
"""
scaledLasso(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ::T,
  ω::AbstractVector{T},
  options::IterLassoOptions=IterLassoOptions();
  kwargs...
  ) where {T<:AbstractFloat} = scaledLasso!(SparseIterate(T, size(X, 2)), X, y, λ, ω, options; kwargs...)

function scaledLasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, λ::Real,
  ω::AbstractVector{<:Real}, options::IterLassoOptions=IterLassoOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  scaledLasso(Xf, yf, T(λ), convert(AbstractVector{T}, ω), options; kwargs...)
end


######################################################################
#
#   Feasible Lasso Interface
#
######################################################################

"""
    feasibleLasso!(x, X, y, λ0, options=IterLassoOptions(); intercept=false, rng=Random.default_rng(), kwargs...)

Heteroscedasticity-robust lasso of Belloni, Chen, Chernozhukov and Hansen
(2012), which uses data-driven penalty loadings. It alternates between the
lasso

    β̂ = argmin  ‖y - β₀ - Xβ‖² / (2n) + λ0 ∑ⱼ Γⱼ |βⱼ|

and the update of the loadings `Γⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² rᵢ² / n)` from the
residuals `r = y - β̂₀ - Xβ̂`, until the relative change in `Γ` is below
`options.optTol` (by `ProximalSolvers.reweight!`), or warns after
`options.maxIter` iterations. The intercept `β₀` is not penalized; it and the
column means `x̄ⱼ` are zero unless `intercept` is true.

The initial residuals are chosen by `options.initProcedure` (see
[`IterLassoOptions`](@ref)); with `:InitStd` they come from a lasso with
penalty `λ0 σinit ∑ⱼ sⱼ |βⱼ|`, `sⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)`. Coordinate descent
starts from and overwrites `x`, and each lasso fit is warm started from the
previous one. Returns a [`LassoSolution`](@ref) whose `penalty.w` holds the
loadings of the last fit and whose `σ` is the sample standard deviation of the
residuals. The keyword arguments `kwargs` are fields of
[`IterLassoOptions`](@ref) that replace those of `options`, as in
`feasibleLasso!(x, X, y, λ0; maxIter=50)`, and the random order of coordinate
descent is drawn from `rng`.
"""
function feasibleLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  λ0::T,
  options::IterLassoOptions=IterLassoOptions();
  intercept::Bool=false,
  rng::AbstractRNG=default_rng(),
  kwargs...
  )  where {T<:AbstractFloat}

  options = _options(options, kwargs)
  X, y = _design(X), _response(y)
  n, p = size(X)
  m = CDLeastSquaresLoss(y, X; intercept)
  xbar = intercept ? _column_means(X) : zeros(T, p)
  passes, cd_converged = 0, true

  # initial residuals; IterLassoOptions allows only these three procedures
  if options.initProcedure == :Screening
    r0 = _findInitResiduals!(X, y, options.sinit, Vector{T}(undef, n); intercept)
  elseif options.initProcedure == :InitStd
    σ = T(options.σinit)
    r = _solve!(x, m, NormL1(λ0 * σ, _loadings(X, intercept)), options.optionsCD; rng)
    passes, cd_converged = r.passes, r.converged
    r0 = ProximalSolvers.residuals(m)
  else    # :WarmStart
    ProximalSolvers.initialize!(m, x)
    r0 = ProximalSolvers.residuals(m)
  end

  # alternate the lasso with loadings Γ with the update of Γ from its residuals
  Γ = _getLoadings!(Vector{T}(undef, p), X, r0, xbar)
  function update(g, x, m)
    Γnew = _getLoadings!(Vector{T}(undef, p), X, ProximalSolvers.residuals(m), xbar)
    NormL1(λ0, Γnew), maximum(abs.(g.w .- Γnew)) / maximum(Γnew)
  end
  res = _reweight!(x, m, NormL1(λ0, Γ), update, options; rng)
  res.converged || @warn "feasibleLasso! did not converge in maxIter = $(options.maxIter) iterations (relative change of the penalty loadings in the last iteration $(round(res.change; sigdigits = 3)) > optTol = $(options.optTol)). Increase maxIter in IterLassoOptions."

  r = ProximalSolvers.residuals(m)
  LassoSolution{T, typeof(res.penalty)}(x, r, res.penalty, std(r), passes + res.passes,
                                        res.converged && res.cd_converged && cd_converged,
                                        ProximalSolvers.intercept(m, x), intercept)
end

feasibleLasso!(
  x::SparseIterate{T},
  X::AbstractMatrix{<:Real},
  y::AbstractVector{<:Real},
  λ0::Real,
  options::IterLassoOptions=IterLassoOptions();
  kwargs...
  ) where {T<:AbstractFloat} =
  feasibleLasso!(x, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y), T(λ0), options; kwargs...)

######################################################################
#
#   Lasso Path Interface
#
######################################################################

"""
    LassoPath(X, y, λpath, options=CDOptions(); max_hat_s=Inf, standardizeX=true, intercept=false, screening=true,
              gram=:auto, weights=nothing, offset=nothing, alpha=1, penalty=nothing, rng=Random.default_rng(),
              kwargs...)
    LassoPath(X, y, options=CDOptions(); nλ=100, λminratio, kwargs...)

Lasso solutions `β0path[k]` and `βpath[k]` of

    minimize  ‖y - β₀ - Xβ‖² / (2n) + λpath[k] ∑ⱼ ωⱼ |βⱼ|

along a decreasing sequence `λpath`, each one warm started from the previous
solution, by `ProximalSolvers.solve_path`. The intercept `β₀` is not penalized, and it is zero unless
`intercept` is true. The loadings are the standard deviations
`ωⱼ = sqrt(∑ᵢ (Xᵢⱼ - x̄ⱼ)² / n)` of the columns when `standardizeX` is true
(centered only with an intercept) and `ωⱼ = 1` otherwise.

Without `λpath`, the path has `nλ` values spaced evenly on the log scale from
`λmax`, the smallest value whose solution is zero, down to `λminratio ⋅ λmax`;
`λminratio` defaults to `0.01` when `X` has more columns than rows and to
`1e-4` otherwise. The first value is `λmax` times `1 + √eps`, so that rounding
cannot make the first solution nonzero.

With `screening`, each λ is solved first over the coordinates kept by the
sequential strong rule of Tibshirani et al. (2012), and the optimality
conditions of the others are checked afterwards; coordinates that violate them
are added and the problem solved again. `kktChecks[k]` is the number of checks
at `λpath[k]`: one when the rule kept every coordinate it should have, zero
without screening. When `options.warmStart` is false, each value is solved
from zero along its own path of `options.numSteps` levels, without screening.

The path stops early, after the first solution with more than `max_hat_s`
nonzero coefficients; the returned `λpath` then contains only the values that
were solved for. The solutions are stored as `SparseVector`s, so a path takes
memory proportional to the number of nonzero coefficients. `iterations[k]` and
`converged[k]` are the passes of coordinate descent at `λpath[k]` and whether it
met its stopping rule. See also [`refitLassoPath`](@ref).

The keyword arguments `kwargs` that are not listed are fields of
[`CDOptions`](@ref) that replace those of `options`, and the random order of
coordinate descent is drawn from `rng`. With `gram`, the path is solved on the
Gram matrix of the columns, and with `weights` and `offset` the loss is the
weighted one, as described for [`lasso`](@ref); `λmax` and the loadings are
weighted too. `fit(LassoPath, X, y, ...)` is the same
as `LassoPath(X, y, ...)`. The functions of StatsAPI apply to the path:
[`coef`](@ref), [`predict`](@ref) and [`nobs`](@ref); `nobs` is the number of
observations, also stored as `nobs`.

`alpha` and `penalty` give the elastic net and other penalties, as for
[`lasso`](@ref): `LassoPath(X, y; alpha=0.5)` is glmnet's elastic-net path at
a fixed `α`, whose `λmax` is that of the lasso divided by `α`, and
`LassoPath(X, y; penalty=MCP(1, 3))` the path of ncvreg, on standardized
columns with `standardizeX`. Along a path of a nonconvex penalty, each solution
is the local one reached from the previous solution, starting from zero at
`λmax`. `penalty` holds the penalty at level one, `NormL1(1, ω)` for the lasso
with loadings `ω`; the solution at `λpath[k]` minimizes the loss plus that
penalty at level `λpath[k]`, applied to `scales .* β` when `scales` is not
`nothing` (`standardizeX` with a penalty other than the ℓ1 norm).
"""
struct LassoPath{T<:AbstractFloat}
  λpath::Vector{T}
  βpath::Vector{SparseVector{T, Int}}
  β0path::Vector{T}
  iterations::Vector{Int}
  converged::Vector{Bool}
  kktChecks::Vector{Int}
  intercept::Bool
  nobs::Int
  penalty::AbstractFunction
  scales::Union{Nothing, Vector{T}}
end

"""
    refitLassoPath(path::LassoPath, X, y)

Least squares refits on the supports along a [`LassoPath`](@ref). Returns a
`Dict` that maps each distinct support `S` (a vector of column indices) to
`X[:, S] \\ y`, or, when the path has an intercept, to `[1 X[:, S]] \\ y`,
whose first entry is the intercept.
"""
function refitLassoPath(
  path::LassoPath{T},
  X::AbstractMatrix{T},
  Y::AbstractVector{T}) where {T<:AbstractFloat}

  λpath = path.λpath
  βpath = path.βpath
  Y = _response(Y)

  out = Dict{Vector{Int},Vector{T}}()
  for i=1:length(λpath)
    S = findall(!iszero, βpath[i])
    if haskey(out, S)
      continue
    end
    XS = Matrix{T}(view(X, :, S))      # the support is small: a dense copy of its columns
    out[S] = path.intercept ? [ones(T, length(Y)) XS] \ Y : XS \ Y
  end
  out
end

refitLassoPath(path::LassoPath{T}, X::AbstractMatrix{<:Real}, Y::AbstractVector{<:Real}) where {T<:AbstractFloat} =
  refitLassoPath(path, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, Y))


function LassoPath(
  X::AbstractMatrix{T},
  Y::AbstractVector{T},
  λpath::Vector{T},
  options=CDOptions();
  max_hat_s=Inf, standardizeX::Bool=true, intercept::Bool=false,
  screening::Bool=true, gram::Union{Bool, Symbol}=:auto,
  weights::Union{Nothing, AbstractVector{<:Real}}=nothing, offset::Union{Nothing, AbstractVector{<:Real}}=nothing,
  alpha::Real=1, penalty::Union{Nothing, AbstractFunction}=nothing,
  rng::AbstractRNG=default_rng(), kwargs...) where {T<:AbstractFloat}

  options = _options(options, kwargs)
  all(λ -> λ >= 0, λpath) || throw(ArgumentError("The penalty levels in λpath must be nonnegative."))
  X, Y = _design(X), _response(Y)
  w, o = _weights(T, weights, size(X, 1)), _offset(T, offset, size(X, 1))
  m, gd, g, s = _pathProblem(X, Y, intercept, standardizeX, gram, w, o, _template(T, penalty, alpha))
  _lassopath(m, gd, g, s, size(X, 2), λpath, options, size(X, 1), max_hat_s, intercept, screening, rng)
end

# the model (gd for the Gram form, see _lsmodel), the penalty at level one and
# the column scales of a path: the ℓ1 norm with the loadings of standardizeX as
# weights, or another penalty on the columns divided by their scales
function _pathProblem(X::AbstractMatrix{T}, Y, intercept::Bool, standardizeX::Bool, gram, w, o, template) where {T}
  if _isl1(template)
    m, _, gd = _lsmodel(X, Y, intercept, _usegram(gram, X, true); w, o)
    ω = standardizeX ? _loadings(X, intercept, w) : ones(T, size(X, 2))
    return m, gd, _l1penalty(template, one(T), ω), nothing
  end
  s = standardizeX ? _scales(X, intercept, w) : nothing
  m, _, gd = _lsmodel(_scaleColumns(X, s), Y, intercept, _usegram(gram, X, true); w, o)
  m, gd, template, s
end

# the path along λpath for the model m (gd for the Gram form, see _lsmodel), the
# penalty g at level one and the column scales s, of n observations and p
# coefficients
function _lassopath(m, gd, g, s, p::Int, λpath::Vector{T}, options::CDOptions, n::Int, max_hat_s,
                    intercept::Bool, screening::Bool, rng::AbstractRNG) where {T}
  x = SparseIterate(T, p)
  options.warmStart || return _coldLassoPath(x, m, gd, g, s, λpath, options, max_hat_s, intercept, n; rng)
  β0path = T[]
  path = solve_path(m, g, λpath, _solver(options; rng); x, screening, maxnnz = max_hat_s,
                    callback = (k, λ, x, m) -> (push!(β0path, _intercept(m, x, gd)); false))
  βpath = s === nothing ? path.solutions : [_unscale(β, s) for β in path.solutions]
  LassoPath{T}(path.λ, βpath, β0path, path.passes, path.converged, path.kkt_checks, intercept, n, g, s)
end

# the path with warmStart = false: each level solved from zero along its own
# path of options.numSteps levels, without screening
function _coldLassoPath(x::SparseIterate{T}, m, gd, g, s, λpath::Vector{T}, options::CDOptions, max_hat_s,
                        intercept::Bool, n::Int; rng::AbstractRNG = default_rng()) where {T}
  βpath = SparseVector{T, Int}[]
  β0path, iterations, converged = T[], Int[], Bool[]
  for λ in λpath
    r = _solveLevels!(x, m, _atlevel(g, λ), options; rng)
    push!(βpath, s === nothing ? _sparsevec(x) : _unscale(_sparsevec(x), s))
    push!(β0path, _intercept(m, x, gd))
    push!(iterations, r.passes)
    push!(converged, r.converged)
    nnz(x) > max_hat_s && break
  end
  k = length(βpath)
  LassoPath{T}(λpath[1:k], βpath, β0path, iterations, converged, zeros(Int, k), intercept, n, g, s)
end

function LassoPath(
  X::AbstractMatrix{T},
  Y::AbstractVector{T},
  options::CDOptions=CDOptions();
  nλ::Integer=100,
  λminratio::Real=(size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
  standardizeX::Bool=true,
  intercept::Bool=false,
  max_hat_s=Inf,
  screening::Bool=true,
  gram::Union{Bool, Symbol}=:auto,
  weights::Union{Nothing, AbstractVector{<:Real}}=nothing,
  offset::Union{Nothing, AbstractVector{<:Real}}=nothing,
  alpha::Real=1,
  penalty::Union{Nothing, AbstractFunction}=nothing,
  rng::AbstractRNG=default_rng(),
  kwargs...) where {T<:AbstractFloat}

  nλ >= 1 || throw(ArgumentError("nλ must be positive"))
  0 < λminratio <= 1 || throw(ArgumentError("λminratio must be in (0, 1]"))
  options = _options(options, kwargs)
  X, Y = _design(X), _response(Y)
  p = size(X, 2)
  w, o = _weights(T, weights, size(X, 1)), _offset(T, offset, size(X, 1))
  m, gd, g, s = _pathProblem(X, Y, intercept, standardizeX, gram, w, o, _template(T, penalty, alpha))
  λmax = lambda_max(m, g, SparseIterate(T, p)) * (1 + sqrt(eps(T)))
  λpath = nλ == 1 ? [λmax] : [λmax * T(λminratio)^(T(k) / (nλ - 1)) for k in 0:nλ-1]
  _lassopath(m, gd, g, s, p, λpath, options, size(X, 1), max_hat_s, intercept, screening, rng)
end

function LassoPath(
  X::AbstractMatrix{<:Real},
  Y::AbstractVector{<:Real},
  λpath::AbstractVector{<:Real},
  options=CDOptions();
  kwargs...)
  T, Xf, Yf = _float_data(X, Y)
  LassoPath(Xf, Yf, convert(Vector{T}, λpath), options; kwargs...)
end

function LassoPath(
  X::AbstractMatrix{<:Real},
  Y::AbstractVector{<:Real},
  options::CDOptions=CDOptions();
  kwargs...)
  T, Xf, Yf = _float_data(X, Y)
  LassoPath(Xf, Yf, options; kwargs...)
end
