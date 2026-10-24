####################################
#
# options
#
####################################

"""
    CDOptions(; maxIter=100_000, optTol=1e-7, order=:random, warmStart=true,
                numSteps=50, relativeTol=true, anderson=0, momentum=nothing)
    CDOptions(o::CDOptions; changes...)

Options for [`coordinateDescent!`](@ref) and the lasso solvers, which run the
coordinate descent of ProximalSolvers (`ProximalSolvers.CDSolver`) with them.
The second form copies `o` with the fields named in `changes` replaced, as in
`CDOptions(o; warmStart=false)`. The solvers also take these fields as keyword
arguments, `lasso(X, y, λ; maxIter=10_000, optTol=1e-8)`, which change the
options passed to them.

- `maxIter`: maximum number of passes over the coordinates; reaching it without
  convergence gives a warning
- `optTol`: convergence tolerance on the largest change of a coordinate in a
  pass. With `relativeTol`, the change of coordinate `k` is measured as
  `|Δxₖ| ⋅ coordinateScale(f, k)` and compared with `optTol ⋅ responseScale(f)`,
  so that the tolerance does not depend on the units of the data; for the
  least squares loss these scales are `‖X[:, k]‖ / √n` and `‖y‖ / √n`.
  Otherwise `|Δxₖ|` is compared with `optTol`
- `order`: the order in which a pass visits the coordinates
  - `:random`: a new random order in every pass, drawn from the random number
    generator `rng` that the solvers take as a keyword argument, by default the
    global one (`Random.default_rng()`)
  - `:cyclic`: the order `1, 2, …`
  The cyclic order can be faster on some designs, but much slower when the
  columns are strongly correlated with each other. `randomize = true` and
  `randomize = false` are the same as `order = :random` and `order = :cyclic`
- `warmStart`: start from the supplied iterate; otherwise start from zero and
  follow a path of decreasing penalties
- `numSteps`: number of penalty levels on that path when `warmStart` is false
- `relativeTol`: see `optTol`; losses that do not define the scales, such as
  user-defined ones, use the absolute rule either way
- `anderson`: the memory `K` of Anderson acceleration, `0` for none (see
  [`coordinateDescent!`](@ref)); it needs `order = :cyclic`
- `momentum`: Nesterov momentum on the passes over the active set (see
  [`coordinateDescent!`](@ref)). The default, `nothing`, uses it where it is
  known to help: with `order = :random`, a `SparseIterate` and a built-in loss.
  `true` asks for it with any loss, and needs `order = :random`; `false` turns
  it off
"""
struct CDOptions
  maxIter::Int64
  optTol::Float64
  order::Symbol             # :random or :cyclic
  warmStart::Bool           # when running CD, should we do path following or not
  numSteps::Int64           # when pathFollowing, how many points are there on the path
  relativeTol::Bool         # changes measured relative to the scales of the loss
  anderson::Int64           # memory of Anderson acceleration, 0 for none
  momentum::Union{Bool, Nothing}   # Nesterov momentum, nothing for the default

  function CDOptions(maxIter, optTol, order::Symbol, warmStart, numSteps, relativeTol, anderson, momentum)
    o = new(maxIter, optTol, order, warmStart, numSteps, relativeTol, anderson, momentum)
    o.maxIter >= 0 || throw(ArgumentError("maxIter must be nonnegative, but it is $(o.maxIter)."))
    o.optTol >= 0 || throw(ArgumentError("optTol must be nonnegative, but it is $(o.optTol)."))
    o.order === :active &&
      throw(ArgumentError("order = :active was removed in version 0.7; use order = :random."))
    o.order in (:random, :cyclic) ||
      throw(ArgumentError("order must be :random or :cyclic, but it is :$(o.order)."))
    o.numSteps >= 0 || throw(ArgumentError("numSteps must be nonnegative, but it is $(o.numSteps)."))
    o.anderson >= 0 || throw(ArgumentError("anderson must be nonnegative, but it is $(o.anderson)."))
    o.anderson == 0 || o.order === :cyclic ||
      throw(ArgumentError("anderson needs order = :cyclic, since the extrapolation assumes the same order in every pass."))
    o.momentum === true && o.order === :cyclic &&
      throw(ArgumentError("momentum needs order = :random; with the cyclic order it can stall far from the minimizer."))
    o
  end
end

# the positional forms of version 0.5.5, without momentum, and of versions
# before it, with randomize in place of order; momentum takes its default
CDOptions(maxIter, optTol, order::Symbol, warmStart, numSteps, relativeTol, anderson) =
  CDOptions(maxIter, optTol, order, warmStart, numSteps, relativeTol, anderson, nothing)
CDOptions(maxIter, optTol, randomize::Bool, warmStart, numSteps, relativeTol) =
  CDOptions(maxIter, optTol, randomize ? :random : :cyclic, warmStart, numSteps, relativeTol, 0, nothing)

function CDOptions(;
  maxIter::Real=100_000,
  optTol::Real=1e-7,
  order::Union{Symbol, Nothing}=nothing,
  randomize::Union{Bool, Nothing}=nothing,
  warmStart::Bool=true,
  numSteps::Real=50,
  relativeTol::Bool=true,
  anderson::Real=0,
  momentum::Union{Bool, Nothing}=nothing)

  order === nothing || randomize === nothing ||
    throw(ArgumentError("Pass order or randomize, not both."))
  o = order !== nothing ? order : randomize === false ? :cyclic : :random
  CDOptions(Int(maxIter), Float64(optTol), o, warmStart, Int(numSteps), relativeTol, Int(anderson), momentum)
end

# randomize, a field before version 0.5.5
Base.getproperty(o::CDOptions, s::Symbol) =
  s === :randomize ? getfield(o, :order) !== :cyclic : getfield(o, s)
Base.propertynames(::CDOptions, private::Bool=false) = (fieldnames(CDOptions)..., :randomize)

const _CDOPTIONS_KEYWORDS = (:maxIter, :optTol, :order, :randomize, :warmStart, :numSteps, :relativeTol,
                             :anderson, :momentum)

# the keyword arguments changes must name fields of options of type O
function _check_keywords(::Type{O}, valid, changes) where {O}
  for k in keys(changes)
    k in valid || throw(ArgumentError("unsupported keyword argument $k: the fields of $O are $(join(valid, ", "))."))
  end
end

function CDOptions(o::CDOptions; changes...)
  _check_keywords(CDOptions, _CDOPTIONS_KEYWORDS, changes)
  # randomize replaces order unless both are given
  order = haskey(changes, :randomize) ? (;) : (; order = o.order)
  CDOptions(; maxIter = o.maxIter, optTol = o.optTol, order..., warmStart = o.warmStart, numSteps = o.numSteps,
            relativeTol = o.relativeTol, anderson = o.anderson, momentum = o.momentum, changes...)
end

# the options with the fields named in the keyword arguments kw replaced, as the
# solvers take them
_options(o, kw) = isempty(kw) ? o : typeof(o)(o; kw...)

# the options with warmStart = true and every other field unchanged
_withWarmStart(o::CDOptions) = CDOptions(o; warmStart = true)

# the coordinate descent of ProximalSolvers with these options, drawing its
# random orders from rng
_solver(o::CDOptions; rng::AbstractRNG = default_rng()) =
  CDSolver(; maxpasses = o.maxIter, tol = o.optTol, order = o.order, relative = o.relativeTol,
           anderson = o.anderson, momentum = o.momentum, rng)

# Minimizes f + g by the coordinate descent of ProximalSolvers, from x when
# options.warmStart is true. Otherwise x is set to zero and, for NormL1 and
# GroupNormL2, the problem is solved along options.numSteps penalty levels from
# λmax down to g. f is a CDModel or a loss of ProximalBase. Returns a CDResults.
function _solve!(x, f, g, options::CDOptions; rng::AbstractRNG = default_rng())
  s = _solver(options; rng)
  options.warmStart && return solve!(x, f, g, s)
  fill!(x, zero(eltype(x)))
  if g isa Union{NormL1, GroupNormL2} && options.numSteps > 0
    solve!(x, f, g, s; continuation = options.numSteps)
  else
    solve!(x, f, g, s)
  end
end


"""
    CDInfo()

Convergence information that [`coordinateDescent!`](@ref) fills in when it is
passed as the last argument. Deprecated: `solve!(x, f, g, options)` (see
[`coordinateDescent!`](@ref)) returns the same information, and more, in a
`ProximalSolvers.CDResults`: `passes` for `iterations`, `converged`,
`maxchange` for `maxChange`, `tolerance`, and the objective `minimum`.

- `iterations`: number of passes over the coordinates, including those on the
  path of penalties when `warmStart` is false
- `converged`: whether the stopping rule was met before `maxIter` passes
- `maxChange`: largest change of a coordinate in the last pass, scaled as
  described for `optTol` in [`CDOptions`](@ref)
- `tolerance`: the threshold `maxChange` is compared with
"""
mutable struct CDInfo
  iterations::Int64
  converged::Bool
  maxChange::Float64
  tolerance::Float64
end

CDInfo() = CDInfo(0, false, NaN, NaN)

function _record!(info::CDInfo, r::CDResults)
  info.iterations = r.passes
  info.converged = r.converged
  info.maxChange = r.maxchange
  info.tolerance = r.tolerance
  info
end


"""
    IterLassoOptions(; maxIter=20, optTol=1e-2, initProcedure=:Screening,
                       sinit=5, σinit=1.0, optionsCD=CDOptions())
    IterLassoOptions(o::IterLassoOptions; changes...)

Options for [`feasibleLasso!`](@ref), which alternates a lasso fit with an
update of the penalty loadings, and for [`scaledLasso!`](@ref), which uses only
`optionsCD` (since version 0.6.1 it solves the square-root lasso in one run of
coordinate descent instead of alternating a lasso fit with an update of the
noise level). The second form copies `o` with the fields named in `changes`
replaced. The two solvers also take these fields as keyword arguments, as in
`feasibleLasso!(x, X, y, λ0; maxIter=50)`.

- `maxIter`: maximum number of outer iterations
- `optTol`: tolerance on the relative change of the noise level or loadings
- `initProcedure`: how the procedure is initialized
  - `:Screening`: least squares on the `sinit` columns of `X` most correlated
    with `y`
  - `:InitStd`: noise level `σinit`
  - `:WarmStart`: the iterate passed in
- `sinit`, `σinit`: used by `:Screening` and `:InitStd`
- `optionsCD`: [`CDOptions`](@ref) for coordinate descent; in
  `feasibleLasso!` each lasso fit is warm started from the previous one, so its
  `warmStart` is not used there
"""
struct IterLassoOptions
  maxIter::Int64
  optTol::Float64
  initProcedure::Symbol    # :Screening, :InitStd, :WarmStart
  sinit::Int64             # how many columns of X to use to estimate initial variance or obtain initial residuals
  σinit::Float64
  optionsCD::CDOptions

  function IterLassoOptions(maxIter, optTol, initProcedure, sinit, σinit, optionsCD)
    o = new(maxIter, optTol, initProcedure, sinit, σinit, optionsCD)
    o.maxIter >= 0 || throw(ArgumentError("maxIter must be nonnegative, but it is $(o.maxIter)."))
    o.optTol >= 0 || throw(ArgumentError("optTol must be nonnegative, but it is $(o.optTol)."))
    o.initProcedure in (:Screening, :InitStd, :WarmStart) ||
      throw(ArgumentError("initProcedure must be :Screening, :InitStd or :WarmStart, but it is :$(o.initProcedure)."))
    o.sinit >= 1 || throw(ArgumentError("sinit must be positive, but it is $(o.sinit)."))
    o.σinit > 0 || throw(ArgumentError("σinit must be positive, but it is $(o.σinit)."))
    o
  end
end

IterLassoOptions(;
  maxIter::Real=20,
  optTol::Real=1e-2,
  initProcedure::Symbol=:Screening,
  sinit::Real=5,
  σinit::Real=1.,
  optionsCD::CDOptions=CDOptions()) =
  IterLassoOptions(Int(maxIter), Float64(optTol), initProcedure, Int(sinit), Float64(σinit), optionsCD)

function IterLassoOptions(o::IterLassoOptions; changes...)
  _check_keywords(IterLassoOptions, fieldnames(IterLassoOptions), changes)
  IterLassoOptions(; maxIter = o.maxIter, optTol = o.optTol, initProcedure = o.initProcedure, sinit = o.sinit,
                   σinit = o.σinit, optionsCD = o.optionsCD, changes...)
end



####################################
#
# helper functions
#
####################################


# the nonzero entries of x as a SparseVector
function _sparsevec(x::SparseIterate{T, 1}) where {T}
  k = nnz(x)
  ind = x.nzval2ind[1:k]
  val = x.nzval[1:k]
  keep = .!iszero.(val)
  sparsevec(ind[keep], val[keep], length(x))
end


# The floating point type T of real data X and y, and X and y converted to it;
# an array that already has element type T is not copied
function _float_data(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
  T = float(promote_type(eltype(X), eltype(y)))
  T, convert(AbstractMatrix{T}, X), convert(AbstractVector{T}, y)
end


# NaN or Inf in the data would give a meaningless solution that is reported as
# converged: shrinking a NaN gives zero, so a coordinate with a NaN gradient
# stays at zero
_check_finite(A::AbstractArray, name::AbstractString) =
  all(isfinite, A) || throw(ArgumentError("$name contains NaN or Inf."))


# the column means of X; a constant column gets its value exactly, so that it is
# exactly zero once centered (the computed mean can differ in the last bits)
function _column_means(X::AbstractMatrix{T}) where {T}
  xbar = Vector{T}(vec(mean(X; dims=1)))
  size(X, 1) > 0 || return xbar
  for k in axes(X, 2)
    c = X[1, k]
    all(==(c), view(X, :, k)) && (xbar[k] = c)
  end
  xbar
end


# the weighted least squares coefficients argmin_β ∑ᵢ wᵢ (yᵢ - Xᵢ'β)², from a
# pivoted QR factorization of √w X rather than the normal equations; when they
# are not unique, as when X has more columns than there are positive weights,
# the one of minimum norm
function _wls(X::AbstractMatrix{T}, w::AbstractVector{T}, y::AbstractVector{T}) where {T <: AbstractFloat}
  sw = sqrt.(w)
  qr(sw .* X, ColumnNorm()) \ (sw .* y)
end


# residuals of the least squares fit of y on the s columns of X most correlated
# with y (both centered with an intercept), and on a constant with an
# intercept; storage is overwritten and returned
function _findInitResiduals!(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int,
  storage::Vector{T};
  intercept::Bool=false) where {T <: AbstractFloat}

  if intercept
    S = _findLargestCorrelations(X, y .- mean(y), s)
    Z = [ones(T, length(y)) X[:, S]]
    mul!(storage, Z, Z \ y)
  else
    S = _findLargestCorrelations(X, y, s)
    Xs = view(X, :, S)
    mul!(storage, Xs, Xs \ y)
  end
  @. storage = y - storage
  return storage
end

# weighted version: correlations and least squares are weighted by w
function _findInitResiduals!(
  w::AbstractVector{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int,
  storage::Vector{T}) where {T <: AbstractFloat}

  S = _findLargestCorrelations(w, X, y, s)

  Xs = view(X, :, S)
  mul!(storage, Xs, _wls(Xs, w, y))
  @. storage = y - storage
  return storage
end


# BitVector marking the s columns of X with the largest |X[:,j]'y|
# (more than s when there are ties)
function _findLargestCorrelations(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  p = size(X, 2)
  storage = Array{T}(undef, p)
  mul!(storage, transpose(X), y)
  @. storage = abs(storage)
  S = storage .>= _sthlargest(storage, s)
end

function _findLargestCorrelations(
  w::AbstractVector{T},
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  n, p = size(X)
  storage = Array{T}(undef, p)
  @inbounds for j=1:p
    val = zero(T)
    @simd for i=1:n
      val += X[i,j] * w[i] * y[i]
    end
    storage[j] = abs(val)
  end
  S = storage .>= _sthlargest(storage, s)
end

# the s-th largest entry of v, or its smallest when s > length(v)
_sthlargest(v::AbstractVector, s::Int) = partialsort(v, min(s, length(v)); rev = true)


# column scales out[j] = sqrt(∑ᵢ X[i,j]² / n), without centering
function _stdX!(out::Vector{T}, X::AbstractMatrix{T}) where {T <: AbstractFloat}
  n, p = size(X)

  @inbounds for j=1:p
    v = zero(T)
    @simd for i=1:n
      v += abs2(X[i, j])
    end
    out[j] = sqrt(v / n)
  end
  out
end

# weighted column scales out[j] = sqrt(∑ᵢ w[i] X[i,j]² / n)
function _stdX!(out::Vector{T}, w::AbstractVector{T}, X::AbstractMatrix{T}) where {T <: AbstractFloat}
  n, p = size(X)

  @inbounds for j=1:p
    v = zero(T)
    @simd for i=1:n
      v += w[i] * abs2(X[i, j])
    end
    out[j] = sqrt(v / n)
  end
  out
end

# heteroscedasticity-robust penalty loadings out[j] = sqrt(∑ᵢ ((X[i,j] - x̄[j]) e[i])² / n),
# for the column means x̄ (zeros without an intercept)
function _getLoadings!(out::Vector{T}, X::AbstractMatrix{T}, e::AbstractVector{T},
                       xbar::AbstractVector{T}=zeros(T, size(X, 2))) where {T <: AbstractFloat}
  n, p = size(X)

  @inbounds for j=1:p
    c = xbar[j]
    v = zero(T)
    @simd for i=1:n
      v += abs2((X[i, j] - c) * e[i])
    end
    out[j] = sqrt(v / n)
  end
  out
end


# weighted noise level sqrt(∑ᵢ w[i] r[i]² / ∑ᵢ w[i])
function _getSigma(w::AbstractVector{T}, r::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(w)
    σ = zero(T)
    for ii = 1:n
        σ += r[ii]^2 * w[ii]
    end
    σ /= sum(w)
    sqrt(σ)
end
