# Normalize deprecated option keywords before applying defaults or copies.
const _OPTION_ALIASES = (:maxIter => :max_iter, :optTol => :opt_tol,
  :warmStart => :warm_start, :numSteps => :num_steps, :relativeTol => :relative_tol,
  :initProcedure => :init_procedure, :optionsCD => :options_cd)

function _option_keywords(kw)
  result = (; kw...)
  for (old, new) in _OPTION_ALIASES
    if haskey(result, old)
      haskey(result, new) && throw(ArgumentError("pass $new or its deprecated spelling $old, not both."))
      Base.depwarn("keyword $old is deprecated; use $new.", old)
      value = getproperty(result, old)
      result = merge(Base.structdiff(result, NamedTuple{(old,)}((nothing,))), NamedTuple{(new,)}((value,)))
    end
  end
  result
end

# Named estimator keywords use nothing to distinguish an omitted value from
# an explicitly supplied one, including one equal to the default.
function _legacy_keyword(value, kwargs, ::Val{old}, ::Val{new}, default) where {old, new}
  kw = (; kwargs...)
  if haskey(kw, old)
    value === nothing || throw(ArgumentError("pass $new or its deprecated spelling $old, not both."))
    Base.depwarn("keyword $old is deprecated; use $new.", old)
    value = getproperty(kw, old)
    kw = Base.structdiff(kw, NamedTuple{(old,)}((nothing,)))
  end
  (value === nothing ? default : value), kw
end

####################################
#
# options
#
####################################

"""
    CDOptions(; max_iter=100_000, opt_tol=1e-7, order=:random, warm_start=true,
                num_steps=50, relative_tol=true, anderson=0, momentum=nothing, gaptol=nothing)
    CDOptions(o::CDOptions; changes...)

Options for [`coordinate_descent!`](@ref) and the lasso solvers, which run the
coordinate descent of ProximalSolvers (`ProximalSolvers.CDSolver`) with them.
The second form copies `o` with the fields named in `changes` replaced, as in
`CDOptions(o; warm_start=false)`. The solvers also take these fields as keyword
arguments, `lasso(X, y, λ; max_iter=10_000, opt_tol=1e-8)`, which change the
options passed to them.

- `max_iter`: maximum number of passes over the coordinates; reaching it without
  convergence gives a warning
- `opt_tol`: convergence tolerance on the largest change of a coordinate in a
  pass. With `relative_tol`, the change of coordinate `k` is measured as
  `|Δxₖ| ⋅ coordinate_scale(f, k)` and compared with `opt_tol ⋅ response_scale(f)`,
  so that the tolerance does not depend on the units of the data; for the
  least squares loss these scales are `‖X[:, k]‖ / √n` and `‖y‖ / √n`.
  Otherwise `|Δxₖ|` is compared with `opt_tol`
- `order`: the order in which a pass visits the coordinates
  - `:random`: a new random order in every pass, drawn from the random number
    generator `rng` that the solvers take as a keyword argument, by default the
    global one (`Random.default_rng()`)
  - `:cyclic`: the order `1, 2, …`
  The cyclic order can be faster on some designs, but much slower when the
  columns are strongly correlated with each other. `randomize = true` and
  `randomize = false` are the same as `order = :random` and `order = :cyclic`
- `warm_start`: start from the supplied iterate; otherwise start from zero and
  follow a path of decreasing penalties
- `num_steps`: number of penalty levels on that path when `warm_start` is false
- `relative_tol`: see `opt_tol`; losses that do not define the scales, such as
  user-defined ones, use the absolute rule either way
- `anderson`: the memory `K` of Anderson acceleration, `0` for none (see
  [`coordinate_descent!`](@ref)); it needs `order = :cyclic`
- `momentum`: Nesterov momentum on the passes over the active set (see
  [`coordinate_descent!`](@ref)). The default, `nothing`, uses it where it is
  known to help: with `order = :random`, a `SparseIterate` and a built-in loss.
  `true` asks for it with any loss, and needs `order = :random`; `false` turns
  it off
- `gaptol`: the relative duality gap at which each level of a path stops, and
  the last level of the path of `warm_start = false` (see ProximalSolvers'
  `solve_path`), for the losses and penalties that have a gap (least squares
  and the GLMs with the ℓ1 norm, its weighted and nonnegative forms, the
  elastic net, the group lasso and the sparse group lasso). `nothing`, the
  default, takes ProximalSolvers' default `PATH_GAPTOL = 1e-4`; `1e-3` is
  the faster setting; `0` stops on the changes of the coordinates with
  `opt_tol`, as before version 0.8.1. A positive `gaptol` also stops a single
  solve (`warm_start = true`), which otherwise uses `opt_tol`. The
  coefficients are within about `gaptol` of the solution, relative to the
  largest, and within about 50 `gaptol` at worst
"""
struct CDOptions
  max_iter::Int
  opt_tol::Float64
  order::Symbol             # :random or :cyclic
  warm_start::Bool           # when running CD, should we do path following or not
  num_steps::Int             # when pathFollowing, how many points are there on the path
  relative_tol::Bool         # changes measured relative to the scales of the loss
  anderson::Int             # memory of Anderson acceleration, 0 for none
  momentum::Union{Bool, Nothing}   # Nesterov momentum, nothing for the default
  gaptol::Union{Float64, Nothing}  # the relative duality gap of the paths, nothing for the default

  function CDOptions(max_iter, opt_tol, order::Symbol, warm_start, num_steps, relative_tol, anderson, momentum,
                     gaptol = nothing)
    o = new(max_iter, opt_tol, order, warm_start, num_steps, relative_tol, anderson, momentum, gaptol)
    o.max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative, but it is $(o.max_iter)."))
    o.opt_tol >= 0 || throw(ArgumentError("opt_tol must be nonnegative, but it is $(o.opt_tol)."))
    o.order === :active &&
      throw(ArgumentError("order = :active was removed in version 0.7; use order = :random."))
    o.order in (:random, :cyclic) ||
      throw(ArgumentError("order must be :random or :cyclic, but it is :$(o.order)."))
    o.num_steps >= 0 || throw(ArgumentError("num_steps must be nonnegative, but it is $(o.num_steps)."))
    o.anderson >= 0 || throw(ArgumentError("anderson must be nonnegative, but it is $(o.anderson)."))
    o.anderson == 0 || o.order === :cyclic ||
      throw(ArgumentError("anderson needs order = :cyclic, since the extrapolation assumes the same order in every pass."))
    o.momentum === true && o.order === :cyclic &&
      throw(ArgumentError("momentum needs order = :random; with the cyclic order it can stall far from the minimizer."))
    o.gaptol === nothing || o.gaptol >= 0 ||
      throw(ArgumentError("gaptol must be nonnegative or nothing, but it is $(o.gaptol)."))
    o
  end
end

# the positional forms of version 0.5.5, without momentum, and of versions
# before it, with randomize in place of order; momentum takes its default
CDOptions(max_iter, opt_tol, order::Symbol, warm_start, num_steps, relative_tol, anderson) =
  CDOptions(max_iter, opt_tol, order, warm_start, num_steps, relative_tol, anderson, nothing)
CDOptions(max_iter, opt_tol, randomize::Bool, warm_start, num_steps, relative_tol) =
  CDOptions(max_iter, opt_tol, randomize ? :random : :cyclic, warm_start, num_steps, relative_tol, 0, nothing)

function CDOptions(; kwargs...)
  kw = _option_keywords(kwargs)
  _check_keywords(CDOptions, _CDOPTIONS_KEYWORDS, kw)
  _cdoptions(; kw...)
end

function _cdoptions(;
  max_iter::Real=100_000, opt_tol::Real=1e-7,
  order::Union{Symbol, Nothing}=nothing, randomize::Union{Bool, Nothing}=nothing,
  warm_start::Bool=true, num_steps::Real=50, relative_tol::Bool=true,
  anderson::Real=0, momentum::Union{Bool, Nothing}=nothing, gaptol::Union{Real, Nothing}=nothing)
  order === nothing || randomize === nothing ||
    throw(ArgumentError("Pass order or randomize, not both."))
  o = order !== nothing ? order : randomize === false ? :cyclic : :random
  CDOptions(Int(max_iter), Float64(opt_tol), o, warm_start, Int(num_steps), relative_tol, Int(anderson), momentum,
            gaptol === nothing ? nothing : Float64(gaptol))
end

const _CDOPTIONS_KEYWORDS = (:max_iter, :opt_tol, :order, :randomize, :warm_start, :num_steps, :relative_tol,
                             :anderson, :momentum, :gaptol)

# the keyword arguments changes must name fields of options of type O
function _check_keywords(::Type{O}, valid, changes) where {O}
  for k in keys(changes)
    k in valid || throw(ArgumentError("unsupported keyword argument $k: the fields of $O are $(join(valid, ", "))."))
  end
end

function CDOptions(o::CDOptions; changes...)
  changes = _option_keywords(changes)
  _check_keywords(CDOptions, _CDOPTIONS_KEYWORDS, changes)
  # randomize replaces order unless both are given
  order = haskey(changes, :randomize) ? (;) : (; order = o.order)
  CDOptions(; max_iter = o.max_iter, opt_tol = o.opt_tol, order..., warm_start = o.warm_start, num_steps = o.num_steps,
            relative_tol = o.relative_tol, anderson = o.anderson, momentum = o.momentum, gaptol = o.gaptol, changes...)
end

# the options with the fields named in the keyword arguments kw replaced, as the
# solvers take them
_options(o, kw) = isempty(kw) ? o : typeof(o)(o; kw...)

# the coordinate descent of ProximalSolvers with these options, drawing its
# random orders from rng; a single solve stops on the duality gap only with a
# positive gaptol
_solver(o::CDOptions; rng::AbstractRNG = default_rng()) =
  CDSolver(; maxpasses = o.max_iter, tol = o.opt_tol, order = o.order, relative = o.relative_tol,
           anderson = o.anderson, momentum = o.momentum, rng, gaptol = something(o.gaptol, 0.0))

# the gap target of the paths and continuation of ProximalSolvers (solve_path
# and solve! with continuation; not the paths of LLA, which stop on the changes)
_gapkw(o::CDOptions) = (; gaptol = o.gaptol)

# Minimizes f + g by the coordinate descent of ProximalSolvers, from x when
# options.warm_start is true. Otherwise x is set to zero and, for NormL1 and
# GroupNormL2, the problem is solved along options.num_steps penalty levels from
# λmax down to g. f is a CDModel or a loss of ProximalBase. Returns a CDResults.
function _solve!(x, f, g, options::CDOptions; rng::AbstractRNG = default_rng())
  s = _solver(options; rng)
  options.warm_start && return solve!(x, f, g, s)
  fill!(x, zero(eltype(x)))
  if g isa Union{NormL1, GroupNormL2} && options.num_steps > 0
    solve!(x, f, g, s; continuation = options.num_steps, _gapkw(options)...)
  else
    solve!(x, f, g, s)
  end
end


"""
    CDInfo()

Convergence information that the deprecated `coordinateDescent!` fills in when it is
passed as the last argument. Deprecated: `coordinate_descent!(x, f, g, options)` (see
[`coordinate_descent!`](@ref)) returns the same information, and more, in a
`ProximalSolvers.CDResults`: `passes` for `iterations`, `converged`,
`maxchange` for `max_change`, `tolerance`, and the objective `minimum`.

- `iterations`: number of passes over the coordinates, including those on the
  path of penalties when `warm_start` is false
- `converged`: whether the stopping rule was met before `max_iter` passes
- `max_change`: largest change of a coordinate in the last pass, scaled as
  described for `opt_tol` in [`CDOptions`](@ref)
- `tolerance`: the threshold `max_change` is compared with
"""
mutable struct CDInfo
  iterations::Int
  converged::Bool
  max_change::Float64
  tolerance::Float64
end

CDInfo() = CDInfo(0, false, NaN, NaN)

function _record!(info::CDInfo, r::CDResults)
  info.iterations = r.passes
  info.converged = r.converged
  info.max_change = r.maxchange
  info.tolerance = r.tolerance
  info
end


"""
    IterLassoOptions(; max_iter=20, opt_tol=1e-2, init_procedure=:Screening,
                       sinit=5, σinit=1.0, options_cd=CDOptions())
    IterLassoOptions(o::IterLassoOptions; changes...)

Options for [`feasible_lasso!`](@ref), which alternates a lasso fit with an
update of the penalty loadings, and for [`scaled_lasso!`](@ref), which uses only
`options_cd` (since version 0.6.1 it solves the square-root lasso in one run of
coordinate descent instead of alternating a lasso fit with an update of the
noise level). The second form copies `o` with the fields named in `changes`
replaced. The two solvers also take these fields as keyword arguments, as in
`feasible_lasso!(x, X, y, λ0; max_iter=50)`.

- `max_iter`: maximum number of outer iterations
- `opt_tol`: tolerance on the relative change of the noise level or loadings
- `init_procedure`: how the procedure is initialized
  - `:Screening`: least squares on the `sinit` columns of `X` most correlated
    with `y`
  - `:InitStd`: noise level `σinit`
  - `:WarmStart`: the iterate passed in
- `sinit`, `σinit`: used by `:Screening` and `:InitStd`
- `options_cd`: [`CDOptions`](@ref) for coordinate descent; in
  `feasible_lasso!` each lasso fit is warm started from the previous one, so its
  `warm_start` is not used there
"""
struct IterLassoOptions
  max_iter::Int
  opt_tol::Float64
  init_procedure::Symbol    # :Screening, :InitStd, :WarmStart
  sinit::Int               # how many columns of X to use to estimate initial variance or obtain initial residuals
  σinit::Float64
  options_cd::CDOptions

  function IterLassoOptions(max_iter, opt_tol, init_procedure, sinit, σinit, options_cd)
    o = new(max_iter, opt_tol, init_procedure, sinit, σinit, options_cd)
    o.max_iter >= 0 || throw(ArgumentError("max_iter must be nonnegative, but it is $(o.max_iter)."))
    o.opt_tol >= 0 || throw(ArgumentError("opt_tol must be nonnegative, but it is $(o.opt_tol)."))
    o.init_procedure in (:Screening, :InitStd, :WarmStart) ||
      throw(ArgumentError("init_procedure must be :Screening, :InitStd or :WarmStart, but it is :$(o.init_procedure)."))
    o.sinit >= 1 || throw(ArgumentError("sinit must be positive, but it is $(o.sinit)."))
    o.σinit > 0 || throw(ArgumentError("σinit must be positive, but it is $(o.σinit)."))
    o
  end
end

function IterLassoOptions(; kwargs...)
  kw = _option_keywords(kwargs)
  _check_keywords(IterLassoOptions, fieldnames(IterLassoOptions), kw)
  _iterlassooptions(; kw...)
end

function _iterlassooptions(;
  max_iter::Real=20, opt_tol::Real=1e-2, init_procedure::Symbol=:Screening,
  sinit::Real=5, σinit::Real=1., options_cd::CDOptions=CDOptions())
  IterLassoOptions(Int(max_iter), Float64(opt_tol), init_procedure, Int(sinit), Float64(σinit), options_cd)
end

function IterLassoOptions(o::IterLassoOptions; changes...)
  changes = _option_keywords(changes)
  _check_keywords(IterLassoOptions, fieldnames(IterLassoOptions), changes)
  IterLassoOptions(; max_iter = o.max_iter, opt_tol = o.opt_tol, init_procedure = o.init_procedure, sinit = o.sinit,
                   σinit = o.σinit, options_cd = o.options_cd, changes...)
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

# The design matrix as coordinate descent reads it, column by column: a strided
# matrix (a Matrix, or a view of a range of rows or of some columns) or a
# SparseMatrixCSC as it is, and any other matrix (a transpose, a view of
# selected rows, …) copied once, into a SparseMatrixCSC if it is sparse and into
# a Matrix otherwise
_design(X::Union{StridedMatrix, SparseMatrixCSC}) = X
_design(X::AbstractMatrix) = issparse(X) ? sparse(X) : Matrix(X)

# the response as the losses read it: a strided vector as it is, any other
# vector (a range, a view of selected entries, a SparseVector, …) copied once
_response(y::StridedVector) = y
_response(y::AbstractVector) = Vector(y)

# Observation weights and an offset as the solvers take them, after checks: a
# Vector{T}, or nothing for unit weights and for a zero offset, which are the
# same as none
function _weights(::Type{T}, w, n::Int) where {T}
  w === nothing && return nothing
  length(w) == n || throw(DimensionMismatch("X has $n rows, but the weights have length $(length(w))"))
  all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("the weights must be finite and nonnegative."))
  sum(w) > 0 || throw(ArgumentError("the weights must have a positive sum."))
  all(isone, w) ? nothing : convert(Vector{T}, w)
end

function _offset(::Type{T}, o, n::Int) where {T}
  o === nothing && return nothing
  length(o) == n || throw(DimensionMismatch("X has $n rows, but the offset has length $(length(o))"))
  all(isfinite, o) || throw(ArgumentError("the offset contains NaN or Inf."))
  all(iszero, o) ? nothing : convert(Vector{T}, o)
end

# y - o, the response that the offset leaves
_minus(y, ::Nothing) = y
_minus(y, o) = y .- o


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

# the column means weighted by w, ∑ᵢ wᵢ Xᵢⱼ / ∑ᵢ wᵢ, with constant columns exact
function _column_means(X::AbstractMatrix{T}, w::AbstractVector{T}) where {T}
  xbar = Vector{T}((transpose(X) * w) ./ sum(w))
  size(X, 1) > 0 || return xbar
  for k in axes(X, 2)
    c = X[1, k]
    all(==(c), view(X, :, k)) && (xbar[k] = c)
  end
  xbar
end
_column_means(X::AbstractMatrix, ::Nothing) = _column_means(X)

# the same over the stored entries: a column is constant when it is all zero
# (whatever it stores) or stores n equal values
function _column_means(X::SparseMatrixCSC{T}) where {T}
  n, p = size(X)
  xbar = zeros(T, p)
  n > 0 || return xbar
  vals = nonzeros(X)
  for k in 1:p
    r = nzrange(X, k)
    a = zero(T)
    for j in r
      a += vals[j]
    end
    xbar[k] = a / n
    if all(j -> iszero(vals[j]), r)
      xbar[k] = zero(T)
    elseif length(r) == n && all(j -> vals[j] == vals[first(r)], r)
      xbar[k] = vals[first(r)]
    end
  end
  xbar
end

# residuals of the least squares fit of y on the s columns of X most correlated
# with y (both centered with an intercept), and on a constant with an
# intercept; storage is overwritten and returned
function _find_init_residuals!(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int,
  storage::Vector{T};
  intercept::Bool=false) where {T <: AbstractFloat}

  if intercept
    S = _find_largest_correlations(X, y .- mean(y), s)
    Z = [ones(T, length(y)) Matrix(X[:, S])]
    mul!(storage, Z, Z \ y)
  else
    S = _find_largest_correlations(X, y, s)
    Xs = issparse(X) ? Matrix(X[:, S]) : view(X, :, S)
    mul!(storage, Xs, Xs \ y)
  end
  @. storage = y - storage
  return storage
end

# BitVector marking the s columns of X with the largest |X[:,j]'y|
# (more than s when there are ties)
function _find_largest_correlations(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  p = size(X, 2)
  storage = Array{T}(undef, p)
  mul!(storage, transpose(X), y)
  @. storage = abs(storage)
  S = storage .>= _sthlargest(storage, s)
end

# the s-th largest entry of v, or its smallest when s > length(v)
_sthlargest(v::AbstractVector, s::Int) = partialsort(v, min(s, length(v)); rev = true)


# heteroscedasticity-robust penalty loadings out[j] = sqrt(∑ᵢ ((X[i,j] - x̄[j]) e[i])² / n),
# for the column means x̄ (zeros without an intercept)
function _get_loadings!(out::Vector{T}, X::AbstractMatrix{T}, e::AbstractVector{T},
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

# the same over the stored entries: the entries that are not stored contribute
# (x̄[j] e[i])², which is x̄[j]² ∑ᵢ e[i]² less what the stored rows took
function _get_loadings!(out::Vector{T}, X::SparseMatrixCSC{T}, e::AbstractVector{T},
                       xbar::AbstractVector{T}=zeros(T, size(X, 2))) where {T <: AbstractFloat}
  n, p = size(X)
  rows, vals = rowvals(X), nonzeros(X)
  E2 = sum(abs2, e)
  @inbounds for j=1:p
    c = xbar[j]
    v = c * c * E2
    for k in nzrange(X, j)
      i = rows[k]
      v += abs2((vals[k] - c) * e[i]) - abs2(c * e[i])
    end
    out[j] = sqrt(max(v, zero(T)) / n)
  end
  out
end


# weighted noise level sqrt(∑ᵢ w[i] r[i]² / ∑ᵢ w[i])
function _get_sigma(w::AbstractVector{T}, r::AbstractVector{T}) where {T <: AbstractFloat}
    n = length(w)
    σ = zero(T)
    for ii = 1:n
        σ += r[ii]^2 * w[ii]
    end
    σ /= sum(w)
    sqrt(σ)
end
