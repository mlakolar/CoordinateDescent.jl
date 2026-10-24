######################################################################
#
#   Varying-coefficient models
#
#     yᵢ = ∑ⱼ Xᵢⱼ βⱼ(zᵢ) + εᵢ
#
#   by local polynomial regression in z: near a point z₀ each βⱼ(z) is a
#   polynomial of degree q in z - z₀, fitted by kernel-weighted least squares
#   (varyingCoefficient) or with a lasso or group lasso penalty
#   (varyingCoefficientLasso). The kernels come from KernelSmoothing.jl, or any
#   function k(u) of the scaled distance u = (zᵢ - z₀)/h.
#
######################################################################

# The coefficients of the local fit at z₀ are ordered by variable: variable j
# has the q + 1 coefficients (j - 1)(q + 1) + 1 : j(q + 1) of the columns
# Xᵢⱼ ((zᵢ - z₀)/h)ʳ, r = 0, …, q. Scaling z - z₀ by h keeps the columns of
# comparable size; the coefficient of r then estimates hʳ βⱼ⁽ʳ⁾(z₀)/r!.
_coefrange(j, q) = (j - 1) * (q + 1) + 1 : j * (q + 1)

# the observations with a positive weight at z₀ and their weights
# wᵢ k((zᵢ - z₀)/h)/h (wᵢ the observation weights, one without), and their
# expanded design
function _localdesign(X::AbstractMatrix{T}, z::AbstractVector{T}, w, kernel, h::T, z0::T, q::Int) where {T}
  idx, v = Int[], T[]
  for i in eachindex(z)
    k = T(kernel((z[i] - z0) / h)) / h * (w === nothing ? one(T) : w[i])
    k > 0 && (push!(idx, i); push!(v, k))
  end
  p = size(X, 2)
  Z = Matrix{T}(undef, length(idx), p * (q + 1))
  for (a, i) in enumerate(idx)
    d = (z[i] - z0) / h
    for j in 1:p
      t = X[i, j]
      for c in _coefrange(j, q)
        Z[a, c] = t
        t *= d
      end
    end
  end
  idx, v, Z
end

# the estimates of βⱼ⁽ʳ⁾(z₀) from the coefficients θ of the local fit, stored in
# B[:, :, g]; the arguments are not typed, so that JET's analysis of the
# callers with an unknown element type finds the method
function _derivatives!(B, g::Int, θ, h, q::Int)
  for j in axes(B, 1), r in 0:q
    B[j, r+1, g] = θ[_coefrange(j, q)[r+1]] * factorial(r) / h^r
  end
  B
end

function _checkvarying(X, z, y, zgrid, h, q, w)
  n = size(X, 1)
  length(z) == n || throw(DimensionMismatch("X has $n rows and z has length $(length(z))"))
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  _check_finite(X, "X"); _check_finite(z, "z"); _check_finite(y, "y"); _check_finite(zgrid, "zgrid")
  h > 0 || throw(ArgumentError("the bandwidth must be positive, but it is $h."))
  q >= 0 || throw(ArgumentError("the degree must be nonnegative, but it is $q."))
  if w !== nothing
    length(w) == n || throw(DimensionMismatch("weights has length $(length(w)), but X has $n rows"))
    _check_finite(w, "weights")
    all(>=(0), w) || throw(ArgumentError("the weights must be nonnegative"))
  end
  nothing
end

_varyingdata(X, z, y, zgrid, w) =
  (T = float(promote_type(eltype(X), eltype(z), eltype(y)));
   (T, convert(Matrix{T}, X), convert(Vector{T}, z), convert(Vector{T}, y), sort!(convert(Vector{T}, zgrid)),
    w === nothing ? nothing : convert(Vector{T}, w)))

"""
    VaryingCoefficientFit

Result returned by [`varyingCoefficient`](@ref).

# Fields
- `zgrid::Vector`: the points `z₀`, sorted
- `coefficients::Array{T,3}`: `coefficients[j, r + 1, g]` estimates the `r`-th
  derivative `βⱼ⁽ʳ⁾(zgrid[g])`, `r = 0, …, degree`
- `stderrors::Array{T,3}`: their heteroskedasticity-robust standard errors
  (`NaN` where the local fit is singular), or an empty array without `se = true`
- `bandwidth`, `degree`, `kernel`: the settings of the fit
- `rank::Vector{Int}`: the rank of the local design at each point; below
  `p(degree + 1)` the local fit is not unique and the one of minimum norm is taken
- `fitted::Vector`, `residuals::Vector`: the fitted values `∑ⱼ Xᵢⱼ β̂ⱼ(zᵢ)` of the
  local fits at the observed `zᵢ`, and the residuals, or empty without `se = true`
- `nobs::Int`: the number of observations

[`coef`](@ref) (`coef(fit; derivative = r)`), [`predict`](@ref), `stderror`,
`confint`, `fitted`, [`residuals`](@ref) and [`nobs`](@ref) apply to it.
"""
struct VaryingCoefficientFit{T, K}
  zgrid::Vector{T}
  coefficients::Array{T, 3}
  stderrors::Array{T, 3}
  bandwidth::T
  degree::Int
  kernel::K
  rank::Vector{Int}
  fitted::Vector{T}
  residuals::Vector{T}
  nobs::Int
end

"""
    varyingCoefficient(X, z, y, zgrid; kernel, bandwidth, degree = 1, weights = nothing, se = false)

The local polynomial estimate of the varying-coefficient model

    yᵢ = ∑ⱼ Xᵢⱼ βⱼ(zᵢ) + εᵢ

at each point `z₀` of `zgrid`. Near `z₀` each `βⱼ` is approximated by a
polynomial of degree `degree` in `z - z₀`, and the polynomials are fitted by
weighted least squares,

    minimize  ∑ᵢ wᵢ Kₕ(zᵢ - z₀) (yᵢ - ∑ⱼ Xᵢⱼ ∑ᵣ θⱼᵣ (zᵢ - z₀)ʳ)²,

with `Kₕ(d) = k(d/h)/h` for the `kernel` `k` and the `bandwidth` `h`, and the
observation `weights` `wᵢ` (one when omitted). `θⱼᵣ r!` estimates the `r`-th
derivative `βⱼ⁽ʳ⁾(z₀)`. A column of ones in `X` gives a varying intercept.

The `kernel` is a function of the scaled distance, `k(u)`, such as the kernels of
KernelSmoothing.jl: `EpanechnikovKernel()`, `GaussianKernel()` (for which `h` is
the standard deviation), `UniformKernel()`, and so on. With a compactly
supported kernel only the observations within `h` of `z₀` enter the fit at
`z₀`; when they do not determine the polynomials (fewer than `p(degree + 1)` of
them, or collinear), the fit of minimum norm is taken and `rank` records it.
[`varyingCoefficientCV`](@ref) chooses the bandwidth by leave-one-out
cross-validation.

With `se = true`, the fit also computes the fitted values and residuals of the
local fits at the observed `zᵢ` (one more local fit per observation, which can
cost far more than the fit on the grid), and the heteroskedasticity-robust (sandwich)
standard errors of the estimates, `(Z'WZ)⁻¹ Z'W diag(ε̂²) W Z (Z'WZ)⁻¹` for the
local design `Z` and the kernel weights `W` (the observation weights count as
frequencies: integer weights give the fit and the standard errors of
replicated rows). They describe the variance of the local
estimate, not its bias: a confidence interval from them covers `βⱼ(z₀)` only
when the bandwidth is small enough for the bias to be negligible
(undersmoothing).

```julia
using CoordinateDescent, KernelSmoothing
fit = varyingCoefficient(X, z, y, range(0, 1; length = 50); kernel = EpanechnikovKernel(), bandwidth = 0.2,
                         se = true)
coef(fit)                      # p × 50: the estimates of βⱼ at the points
coef(fit; derivative = 1)      # of their first derivatives
confint(fit)                   # pointwise intervals, from the robust standard errors
predict(fit, Xnew, znew)       # ∑ⱼ Xnewᵢⱼ βⱼ(znewᵢ), interpolated linearly between the points
```
"""
function varyingCoefficient(X::AbstractMatrix{<:Real}, z::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                            zgrid::AbstractVector{<:Real}; kernel, bandwidth::Real, degree::Integer = 1,
                            weights::Union{Nothing, AbstractVector{<:Real}} = nothing, se::Bool = false)
  T, Xf, zf, yf, g, w = _varyingdata(X, z, y, zgrid, weights)
  h, q = T(bandwidth), Int(degree)
  _checkvarying(Xf, zf, yf, g, h, q, w)
  n, p = size(Xf)
  B = Array{T}(undef, p, q + 1, length(g))
  rk = Vector{Int}(undef, length(g))
  fitted, res = se ? _localfitted(Xf, zf, yf, w, kernel, h, q) : (T[], T[])
  S = se ? fill(T(NaN), p, q + 1, length(g)) : Array{T}(undef, 0, 0, 0)
  for (k, z0) in enumerate(g)
    idx, v, Z = _localdesign(Xf, zf, w, kernel, h, z0, q)
    F = qr(sqrt.(v) .* Z, ColumnNorm())
    rk[k] = _qrrank(F, size(Z))
    θ = F \ (sqrt.(v) .* yf[idx])
    _derivatives!(B, k, θ, h, q)
    se && rk[k] == p * (q + 1) && _sandwich!(S, k, Z, v, w === nothing ? nothing : w[idx], res[idx], h, q)
  end
  VaryingCoefficientFit{T, typeof(kernel)}(g, B, S, h, q, kernel, rk, fitted, res, n)
end

# the numerical rank of the pivoted QR factorization of an m × c matrix
function _qrrank(F, (m, c))
  d = abs.(diag(F.R))
  isempty(d) || iszero(d[1]) ? 0 : count(>(max(m, c) * eps(eltype(d)) * d[1]), d)
end

# the robust standard errors of the estimates at point k, from the local design
# Z, the weights v = w K of its observations (w the observation weights, a
# frequency: an observation of weight wᵢ counts as wᵢ copies, each contributing
# (Kᵢeᵢ)² to the middle of the sandwich) and their residuals e
function _sandwich!(S, k::Int, Z::Matrix{T}, v::Vector{T}, w, e::AbstractVector{T}, h::T, q::Int) where {T}
  A = inv(Symmetric(Z' * (v .* Z)))
  m = w === nothing ? v .^ 2 .* e .^ 2 : v .^ 2 ./ w .* e .^ 2
  M = Z' * (m .* Z)
  V = A * M * A
  for j in axes(S, 1), r in 0:q
    c = _coefrange(j, q)[r+1]
    S[j, r+1, k] = sqrt(max(V[c, c], zero(T))) * factorial(r) / h^r
  end
  S
end

# the fitted values ∑ⱼ Xᵢⱼ β̂ⱼ(zᵢ) of the local fits at the observed zᵢ, and the
# residuals
function _localfitted(X::Matrix{T}, z::Vector{T}, y::Vector{T}, w, kernel, h::T, q::Int) where {T}
  n, p = size(X)
  ŷ = Vector{T}(undef, n)
  for i in 1:n
    idx, v, Z = _localdesign(X, z, w, kernel, h, z[i], q)
    θ = qr(sqrt.(v) .* Z, ColumnNorm()) \ (sqrt.(v) .* y[idx])
    ŷ[i] = sum(X[i, j] * θ[_coefrange(j, q)[1]] for j in 1:p)
  end
  ŷ, y .- ŷ
end

# the coefficients of the columns of X at the points z, interpolated linearly in
# z between the points of the sorted grid g (the nearest point outside it), from
# the p × G matrix C
function _interpolate(C::AbstractMatrix{T}, g::Vector{T}, z0::Real) where {T}
  b = searchsortedfirst(g, z0)
  b == 1 && return C[:, 1]
  b > length(g) && return C[:, end]
  g[b] == z0 && return C[:, b]
  t = T((z0 - g[b-1]) / (g[b] - g[b-1]))
  (1 - t) .* C[:, b-1] .+ t .* C[:, b]
end

function _predictvarying(C::AbstractMatrix, g::Vector, X::AbstractMatrix, z::AbstractVector)
  size(X, 2) == size(C, 1) || throw(DimensionMismatch("X has $(size(X, 2)) columns, but the model has $(size(C, 1)) coefficient functions."))
  length(z) == size(X, 1) || throw(DimensionMismatch("X has $(size(X, 1)) rows and z has length $(length(z))"))
  [dot(view(X, i, :), _interpolate(C, g, z[i])) for i in axes(X, 1)]
end

"""
    coef(fit::VaryingCoefficientFit; derivative = 0)
    stderror(fit::VaryingCoefficientFit; derivative = 0)
    confint(fit::VaryingCoefficientFit; level = 0.95, derivative = 0)
    predict(fit::VaryingCoefficientFit, X, z)

The `p × length(zgrid)` estimates of `βⱼ⁽ʳ⁾` at the points of `fit.zgrid`
(`r = derivative`), their robust standard errors, and the pointwise normal
intervals `estimate ± z stderror` as a pair of matrices (lower, upper), which
cover `βⱼ⁽ʳ⁾(z₀)` only when the bias of the local fit is negligible (see
[`varyingCoefficient`](@ref)). `predict` gives `∑ⱼ Xᵢⱼ βⱼ(zᵢ)` with `β` interpolated
linearly between the points of the grid (and constant outside it).
`fitted` and `residuals` are those of the local fits at the observations.
"""
StatsAPI.coef(fit::VaryingCoefficientFit; derivative::Integer = 0) = fit.coefficients[:, _derivindex(fit, derivative), :]

function StatsAPI.stderror(fit::VaryingCoefficientFit; derivative::Integer = 0)
  isempty(fit.stderrors) && throw(ArgumentError("the fit was computed with se = false."))
  fit.stderrors[:, _derivindex(fit, derivative), :]
end

function StatsAPI.confint(fit::VaryingCoefficientFit{T}; level::Real = 0.95, derivative::Integer = 0) where {T}
  0 < level < 1 || throw(ArgumentError("level must be in (0, 1), but it is $level."))
  zq = _qnorm(1 - (1 - T(level)) / 2)
  b, s = coef(fit; derivative), stderror(fit; derivative)
  b .- zq .* s, b .+ zq .* s
end

function _derivindex(fit, r::Integer)
  0 <= r <= fit.degree || throw(ArgumentError("derivative must be between 0 and the degree $(fit.degree), but it is $r."))
  r + 1
end

StatsAPI.predict(fit::VaryingCoefficientFit, X::AbstractMatrix{<:Real}, z::AbstractVector{<:Real}) =
  _predictvarying(coef(fit), fit.zgrid, X, z)

function StatsAPI.fitted(fit::VaryingCoefficientFit)
  isempty(fit.fitted) && throw(ArgumentError("the fit was computed with se = false."))
  fit.fitted
end

function StatsAPI.residuals(fit::VaryingCoefficientFit)
  isempty(fit.residuals) && throw(ArgumentError("the fit was computed with se = false."))
  fit.residuals
end

StatsAPI.nobs(fit::VaryingCoefficientFit) = fit.nobs

######################################################################
#
#   Bandwidth selection by leave-one-out cross-validation
#
######################################################################

"""
    VaryingCoefficientCV

Result returned by [`varyingCoefficientCV`](@ref) and
[`varyingCoefficientLassoCV`](@ref): the `bandwidths` tried, the
cross-validated mean squared prediction error `cvm` at each (weighted by the
observation weights), and the `bandwidth` that minimizes it.
"""
struct VaryingCoefficientCV{T}
  bandwidths::Vector{T}
  cvm::Vector{T}
  bandwidth::T
end

"""
    varyingCoefficientCV(X, z, y, bandwidths; kernel, degree = 1, weights = nothing)

Leave-one-out cross-validation of the bandwidth of
[`varyingCoefficient`](@ref): observation `i` is predicted from the local fit at
`zᵢ` computed without it, and `cvm` is the mean squared error of these
predictions, weighted by the observation weights. The local fit is linear in
`y`, so the prediction without `i` comes from the fit with it,
`(ŷᵢ - Hᵢᵢ yᵢ) / (1 - Hᵢᵢ)` with the leverage `Hᵢᵢ`, at the cost of one local fit
per observation. An observation alone in its window, or in a window that does
not determine the local polynomials, cannot be predicted, and its bandwidth gets
`cvm = Inf`. Returns a [`VaryingCoefficientCV`](@ref).
"""
function varyingCoefficientCV(X::AbstractMatrix{<:Real}, z::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                              bandwidths::AbstractVector{<:Real}; kernel, degree::Integer = 1,
                              weights::Union{Nothing, AbstractVector{<:Real}} = nothing)
  T, Xf, zf, yf, _, w = _varyingdata(X, z, y, Real[], weights)
  hs = convert(Vector{T}, bandwidths)
  isempty(hs) && throw(ArgumentError("give at least one bandwidth"))
  q = Int(degree)
  foreach(h -> _checkvarying(Xf, zf, yf, T[], h, q, w), hs)
  cvm = map(h -> _loocv(Xf, zf, yf, w, kernel, h, q), hs)
  VaryingCoefficientCV{T}(hs, cvm, hs[argmin(cvm)])
end

function _loocv(X::Matrix{T}, z::Vector{T}, y::Vector{T}, w, kernel, h::T, q::Int) where {T}
  n, p = size(X)
  total, wsum = zero(T), zero(T)
  for i in 1:n
    wi = w === nothing ? one(T) : w[i]
    iszero(wi) && continue
    idx, v, Z = _localdesign(X, z, w, kernel, h, z[i], q)
    a = searchsortedfirst(idx, i)
    sw = sqrt.(v)
    F = qr(sw .* Z, ColumnNorm())
    _qrrank(F, size(Z)) == p * (q + 1) || return T(Inf)
    Q = Matrix(F.Q)
    H = sum(abs2, view(Q, a, :))                # the leverage of observation i
    H < 1 - sqrt(eps(T)) || return T(Inf)
    θ = F \ (sw .* y[idx])
    ŷ = sum(X[i, j] * θ[_coefrange(j, q)[1]] for j in 1:p)
    total += wi * abs2((y[i] - ŷ) / (1 - H))
    wsum += wi
  end
  total / wsum
end

######################################################################
#
#   The varying-coefficient lasso
#
######################################################################

"""
    VaryingCoefficientLassoFit

Result returned by [`varyingCoefficientLasso`](@ref).

# Fields
- `zgrid::Vector`: the points `z₀`, sorted
- `coefficients::Array{T,3}`: `coefficients[j, r + 1, g]`, the penalized estimate
  of `βⱼ⁽ʳ⁾(zgrid[g])`
- `refitted::Array{T,3}`: the same from the unpenalized local fit on the selected
  variables, or an empty array without `refit`
- `selected::BitMatrix`: `selected[j, g]` when variable `j` has a nonzero
  coefficient at `zgrid[g]`
- `λ`, `penalty`, `bandwidth`, `degree`, `kernel`: the settings of the fit
- `nobs::Int`: the number of observations

[`coef`](@ref) (`coef(fit; derivative = r, refit = false)`), [`predict`](@ref)
and [`nobs`](@ref) apply to it.
"""
struct VaryingCoefficientLassoFit{T, K}
  zgrid::Vector{T}
  coefficients::Array{T, 3}
  refitted::Array{T, 3}
  selected::BitMatrix
  λ::T
  penalty::Symbol
  bandwidth::T
  degree::Int
  kernel::K
  nobs::Int
end

"""
    varyingCoefficientLasso(X, z, y, zgrid, λ, options = CDOptions(); kernel, bandwidth, degree = 1,
                            penalty = :group, refit = true, weights = nothing)

A sparse varying-coefficient model: at each point `z₀` of `zgrid`, the local
polynomial fit of [`varyingCoefficient`](@ref) with a penalty that selects the
variables `Xⱼ` whose coefficient functions are not zero near `z₀`,

    minimize  ∑ᵢ vᵢ (yᵢ - ∑ⱼ Xᵢⱼ ∑ᵣ θⱼᵣ ((zᵢ - z₀)/h)ʳ)² / (2 ∑ᵢ vᵢ) + λ P(θ),

with the local weights `vᵢ = wᵢ Kₕ(zᵢ - z₀)`. The loss is the mean squared error
on the local sample, so `λ` is on the scale of an ordinary lasso.

- `penalty = :group`, the default, is the group lasso over the `degree + 1`
  coefficients of each variable, `P(θ) = ∑ⱼ √(degree + 1) ‖Z̃ⱼθⱼ‖ / √n`, with the
  locally weighted columns `Z̃ⱼ` of variable `j` orthonormalized (the
  `standardize = :orthonormal` of [`groupLasso`](@ref)), so that the penalty is
  on the fitted values of each variable and a variable enters or leaves with all
  its polynomial terms.
- `penalty = :lasso` is the lasso on the expanded coefficients, with the
  loadings of the standardized local columns, as [`lasso`](@ref) with
  `standardizeX`; a variable is selected when any of its coefficients is not
  zero.

With `refit`, the selected variables are refitted at each point by the
unpenalized local fit (all their polynomial terms), which removes the shrinkage
of the penalty. Coordinate descent runs with `options`. The `kernel` and the
`bandwidth` are as for [`varyingCoefficient`](@ref);
[`varyingCoefficientLassoCV`](@ref) chooses the bandwidth. Returns a
[`VaryingCoefficientLassoFit`](@ref).
"""
function varyingCoefficientLasso(X::AbstractMatrix{<:Real}, z::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                                 zgrid::AbstractVector{<:Real}, λ::Real, options::CDOptions = CDOptions();
                                 kernel, bandwidth::Real, degree::Integer = 1, penalty::Symbol = :group,
                                 refit::Bool = true, weights::Union{Nothing, AbstractVector{<:Real}} = nothing)
  T, Xf, zf, yf, g, w = _varyingdata(X, z, y, zgrid, weights)
  h, q = T(bandwidth), Int(degree)
  _checkvarying(Xf, zf, yf, g, h, q, w)
  λ >= 0 || throw(ArgumentError("λ must be nonnegative, but it is $λ."))
  penalty in (:group, :lasso) || throw(ArgumentError("penalty must be :group or :lasso, but it is :$penalty."))
  n, p = size(Xf)
  B = zeros(T, p, q + 1, length(g))
  R = refit ? zeros(T, p, q + 1, length(g)) : Array{T}(undef, 0, 0, 0)
  sel = falses(p, length(g))
  x = SparseIterate(T, p * (q + 1))         # warm starts the lasso at each point from the previous one
  for (k, z0) in enumerate(g)
    θ, s, θr = _locallasso(Xf, zf, yf, w, kernel, h, z0, q, T(λ), penalty, refit, options; x)
    _derivatives!(B, k, θ, h, q)
    sel[:, k] .= s
    refit && _derivatives!(R, k, θr, h, q)
  end
  VaryingCoefficientLassoFit{T, typeof(kernel)}(g, B, R, sel, T(λ), penalty, h, q, kernel, n)
end

# The penalized local fit at z₀ and the variables it selects, and the unpenalized
# refit on them (zero without refit), as coefficient vectors of the local design
function _locallasso(X::Matrix{T}, z::Vector{T}, y::Vector{T}, w, kernel, h::T, z0::T, q::Int, λ::T,
                     penalty::Symbol, refit::Bool, options::CDOptions; skip::Int = 0,
                     x::SparseIterate{T} = SparseIterate(T, size(X, 2) * (q + 1))) where {T}
  idx, v, Z = _localdesign(X, z, w, kernel, h, z0, q)
  if skip > 0                      # leave observation skip out
    keep = idx .!= skip
    idx, v, Z = idx[keep], v[keep], Z[keep, :]
  end
  p, m = size(X, 2), length(idx)
  c = p * (q + 1)
  θ = zeros(T, c)
  m == 0 && return θ, falses(p), zeros(T, c)
  # rows scaled in place so that ‖yₛ - Zₛθ‖²/(2m) = ∑ vᵢ (yᵢ - Zᵢθ)² / (2 ∑ vᵢ);
  # the scales are proportional to √vᵢ, so least squares on the scaled rows is
  # the weighted least squares of the refit
  r = sqrt.(m .* v ./ sum(v))
  Zs = (Z .*= r)
  ys = r .* view(y, idx)
  if penalty === :group
    groups = [collect(_coefrange(j, q)) for j in 1:p]
    θ .= Vector(groupLasso(Zs, ys, λ, groups, options; intercept = false, standardize = :orthonormal,
                           groupweights = fill(sqrt(T(q + 1)), p)).x)
  else
    # the lasso with the loadings of the standardized columns (zero columns get
    # weight one and stay at zero), from the iterate x, which keeps the solution
    ω = sqrt.(vec(sum(abs2, Zs; dims = 1)) ./ m)
    ω[iszero.(ω)] .= one(T)
    _solve!(x, CDLeastSquaresLoss(ys, Zs), NormL1(λ, ω), options)
    θ .= Vector(x)
  end
  sel = BitVector(any(!iszero, view(θ, _coefrange(j, q))) for j in 1:p)
  θr = zeros(T, c)
  if refit && any(sel)
    cols = reduce(vcat, [collect(_coefrange(j, q)) for j in findall(sel)])
    θr[cols] .= qr(Zs[:, cols], ColumnNorm()) \ ys
  end
  θ, sel, θr
end

"""
    coef(fit::VaryingCoefficientLassoFit; derivative = 0, refit = false)
    predict(fit::VaryingCoefficientLassoFit, X, z; refit = false)

The `p × length(zgrid)` penalized estimates of `βⱼ⁽ʳ⁾` (`r = derivative`) at the
points of `fit.zgrid`, or with `refit` those of the unpenalized refit on the
selected variables; and the predictions `∑ⱼ Xᵢⱼ βⱼ(zᵢ)`, interpolated linearly
between the points, as for [`varyingCoefficient`](@ref).
"""
function StatsAPI.coef(fit::VaryingCoefficientLassoFit; derivative::Integer = 0, refit::Bool = false)
  r = _derivindex(fit, derivative)
  if refit
    isempty(fit.refitted) && throw(ArgumentError("the fit was computed with refit = false."))
    return fit.refitted[:, r, :]
  end
  fit.coefficients[:, r, :]
end

StatsAPI.predict(fit::VaryingCoefficientLassoFit, X::AbstractMatrix{<:Real}, z::AbstractVector{<:Real};
                 refit::Bool = false) = _predictvarying(coef(fit; refit), fit.zgrid, X, z)
StatsAPI.nobs(fit::VaryingCoefficientLassoFit) = fit.nobs

"""
    varyingCoefficientLassoCV(X, z, y, bandwidths, λ, options = CDOptions(); kernel, degree = 1,
                              penalty = :group, refit = true, weights = nothing)

Leave-one-out cross-validation of the bandwidth of
[`varyingCoefficientLasso`](@ref) at the level `λ`: observation `i` is predicted
from the penalized local fit at `zᵢ` computed without it (refitted on the
selected variables with `refit`), and `cvm` is the mean squared error of these
predictions, weighted by the observation weights. It solves one local lasso per
observation and bandwidth. Returns a [`VaryingCoefficientCV`](@ref).
"""
function varyingCoefficientLassoCV(X::AbstractMatrix{<:Real}, z::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                                   bandwidths::AbstractVector{<:Real}, λ::Real, options::CDOptions = CDOptions();
                                   kernel, degree::Integer = 1, penalty::Symbol = :group, refit::Bool = true,
                                   weights::Union{Nothing, AbstractVector{<:Real}} = nothing)
  T, Xf, zf, yf, _, w = _varyingdata(X, z, y, Real[], weights)
  hs = convert(Vector{T}, bandwidths)
  isempty(hs) && throw(ArgumentError("give at least one bandwidth"))
  q = Int(degree)
  foreach(h -> _checkvarying(Xf, zf, yf, T[], h, q, w), hs)
  penalty in (:group, :lasso) || throw(ArgumentError("penalty must be :group or :lasso, but it is :$penalty."))
  n, p = size(Xf)
  x = SparseIterate(T, p * (q + 1))        # warm starts each local lasso from the previous one
  cvm = map(hs) do h
    total, wsum = zero(T), zero(T)
    for i in 1:n
      wi = w === nothing ? one(T) : w[i]
      iszero(wi) && continue
      θ, _, θr = _locallasso(Xf, zf, yf, w, kernel, h, zf[i], q, T(λ), penalty, refit, options; skip = i, x)
      t = refit ? θr : θ
      ŷ = sum(Xf[i, j] * t[_coefrange(j, q)[1]] for j in 1:p)
      total += wi * abs2(yf[i] - ŷ)
      wsum += wi
    end
    total / wsum
  end
  VaryingCoefficientCV{T}(hs, cvm, hs[argmin(cvm)])
end
