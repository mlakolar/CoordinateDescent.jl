######################################################################
#
#   Cross-validation and information criteria along a lasso path
#
######################################################################

"""
    CVLassoPath

Result returned by [`CVLassoPath`](@ref)`(X, y)`.

# Fields
- `path::LassoPath`: the path on all the data
- `λpath::Vector`: its levels, the grid of every fold
- `cvm::Vector`, `cvse::Vector`: the cross-validated mean squared prediction
  error at each level, and its standard error
- `nnz::Vector{Int}`: the number of nonzero coefficients of `path` at each level
- `λmin`, `λ1se`: the level with the smallest `cvm`, and the largest level
  whose `cvm` is within one standard error of it
- `folds::Vector{Int}`: the fold of each observation
"""
struct CVLassoPath{T<:AbstractFloat}
  path::LassoPath{T}
  λpath::Vector{T}
  cvm::Vector{T}
  cvse::Vector{T}
  nnz::Vector{Int}
  λmin::T
  λ1se::T
  folds::Vector{Int}
end

# a balanced random assignment of n observations to K folds
function _folds(rng::AbstractRNG, n::Integer, K::Integer)
  2 <= K <= n || throw(ArgumentError("nfolds must be between 2 and the number of observations $n, but it is $K."))
  f = Vector{Int}(undef, n)
  for (i, j) in enumerate(randperm(rng, n))
    f[j] = (i - 1) % K + 1
  end
  f
end

"""
    CVLassoPath(X, y, options = CDOptions(); nfolds = 10, folds = nothing, rng = Random.default_rng(),
                threads = true, nλ = 100, λminratio, intercept = false, standardizeX = true, kwargs...)

K-fold cross-validation of the lasso path of [`LassoPath`](@ref) (with the
same options and keywords): the path on all the data gives the levels, each
fold's path is computed on the other folds at those levels, and the mean
squared error on the held-out fold estimates the prediction error at each
level. Standardization and the intercept are computed on each training fold,
so nothing leaks from the held-out one.

`cvm` is the average of the fold errors weighted by the fold sizes, and `cvse`
its standard error, `sqrt(∑ₖ nₖ (eₖ - cvm)² / n / (K - 1))`, as in glmnet.
`λmin` minimizes `cvm`, and `λ1se` is the largest level whose `cvm` is within
`cvse` of the minimum (the one-standard-error rule). A fold path that stops
early (`max_hat_s`) leaves the remaining levels out of the averages.

The folds are a balanced random assignment drawn from `rng`, or the vector
`folds` of fold labels `1, …, K` given, for example grouped or blocked folds for
dependent data. With `threads`, the folds run as tasks on the available
threads; each fold draws its random coordinate orders from a generator seeded
from `rng`, so that the result does not depend on the number of threads.

Returns a [`CVLassoPath`](@ref); `coef(cv)` and `predict(cv, X)` use the
solution at `λ1se`, or at `λmin` with `λ = :λmin`.
"""
function CVLassoPath(X::AbstractMatrix{T}, y::AbstractVector{T}, options::CDOptions = CDOptions();
                     nfolds::Integer = 10, folds::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                     rng::AbstractRNG = default_rng(), threads::Bool = true, nλ::Integer = 100,
                     λminratio::Real = (size(X, 1) < size(X, 2) ? 0.01 : 1e-4),
                     kwargs...) where {T<:AbstractFloat}
  n = size(X, 1)
  length(y) == n || throw(DimensionMismatch("X has $n rows and y has length $(length(y))"))
  fl = folds === nothing ? _folds(rng, n, nfolds) : collect(Int, folds)
  length(fl) == n || throw(DimensionMismatch("folds must have one label per observation"))
  K = maximum(fl)
  K >= 2 && all(k -> any(==(k), fl), 1:K) ||
    throw(ArgumentError("the fold labels must be 1, …, K for some K ≥ 2, each used at least once"))

  seeds = rand(rng, UInt64, K + 1)
  path = LassoPath(X, y, options; nλ, λminratio, rng = Xoshiro(seeds[end]), kwargs...)
  λs = path.λpath
  L = length(λs)
  errs = fill(T(NaN), K, L)
  function fold!(k)
    train, test = findall(!=(k), fl), findall(==(k), fl)
    fp = LassoPath(X[train, :], y[train], λs, options; rng = Xoshiro(seeds[k]), kwargs...)
    Ŷ = predict(fp, X[test, :])
    for j in axes(Ŷ, 2)
      errs[k, j] = sum(i -> abs2(y[test[i]] - Ŷ[i, j]), eachindex(test)) / length(test)
    end
  end
  if threads
    foreach(wait, [Threads.@spawn fold!(k) for k in 1:K])
  else
    foreach(fold!, 1:K)
  end

  nk = [count(==(k), fl) for k in 1:K]
  cvm, cvse = Vector{T}(undef, L), Vector{T}(undef, L)
  for j in 1:L
    ok = findall(k -> !isnan(errs[k, j]), 1:K)
    w = nk[ok]
    m = sum(w .* errs[ok, j]) / sum(w)
    cvm[j] = m
    cvse[j] = length(ok) > 1 ? sqrt(sum(w .* abs2.(errs[ok, j] .- m)) / sum(w) / (length(ok) - 1)) : T(NaN)
  end
  jmin = argmin(cvm)
  bound = cvm[jmin] + (isnan(cvse[jmin]) ? zero(T) : cvse[jmin])
  j1se = findfirst(j -> cvm[j] <= bound, 1:jmin)
  CVLassoPath{T}(path, λs, cvm, cvse, [nnz(β) for β in path.βpath], λs[jmin], λs[j1se], fl)
end

function CVLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, options::CDOptions = CDOptions(); kwargs...)
  T, Xf, yf = _float_data(X, y)
  CVLassoPath(Xf, yf, options; kwargs...)
end

_cvlevel(cv::CVLassoPath, λ::Symbol) = λ === :λ1se ? cv.λ1se : λ === :λmin ? cv.λmin :
  throw(ArgumentError("λ must be :λ1se, :λmin or a number, but it is :$λ."))
_cvlevel(cv::CVLassoPath, λ::Real) = λ

"""
    coef(cv::CVLassoPath; λ = :λ1se)
    predict(cv::CVLassoPath, X; λ = :λ1se)

The coefficients (`[β₀; β]` with an intercept), and the predictions at the rows
of `X`, of the path on all the data at `cv.λ1se`, at `cv.λmin` with
`λ = :λmin`, or at a level `λ` within the path.
"""
StatsAPI.coef(cv::CVLassoPath; λ::Union{Symbol, Real} = :λ1se) = coef(cv.path, _cvlevel(cv, λ))

function StatsAPI.predict(cv::CVLassoPath, X::AbstractMatrix{<:Real}; λ::Union{Symbol, Real} = :λ1se)
  b = coef(cv; λ)
  cv.path.intercept ? Vector(X * b[2:end]) .+ b[1] : Vector(X * b)
end

# the residual sums of squares and the numbers of nonzero coefficients along the path
function _rss_df(path::LassoPath, X::AbstractMatrix, y::AbstractVector)
  size(X, 1) == length(y) || throw(DimensionMismatch("X has $(size(X, 1)) rows and y has length $(length(y))"))
  Ŷ = predict(path, X)
  [sum(abs2, y .- view(Ŷ, :, j)) for j in axes(Ŷ, 2)], [nnz(β) for β in path.βpath]
end

function _criterion(path::LassoPath{T}, X, y, penalty) where {T}
  n = length(y)
  rss, s = _rss_df(path, X, y)
  map(rss, s) do r, sk
    df = sk + path.intercept
    df < n ? n * log(r / n) + penalty(df, sk) : T(NaN)
  end
end

"""
    ebic(path::LassoPath, X, y; γ = 0.5)
    aic(path::LassoPath, X, y)
    bic(path::LassoPath, X, y)

Information criteria along the lasso `path` fitted to `X` and `y`, one per
level of `path.λpath`:

- `aic = n log(RSS/n) + 2 df`;
- `bic = n log(RSS/n) + log(n) df`;
- `ebic = bic + 2γ log C(p, s)`, the extended BIC of Chen and Chen (2008) for
  large model spaces, with `s` nonzero coefficients among `p`.

`RSS` is the residual sum of squares and `df` the number of nonzero
coefficients plus one for an intercept, an unbiased estimate of the degrees of
freedom of the lasso (Zou, Hastie and Tibshirani, 2007). A level with `df ≥ n`
gets `NaN`. The level that minimizes a criterion `c` is
`path.λpath[argmin(c)]`, with the `NaN`s left out.
"""
function ebic(path::LassoPath, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}; γ::Real = 0.5)
  0 <= γ <= 1 || throw(ArgumentError("γ must be in [0, 1], but it is $γ."))
  p = size(X, 2)
  logbinom(s) = s == 0 ? 0.0 : sum(log(p - s + i) - log(i) for i in 1:s)
  _criterion(path, X, y, (df, s) -> log(length(y)) * df + 2γ * logbinom(s))
end

"""
    aic(path::LassoPath, X, y)

The Akaike criterion `n log(RSS/n) + 2 df` at each level of the lasso path; see
[`ebic`](@ref).
"""
StatsAPI.aic(path::LassoPath, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}) =
  _criterion(path, X, y, (df, s) -> 2df)

"""
    bic(path::LassoPath, X, y)

The Bayesian criterion `n log(RSS/n) + log(n) df` at each level of the lasso
path; see [`ebic`](@ref).
"""
StatsAPI.bic(path::LassoPath, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}) =
  _criterion(path, X, y, (df, s) -> log(length(y)) * df)
