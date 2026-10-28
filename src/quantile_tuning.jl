# Fixed-quantile pivotal calibration and held-out check-loss selection.

function _quantile_score_maxima(X, w, tau, intercept, uniforms)
  n = size(X, 1)
  scores = tau .- (uniforms .< tau)
  gradients = X' * scores
  if intercept
    means = vec(sum(X; dims=1)) ./ n
    gradients .-= means * sum(scores; dims=1)
  end
  vec(maximum(abs.(gradients) ./ (n .* w); dims=1))
end

"""
    quantile_lambda(X; tau=0.5, c=2.0, gamma=0.1, nsim=1000,
                    penalty_factors=nothing, standardize_x=true,
                    intercept=true, rng=Random.default_rng())

Simulate a fixed-quantile pivotal penalty in the normalization of
[`quantile_lasso`](@ref): mean check loss plus `λ*sum(w .* abs.(β))`.
For independent uniforms `Uᵢ`, simulate
`max_j abs(sum_i Zᵢⱼ*(tau - (Uᵢ < tau)))/(n*wⱼ)` and return `c` times
its empirical `(1-gamma)` quantile (linear interpolation, R's type 7).
`Z` is `X` centered by column when fitting an unpenalized intercept, and `X`
otherwise. `w` contains the effective loadings used by the fit, including its
population-standard-deviation scaling when `standardize_x=true`.

This is the single-quantile score simulation of Belloni and Chernozhukov
(2011), translated to this objective's scale. It depends on `X`, not `y`,
estimates no noise scale, and does not calibrate a simultaneous quantile process.
The paper's defaults here are `c=2` and `gamma=0.1`; `nsim` controls Monte Carlo
precision. Use a seeded `rng` for reproducibility and the same loading/intercept
settings in the subsequent fit. All effective slope loadings must be positive;
unpenalized slopes require a separately justified calibration or an explicit λ.
The rule calibrates exact check-loss scores; it is not a bandwidth calibration
or a finite-sample guarantee for a convolution-smoothed fit.
"""
function quantile_lambda(X::AbstractMatrix{<:Real}; tau::Real=0.5, c::Real=2.0,
                         gamma::Real=0.1, nsim::Integer=1000,
                         penalty_factors=nothing, standardize_x::Bool=true,
                         intercept::Bool=true, rng::AbstractRNG=default_rng())
  isfinite(tau) && 0 < tau < 1 || throw(ArgumentError("tau must lie in (0,1)"))
  isfinite(c) && c > 0 || throw(ArgumentError("c must be positive and finite"))
  isfinite(gamma) && 0 < gamma < 1 || throw(ArgumentError("gamma must lie in (0,1)"))
  nsim >= 2 || throw(ArgumentError("nsim must be at least two"))
  Xf, _, w = _qdata(X, zeros(size(X,1)), penalty_factors, standardize_x, intercept, nothing)
  all(>(0), w) || throw(ArgumentError("pivotal calibration needs positive effective slope loadings"))
  draws = Vector{Float64}(undef, nsim)
  # Bound temporary memory and use matrix products for batches of simulations.
  for first in 1:64:nsim
    last = min(first+63, nsim)
    uniforms = rand(rng, size(X,1), last-first+1)
    draws[first:last] = _quantile_score_maxima(Xf, w, Float64(tau), intercept, uniforms)
  end
  Float64(c) * quantile(draws, 1-Float64(gamma))
end

"""
    CVQuantileLassoPath(X, y[, λpath]; nfolds=10, folds=nothing,
                       rng=Random.default_rng(), threads=true, tau=0.5,
                       nλ=100, λminratio=0.01, offset=nothing, kwargs...)

Select a quantile-lasso penalty by held-out **check loss**, for either exact
or convolution-smoothed fits. `kwargs` go to [`QuantileLassoPath`](@ref),
including `method`, `bandwidth`, loadings, standardization and solver controls.
`max_hat_s` is forwarded to the full-data path and every fold path. A support
stop on the full data truncates the shared grid. A fold that stops earlier
has `NaN` errors at its unreached levels; these are ineligible, with `cvm=Inf`
and `cvse=NaN`, even when no fold reaches them.
An explicit decreasing `λpath` is shared by all folds; otherwise the full-data
path supplies the common grid. Scaling and centering are computed using each
training fold. The automatic common grid itself uses the full data.

`cvm` and `cvse` use fold-size weighting and the same standard-error convention
as [`CVLassoPath`](@ref). `λmin` minimizes `cvm`; `λ1se` is the largest eligible
penalty within one standard error of that minimum. A level is eligible only
when the full-data fit and **every** fold's fit pass numerical convergence.
Ineligible levels have `cvm=Inf`; `fold_errors` is a folds-by-level matrix with
`NaN` for failed/missing fold fits. If no level is eligible, an error is raised.
`time_limit`, when supplied, applies separately to each exact path.

The result retains the full-data `path`, `λpath`, `cvm`, `cvse`, `nnz`, `λmin`,
`λ1se`, `folds` and `fold_errors`. `coef(cv)`, `predict(cv, X)` and
`dual_scores(cv)` select the saved full-data fit at `λ1se`, or at `λmin` with
`λ=:λmin`. Duals are never averaged across folds or interpolated.
Use `dual_scores(cv.path)` to access the whole full-data path.
Offsets follow observations into training and held-out evaluation; pass new
prediction offsets explicitly. Observation weights are not supported.
"""
struct CVQuantileLassoPath
  path::QuantileLassoPath
  λpath::Vector{Float64}
  cvm::Vector{Float64}
  cvse::Vector{Float64}
  nnz::Vector{Int}
  λmin::Float64
  λ1se::Float64
  folds::Vector{Int}
  fold_errors::Matrix{Float64}
end

function _quantile_cvsummary(path, errors, weights)
  L = length(path)
  eligible = [path[j].converged && all(isfinite, view(errors,:,j)) for j in 1:L]
  any(eligible) || error("no quantile CV level has a converged full-data fit and every fold fit; inspect solver limits and tolerances")
  cvm, cvse = fill(Inf,L), fill(NaN,L)
  means, ses, _, _ = _cvsummary(errors[:,eligible], weights)
  cvm[eligible] = means
  cvse[eligible] = ses
  jmin = argmin(cvm)
  bound = cvm[jmin] + cvse[jmin]
  j1se = findfirst(j -> eligible[j] && cvm[j] <= bound, 1:jmin)
  cvm, cvse, jmin, j1se
end

function CVQuantileLassoPath(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                            λpath::Union{Nothing,AbstractVector{<:Real}}=nothing;
                            nfolds::Integer=10, folds::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                            rng::AbstractRNG=default_rng(), threads::Bool=true,
                            tau::Real=0.5, nλ::Integer=100, λminratio::Real=0.01,
                            offset=nothing, kwargs...)
  n = size(X,1)
  n > 0 || throw(ArgumentError("X must have observations"))
  length(y) == n || throw(DimensionMismatch("X and y must have the same observation count"))
  fl, K = _foldlabels(rng, n, nfolds, folds)
  all(>(0), fl) || throw(ArgumentError("fold labels must be positive integers"))
  seeds = rand(rng, UInt64, K+1)
  path = if λpath === nothing
    QuantileLassoPath(X,y; nλ, λminratio, tau, offset, rng=Xoshiro(seeds[end]), kwargs...)
  else
    QuantileLassoPath(X,y,λpath; tau, offset, rng=Xoshiro(seeds[end]), kwargs...)
  end
  ls = path.λpath
  errors = fill(NaN, K, length(ls))
  function fold!(k)
    train, test = findall(!=(k),fl), findall(==(k),fl)
    fp = QuantileLassoPath(X[train,:],y[train],ls; tau, offset=_rows(offset,train),
                           rng=Xoshiro(seeds[k]), kwargs...)
    for j in eachindex(fp.λpath)
      fp[j].converged || continue
      predictions = predict(fp[j], X[test,:]; offset=_rows(offset,test))
      errors[k,j] = sum(_qcheck(y[test[i]]-predictions[i],tau) for i in eachindex(test))/length(test)
    end
  end
  isempty(ls) && error("the full-data quantile path returned no fits; inspect solver limits")
  _runfolds(fold!, K, threads)
  cvm, cvse, jmin, j1se = _quantile_cvsummary(path, errors, _foldweights(Float64,fl,K,nothing))
  CVQuantileLassoPath(path,copy(ls),cvm,cvse,[count(!iszero,s.β) for s in path.solutions],
                       ls[jmin],ls[j1se],fl,errors)
end

function _quantile_cvsolution(cv::CVQuantileLassoPath, λ)
  level = _cvlevel(cv,λ)
  j = findfirst(==(Float64(level)),cv.λpath)
  j === nothing && throw(ArgumentError("quantile CV does not interpolate; request a saved penalty"))
  isfinite(cv.cvm[j]) || throw(ArgumentError("the requested penalty failed full-data or fold convergence"))
  cv.path[j]
end

StatsAPI.coef(cv::CVQuantileLassoPath; λ::Union{Symbol,Real}=:λ1se) = coef(_quantile_cvsolution(cv,λ))
StatsAPI.predict(cv::CVQuantileLassoPath, X::AbstractMatrix{<:Real}; λ::Union{Symbol,Real}=:λ1se, offset=nothing) =
  predict(_quantile_cvsolution(cv,λ),X;offset)
dual_scores(cv::CVQuantileLassoPath; λ::Union{Symbol,Real}=:λ1se) = dual_scores(_quantile_cvsolution(cv,λ))
StatsAPI.nobs(cv::CVQuantileLassoPath) = nobs(cv.path)
StatsAPI.fit(::Type{CVQuantileLassoPath}, X::AbstractMatrix, y::AbstractVector, args...; kwargs...) =
  CVQuantileLassoPath(X,y,args...;kwargs...)
