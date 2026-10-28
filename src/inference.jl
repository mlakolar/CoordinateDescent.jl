# Approximate inverses are stored by rows, without symmetrization. Symmetrizing
# would change both the bias correction and the constraints of the QP.
"""
    DebiasingMatrix

An approximate inverse on `targets`: `theta` has one row per target and one
column per predictor. `scores = (X .- means') * theta'`. `converged`,
`iterations`, and `error` report each solve; `error` is the largest off-diagonal
Gram residual for nodewise regression, or the constraint residual for the QP.
`lambda` is the penalty on the standardized design and `scales` its column
scales. The matrix can be reused with another response on the **same design**.
For nodewise rows, `bias_factor` is max_k!=j |A_k'z|/norm(z) on the working
columns and `noise_factor` is the original-unit SD per unit Gaussian noise.
`reference_noise` holds the LDPE stage1 noise; it is NaN for other rules.
`bias_target_met` records whether the original LDPE bias bound was attained,
`path_boundary` flags selection of a grid endpoint, and `tuning` identifies
the rule. These diagnostics do not establish confidence coverage.
"""
struct DebiasingMatrix{T}
  theta::SparseMatrixCSC{T, Int}
  scores::Matrix{T}
  targets::Vector{Int}
  means::Vector{T}
  scales::Vector{T}
  lambda::Vector{T}
  converged::Vector{Bool}
  iterations::Vector{Int}
  error::Vector{T}
  method::Symbol
  bias_factor::Vector{T}
  noise_factor::Vector{T}
  reference_noise::Vector{T}
  bias_target_met::Vector{Bool}
  path_boundary::Vector{Bool}
  tuning::Symbol
end

function _inference_data(X::AbstractMatrix{<:Real}, y = nothing)
  n, p = size(X)
  n >= 2 && p >= 1 || throw(ArgumentError("inference needs at least two rows and one predictor"))
  T = y === nothing ? float(eltype(X)) : float(promote_type(eltype(X), eltype(y)))
  A = Matrix{T}(X)
  _check_finite(A, "X")
  y === nothing && return A
  length(y) == n || throw(DimensionMismatch("X and y must have the same number of observations"))
  b = Vector{T}(y)
  _check_finite(b, "y")
  A, b
end

function _inference_targets(targets, p)
  t = collect(Int, targets)
  !isempty(t) && all(j -> 1 <= j <= p, t) && allunique(t) ||
    throw(ArgumentError("targets must be distinct predictor indices in 1:$p"))
  t
end

function _inference_level(x, name)
  isfinite(x) && x >= 0 || throw(ArgumentError("$name must be finite and nonnegative"))
  x
end

# A bounded number of tasks: sharing the Gram matrix must not allocate a solver
# state per predictor. Seeds belong to problems, not to threads.
function _inference_workers(f, count, threads)
  workers = threads ? min(Threads.nthreads(), count) : 1
  if workers == 1
    f(1:count)
  else
    @sync for w in 1:workers
      Threads.@spawn f(w:workers:count)
    end
  end
end

# Pooled prediction CV over a bounded, response-independent sample of nodes.
# Every fold estimates its own centering/scaling. The selected penalty belongs
# to the standardized full-data design; targets do not affect node selection.
function _nodewise_cv(A::Matrix{T}, options, rng; nfolds=5, nlambda=20, nodes=20,
                      intercept=true, standardize_x=true, gram=true) where {T}
  n, p = size(A)
  2 <= nfolds <= n || throw(ArgumentError("nodewise_cv_folds must be between 2 and n"))
  nlambda >= 2 || throw(ArgumentError("nodewise_cv_nlambda must be at least 2"))
  nodes >= 1 || throw(ArgumentError("nodewise_cv_nodes must be positive"))
  p == 1 && return zero(T)
  js = sort!(randperm(rng, p)[1:min(nodes, p)])
  fl, K = _foldlabels(rng, n, nfolds, nothing)
  # Columns are already standardized according to the caller's convention.
  G = transpose(A) * A[:,js] / n
  top = maximum(abs(G[k,a]) for (a,j) in enumerate(js) for k in 1:p if k != j)
  top <= eps(T) && return zero(T)
  levels = exp.(range(log(top), log(top * T(0.05)); length=nlambda))
  errs = zeros(T, nlambda)
  for k in 1:K
    train, test = findall(!=(k), fl), findall(==(k), fl)
    B, C = copy(A[train,:]), copy(A[test,:])
    center = intercept ? vec(mean(B; dims=1)) : zeros(T,p)
    B .-= transpose(center); C .-= transpose(center)
    scale = standardize_x ? vec(sqrt.(sum(abs2,B; dims=1) ./ length(train))) : ones(T,p)
    all(>(eps(T)), scale) || throw(ArgumentError("constant predictor in nodewise CV training fold"))
    B ./= transpose(scale); C ./= transpose(scale)
    H = gram ? transpose(B) * B / length(train) : nothing
    b = zeros(T,p)
    model = gram ? ProximalSolvers.cdmodel(QuadraticFunction(H,b,zero(T)), nothing) : nothing
    for j in js
      if gram
        b .= .-view(H,:,j); model.c = H[j,j]/2
      else
        model, _, _ = _lsmodel(B, Vector(B[:,j]), false, false)
      end
      lo, hi = fill(T(-Inf),p), fill(T(Inf),p); lo[j] = hi[j] = 0
      box = IndBox(lo,hi); x = SparseIterate(T,p)
      for (a,level) in enumerate(levels)
        r = coordinate_descent!(x,model,Constrained(NormL1(level),box),options; rng)
        r.converged || throw(ArgumentError("nodewise CV did not converge; adjust options"))
        e = C[:,j] - C * _sparsevec(x)
        errs[a] += sum(abs2,e)
      end
    end
  end
  levels[argmin(errs)]
end

# Zhang and Zhang (2014), Table 1, evaluated on a finite decreasing grid.
# The choice uses X only. kappa bounds the relative increase in the noise SD,
# not its variance. Return the actual selected penalty and both diagnostics.
function _ldpe_node!(x, model, box, options, rng, A::Matrix{T}, G, j;
                     path=nothing, nlambda=60, minratio=0.001, kappa=0.25,
                     bias_target=sqrt(2log(max(size(A,2),2))), relax=0.1, unpenalized=0) where {T}
  n,p=size(A)
  cross = G === nothing ? transpose(A)*A[:,j]/n : view(G,:,j)
  weights = ones(T,p)
  conditional = cross
  if unpenalized > 0
    others = sort!([k for k in 1:p if k!=j];by=k -> (-abs(cross[k]),k))
    selected = others[1:unpenalized]
    B = A[:,selected]
    rank(B)==unpenalized || throw(ArgumentError("linearly dependent unpenalized predictors for target $j"))
    residual = A[:,j] - B*(B\A[:,j])
    conditional = transpose(A)*residual/n
    weights[selected] .= 0
  end
  top = maximum((abs(conditional[k]) for k in 1:p if k!=j && weights[k]>0);init=zero(T))
  levels = path === nothing ? (top <= eps(T) ? T[0] :
    exp.(range(log(top),log(top*T(minratio));length=nlambda))) : collect(T,path)
  vectors=SparseVector{T,Int}[]; penalties=T[]; eta=T[]; noise=T[]
  passes=0; converged=true
  for level in levels
    result=coordinate_descent!(x,model,Constrained(NormL1(level,weights),box),options;rng)
    passes+=result.passes;converged &= result.converged
    v=-_sparsevec(x);v[j]=1
    z=A*v; normz=norm(z); denom=dot(A[:,j],z)
    denom > eps(T)*dot(A[:,j],A[:,j]) && normz>eps(T) || continue
    gz=G===nothing ? transpose(A)*z/n : G*v
    push!(vectors,v);push!(penalties,level)
    push!(eta,n*maximum((abs(gz[k]) for k in 1:p if k!=j);init=zero(T))/normz)
    push!(noise,normz/abs(denom))
  end
  isempty(vectors) && throw(ArgumentError("degenerate LDPE path for target $j"))
  target=T(bias_target)
  ref=findfirst(<=(target),eta)
  if ref===nothing
    target=(1+T(relax))*minimum(eta)
    ref=findfirst(<=(target),eta)
  end
  chosen=findlast(i -> noise[i] <= (1+T(kappa))*noise[ref]*(1+10eps(T)),ref:length(noise))
  # findlast over a range returns its positional index, not the value at it.
  chosen=ref+chosen-1
  vectors[chosen],penalties[chosen],eta[chosen],noise[chosen],noise[ref],
    eta[chosen] <= bias_target*(1+10sqrt(eps(T))),
    chosen==1 || chosen==length(vectors),converged,passes
end

"""
    nodewise_lasso(X, lambda=nothing; kwargs...)
    debiasing_matrix(X; method=:nodewise, lambda, mu, kwargs...)

Regress each target column on all other columns by the lasso, and divide its
residual by `X_j'residual/n`. This gives the nodewise inverse of Zhang and Zhang
(2014) and van de Geer et al. (2014), with `diag(theta * Gram) = 1`.

`intercept=true` centers the design; `standardize_x=true` divides columns by
their population standard deviations. The returned inverse and scores are on
the original scale. `targets=1:p` may select a subset. `lambda` is a common
penalty on the working design. A decreasing `lambda_path` ending at `lambda`
warm starts each regression through those levels. `options=CDOptions()` controls
coordinate descent; `threads=true` runs independent regressions in parallel with
problem-specific seeds drawn from `rng`.

`gram=:auto` shares a single dense Gram matrix if it fits `max_gram_bytes=2^30`;
`gram=true` requests it explicitly and `false` works on the residuals. Both
design and output scores are dense even for sparse input. Memory is O(np+p²)
with the Gram matrix, O(np) plus solver states without it. No p-by-p matrix is
copied per node. An omitted `lambda` (or `:cv`) selects a common penalty by
pooled five-fold prediction CV over 20 randomly sampled columns (or all columns
if fewer). The geometric grid has 20 levels from the largest sampled absolute
correlation to 5% of that value. Every training fold estimates its own centering
and scaling. `nodewise_cv_folds`, `nodewise_cv_nodes`, and
`nodewise_cv_nlambda` control this bounded search. Sampling does not depend on
`targets`. `lambda=:universal` retains `sqrt(2log(max(p,2))/n)`; a number fixes
the penalty and bypasses CV. Returned `lambda` records the selected penalty.
CV optimizes prediction error and does not guarantee coverage.

`lambda=:ldpe` follows Zhang and Zhang's Table1 on a finite node-specific grid.
It first chooses the largest penalty meeting `ldpe_bias_target=sqrt(2log(p))`;
if unattainable it uses `(1+ldpe_relax)*minimum(bias_factor)` and records the
original bound as unmet. It then chooses the smallest grid penalty with noise
SD at most `(1+ldpe_kappa)` times that reference. Defaults are
`ldpe_kappa=0.25`, `ldpe_relax=0.1`, `ldpe_nlambda=60`, and
`ldpe_minratio=0.001`. The grid starts at that node's maximum correlation.
`lambda_path` may instead supply a common decreasing grid. `lambda` in the
result stores each row's selected penalty. The noise budget applies to SD,
not variance. `ldpe_unpenalized=m` implements the restricted weighted variant:
the m largest absolute working correlations with each target receive zero
penalty. The remaining path starts at the largest conditional correlation after
regressing the target on those m columns. The default m=0 gives ordinary LDPE;
m must be at most min(p-1,n-2). This experimental expert option uses the design
only. Inspect the bias and boundary diagnostics before inference.

With `method=:qp`, `mu` defaults to the universal level (no CV). Solve Javanmard and Montanari's (2014) program
`min m'Gram*m/2` subject to `norm(Gram*m-e_j, Inf) <= mu`, with `0 < mu < 1`.
The dual is `min u'Gram*u/2-u_j+mu*norm(u,1)`, solved by coordinate descent.
The returned row is its primal solution `m=u`; feasibility and primal-dual gap
are checked. An infeasible or unconverged program throws, rather than silently
relaxing `mu`. `mu` applies to the working (standardized) Gram matrix. QP rows
are not normalized to unit diagonal and are not symmetrized. The QP requires
the shared Gram form. Inference needs the sparsity and regularity assumptions
of the cited methods; an approximate inverse alone does not ensure coverage.
"""
nodewise_lasso(X::AbstractMatrix{<:Real}, lambda::Union{Nothing, Real, Symbol} = nothing; kwargs...) =
  debiasing_matrix(X; method = :nodewise, lambda, kwargs...)

function debiasing_matrix(X::AbstractMatrix{<:Real}; method::Symbol = :nodewise,
    lambda::Union{Nothing, Real, Symbol} = nothing, mu::Union{Nothing, Real} = nothing,
    intercept::Bool = true, standardize_x::Bool = true, targets = 1:size(X, 2),
    lambda_path = nothing, gram::Union{Bool, Symbol} = :auto,
    max_gram_bytes::Integer = 2^30, threads::Bool = true,
    nodewise_cv_folds::Integer = min(5, size(X,1)), nodewise_cv_nlambda::Integer = 20, nodewise_cv_nodes::Integer = 20,
    ldpe_nlambda::Integer = 60, ldpe_minratio::Real = 0.001,
    ldpe_kappa::Real = 0.25, ldpe_bias_target::Real = sqrt(2log(max(size(X,2),2))),
    ldpe_relax::Real = 0.1, ldpe_unpenalized::Integer = 0,
    options::CDOptions = CDOptions(), rng::AbstractRNG = default_rng())
  method in (:nodewise, :qp) || throw(ArgumentError("method must be :nodewise or :qp"))
  lambda isa Real && _inference_level(lambda, "lambda")
  lambda isa Symbol && !(lambda in (:cv, :universal, :ldpe)) && throw(ArgumentError("lambda must be :cv, :universal, :ldpe or a number"))
  method == :qp && lambda === :ldpe && throw(ArgumentError("LDPE tuning applies to nodewise projections"))
  ldpe = method == :nodewise && lambda === :ldpe
  ldpe && !(ldpe_nlambda >= 2 && 0 < ldpe_minratio < 1 && 0 <= ldpe_kappa <= 1 &&
    isfinite(ldpe_bias_target) && ldpe_bias_target > 0 && 0 < ldpe_relax <= 1) &&
      throw(ArgumentError("invalid LDPE grid, noise budget or bias target"))
  gram in (true, false, :auto) || throw(ArgumentError("gram must be true, false or :auto"))
  max_gram_bytes >= 0 || throw(ArgumentError("max_gram_bytes must be nonnegative"))
  A = _inference_data(X)
  T = eltype(A)
  n, p = size(A)
  ts = _inference_targets(targets, p)
  0 <= ldpe_unpenalized <= min(p-1,n-2) || throw(ArgumentError("ldpe_unpenalized must be between 0 and min(p-1,n-2)"))
  ldpe_unpenalized > 0 && !ldpe && throw(ArgumentError("ldpe_unpenalized requires lambda=:ldpe"))
  means = intercept ? vec(mean(A; dims = 1)) : zeros(T, p)
  A .-= transpose(means)
  scales = standardize_x ? vec(sqrt.(sum(abs2, A; dims = 1) ./ n)) : ones(T, p)
  all(j -> sum(abs2, view(A, :, j)) > 0, 1:p) || throw(ArgumentError("X has a constant or zero predictor"))
  A ./= transpose(scales)
  usegram = gram === true || (gram === :auto && sizeof(T) * big(p)^2 <= max_gram_bytes)
  automatic = lambda === nothing || lambda === :cv
  tuning = method == :qp ? :qp : ldpe ? :ldpe : automatic ? :cv : lambda === :universal ? :universal : :fixed
  universal = T(sqrt(2log(max(p,2))/n))
  if method == :nodewise
    lambda = automatic ? _nodewise_cv(A, options, rng; nfolds=nodewise_cv_folds,
      nlambda=nodewise_cv_nlambda, nodes=nodewise_cv_nodes, intercept, standardize_x, gram=usegram) :
      (lambda === :universal || ldpe) ? universal : T(lambda)
  else
    lambda = lambda isa Real ? T(lambda) : universal
  end
  mu = mu === nothing ? lambda : T(mu)
  method == :qp && !(isfinite(mu) && 0 < mu < 1) && throw(ArgumentError("mu must be in (0,1)"))
  automatic && lambda_path !== nothing && throw(ArgumentError("lambda_path requires an explicit numerical lambda"))
  method == :qp && !usegram && throw(ArgumentError("the QP requires gram=true or sufficient max_gram_bytes"))
  G = usegram ? _gram(A, zeros(T, p)) : nothing
  levels = lambda_path === nothing ? T[lambda] : collect(T, lambda_path)
  !isempty(levels) && all(x -> isfinite(x) && x >= 0, levels) &&
    issorted(levels; rev = true) && (ldpe || last(levels) == T(lambda)) ||
    throw(ArgumentError("lambda_path must decrease to lambda"))
  method == :qp && lambda_path !== nothing && throw(ArgumentError("lambda_path is for nodewise regression"))
  cols = Vector{SparseVector{T, Int}}(undef, length(ts))
  ok = fill(false, length(ts))       # Bool vectors, not bit arrays: concurrent writes
  passes = zeros(Int, length(ts))
  errors = zeros(T, length(ts))
  actual_lambda=fill(T(method == :qp ? mu : lambda),length(ts))
  bias_factors=fill(T(NaN),length(ts));noise_factors=fill(T(NaN),length(ts))
  reference_noise=fill(T(NaN),length(ts));bias_met=fill(false,length(ts));boundary=fill(false,length(ts))
  # Draw by physical predictor: changing the requested subset/order must not
  # change a predictor's descent stream (including Xoshiro bulk generation).
  seeds = rand(rng, UInt64, p)
  _inference_workers(length(ts), threads) do ids
    # Construct once: QuadModel checks G for finite entries at construction.
    b = zeros(T, p)
    m = usegram ? ProximalSolvers.cdmodel(QuadraticFunction(G, b, zero(T)), nothing) : nothing
    for a in ids
      j = ts[a]
      x = SparseIterate(T, p)
      gen = Xoshiro(seeds[j])
      if method == :qp
        fill!(b, 0); b[j] = -1
        m.c = zero(T)
        r = coordinate_descent!(x, m, NormL1(T(mu)), options; rng = gen)
        v = _sparsevec(x)
        gv = G * v
        err = maximum(abs(gv[k] - (k == j)) for k in 1:p)
        gap = dot(v, gv) - v[j] + T(mu) * sum(abs, v)
        valid = r.converged && err <= mu + 10options.opt_tol &&
          abs(gap) <= 100options.opt_tol * max(one(T), dot(v, gv))
        valid || throw(ArgumentError("QP for target $j did not converge to a feasible optimum (residual=$err, gap=$gap)"))
        cols[a] = sparse(v ./ (scales .* scales[j]))
        ok[a], passes[a], errors[a] = true, r.passes, err
      else
        lo, hi = fill(T(-Inf), p), fill(T(Inf), p)
        lo[j] = hi[j] = 0
        box = IndBox(lo, hi)
        if usegram
          b .= .-view(G, :, j)
          m.c = G[j, j] / 2
        else
          m, _, _ = _lsmodel(A, Vector(view(A, :, j)), false, false)
        end
        if ldpe
          v,actual_lambda[a],bias_factors[a],noise_factors[a],reference_noise[a],
            bias_met[a],boundary[a],valid,passes[a] = _ldpe_node!(x,m,box,options,gen,A,G,j;
              path=lambda_path,nlambda=ldpe_nlambda,minratio=ldpe_minratio,kappa=ldpe_kappa,
              bias_target=ldpe_bias_target,relax=ldpe_relax,unpenalized=ldpe_unpenalized)
          noise_factors[a] /= scales[j];reference_noise[a] /= scales[j]
        else
          valid = true
          for level in levels
            r = coordinate_descent!(x, m, Constrained(NormL1(level), box), options; rng = gen)
            valid &= r.converged
            passes[a] += r.passes
          end
          v = -_sparsevec(x)
          v[j] = 1
        end
        z = A * v
        tau = dot(view(A, :, j), z) / n
        tau > eps(T) * sum(abs2, view(A, :, j)) / n ||
          throw(ArgumentError("degenerate nodewise residual for target $j"))
        cols[a] = sparse(v ./ (tau .* scales .* scales[j]))
        # Off-diagonal error is computed without forming theta*G for all rows.
        gz = usegram ? G * v : transpose(A) * z ./ n
        errors[a] = maximum((abs(gz[k] / tau) for k in 1:p if k != j); init = zero(T))
        if !ldpe
          bias_factors[a] = n*maximum((abs(gz[k]) for k in 1:p if k!=j);init=zero(T))/norm(z)
          noise_factors[a] = norm(z)/(n*tau*scales[j])
          bias_met[a] = bias_factors[a] <= ldpe_bias_target
        end
        ok[a] = valid
      end
    end
  end
  # Splatting p sparse vectors into hcat creates a p-argument method instance;
  # at p=10^4 Julia recursively infers the sparse concatenation. Assemble CSC
  # from triplets instead, with compilation independent of the predictor count.
  count = sum(nnz, cols)
  rowidx, colidx = Vector{Int}(undef, count), Vector{Int}(undef, count)
  values = Vector{T}(undef, count)
  at = 0
  for a in eachindex(cols)
    idx, val = findnz(cols[a])
    for k in eachindex(idx)
      at += 1
      rowidx[at], colidx[at], values[at] = a, idx[k], val[k]
    end
  end
  theta = sparse(rowidx, colidx, values, length(ts), p)
  scores = Matrix((A .* transpose(scales)) * transpose(theta))
  if method == :qp
    noise_factors .= vec(sqrt.(sum(abs2,scores;dims=1)))/n
  end
  DebiasingMatrix(theta, scores, ts, means, scales, actual_lambda,
                  ok, passes, errors, method,bias_factors,noise_factors,reference_noise,bias_met,boundary,tuning)
end

@doc (@doc nodewise_lasso) debiasing_matrix

"""
    DebiasedLassoResult

Debiased slope estimates `beta`, their standard errors `se`, and marginal
normal `pvalues`, ordered by `projection.targets`. `initial` holds all initial
slopes, `intercept` the fitted intercept, `residuals` the response residuals,
`influence` the n-by-target influence vectors, and `lambda` the initial penalty.
`converged` includes the initial and all projection solves. `coef`, `stderror`,
`vcov`, `confint` and `nobs` follow StatsAPI. Intercept inference is not included.
"""
struct DebiasedLassoResult{T}
  beta::Vector{T}
  se::Vector{T}
  pvalues::Vector{T}
  initial::Vector{T}
  intercept::T
  residuals::Vector{T}
  influence::Matrix{T}
  projection::DebiasingMatrix{T}
  lambda::T
  sigma::T
  family::Symbol
  hasintercept::Bool
  standardize_x::Bool
  robust::Bool
  variance::Symbol
  converged::Bool
end

# HC corrections use the selected initial design, including its intercept.
# This is a finite-sample variance policy, not a post-selection coverage proof.
function _inference_influence(A, residuals, scores, beta, intercept, sd, variance)
  variance == :model && return sd .* scores
  n = size(A, 1)
  adjusted = residuals
  if variance in (:hc1, :hc2, :hc3)
    ids = findall(!iszero, beta)
    M = intercept ? hcat(ones(eltype(A), n), A[:, ids]) : A[:, ids]
    k = size(M, 2)
    k < n && rank(M) == k || throw(ArgumentError("HC variance needs a full-rank initial selected design with residual degrees of freedom"))
    if variance == :hc1
      T = eltype(A)
      adjusted = residuals .* sqrt(T(n) / T(n - k))
    else
      h = k == 0 ? zeros(eltype(A), n) : vec(sum(abs2, Matrix(qr(M).Q)[:, 1:k]; dims=2))
      all(x -> 0 <= x < 1 - 1e-8, h) || throw(ArgumentError("HC variance needs leverages below one"))
      adjusted = residuals ./ (variance == :hc2 ? sqrt.(1 .- h) : 1 .- h)
    end
  end
  influence = scores .* adjusted
  influence .-= mean(influence; dims=1)
  influence
end

function _glm_moments(eta, family)
  if family == :binomial
    m = map(t -> t >= 0 ? inv(1 + exp(-t)) : exp(t) / (1 + exp(t)), eta)
    return m, m .* (1 .- m)
  elseif family == :poisson
    m = exp.(eta)
    all(isfinite, m) || throw(ArgumentError("nonfinite Poisson mean"))
    return m, m
  end
  throw(ArgumentError("family must be :gaussian, :binomial or :poisson"))
end

"""
    debiased_lasso(X, y, lambda=nothing; family=:gaussian, kwargs...)

The desparsified lasso, `beta + theta * X'(y-mu)/n`, for Gaussian, logistic
(`:binomial`) and Poisson regression. An omitted `lambda` selects the initial
penalty by five-fold prediction CV (`initial_tuning=:cv`), using the minimum
held-out MSE for Gaussian data or deviance for GLMs on a 40-level lasso path.
`initial_cv_folds` and `initial_cv_nlambda` control this search. Unless an
explicit `options.gaptol` is supplied, inference CV paths use `gaptol=0` and
stop using `options.opt_tol`, preserving the validated numerical policy. A numerical
penalty bypasses CV. Gaussian noise uses residual RMS with
`n - nnz(beta) - intercept` degrees of freedom; `sigma` may supply the noise SD.
For Gaussian data, `initial_tuning=:scaled` with an omitted penalty retains the
scaled lasso at universal level `sqrt(2log(max(p,2))/n)` and its noise estimate.
For an otherwise automatic Gaussian call, the default projection is restricted
LDPE: four X-only correlated controls (capped by the available dimension),
`ldpe_kappa=0.5`, 60 levels down to 0.001 of the conditional maximum,
and bias target `sqrt(2log(p))`. The default variance is centered HC3, using
leverages of the selected initial design. This policy passed pooled calibration
checks on 500 fresh datasets in each of ten Gaussian designs. Requested-role
coverage ranged from 93.2% to 97.8%; correlated designs had long intervals and
low power. These are empirical results, not uniform coverage guarantees.
The initial estimator remains regularized: post-selection refits did not
uniformly improve development calibration.

An explicit initial penalty, initial fit, projection, `nodewise_lambda`, noise
SD, or nonstandard centering/scaling requests the expert path. Its variance
defaults to model-based. `nodewise_lambda=:cv` restores pooled projection CV;
`:universal`, `:ldpe`, or a number specifies another projection rule. For GLMs
the Hessian-weighted, weighted-centered design supplies the projection and the
model variance is one. GLM and QP defaults remain experimental: fresh studies
found undercoverage. Numerical convergence does not establish calibration.

`initial` can supply all p slopes; its intercept is `mean(y-X*initial)` for
Gaussian models, or `initial_intercept` for GLMs. `projection` can reuse an
approximate inverse (caller must ensure it belongs to the same working design).
`variance` selects `:model`, `:hc0`, `:hc1`, `:hc2`, or `:hc3`. HC1--HC3 are
Gaussian only and require a full-rank initial selected design. `robust=true`
is the compatible alias for `variance=:hc0`, and `robust=false` for `:model`.
Explicit policies override the automatic variance choice. Other keywords pass to
[`debiasing_matrix`](@ref), including `method`, `lambda` there named
`nodewise_lambda`, `mu`, `targets`, `gram`, `threads`, and `lambda_path`.
`intercept=true`, `standardize_x=true`, `options` and `rng` also control the
initial fit. Automatic Gaussian fits use `opt_tol=1e-8, gaptol=0`; explicit
`options` are respected. Other calls retain `CDOptions()` as their default.
Nonconvergence is reported in the result, and must be
resolved before using its intervals. Confidence intervals are asymptotic.
"""
function debiased_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, lambda::Union{Nothing, Real} = nothing;
    family::Symbol = :gaussian, initial = nothing, initial_intercept::Real = 0,
    sigma::Union{Nothing, Real} = nothing, projection = nothing, robust::Union{Nothing, Bool} = nothing,
    variance::Symbol = :auto,
    intercept::Bool = true, standardize_x::Bool = true,
    nodewise_lambda::Union{Nothing, Real, Symbol} = nothing,
    initial_tuning::Symbol = :cv, initial_cv_folds::Integer = min(5, size(X,1)), initial_cv_nlambda::Integer = 40,
    options::Union{Nothing, CDOptions} = nothing, rng::AbstractRNG = default_rng(), kwargs...)
  A, b = _inference_data(X, y)
  T = eltype(A); n, p = size(A)
  family in (:gaussian, :binomial, :poisson) || throw(ArgumentError("unknown family $family"))
  automatic = family == :gaussian && lambda === nothing && initial === nothing &&
    projection === nothing && nodewise_lambda === nothing && sigma === nothing &&
    initial_tuning == :cv && intercept && standardize_x && get(kwargs, :method, :nodewise) == :nodewise
  options = options === nothing ? (automatic ? CDOptions(; opt_tol=1e-8, gaptol=0) : CDOptions()) : options
  if automatic
    nodewise_lambda = :ldpe
    kwargs = merge((ldpe_unpenalized=min(4, p-1, n-2), ldpe_kappa=0.5), (; kwargs...))
  end
  variance in (:auto, :model, :hc0, :hc1, :hc2, :hc3) || throw(ArgumentError("unknown variance policy $variance"))
  if variance == :auto
    variance = robust === nothing ? (automatic ? :hc3 : :model) : (robust ? :hc0 : :model)
  elseif robust !== nothing && (robust ? variance != :hc0 : variance != :model)
    throw(ArgumentError("specify variance or the compatible robust alias, not conflicting policies"))
  end
  family != :gaussian && variance in (:hc1, :hc2, :hc3) &&
    throw(ArgumentError("HC1/HC2/HC3 variance policies are Gaussian only"))
  robust = variance != :model
  lambda === nothing || _inference_level(lambda, "lambda")
  sigma === nothing || (isfinite(sigma) && sigma > 0) || throw(ArgumentError("sigma must be positive and finite"))
  family != :gaussian && sigma !== nothing && sigma != 1 && throw(ArgumentError("GLM dispersion is one"))
  initial_tuning in (:cv, :scaled) || throw(ArgumentError("initial_tuning must be :cv or :scaled"))
  initial_cv_nlambda >= 2 || throw(ArgumentError("initial_cv_nlambda must be at least 2"))
  family != :gaussian && lambda === nothing && initial_tuning == :scaled &&
    throw(ArgumentError("scaled initial tuning is Gaussian only"))
  sigmafit = nothing
  if initial === nothing
    if lambda === nothing && initial_tuning == :cv
      # Inference keeps the fit-accuracy stopping rule used in validation.
      # Generic prediction paths may otherwise use a looser default gap target.
      cvoptions = options.gaptol === nothing ? CDOptions(options; gaptol=0) : options
      cv = if family == :gaussian
        CVLassoPath(A,b,cvoptions; nfolds=initial_cv_folds,nλ=initial_cv_nlambda,
                    intercept,standardize_x,threads=false,rng)
      else
        loss = family == :binomial ? ProximalBase.Logistic() : ProximalBase.Poisson()
        CVGLMLassoPath(A,b,loss,cvoptions; nfolds=initial_cv_folds,nλ=initial_cv_nlambda,
                       intercept,standardize_x,threads=false,rng)
      end
      lambda = cv.λmin
    end
    sol = if family == :gaussian && lambda === nothing
      scaled_lasso(A, b, T(sqrt(2log(max(p, 2))/n)), standardize_x ? _loadings(A, intercept) : ones(T, p),
                   IterLassoOptions(; options_cd = options); intercept, rng)
    elseif family == :gaussian
      lasso(A, b, T(lambda), options; intercept, standardize_x, rng)
    else
      loss = family == :binomial ? ProximalBase.Logistic() : ProximalBase.Poisson()
      glm_lasso(A, b, T(lambda), loss, options; intercept, standardize_x, rng)
    end
    beta = Vector(_sparsevec(sol.x)); beta0 = sol.intercept
    conv = sol.converged
    if family == :gaussian && lambda === nothing
      sigmafit = sol.σ
      lambda = sol.penalty.λ0
    end
  else
    length(initial) == p || throw(DimensionMismatch("initial must contain p slopes"))
    beta = Vector{T}(initial); _check_finite(beta, "initial")
    beta0 = intercept ? (family == :gaussian ? mean(b - A * beta) : T(initial_intercept)) : zero(T)
    isfinite(beta0) || throw(ArgumentError("initial_intercept must be finite"))
    conv = true
    lambda = lambda === nothing ? zero(T) : T(lambda)
  end
  eta = A * beta .+ beta0
  if family == :gaussian
    res = b - eta
    df = n - count(!iszero, beta) - intercept
    sigma === nothing && sigmafit === nothing && df <= 0 &&
      throw(ArgumentError("no residual degrees of freedom: supply sigma"))
    sd = T(sigma === nothing ? (sigmafit === nothing ? norm(res) / sqrt(df) : sigmafit) : sigma)
    isfinite(sd) && sd > 0 || throw(ArgumentError("noise estimate must be positive and finite"))
    proj = projection === nothing ? debiasing_matrix(A; lambda = nodewise_lambda, intercept, standardize_x, options, rng, kwargs...) : projection
    scores = proj.scores
    influence = _inference_influence(A, res, scores, beta, intercept, sd, variance)
  else
    _check_response(family == :binomial ? ProximalBase.Logistic() : ProximalBase.Poisson(), b)
    meanfit, h = _glm_moments(eta, family)
    sum(h) > eps(T) || throw(ArgumentError("degenerate GLM Hessian"))
    center = intercept ? vec(transpose(A) * h) ./ sum(h) : zeros(T, p)
    Ac = A .- transpose(center)
    weighted = sqrt.(h) .* Ac
    proj = projection === nothing ? debiasing_matrix(weighted; lambda = nodewise_lambda, intercept = false, standardize_x, options, rng, kwargs...) : projection
    scores = Matrix(Ac * transpose(proj.theta))
    res = b - meanfit
    influence = robust ? scores .* res : scores .* sqrt.(h)
    sd = one(T)
  end
  size(scores) == (n, length(proj.targets)) && size(proj.theta, 2) == p ||
    throw(DimensionMismatch("projection does not match X"))
  family != :gaussian && robust && (influence .-= mean(influence; dims = 1))
  se = vec(sqrt.(sum(abs2, influence; dims = 1))) ./ n
  all(x -> isfinite(x) && x > 0, se) || throw(ArgumentError("degenerate standard errors"))
  bhat = beta[proj.targets] + transpose(scores) * res ./ n
  pv = 2 .* _pnorm.(-abs.(bhat ./ se))
  DebiasedLassoResult(bhat, se, pv, beta, T(beta0), res, influence, proj, T(lambda), sd, family, intercept, standardize_x, robust, variance,
                      conv && all(proj.converged))
end

StatsAPI.coef(r::DebiasedLassoResult) = copy(r.beta)
StatsAPI.stderror(r::DebiasedLassoResult) = copy(r.se)
StatsAPI.nobs(r::DebiasedLassoResult) = length(r.residuals)
StatsAPI.vcov(r::DebiasedLassoResult) = transpose(r.influence) * r.influence / nobs(r)^2
function StatsAPI.confint(r::DebiasedLassoResult; level::Real = 0.95)
  0 < level < 1 || throw(ArgumentError("level must be in (0,1)"))
  z = _qnorm((1 + float(level)) / 2)
  [r.beta .- z .* r.se r.beta .+ z .* r.se]
end

# Shared max-T adjustment. Stepdown orders hypotheses by observed |t| and
# enforces monotonicity of adjusted p-values in that order.
function _inference_adjust(draws, observed, stepdown)
  m, B = size(draws)
  order = sortperm(abs.(observed); rev = true)
  adj = zeros(eltype(draws), m)
  maxima = zeros(eltype(draws), B)
  for a in reverse(eachindex(order))
    j = order[a]
    maxima .= max.(maxima, abs.(view(draws, j, :)))
    stepdown && (adj[j] = (1 + count(>=(abs(observed[j])), maxima)) / (B + 1))
  end
  if stepdown
    for a in 2:m
      adj[order[a]] = max(adj[order[a]], adj[order[a-1]])
    end
  else
    for j in 1:m
      adj[j] = (1 + count(>=(abs(observed[j])), maxima)) / (B + 1)
    end
  end
  adj, maxima
end

function _inference_boot_result(r, draws, null, level, stepdown)
  0 < level < 1 || throw(ArgumentError("level must be in (0,1)"))
  m, B = size(draws)
  B >= 1 || throw(ArgumentError("at least one bootstrap draw is required"))
  all(isfinite, draws) || throw(ArgumentError("bootstrap draws must be finite"))
  h0 = null isa Real ? fill(null, m) : collect(null)
  length(h0) == m && all(isfinite, h0) || throw(ArgumentError("null must be finite with one value per target"))
  obs = (r.beta - h0) ./ r.se
  p = [(1 + count(>=(abs(obs[j])), abs.(view(draws, j, :)))) / (B + 1) for j in 1:m]
  adjusted, maxima = _inference_adjust(draws, obs, stepdown)
  # Order statistic with finite-Monte-Carlo correction; Inf if B cannot resolve alpha.
  k = ceil(Int, (B + 1) * level)
  critical = k > B ? Inf : sort(maxima)[k]
  (; pvalues = p, adjusted_pvalues = max.(p, adjusted), critical,
     intervals = [r.beta .- critical .* r.se r.beta .+ critical .* r.se], draws, maxima,
     level, stepdown)
end

"""
    multiplier_bootstrap(result; B=1000, rng=Random.default_rng(), multipliers=nothing,
                         null=0, level=0.95, stepdown=false)

Gaussian multiplier bootstrap of the joint influence vectors of a
[`debiased_lasso`](@ref) result (Zhang and Cheng, 2017). One common multiplier
per observation preserves dependence across targets. Optional `multipliers`
is a finite n-by-B matrix, useful for reproducible comparisons. Returns marginal
and Westfall--Young max-T adjusted p-values, simultaneous intervals, and the
studentized draws. `stepdown=true` gives stepdown max-T adjusted p-values;
intervals always use the single-step maximum. Monte Carlo p-values use the
add-one correction. Homoskedastic fits use model influences; robust fits use
centered empirical influences. This does not refit bootstrap samples; use
[`bootstrap_lasso`](@ref) for a residual/wild bootstrap with refitting.
"""
function multiplier_bootstrap(r::DebiasedLassoResult; B::Integer = 1000,
    rng::AbstractRNG = default_rng(), multipliers = nothing, null = 0,
    level::Real = 0.95, stepdown::Bool = false)
  r.converged || throw(ArgumentError("resolve nonconvergence before bootstrapping"))
  B >= 1 || throw(ArgumentError("B must be positive"))
  E = multipliers === nothing ? randn(rng, eltype(r.beta), nobs(r), B) : Matrix{eltype(r.beta)}(multipliers)
  size(E, 1) == nobs(r) && size(E, 2) >= 1 || throw(DimensionMismatch("multipliers must be n-by-B"))
  all(isfinite, E) || throw(ArgumentError("multipliers must be finite"))
  draws = (transpose(r.influence) * E) ./ (nobs(r) .* r.se)
  _inference_boot_result(r, draws, null, level, stepdown)
end

"""
    bootstrap_lasso(X, y, result; B=1000, wild=false, resamples=nothing,
                    rng=Random.default_rng(), level=0.95, stepdown=false, options=CDOptions())

Residual bootstrap of a Gaussian debiased lasso with lasso refitting at the
fixed initial penalty (`hdi::boot.lasso.proj`'s `boot.shortcut=true`). Centered
residuals are sampled with replacement, or multiplied by independent normal
variables with `wild=true`. `resamples` may supply an n-by-B matrix of these
already resampled centered errors. Reuses the original nodewise scores.
Studentization uses the residual SD with residual degrees of freedom for each
refit. A second refit under the complete null supplies the Westfall--Young
single-step or stepdown max-T adjustment. Returns centered studentized draws
and null draws, plus `equal_tail_pvalues` using hdi's marginal convention.
`pvalues` uses absolute tails, as in [`multiplier_bootstrap`](@ref).
This method is Gaussian-only; it is separate from multiplier
inference. Uses the original `sigma` only for the observed statistic.
"""
function bootstrap_lasso(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, r::DebiasedLassoResult;
    B::Integer = 1000, wild::Bool = false, resamples = nothing,
    rng::AbstractRNG = default_rng(), level::Real = 0.95, stepdown::Bool = false,
    options::CDOptions = CDOptions())
  r.family == :gaussian || throw(ArgumentError("residual bootstrap requires a Gaussian model"))
  r.converged || throw(ArgumentError("resolve nonconvergence before bootstrapping"))
  A, b = _inference_data(X, y)
  size(A) == (nobs(r), length(r.initial)) || throw(DimensionMismatch("X does not match the fit"))
  isapprox(b - A * r.initial .- r.intercept, r.residuals; rtol = sqrt(eps(eltype(A))), atol = sqrt(eps(eltype(A)))) ||
    throw(ArgumentError("X and y must be the data used for the original fit"))
  n = nobs(r); m = length(r.beta)
  B >= 1 || throw(ArgumentError("B must be positive"))
  rc = r.residuals .- mean(r.residuals)
  E = resamples === nothing ? (wild ? rc .* randn(rng, n, B) : rc[rand(rng, 1:n, n, B)]) : Matrix(resamples)
  size(E, 1) == n && size(E, 2) >= 1 && all(isfinite, E) || throw(ArgumentError("resamples must be a finite n-by-B matrix"))
  B = size(E, 2)
  draws, nulldraws = zeros(eltype(A), m, B), zeros(eltype(A), m, B)
  hasintercept = r.hasintercept
  fitted = A * r.initial .+ r.intercept
  for k in 1:B
    for undernull in (false, true)
      ys = undernull ? E[:, k] : fitted + E[:, k]
      fit = lasso(A, ys, r.lambda, options; intercept = hasintercept, standardize_x = r.standardize_x, rng)
      fit.converged || throw(ArgumentError("bootstrap refit $k did not converge"))
      beta = Vector(_sparsevec(fit.x))
      df = n - count(!iszero, beta) - hasintercept
      df > 0 || throw(ArgumentError("bootstrap refit has no residual degrees of freedom"))
      sd = norm(fit.residuals) / sqrt(df)
      sd > 0 || throw(ArgumentError("degenerate bootstrap residuals"))
      inf = _inference_influence(A, fit.residuals, r.projection.scores, beta, hasintercept, sd, r.variance)
      se = vec(sqrt.(sum(abs2, inf; dims = 1))) ./ n
      bhat = beta[r.projection.targets] + transpose(r.projection.scores) * fit.residuals ./ n
      truth = undernull ? zeros(m) : r.initial[r.projection.targets]
      (undernull ? nulldraws : draws)[:, k] = (bhat - truth) ./ se
    end
  end
  out = _inference_boot_result(r, draws, 0, level, stepdown)
  observed = r.beta ./ r.se
  equal_tail = map(1:m) do j
    countleft = count(<=(observed[j]), view(draws, j, :))
    (2min(countleft, B - countleft) + 1) / (B + 1)
  end
  adjusted, maxima = _inference_adjust(nulldraws, r.beta ./ r.se, stepdown)
  k = ceil(Int, (B + 1) * level)
  critical = k > B ? Inf : sort(maxima)[k]
  merge(out, (; adjusted_pvalues = max.(out.pvalues, adjusted), equal_tail_pvalues = equal_tail, null_draws = nulldraws, maxima, critical,
               intervals = [r.beta .- critical .* r.se r.beta .+ critical .* r.se]))
end

"""
    decorrelated_score(X, y, target, lambda; family=:binomial, null=0,
                       projection_lambda=lambda, intercept=true, sigma=nothing,
                       options=CDOptions(), rng=Random.default_rng())

Ning and Liu (2017) decorrelated score test of one slope. Fit the nuisance
coefficients under `beta[target]=null`, then project the target on the nuisance
design by Hessian-weighted lasso (their penalized least-squares alternative).
The intercept is unpenalized in both fits. With residualized target `z`, the
score is `mean(z .* (mu-y))` and information `mean(h .* z .* X[:,target])`.
Return the score, information, normal statistic, two-sided p-value, nuisance
fit and projection, and convergence flag. Gaussian models use `sigma` or the
restricted residual SD; GLMs have dispersion one. The weighted lasso penalty
is normalized by n, including with a non-unit Hessian. No target selection or
multiplicity adjustment is performed.
"""
function decorrelated_score(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, target::Integer, lambda::Real;
    family::Symbol = :binomial, null::Real = 0, projection_lambda::Real = lambda,
    intercept::Bool = true, sigma::Union{Nothing, Real} = nothing,
    options::CDOptions = CDOptions(), rng::AbstractRNG = default_rng())
  A, b = _inference_data(X, y)
  n, p = size(A); T = eltype(A)
  1 <= target <= p || throw(ArgumentError("target must be in 1:p"))
  isfinite(null) || throw(ArgumentError("null must be finite"))
  _inference_level(lambda, "lambda"); _inference_level(projection_lambda, "projection_lambda")
  family in (:gaussian, :binomial, :poisson) || throw(ArgumentError("unknown family"))
  sigma === nothing || (isfinite(sigma) && sigma > 0) || throw(ArgumentError("sigma must be positive and finite"))
  family != :gaussian && sigma !== nothing && sigma != 1 && throw(ArgumentError("GLM dispersion is one"))
  others = [j for j in 1:p if j != target]
  # A zero nuisance column allows the existing solvers to handle p=1 too.
  N = isempty(others) ? zeros(T, n, 1) : A[:, others]
  xj = A[:, target]
  offset = T(null) .* xj
  fit = if family == :gaussian
    lasso(N, b, T(lambda), options; intercept, offset, rng)
  else
    loss = family == :binomial ? ProximalBase.Logistic() : ProximalBase.Poisson()
    glm_lasso(N, b, T(lambda), loss, options; intercept, offset, rng)
  end
  eta = N * _sparsevec(fit.x) .+ fit.intercept .+ offset
  meanfit, h = family == :gaussian ? (eta, ones(T, n)) : _glm_moments(eta, family)
  sum(h) > eps(T) || throw(ArgumentError("degenerate Hessian"))
  proj = lasso(N, xj, T(projection_lambda * n / sum(h)), options; intercept, weights = h, rng)
  z = proj.residuals
  information = dot(h .* z, xj) / n
  information > 0 || throw(ArgumentError("nonpositive decorrelated information"))
  df = n - count(!iszero, _sparsevec(fit.x)) - intercept
  family == :gaussian && sigma === nothing && df <= 0 && throw(ArgumentError("supply sigma: no residual degrees of freedom"))
  sd = family == :gaussian ? (sigma === nothing ? norm(b - eta) / sqrt(df) : sigma) : one(T)
  isfinite(sd) && sd > 0 || throw(ArgumentError("degenerate noise estimate"))
  score = dot(z, meanfit - b) / (n * sd^2)
  information /= sd^2
  statistic = sqrt(T(n)) * score / sqrt(information)
  (; score, information, statistic, pvalue = 2 * _pnorm(-abs(statistic)), null, target, family,
     nuisance = fit, projection = proj, converged = fit.converged && proj.converged)
end
