# Sparse singular-vector regression by exclusive extraction. Each component
# solves a weighted rank-one problem without forming Kronecker designs.
_soft_lr(x, c) = sign(x) * max(abs(x) - c, zero(x))
function _lr_lasso_gram!(u, H, z, λ; max_iter, tol)
  g = H * u - z
  for sweep in 1:max_iter
    for j in eachindex(u)
      h = H[j, j]
      v = h > 0 ? _soft_lr(h*u[j] - g[j], λ) / h : zero(eltype(u))
      δ = v - u[j]
      u[j] = v
      g .+= δ .* view(H, :, j)
    end
    err = maximum(u[j] == 0 ? max(0, abs(g[j])-λ) : abs(g[j]+λ*sign(u[j])) for j in eachindex(u))
    err <= tol * max(1, norm(z)) && return true
  end
  false
end

function _lr_unit(d, λ, wu, wv, u0, v0; max_iter, tol, inner_max_iter, inner_tol)
  Xw = d.X .* wu'
  H, C = Xw' * Xw / size(d.X, 1), Xw' * d.Y / size(d.X, 1)
  u, v = u0 ./ wu, v0 ./ wv
  u ./= sum(abs, u); v ./= sum(abs, v)
  B = zeros(eltype(u), length(u), length(v))
  converged, iterations = false, 0
  for k in 1:max_iter
    iterations = k
    old = B
    a = wv .* v
    inner = _lr_lasso_gram!(u, H .* sum(abs2, a), C * a, λ; max_iter = inner_max_iter, tol = inner_tol)
    du = sum(abs, u)
    if du == 0
      return B .* 0, true, k, 0
    end
    u ./= du
    z = wv .* (C' * u)
    h = sum(abs2, Xw * u) / size(d.X, 1)
    for j in eachindex(v)
      hj = h * wv[j]^2
      v[j] = hj > 0 ? _soft_lr(z[j], λ) / hj : zero(eltype(v))
    end
    dv = sum(abs, v)
    if dv == 0
      return B .* 0, true, k, 0
    end
    v ./= dv
    B = (wu .* u) * (wv .* v)' .* dv
    if inner && norm(B-old) <= tol * max(1, norm(B))
      converged = true
      break
    end
  end
  df = count(!iszero, u) + count(!iszero, v) - 1
  B, converged, iterations, df
end

"""
    sparse_svd_regression(X, Y, rank; intercept = true, nλ = 50,
                          λminratio = 1e-5, adaptive = true, gamma = 2,
                          criterion = :bic, max_iter = 1000, tol = 1e-7,
                          inner_max_iter = 10_000, inner_tol = 1e-9)

Sparse singular-vector regression by one round of iterative exclusive
extraction (Chen, Chan and Stenseth, 2012; rrpack's `rssvd`). From the SVD of
the rank-constrained least-squares coefficient matrix, form the exclusive
response layer for each component. Fit each layer with a weighted rank-one
lasso, selecting its penalty by `:bic`, `:bicp` or `:aic`, including the zero
component. Both predictor and response loadings can be sparse. The rank is an
upper bound; selected layers need not remain mutually orthogonal.

Adaptive inverse weights use the absolute initial singular vectors to power
`gamma`, with zero entries replaced by `eps(T)`. Each component's grid runs
from `0.9λmax` to `λminratio*λmax`, followed by zero, where
`λmax = maximum(abs.(wu .* (Xc'Ylayer/n) .* wv'))`. The loss is divided by
`2n`. Optimization is nonconvex and convergence is local. Returns
`(solution, components, λ, scores, convergedpath, selected)`; the scores include every grid value and
the zero model. Nonconverged candidates have infinite scores and their
flags are preserved in `convergedpath`. Component scores use `nnz(u)+nnz(v)-1` parameters.
"""
function sparse_svd_regression(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, r::Integer;
                               intercept::Bool = true, nλ::Integer = 50, λminratio::Real = 1e-5,
                               adaptive::Bool = true, gamma::Real = 2, criterion::Symbol = :bic,
                               max_iter::Integer = 1000, tol::Real = 1e-7,
                               inner_max_iter::Integer = 10_000, inner_tol::Real = 1e-9)
  _lr_options(0, max_iter, tol); _lr_options(0, inner_max_iter, inner_tol)
  nλ >= 2 || throw(ArgumentError("nλ must be at least 2"))
  isfinite(λminratio) && 0 < λminratio <= 0.9 || throw(ArgumentError("λminratio must be in (0, 0.9]"))
  isfinite(gamma) && gamma >= 0 || throw(ArgumentError("gamma must be finite and nonnegative"))
  criterion in (:bic, :bicp, :aic) || throw(ArgumentError("criterion must be :bic, :bicp or :aic"))
  d = _lr_data(X, Y, intercept)
  f = _lr_svd(d)
  _lr_rank(r, min(f.xrank, size(Y, 2)))
  T = eltype(d.X)
  initial = _lr_rrcoef(d, f, r)
  F = svd(initial; full = false)
  components = Matrix{T}[]
  levels, scores = Vector{T}[], Vector{T}[]
  convergedpath, selected = Vector{Bool}[], Int[]
  total = zeros(T, size(initial))
  iterations, converged, df = 0, true, 0
  n, q, p = size(X, 1), size(Y, 2), size(X, 2)
  pen = criterion == :bic ? log(n*q) : criterion == :bicp ? 2log(p*q) : 2
  for k in 1:r
    layer = F.U[:, k] * F.Vt[k:k, :] .* F.S[k]
    dk = merge(d, (; Y = d.Y - d.X * (initial - layer), intercept = false, my = zeros(T, q)))
    wu = adaptive ? max.(abs.(F.U[:, k]).^gamma, eps(T)) : ones(T, p)
    wv = adaptive ? max.(abs.(F.V[:, k]).^gamma, eps(T)) : ones(T, q)
    C = (dk.X' * dk.Y / n) .* wu .* wv'
    ij = argmax(abs.(C))
    u, v = zeros(T, p), zeros(T, q)
    u[ij[1]], v[ij[2]] = one(T), one(T)
    λmax = abs(C[ij])
    λs = [T(λmax * 0.9 * (λminratio / 0.9)^(j/max(1, nλ-2))) for j in 0:nλ-2]
    push!(λs, zero(T))
    candidates = Matrix{T}[]
    cs, ds = Bool[], Int[]
    vals = T[]
    for λ in λs
      B, ok, it, countdf = _lr_unit(dk, λ, wu, wv, u, v; max_iter, tol, inner_max_iter, inner_tol)
      push!(candidates, B); push!(cs, ok); push!(ds, countdf)
      push!(vals, ok ? log(sum(abs2, dk.Y-dk.X*B)) + pen*countdf/(n*q) : T(Inf))
      iterations += it
      if any(!iszero, B)
        S = svd(B; full = false)
        u, v = S.U[:, 1], S.V[:, 1]
      end
    end
    push!(vals, log(sum(abs2, dk.Y)))
    push!(cs, true)
    push!(λs, λmax)
    j = argmin(vals)
    B = j <= nλ ? candidates[j] : zeros(T, p, q)
    j <= nλ && (df += ds[j]; converged &= cs[j])
    total .+= B
    push!(components, B); push!(levels, λs); push!(scores, vals)
    push!(convergedpath, cs); push!(selected, j)
  end
  sol = _lr_solution(d, total, 0, iterations, converged, sum(abs2, d.Y-d.X*total)/(2n), :sparse_svd;
                     df = df + q*intercept)
  (; solution = sol, components, λ = levels, scores, convergedpath, selected)
end
