# Accuracy certificates of the benchmark cases, computed from the raw inputs
# and the returned point only. Each states the documented objective of an
# estimator (its scaling, weights, offset, loadings, intercept and
# standardization), evaluates it at the returned coefficients and intercept,
# and bounds the distance to optimality independently of the solver's stopping
# rule. Nothing here calls HighDimensionalModels; HiGHS is called through its C
# API only to complete the dual of exact quantile regression.
#
# Every certificate returns (; value, measure, status, note), with status :ok
# or :uncertifiable (value NaN, and a note saying why). Nonfinite inputs are
# uncertifiable, never a zero gap. All arithmetic is in Float64.
#
# Relative gap (measure :relative_gap), as in ProximalSolvers' benchmark:
# (P - D)/max(|P|, floatmin) for the primal value P at the returned point and a
# dual (lower) bound D. With the rounding bound tol = 16 γₘ (|P| + |D| + scale),
# γₘ = m eps/(1 - m eps) for sums of m terms and scale the sum of the
# magnitudes of their terms, P - D < -tol is a weak-duality violation, which
# makes the case uncertifiable, and -tol ≤ P - D < 0 counts as a zero gap.
#
# Residuals (:kkt_residual, :fixed_point_residual) are relative to
# max(1, ‖∇f(0)‖∞), ∇f(0) the gradient of the loss at β = 0 with the intercept
# at its optimum for β = 0 (zero without an intercept), in the coordinates in
# which the penalty acts (by blocks, the largest Euclidean norm of a block).
# A case whose penalty leaves some coefficients free (a zero loading or group
# weight, or λ = 0) has a dual with equality constraints that no scaled point
# meets, so it reports the KKT residual of the whole problem instead of a gap.

using LinearAlgebra, SparseArrays, Statistics
import HiGHS

const CERTIFICATE_MEASURES = (:relative_gap, :kkt_residual, :fixed_point_residual, :objective_excess,
                              :coefficient_error, :feasibility, :consistency)

cert_ok(value, measure; note = "") = (; value = Float64(value), measure, status = :ok, note = String(note))
uncertifiable(note, measure = :relative_gap) = (; value = NaN, measure, status = :uncertifiable, note = String(note))

cert_finite(::Nothing) = true
cert_finite(x::Number) = isfinite(x)
cert_finite(x::SparseMatrixCSC) = all(isfinite, nonzeros(x))
cert_finite(x::AbstractArray) = all(isfinite, x)
cert_finite(xs...) = all(cert_finite, xs)

f64(::Nothing) = nothing
f64(x::Number) = Float64(x)
f64(x::SparseMatrixCSC) = SparseMatrixCSC{Float64, Int}(x)
f64(x::AbstractArray) = Array{Float64}(x)

cert_soft(x, c) = sign(x) * max(abs(x) - c, 0.0)
cert_maxabs(x) = maximum(abs, x; init = 0.0)
colmeans(X, w) = (transpose(X) * w) ./ sum(w)

function relative_gap(P, D; scale = 0.0, terms::Integer = 1)
  cert_finite(P, D, scale) && scale >= 0 || return uncertifiable("nonfinite primal or dual value")
  γ = terms * eps() / (1 - terms * eps())
  tol = 16γ * (abs(P) + abs(D) + scale)
  P - D < -tol && return uncertifiable("weak duality violation: P - D = $(P - D) < -$tol")
  cert_ok(max(P - D, 0.0) / max(abs(P), floatmin(Float64)), :relative_gap)
end

# a residual relative to max(1, g0), g0 = ‖∇f(0)‖∞
relative_residual(value, g0, measure) =
  cert_finite(value, g0) ? cert_ok(value / max(1.0, g0), measure) : uncertifiable("nonfinite residual", measure)

# the largest t ≤ 1 with |t zⱼ| ≤ cⱼ for every j (c ≥ 0)
function box_scale(z, c)
  t = 1.0
  for j in eachindex(z, c)
    abs(z[j]) > 0 && (t = min(t, c[j] / abs(z[j])))
  end
  t
end

# The inputs of a regression certificate in Float64, with the observation
# weights w (ones for nothing), their sum W and the offset o (zeros for
# nothing); or a note saying why there is no certificate. X = nothing is the
# identity design of a signal.
function cert_regression(X, y, β0, β; weights = nothing, offset = nothing)
  n = length(y)
  X === nothing || size(X) == (n, length(β)) || return "the dimensions of X, y and β disagree"
  weights === nothing || length(weights) == n || return "weights and y have different lengths"
  offset === nothing || length(offset) == n || return "offset and y have different lengths"
  cert_finite(X, y, β0, β, weights, offset) || return "nonfinite input"
  w = weights === nothing ? ones(n) : f64(weights)
  all(>=(0), w) && sum(w) > 0 || return "the weights must be nonnegative with a positive sum"
  (; X = f64(X), y = f64(y), β0 = Float64(β0), β = f64(β), w, W = sum(w),
     o = offset === nothing ? zeros(n) : f64(offset), n, p = length(β))
end

cert_levels(λ, ω, p) = length(ω) == p && cert_finite(λ, ω) && λ >= 0 && all(>=(0), ω)

# The distance of -g from the subdifferential of the separable penalty
# c1ⱼ|βⱼ| + c2ⱼβⱼ²/2 at β, the largest over the coordinates
function l1_kkt(g, β, c1, c2)
  maximum((β[j] == 0 ? max(abs(g[j]) - c1[j], 0.0) : abs(g[j] + c1[j] * sign(β[j]) + c2[j] * β[j])
           for j in eachindex(g, β)); init = 0.0)
end

# the distance of -g from the subdifferential of c‖b‖₂ at b
block_kkt(g, b, c) = (nb = norm(b); nb == 0 ? max(norm(g) - c, 0.0) : norm(g .+ (c / nb) .* b))

# the root of a continuous nonincreasing φ that changes sign, bracketed from
# start by doubling steps and bisected down to adjacent floating-point numbers:
# the end with the smaller |φ|, or nothing when no sign change is found
function decreasing_root(φ, start)
  lo = hi = start
  φlo = φhi = φ(start)
  φlo == 0 && return start
  step = 1.0
  for _ in 1:1100
    if φlo > 0 && φhi > 0
      lo, φlo = hi, φhi
      hi = lo + step
      φhi = φ(hi)
    elseif φlo < 0 && φhi < 0
      hi, φhi = lo, φlo
      lo = hi - step
      φlo = φ(lo)
    else
      break
    end
    step *= 2
  end
  φlo >= 0 && φhi <= 0 || return nothing
  while true
    mid = lo + (hi - lo) / 2
    lo < mid < hi || break
    φm = φ(mid)
    φm == 0 && return mid
    φm > 0 ? (lo = mid; φlo = φm) : (hi = mid; φhi = φm)
  end
  abs(φlo) <= abs(φhi) ? lo : hi
end

"""
    column_loadings(X; intercept, weights = nothing)

The population standard deviations `sqrt(∑ᵢ wᵢ (Xᵢⱼ - x̄ⱼ)² / ∑ᵢ wᵢ)` of the
columns, centered at their weighted means only with an intercept (`wᵢ = 1`
without weights): the loadings of `standardize_x` in `src/lasso.jl`.
`column_scales` replaces a zero loading by one, as the scales of a penalty
applied to `s .* β` do.
"""
function column_loadings(X::AbstractMatrix; intercept::Bool, weights = nothing)
  A = f64(X)
  n, p = size(A)
  w = weights === nothing ? ones(n) : f64(weights)
  c = intercept ? colmeans(A, w) : zeros(p)
  [sqrt(sum(w .* (Vector(A[:, j]) .- c[j]) .^ 2) / sum(w)) for j in 1:p]
end

function column_scales(X::AbstractMatrix; intercept::Bool, weights = nothing)
  s = column_loadings(X; intercept, weights)
  s[iszero.(s)] .= 1
  s
end

######################################################################
#   Least squares with ℓ1 penalties
######################################################################

# ∑wᵢrᵢ²/(2∑wᵢ) + λ∑ωⱼ|βⱼ|, r = y - o - β₀ - Xβ. The dual point is
# θ = t w∘r_c/∑w, r_c the residuals centered at their weighted mean with an
# intercept (so that 1'θ = 0 and X'θ = X̃'θ), with t = min(1, minⱼ λωⱼ/|xⱼ'θ|);
# the dual value is t ỹ_c'θ - t²∑wᵢr_cᵢ²/(2∑w), ỹ_c = y - o centered likewise.
function ls_l1_parts(X, y, β0, β, λ, ω; intercept::Bool, weights = nothing, offset = nothing)
  d = cert_regression(X, y, β0, β; weights, offset)
  d isa String && return d
  cert_levels(λ, ω, d.p) || return "λ and the loadings must be finite and nonnegative, one loading per column"
  (; X, β, w, W, n, p) = d
  ỹ = d.y .- d.o
  fit = X * β
  r = ỹ .- d.β0 .- fit
  c1 = λ .* f64(ω)
  penalty = dot(c1, abs.(β))
  P = dot(w, r .^ 2) / (2W) + penalty
  yc = intercept ? ỹ .- dot(w, ỹ) / W : ỹ
  if any(iszero, c1)
    g = -(transpose(X) * (w .* r)) ./ W
    kkt = max(intercept ? abs(dot(w, r)) / W : 0.0, l1_kkt(g, β, c1, zeros(p)))
    return (; P, kkt, g0 = cert_maxabs(transpose(X) * (w .* yc)) / W)
  end
  rc = intercept ? r .- dot(w, r) / W : r
  θ = w .* rc ./ W
  t = box_scale(transpose(X) * θ, c1)
  D = t * dot(yc, θ) - t^2 * dot(w, rc .^ 2) / (2W)
  scale = dot(w, (abs.(ỹ) .+ abs(d.β0) .+ abs.(fit)) .^ 2) / (2W) + penalty +
          t * dot(abs.(yc), abs.(θ)) + t^2 * dot(w, rc .^ 2) / (2W)
  (; P, D, θ = t .* θ, t, scale, terms = max(n, p), kkt = nothing)
end

function ls_l1_gap(X, y, β0, β, λ, ω; intercept::Bool, weights = nothing, offset = nothing)
  a = ls_l1_parts(X, y, β0, β, λ, ω; intercept, weights, offset)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

# An exact fit, ‖r_c‖_w at most SQRT_FIT_FLOOR ‖ỹ_c‖_w, leaves the direction
# of the square-root lasso's dual point to rounding.
const SQRT_FIT_FLOOR = 1e-10

# ‖r‖_w + λ∑ωⱼ|βⱼ|, ‖r‖_w = sqrt(∑wᵢrᵢ²), r = y - β₀ - Xβ. The dual point is
# θ = t w∘r_c/‖r_c‖_w, so that ∑θᵢ²/wᵢ ≤ 1 and 1'θ = 0, with t as for the lasso;
# the dual value is ỹ_c'θ.
function sqrt_l1_parts(X, y, β0, β, λ, ω; intercept::Bool, weights = nothing)
  d = cert_regression(X, y, β0, β; weights)
  d isa String && return d
  cert_levels(λ, ω, d.p) || return "λ and the loadings must be finite and nonnegative, one loading per column"
  (; X, y, β, w, W, n, p) = d
  fit = X * β
  r = y .- d.β0 .- fit
  c1 = λ .* f64(ω)
  penalty = dot(c1, abs.(β))
  P = sqrt(dot(w, r .^ 2)) + penalty
  yc = intercept ? y .- dot(w, y) / W : y
  rc = intercept ? r .- dot(w, r) / W : r
  nr, ny = sqrt(dot(w, rc .^ 2)), sqrt(dot(w, yc .^ 2))
  nr > SQRT_FIT_FLOOR * ny || return "an exact fit: ‖r‖ ≤ $SQRT_FIT_FLOOR ‖y‖"
  if any(iszero, c1)
    g = -(transpose(X) * (w .* r)) ./ sqrt(dot(w, r .^ 2))
    kkt = max(intercept ? abs(dot(w, r)) / sqrt(dot(w, r .^ 2)) : 0.0, l1_kkt(g, β, c1, zeros(p)))
    return (; P, kkt, g0 = cert_maxabs(transpose(X) * (w .* yc)) / ny)
  end
  θ = w .* rc ./ nr
  t = box_scale(transpose(X) * θ, c1)
  scale = sqrt(dot(w, (abs.(y) .+ abs(d.β0) .+ abs.(fit)) .^ 2)) + penalty + t * dot(abs.(yc), abs.(θ))
  (; P, D = t * dot(yc, θ), θ = t .* θ, t, scale, terms = max(n, p), kkt = nothing)
end

function sqrt_l1_gap(X, y, β0, β, λ, ω; intercept::Bool, weights = nothing)
  a = sqrt_l1_parts(X, y, β0, β, λ, ω; intercept, weights)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

# Least-squares stationarity on the support S of β and the intercept column,
# max(|1'r|, ‖X_S'r‖∞) relative to ‖[1 X_S]‖_F ‖y‖ (the Frobenius norm bounds
# the spectral one), r = y - β₀ - Xβ: the refit of a selected support.
function ls_refit_stationarity(X, y, β0, β; intercept::Bool)
  d = cert_regression(X, y, β0, β)
  d isa String && return uncertifiable(d, :kkt_residual)
  S = findall(!iszero, d.β)
  r = d.y .- d.β0 .- d.X * d.β
  XS = d.X[:, S]
  num = max(intercept ? abs(sum(r)) : 0.0, cert_maxabs(transpose(XS) * r))
  den = sqrt((intercept ? d.n : 0) + sum(abs2, XS)) * norm(d.y)
  cert_ok(num / (den > 0 ? den : 1.0), :kkt_residual)
end

######################################################################
#   Generalized linear models
######################################################################

cert_sigmoid(t) = t >= 0 ? inv(1 + exp(-t)) : exp(t) / (1 + exp(t))
cert_mean(family, η) = family === :logistic ? cert_sigmoid(η) : exp(η)
cert_xlogx(p) = p > 0 ? p * log(p) : 0.0

# ℓ(η, y): log(1 + exp(η)) - yη (logistic) and exp(η) - yη (Poisson, the
# negative log-likelihood up to a term in y), ProximalBase's losses
cert_loss(family, η, y) = family === :logistic ?
  (η > 0 ? (1 - y) * η + log1p(exp(-η)) : log1p(exp(η)) - y * η) : exp(η) - y * η
cert_loss_size(family, η, y) = family === :logistic ? abs(η) * (1 + abs(y)) + log(2.0) : exp(η) + abs(y * η)
# ∂ℓ/∂η, with the squared loss (η - y)²/2 as :gaussian
cert_dloss(family, η, y) = family === :gaussian ? η - y : cert_mean(family, η) - y
# the conjugate ℓ*(v) = sup_η vη - ℓ(η, y) at the mean parameter q = v + y:
# the negative entropy q log q + (1 - q) log(1 - q) on [0, 1], and q log q - q
cert_conjugate(family, q) = family === :logistic ? cert_xlogx(q) + cert_xlogx(1 - q) : cert_xlogx(q) - q

# the intercept c that solves ∑wᵢ(yᵢ - μ(c + η₀ᵢ)) = 0, or nothing when the
# responses are all at a bound and no finite c does
function glm_intercept(family, η0, y, w, start)
  sy = dot(w, y)
  sy > 0 && (family === :poisson || dot(w, 1 .- y) > 0) || return nothing
  φ(c) = sy - sum((w[i] == 0 ? 0.0 : w[i] * cert_mean(family, c + η0[i]) for i in eachindex(w)); init = 0.0)
  decreasing_root(φ, isfinite(start) ? start : 0.0)
end

cert_glm_domain(family, y) = family === :logistic ? all(v -> 0 <= v <= 1, y) : all(>=(0), y)

# ∑wᵢℓ(ηᵢ, yᵢ)/∑w + λ∑ⱼ(αωⱼ|βⱼ| + (1 - α)ωⱼ²βⱼ²/2), η = β₀ + Xβ + o: the
# objective of src/glm.jl. ω are the scales on which the penalty acts. For
# α = 1 that is λ∑ⱼωⱼ|βⱼ| with ω the loadings of standardize_x (a zero loading
# leaves a coefficient free) or the given penalty weights; for α < 1 the
# elastic net λ(α‖s∘β‖₁ + (1 - α)‖s∘β‖²/2) of the standardized coefficients
# s∘β, with ω = s the scales of standardize_x (a zero spread replaced by one)
# or ones.
#
# The dual: with u = w∘(y - μ)/∑w and 1'u = 0 (intercept),
# D(u) = -u'o - ∑(wᵢ/∑w) ℓ*(-∑w uᵢ/wᵢ) - G*(X'u). The point comes from the
# returned β with the intercept re-optimized (a monotone root find), so that
# 1'u = 0 holds; for α = 1 it is scaled by t = min(1, minⱼ λωⱼ/|xⱼ'u|), which
# keeps the mean parameter (1 - t)y + tμ in the domain of ℓ*; for α < 1 the
# conjugate G*(z) = ∑ⱼ(|zⱼ| - λαωⱼ)₊²/(2λ(1 - α)ωⱼ²) of the elastic net needs
# no scaling. The primal is still evaluated at the returned (β₀, β).
function glm_parts(family::Symbol, X, y, β0, β, λ, ω; alpha = 1, intercept::Bool, weights = nothing,
                   offset = nothing)
  family in (:logistic, :poisson) || return "family must be :logistic or :poisson"
  d = cert_regression(X, y, β0, β; weights, offset)
  d isa String && return d
  cert_glm_domain(family, d.y) || return "responses outside the domain of the $family loss"
  cert_levels(λ, ω, d.p) || return "λ and ω must be finite and nonnegative, one per column"
  α = Float64(alpha)
  0 < α <= 1 || return "alpha must be in (0, 1]"
  (; X, y, β, w, W, o, n, p) = d
  ωf = f64(ω)
  c1, c2 = λ * α .* ωf, λ * (1 - α) .* ωf .^ 2
  η0 = X * β .+ o
  η = d.β0 .+ η0
  penalty = dot(c1, abs.(β)) + dot(c2, β .^ 2) / 2
  P = sum(w[i] == 0 ? 0.0 : w[i] * cert_loss(family, η[i], y[i]) for i in 1:n) / W + penalty
  isfinite(P) || return "nonfinite primal value"
  if λ == 0 || any(iszero, ωf)
    gη = w .* (cert_mean.(family, η) .- y) ./ W
    kkt = max(intercept ? abs(sum(gη)) : 0.0, l1_kkt(transpose(X) * gη, β, c1, c2))
    c0 = intercept ? glm_intercept(family, o, y, w, 0.0) : 0.0
    c0 === nothing && return "the intercept has no finite optimum: the responses are all at a bound"
    g0 = cert_maxabs(transpose(X) * (w .* (cert_mean.(family, c0 .+ o) .- y))) / W
    return (; P, kkt, g0)
  end
  c = intercept ? glm_intercept(family, η0, y, w, d.β0) : 0.0
  c === nothing && return "the intercept has no finite optimum: the responses are all at a bound"
  μ = cert_mean.(family, c .+ η0)
  u = w .* (y .- μ) ./ W
  z = transpose(X) * u
  t, Gstar = α == 1 ? (box_scale(z, c1), 0.0) :
    (1.0, sum(max(abs(z[j]) - c1[j], 0.0)^2 / (2c2[j]) for j in 1:p))
  q = y .+ t .* (μ .- y)
  q = family === :logistic ? clamp.(q, 0.0, 1.0) : max.(q, 0.0)
  conj = [w[i] == 0 ? 0.0 : w[i] * cert_conjugate(family, q[i]) / W for i in 1:n]
  D = -t * dot(u, o) - sum(conj) - Gstar
  # the last term bounds the effect of rounding in 1'u = 0
  scale = sum(w[i] * cert_loss_size(family, η[i], y[i]) for i in 1:n) / W + penalty + sum(abs, conj) +
          t * dot(abs.(u), abs.(o)) + Gstar + t * sum(abs, u) * cert_maxabs(c .+ η0)
  (; P, D, u = t .* u, t, intercept = c, scale, terms = max(n, p), kkt = nothing)
end

function glm_l1_gap(family::Symbol, X, y, β0, β, λ, ω; alpha = 1, intercept::Bool, weights = nothing,
                    offset = nothing)
  a = glm_parts(family, X, y, β0, β, λ, ω; alpha, intercept, weights, offset)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

# The firm and SCAD thresholding rules, the proximal maps of γ MCP(λ, a) and
# γ SCAD(λ, a), single valued for γ < a and γ < a - 1
function cert_mcp_prox(v, λ, a, γ)
  t = abs(v)
  t <= γ * λ && return 0.0
  t > a * λ && return v
  sign(v) * (t - γ * λ) / (1 - γ / a)
end

function cert_scad_prox(v, λ, a, γ)
  t = abs(v)
  t <= (1 + γ) * λ && return cert_soft(v, γ * λ)
  t > a * λ && return v
  sign(v) * ((a - 1) * t - γ * a * λ) / (a - 1 - γ)
end

# the largest eigenvalue of Ã'diag(v)Ã, Ã = [1 X./ω'] with an intercept and
# X./ω' without
function curvature_bound(X, ω, v, intercept::Bool)
  A = Matrix(X) ./ transpose(ω)
  intercept && (A = hcat(ones(size(A, 1)), A))
  S = sqrt.(v) .* A
  eigmax(Symmetric(size(S, 1) >= size(S, 2) ? transpose(S) * S : S * transpose(S)))
end

"""
    nonconvex_stationarity(family, penalty, X, y, β0, β, λ, a, ω; intercept, weights = nothing, offset = nothing)

The proximal-gradient fixed-point residual of `∑wᵢℓ(ηᵢ, yᵢ)/∑w + ∑ⱼ pen(ωⱼβⱼ)`,
`pen` MCP or SCAD at level `λ` and parameter `a`, in the coordinates `b = ω∘β`
in which the penalty acts (`ω` the scales of `standardize_x`, or ones):
`max(‖b - prox_{γ pen}(b - γ∇_b f)‖∞ / γ, |∂f/∂β₀|)`. The step is `γ = 1/L`
with `L` the largest eigenvalue of `Ã'diag(v)Ã`, `Ã = [1 X./ω']`, where
`v = w/∑w` for least squares, `w/(4∑w)` for logistic regression (a global
Lipschitz constant of the gradient) and `w∘exp(η)/∑w` for Poisson regression
(the curvature at the returned point); it is reduced to `a/2` (MCP) or
`(a - 1)/2` (SCAD) when larger, so that the proximal map is single valued. It
is zero exactly at stationary points; it says nothing about global optimality.
"""
function nonconvex_stationarity(family::Symbol, penalty::Symbol, X, y, β0, β, λ, a, ω; intercept::Bool,
                                weights = nothing, offset = nothing)
  m = :fixed_point_residual
  family in (:gaussian, :logistic, :poisson) ||
    return uncertifiable("family must be :gaussian, :logistic or :poisson", m)
  penalty in (:mcp, :scad) || return uncertifiable("penalty must be :mcp or :scad", m)
  d = cert_regression(X, y, β0, β; weights, offset)
  d isa String && return uncertifiable(d, m)
  family === :gaussian || cert_glm_domain(family, d.y) ||
    return uncertifiable("responses outside the domain of the loss", m)
  cert_finite(λ, a, ω) && λ >= 0 && length(ω) == d.p && all(>(0), ω) ||
    return uncertifiable("λ must be nonnegative and ω positive, one per column", m)
  (penalty === :mcp ? a > 0 : a > 2) || return uncertifiable("a must be positive (MCP) or exceed two (SCAD)", m)
  (; X, y, β, w, W, o, p) = d
  ωf = f64(ω)
  b = ωf .* β
  η = d.β0 .+ X * β .+ o
  gη = w .* cert_dloss.(family, η, y) ./ W
  gb = (transpose(X) * gη) ./ ωf
  v = family === :gaussian ? w ./ W : family === :logistic ? w ./ (4W) : w .* exp.(η) ./ W
  L = curvature_bound(X, ωf, v, intercept)
  γ = min(L > 0 ? 1 / L : Inf, (penalty === :mcp ? a : a - 1) / 2)
  prox = penalty === :mcp ? cert_mcp_prox : cert_scad_prox
  z = [prox(b[j] - γ * gb[j], λ, a, γ) for j in 1:p]
  residual = max(cert_maxabs(b .- z) / γ, intercept ? abs(sum(gη)) : 0.0)
  c0 = !intercept ? 0.0 : family === :gaussian ? dot(w, y .- o) / W : glm_intercept(family, o, y, w, 0.0)
  c0 === nothing && return uncertifiable("the intercept has no finite optimum: the responses are all at a bound", m)
  g0 = cert_maxabs((transpose(X) * (w .* cert_dloss.(family, c0 .+ o, y) ./ W)) ./ ωf)
  relative_residual(residual, g0, m)
end

######################################################################
#   Group penalties
######################################################################

function cert_partition(groups, p)
  seen = falses(p)
  for G in groups, j in G
    1 <= j <= p && !seen[j] || return false
    seen[j] = true
  end
  all(seen)
end

# An orthonormal basis Q of the span of the centered columns X̃_G: the left
# singular vectors whose singular values exceed max(n, |G|) eps σ₁, the rank
# rule of src/group.jl. Directions at or below it are treated as absent, as in
# an exactly rank-deficient group.
function group_basis(XcG)
  isempty(XcG) && return zeros(size(XcG, 1), 0)
  F = svd(XcG)
  (isempty(F.S) || F.S[1] == 0) && return zeros(size(XcG, 1), 0)
  F.U[:, F.S .> max(size(XcG)...) * eps() * F.S[1]]
end

# ‖y - β₀ - Xβ‖²/(2n) + λ∑ₖwₖPₖ(β), Pₖ = ‖X̃_Gβ_G‖₂/√n with orthonormal (X̃
# centered only with an intercept; src/group.jl) and ‖β_G‖₂ otherwise.
# The polar of λwₖ‖X̃_G·‖/√n: X̃_G'θ = λwₖX̃_G'v/√n for some ‖v‖ ≤ 1, whose
# smallest v is the multiple of the projection Qθ of θ on span(X̃_G); so
# θ is feasible when ‖Z_G'θ‖ ≤ λwₖ, Z_G = √n Q, also for rank-deficient groups.
# The dual point θ = t r_c/n and the dual value t ỹ_c'θ - t²‖r_c‖²/(2n) are
# those of the lasso.
function group_parts(X, y, β0, β, λ, groups, groupweights; intercept::Bool, orthonormal::Bool)
  d = cert_regression(X, y, β0, β)
  d isa String && return d
  cert_partition(groups, d.p) || return "the groups must partition the columns"
  length(groupweights) == length(groups) && cert_finite(λ, groupweights) && λ >= 0 && all(>=(0), groupweights) ||
    return "λ and the group weights must be finite and nonnegative, one weight per group"
  (; X, y, β, n, p) = d
  wg = f64(groupweights)
  fit = X * β
  r = y .- d.β0 .- fit
  xbar = intercept ? colmeans(X, ones(n)) : zeros(p)
  blocks = map(groups) do G
    XcG = Matrix(X[:, G]) .- transpose(xbar[G])
    orthonormal ? (; XcG, Q = group_basis(XcG)) : (; XcG, Q = nothing)
  end
  sizes = [orthonormal ? norm(blocks[k].XcG * β[G]) / sqrt(n) : norm(β[G]) for (k, G) in enumerate(groups)]
  penalty = λ * dot(wg, sizes)
  P = sum(abs2, r) / (2n) + penalty
  yc = intercept ? y .- mean(y) : y
  # the coordinates of the fit (γ_G = Q'X̃_Gβ_G/√n with orthonormal) and the
  # gradient of the loss in them at the residuals v
  coords(k, G) = orthonormal ? transpose(blocks[k].Q) * (blocks[k].XcG * β[G]) ./ sqrt(n) : β[G]
  gradient(k, G, v) = orthonormal ? -(transpose(blocks[k].Q) * v) ./ sqrt(n) : -(transpose(X[:, G]) * v) ./ n
  if λ == 0 || any(iszero, wg)
    kkt = maximum(block_kkt(gradient(k, G, r), coords(k, G), λ * wg[k]) for (k, G) in enumerate(groups))
    kkt = max(kkt, intercept ? abs(sum(r)) / n : 0.0)
    return (; P, kkt, g0 = maximum(norm(gradient(k, G, yc)) for (k, G) in enumerate(groups)))
  end
  rc = intercept ? r .- mean(r) : r
  θ = rc ./ n
  t = 1.0
  for (k, G) in enumerate(groups)
    a = orthonormal ? sqrt(n) * norm(transpose(blocks[k].Q) * θ) : norm(transpose(X[:, G]) * θ)
    a > 0 && (t = min(t, λ * wg[k] / a))
  end
  D = t * dot(yc, θ) - t^2 * sum(abs2, rc) / (2n)
  scale = sum((abs.(y) .+ abs(d.β0) .+ abs.(fit)) .^ 2) / (2n) + penalty + t * dot(abs.(yc), abs.(θ)) +
          t^2 * sum(abs2, rc) / (2n)
  (; P, D, θ = t .* θ, t, scale, terms = max(n, p), kkt = nothing, blocks)
end

function group_gap(X, y, β0, β, λ, groups, groupweights; intercept::Bool, orthonormal::Bool)
  a = group_parts(X, y, β0, β, λ, groups, groupweights; intercept, orthonormal)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

# the distance of -g from the subdifferential of c1‖b‖₁ + c2‖b‖₂ at b; at b = 0
# the subdifferential is the sum of a box and a ball, at distance
# max(‖S_c1(g)‖ - c2, 0)
function sparse_group_kkt(g, b, c1, c2)
  nb = norm(b)
  nb == 0 && return max(norm(cert_soft.(g, c1)) - c2, 0.0)
  v = -g .- (c2 / nb) .* b
  norm([b[j] == 0 ? max(abs(v[j]) - c1, 0.0) : abs(v[j] - c1 * sign(b[j])) for j in eachindex(b)])
end

# the largest t ≤ 1 with ‖S_c1(t g_G)‖₂ ≤ c2ₖ for every group, by bisection; the
# feasible set is an interval [0, t*], as ‖S_c1(t g)‖ is nondecreasing in t
function sparse_group_scale(g, groups, c1, c2)
  feasible(t) = all(norm(cert_soft.(t .* g[G], c1)) <= c2[k] for (k, G) in enumerate(groups))
  feasible(1.0) && return 1.0
  lo, hi = 0.0, 1.0
  for _ in 1:100
    mid = (lo + hi) / 2
    feasible(mid) ? (lo = mid) : (hi = mid)
  end
  lo
end

# ‖y - β₀ - Xβ‖²/(2n) + λ(α∑ⱼ|β̃ⱼ| + (1 - α)∑ₖwₖ‖β̃_Gₖ‖₂), β̃ = s∘β with s the
# scales of standardize_x (centered with an intercept, a zero spread replaced
# by one) and β̃ = β otherwise (src/sparsegroup.jl). The polar constraint is
# ‖S_{λα}(g_G)‖₂ ≤ λ(1 - α)wₖ on g = (X̃'θ)./s, met by t θ for the largest such
# t; the dual value is that of the lasso.
function sparse_group_parts(X, y, β0, β, λ, α, groups, groupweights; intercept::Bool, standardize_x::Bool)
  d = cert_regression(X, y, β0, β)
  d isa String && return d
  cert_partition(groups, d.p) || return "the groups must partition the columns"
  length(groupweights) == length(groups) && cert_finite(λ, α, groupweights) && λ >= 0 && 0 <= α <= 1 &&
    all(>=(0), groupweights) || return "λ, α and the group weights must be finite, nonnegative, α ≤ 1"
  (; X, y, β, n, p) = d
  wg = f64(groupweights)
  s = standardize_x ? column_scales(X; intercept) : ones(p)
  b = s .* β
  c1, c2 = λ * α, λ * (1 - α) .* wg
  penalty = c1 * sum(abs, b) + sum(c2[k] * norm(b[G]) for (k, G) in enumerate(groups))
  fit = X * β
  r = y .- d.β0 .- fit
  P = sum(abs2, r) / (2n) + penalty
  yc = intercept ? y .- mean(y) : y
  if λ == 0 || (α == 0 && any(iszero, wg))
    g = -(transpose(X) * r) ./ (n .* s)
    kkt = maximum(sparse_group_kkt(g[G], b[G], c1, c2[k]) for (k, G) in enumerate(groups))
    kkt = max(kkt, intercept ? abs(sum(r)) / n : 0.0)
    g0 = (transpose(X) * yc) ./ (n .* s)
    return (; P, kkt, g0 = maximum(norm(cert_soft.(g0[G], c1)) for G in groups))
  end
  rc = intercept ? r .- mean(r) : r
  θ = rc ./ n
  t = sparse_group_scale((transpose(X) * θ) ./ s, groups, c1, c2)
  D = t * dot(yc, θ) - t^2 * sum(abs2, rc) / (2n)
  scale = sum((abs.(y) .+ abs(d.β0) .+ abs.(fit)) .^ 2) / (2n) + penalty + t * dot(abs.(yc), abs.(θ)) +
          t^2 * sum(abs2, rc) / (2n)
  (; P, D, θ = t .* θ, t, s, scale, terms = max(n, p), kkt = nothing)
end

function sparse_group_gap(X, y, β0, β, λ, α, groups, groupweights; intercept::Bool, standardize_x::Bool)
  a = sparse_group_parts(X, y, β0, β, λ, α, groups, groupweights; intercept, standardize_x)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

######################################################################
#   Multiple responses and low rank
######################################################################

# the inputs of a multiple-response certificate in Float64, or a note
function cert_multi(X, Y, β0, B)
  n, p = size(X)
  q = size(Y, 2)
  size(Y, 1) == n && length(β0) == q && size(B) == (p, q) || return "the dimensions of X, Y, β₀ and B disagree"
  cert_finite(X, Y, β0, B) || return "nonfinite input"
  X, Y, b0, B = f64(X), f64(Y), f64(β0), f64(B)
  XB = X * B
  (; X, Y, b0, B, XB, R = Y .- transpose(b0) .- XB, n, p, q)
end

# ‖Y - 1β₀' - XB‖²_F/(2n) + λ∑ₖωₖ‖B[k, :]‖₂. Θ = t R_c/n, R_c the column-
# centered residuals with an intercept, t = min(1, minₖ λωₖ/‖(X'Θ)[k, :]‖); the
# dual value is t⟨Y_c, Θ⟩ - t²‖R_c‖²/(2n).
function multi_parts(X, Y, β0, B, λ, ω; intercept::Bool)
  d = cert_multi(X, Y, β0, B)
  d isa String && return d
  cert_levels(λ, ω, d.p) || return "λ and the loadings must be finite and nonnegative, one loading per row"
  (; X, Y, B, R, n, p) = d
  c = λ .* f64(ω)
  penalty = sum(c[k] * norm(B[k, :]) for k in 1:p)
  P = sum(abs2, R) / (2n) + penalty
  Yc = intercept ? Y .- mean(Y; dims = 1) : Y
  if any(iszero, c)
    G = -(transpose(X) * R) ./ n
    kkt = max(maximum(block_kkt(G[k, :], B[k, :], c[k]) for k in 1:p),
              intercept ? norm(sum(R; dims = 1)) / n : 0.0)
    return (; P, kkt, g0 = maximum(norm, eachrow(transpose(X) * Yc ./ n)))
  end
  Rc = intercept ? R .- mean(R; dims = 1) : R
  Θ = Rc ./ n
  Z = transpose(X) * Θ
  t = 1.0
  for k in 1:p
    a = norm(Z[k, :])
    a > 0 && (t = min(t, c[k] / a))
  end
  D = t * dot(Yc, Θ) - t^2 * sum(abs2, Rc) / (2n)
  scale = sum((abs.(Y) .+ abs.(transpose(d.b0)) .+ abs.(d.XB)) .^ 2) / (2n) + penalty + t * dot(abs.(Yc), abs.(Θ)) +
          t^2 * sum(abs2, Rc) / (2n)
  (; P, D, Θ = t .* Θ, t, scale, terms = max(n, p, d.q), kkt = nothing)
end

function multi_gap(X, Y, β0, B, λ, ω; intercept::Bool)
  a = multi_parts(X, Y, β0, B, λ, ω; intercept)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

"""
    rrr_excess(X, Y, β0, B, rank; intercept)

The relative excess `(L - L*)/max(L*, eps ‖Ỹ‖²)` of the loss
`L = ‖Y - 1β₀' - XB‖²_F` at the returned point (src/lowrank.jl; the scaling
cancels) over the rank-`rank` optimum `L* = ‖(I - UU')Ỹ‖² + ∑_{i>rank} σᵢ²`,
`U` an orthonormal basis of the column space of the centered `X̃` (singular
values above `max(n, p) eps σ₁`, as `reduced_rank`'s default `rtol`) and `σ`
the singular values of `U'Ỹ`, the fitted values of least squares. `B` must
have numerical rank at most `rank` (singular values above
`8 max(p, q) eps(T) σ₁(B)`, `T` the element type of `B`), else the case is
uncertifiable. An excess below `-tol` (the rounding bound of the relative gap
with the precision `eps(T)` of the inputs, `T` the least precise of their
element types: a Float32 `B` of numerical rank `rank` keeps
singular values at the Float32 rounding level, which can lower the loss below
`L*` by that order) is uncertifiable too. The floor `eps ‖Ỹ‖²` of the
denominator, the rounding level of the loss, keeps the measure finite when `Y`
can be fitted exactly.
"""
function rrr_excess(X, Y, β0, B, rank; intercept::Bool)
  m = :objective_excess
  d = cert_multi(X, Y, β0, B)
  d isa String && return uncertifiable(d, m)
  rank isa Integer && rank >= 0 || return uncertifiable("the rank bound must be a nonnegative integer", m)
  precision = maximum(v -> eps(float(eltype(v))), (X, Y, β0, B))
  (; X, Y, n, p, q) = d
  σB = svdvals(d.B)
  rB = isempty(σB) || σB[1] == 0 ? 0 : count(>(8 * max(p, q) * eps(float(eltype(B))) * σB[1]), σB)
  rB <= rank || return uncertifiable("B has numerical rank $rB, above the bound $rank", m)
  Xc = Matrix(X)
  intercept && (Xc .-= mean(Xc; dims = 1))
  Yc = intercept ? Y .- mean(Y; dims = 1) : Y
  F = svd(Xc)
  U = isempty(F.S) || F.S[1] == 0 ? zeros(n, 0) : F.U[:, F.S .> max(n, p) * eps() * F.S[1]]
  M = transpose(U) * Yc
  σ = svdvals(M)
  L = sum(abs2, d.R)
  Lstar = sum(abs2, Yc .- U * M) + sum(abs2, σ[min(rank, length(σ))+1:end])
  terms = max(n, p, q)
  γ = terms * precision / (1 - terms * precision)
  tol = 16γ * (L + Lstar + sum(abs2, abs.(Y) .+ abs.(transpose(d.b0)) .+ abs.(d.XB)) + sum(abs2, Yc))
  L - Lstar < -tol && return uncertifiable("loss below the rank-$rank optimum: L - L* = $(L - Lstar)", m)
  cert_ok(max(L - Lstar, 0.0) / max(Lstar, eps() * sum(abs2, Yc), floatmin(Float64)), m)
end

# ‖Y - 1β₀' - XB‖²_F/(2n) + λ‖B‖_*, whose polar is the spectral norm:
# Θ = t R_c/n with t = min(1, λ/‖X'Θ‖₂); the dual value is that of multi_parts.
function nuclear_parts(X, Y, β0, B, λ; intercept::Bool)
  d = cert_multi(X, Y, β0, B)
  d isa String && return d
  cert_finite(λ) && λ >= 0 || return "λ must be finite and nonnegative"
  (; X, Y, R, n, p, q) = d
  penalty = λ * sum(svdvals(d.B))
  P = sum(abs2, R) / (2n) + penalty
  Yc = intercept ? Y .- mean(Y; dims = 1) : Y
  if λ == 0
    kkt = max(cert_maxabs(transpose(X) * R) / n, intercept ? cert_maxabs(sum(R; dims = 1)) / n : 0.0)
    return (; P, kkt, g0 = cert_maxabs(transpose(X) * Yc) / n)
  end
  Rc = intercept ? R .- mean(R; dims = 1) : R
  Θ = Rc ./ n
  a = opnorm(transpose(X) * Θ)
  t = a > 0 ? min(1.0, λ / a) : 1.0
  D = t * dot(Yc, Θ) - t^2 * sum(abs2, Rc) / (2n)
  scale = sum((abs.(Y) .+ abs.(transpose(d.b0)) .+ abs.(d.XB)) .^ 2) / (2n) + penalty + t * dot(abs.(Yc), abs.(Θ)) +
          t^2 * sum(abs2, Rc) / (2n)
  (; P, D, Θ = t .* Θ, t, scale, terms = max(n, p, q), kkt = nothing)
end

function nuclear_gap(X, Y, β0, B, λ; intercept::Bool)
  a = nuclear_parts(X, Y, β0, B, λ; intercept)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

######################################################################
#   Quantile regression
######################################################################

cert_check(r, τ) = max(τ * r, (τ - 1) * r)

# HiGHS's default primal feasibility tolerance, which bounds how far from zero
# the residuals of the basic observations of an exact (LP) fit can be; residuals
# within QUANTILE_ZERO (1 + ‖y - o‖∞) of zero are completed by the LP below.
# Enlarging this set never loosens the bound, as the LP maximizes over a larger
# family that still contains the sign pattern of the optimum.
const HIGHS_FEASIBILITY = 1e-7
const QUANTILE_ZERO = 100 * HIGHS_FEASIBILITY
# the tolerances of the completion LP; its solution is made exactly feasible
# afterwards, so they affect only how tight the bound is
const COMPLETION_TOLERANCE = 1e-10
# HiGHS sizes one process-wide thread scheduler at its first run, and a later
# instance that asks for another number of threads fails; the package's HiGHS
# extension asks for one, so the completion LP does too, whichever runs first
const HIGHS_THREADS = 1

# The completion LP of quantile_exact_gap: over a_Z ∈ [τ - 1, τ], maximize
# ∑_Z ỹᵢaᵢ subject to ∑_Z aᵢ = -∑_F fᵢ (intercept) and |X_Z'a_Z + X_F'f| ≤ nλw,
# with f the sign-fixed values off Z. Returns (a_Z, note), a_Z = nothing when
# there is nothing to complete or HiGHS does not find an optimum.
function quantile_completion(X, ỹ, Z, F, f, λ, w, τ, intercept::Bool)
  isempty(Z) && return nothing, "no zero residuals to complete"
  n, p = size(X)
  k = length(Z)
  XZ = Matrix(X[Z, :])
  xf = transpose(X[F, :]) * f
  start, index, value = Cint[], Cint[], Float64[]
  for i in 1:k
    push!(start, length(index))
    if intercept
      push!(index, 0)
      push!(value, 1.0)
    end
    for j in 1:p
      XZ[i, j] == 0 && continue
      push!(index, intercept + j - 1)
      push!(value, XZ[i, j])
    end
  end
  bound = n * λ .* w
  rlower = [intercept ? [-sum(f)] : Float64[]; -bound .- xf]
  rupper = [intercept ? [-sum(f)] : Float64[]; bound .- xf]
  nrow = length(rlower)
  col, coldual, row, rowdual = zeros(k), zeros(k), zeros(nrow), zeros(nrow)
  h = HiGHS.Highs_create()
  model = -1
  try
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", 0)
    HiGHS.Highs_setIntOptionValue(h, "threads", HIGHS_THREADS)
    HiGHS.Highs_setDoubleOptionValue(h, "primal_feasibility_tolerance", COMPLETION_TOLERANCE)
    HiGHS.Highs_setDoubleOptionValue(h, "dual_feasibility_tolerance", COMPLETION_TOLERANCE)
    status = HiGHS.Highs_passLp(h, k, nrow, length(index), HiGHS.kHighsMatrixFormatColwise,
                                HiGHS.kHighsObjSenseMaximize, 0.0, ỹ[Z], fill(τ - 1, k), fill(τ, k),
                                rlower, rupper, start, index, value)
    if status != HiGHS.kHighsStatusError
      HiGHS.Highs_run(h)
      model = HiGHS.Highs_getModelStatus(h)
      HiGHS.Highs_getSolution(h, col, coldual, row, rowdual)
    end
  finally
    HiGHS.Highs_destroy(h)
  end
  model == HiGHS.kHighsModelStatusOptimal && cert_finite(col) && return col, ""
  nothing, "the completion LP ended with HiGHS model status $model"
end

# Makes ∑a zero up to rounding by moving the coordinates in Z, or else all of
# them, toward the bound on the side that lowers |∑a|, in proportion to their
# distance from it; a stays in [τ - 1, τ]. Over all coordinates there is always
# room, as ∑a lies strictly between n(τ - 1) and nτ.
function rebalance!(a, Z, τ)
  for set in (Z, eachindex(a))
    for _ in 1:2
      e = sum(a)
      e == 0 && return a
      room = e > 0 ? a[set] .- (τ - 1) : τ .- a[set]
      R = sum(room; init = 0.0)
      R >= abs(e) || break
      a[set] .-= e .* room ./ R
      clamp!(a, τ - 1, τ)
    end
  end
  a
end

# mean(ρ_τ(y - o - β₀ - Xβ)) + λ∑wⱼ|βⱼ|, whose dual is: maximize (1/n)(y - o)'a
# subject to aᵢ ∈ [τ - 1, τ], ∑aᵢ = 0 (intercept) and |X'a/n| ≤ λw. The point
# takes aᵢ = τ or τ - 1 by the sign of rᵢ where |rᵢ| exceeds the zero tolerance,
# and completes the others by the LP above (zero when it is infeasible). It is
# then made exactly feasible: clamped to the box, rebalanced to ∑a = 0, and
# contracted toward the feasible a = 0 (τ - 1 ≤ 0 ≤ τ) by the largest step
# t = min(1, minⱼ nλwⱼ/|X'a|ⱼ), which keeps both; so a bound always exists.
function quantile_parts(X, y, β0, β, λ, w, τ; intercept::Bool, offset = nothing)
  d = cert_regression(X, y, β0, β; offset)
  d isa String && return d
  cert_levels(λ, w, d.p) && cert_finite(τ) && 0 < τ < 1 ||
    return "λ and the penalty factors must be finite and nonnegative, one per column, and τ in (0, 1)"
  λ > 0 && all(>(0), w) ||
    return "λ = 0 or a zero penalty factor makes a dual constraint an equality; no gap is certified"
  (; X, β, n, p) = d
  wf = f64(w)
  ỹ = d.y .- d.o
  fit = X * β
  r = ỹ .- d.β0 .- fit
  penalty = λ * dot(wf, abs.(β))
  P = sum(cert_check.(r, τ)) / n + penalty
  ζ = QUANTILE_ZERO * (1 + cert_maxabs(ỹ))
  Z = findall(v -> abs(v) <= ζ, r)
  F = findall(v -> abs(v) > ζ, r)
  f = [r[i] > 0 ? τ : τ - 1 for i in F]
  aZ, note = quantile_completion(X, ỹ, Z, F, f, λ, wf, τ, intercept)
  a = zeros(n)
  a[F] .= f
  aZ === nothing || (a[Z] .= aZ)
  clamp!(a, τ - 1, τ)
  intercept && rebalance!(a, Z, τ)
  t = box_scale(transpose(X) * a, n * λ .* wf)
  a .*= t
  yc = intercept ? ỹ .- mean(ỹ) : ỹ
  D = dot(yc, a) / n
  scale = max(τ, 1 - τ) * sum(abs.(ỹ) .+ abs(d.β0) .+ abs.(fit)) / n + penalty + dot(abs.(yc), abs.(a)) / n
  (; P, D, a, t, Z, note, scale, terms = max(n, p))
end

function quantile_exact_gap(X, y, β0, β, λ, w, τ; intercept::Bool, offset = nothing)
  a = quantile_parts(X, y, β0, β, λ, w, τ; intercept, offset)
  a isa String && return uncertifiable(a)
  c = relative_gap(a.P, a.D; a.scale, a.terms)
  c.status === :ok && !isempty(a.note) ? cert_ok(c.value, c.measure; note = a.note) : c
end

# The check loss convolved with the Epanechnikov kernel K(v) = 3(1 - v²)/4 on
# [-1, 1] at bandwidth h, at the residual u: ρ_τ(u) for |u| ≥ h, and inside,
# with s = u/h, (τ - 1/2)u + (h/2)(3/8 + 3s²/4 - s⁴/8), from
# ρ_τ(u) = (τ - 1/2)u + |u|/2 and ∫|u - v|K_h(v)dv. Its derivative in u is
# G(s) - (1 - τ), G(s) = (2 + 3s - s³)/4 the distribution function of K; its
# second derivative is at most 3/(4h).
function cert_conv_check(u, τ, h)
  abs(u) >= h && return cert_check(u, τ)
  s = u / h
  (τ - 0.5) * u + h * (3 / 8 + 3s^2 / 4 - s^4 / 8) / 2
end
cert_conv_score(u, τ, h) = u >= h ? τ : u <= -h ? τ - 1 : (s = u / h; τ - 1 + (2 + 3s - s^3) / 4)

"""
    quantile_smooth_residual(X, y, β0, β, λ, w, τ, h; intercept, offset = nothing)

The proximal-gradient fixed-point residual of `mean(ℓ_h(y - o - β₀ - Xβ)) +
λ∑wⱼ|βⱼ|`, `ℓ_h` the check loss convolved with the Epanechnikov kernel at
bandwidth `h` (`method = :smooth`): `max(‖β - S_{γλw}(β - γ∇)‖∞/γ, |∂/∂β₀|)`,
with `γ = 1/L`, `L = (3/(4h)) λmax(Ã'Ã/n)` and `Ã = [1 X]` (a Lipschitz
constant of the gradient).
"""
function quantile_smooth_residual(X, y, β0, β, λ, w, τ, h; intercept::Bool, offset = nothing)
  m = :fixed_point_residual
  d = cert_regression(X, y, β0, β; offset)
  d isa String && return uncertifiable(d, m)
  cert_levels(λ, w, d.p) && cert_finite(τ, h) && 0 < τ < 1 && h > 0 ||
    return uncertifiable("λ, the penalty factors, τ in (0, 1) and h > 0 must be finite", m)
  (; X, β, n, p) = d
  wf = f64(w)
  ỹ = d.y .- d.o
  ψ = cert_conv_score.(ỹ .- d.β0 .- X * β, τ, h)
  g = -(transpose(X) * ψ) ./ n
  L = 3 / (4h) * curvature_bound(X, ones(p), fill(1 / n, n), intercept)
  γ = 1 / L
  z = cert_soft.(β .- γ .* g, γ * λ .* wf)
  residual = max(cert_maxabs(β .- z) / γ, intercept ? abs(sum(ψ)) / n : 0.0)
  c0 = 0.0
  if intercept
    c0 = decreasing_root(c -> sum(u -> cert_conv_score(u - c, τ, h), ỹ), median(ỹ))
    c0 === nothing && return uncertifiable("no intercept optimum at β = 0", m)
  end
  g0 = cert_maxabs(transpose(X) * cert_conv_score.(ỹ .- c0, τ, h)) / n
  relative_residual(residual, g0, m)
end

######################################################################
#   Generalized lasso
######################################################################

# A point of the box |uₖ| ≤ bₖ that minimizes ‖D'u - g‖²/2, by projected Newton
# (Bertsekas, 1982): a Newton step on the coordinates off the active faces of
# the box, gradient steps on the others, and an Armijo search along the
# projection arc. The Newton matrix D_F D_F' is regularized by δI,
# δ = 1e-12 ‖D‖₁‖D‖∞ (an upper bound of ‖D‖₂², scaled), as the rows of stacked
# or graph D are dependent; on a face the steps then converge at the rate
# δ/(μ + δ), μ the smallest positive eigenvalue of D_F D_F', which is small for
# divided differences on irregular locations, so the iteration continues until
# the objective is at rounding level or stops decreasing, or for maxiter steps.
# Every iterate is in the box, so the bound below is valid wherever it stops.
function box_lsq(D::SparseMatrixCSC{Float64}, g::Vector{Float64}, b::Vector{Float64}; maxiter::Integer = 500)
  m = size(D, 1)
  u = zeros(m)
  Dt = sparse(transpose(D))
  L = m == 0 ? 0.0 : maximum(sum(abs, D; dims = 1)) * maximum(sum(abs, D; dims = 2))
  L > 0 || return u
  objective(v) = sum(abs2, Dt * v .- g) / 2
  floor = (4eps())^2 * sum(abs2, g) / 2
  φu = objective(u)
  for _ in 1:maxiter
    φu <= floor && break
    grad = D * (Dt * u .- g)
    pg = cert_maxabs(u .- clamp.(u .- grad ./ L, -b, b))
    pg == 0 && break
    ε = min(1e-8 * maximum(b), pg)
    active = [b[k] == 0 || (u[k] <= -b[k] + ε && grad[k] > 0) || (u[k] >= b[k] - ε && grad[k] < 0) for k in 1:m]
    free, fixed = findall(!, active), findall(active)
    step = zeros(m)
    step[fixed] .= -grad[fixed] ./ L
    if !isempty(free)
      DF = D[free, :]
      H = DF * transpose(DF) + (1e-12 * L) * I
      factor = try cholesky(Symmetric(H)) catch; nothing end
      factor === nothing && break
      step[free] .= -(factor \ grad[free])
    end
    α, accepted, candidate, φc = 1.0, false, u, φu
    for _ in 1:60
      candidate = clamp.(u .+ α .* step, -b, b)
      decrease = α * dot(grad[free], -step[free]) + dot(grad[fixed], u[fixed] .- candidate[fixed])
      decrease > 0 || break
      φc = objective(candidate)
      if φc <= φu - 1e-4 * decrease
        accepted = true
        break
      end
      α /= 2
    end
    accepted || break
    progress = φu - φc
    u, φu = candidate, φc
    progress <= 1e-15 * φu && break
  end
  u
end

# ‖y - β₀ - Xβ‖²/(2n) + ∑ₖλrowsₖ|(Dβ)ₖ|, with X = nothing for a signal (the
# identity design, n = length(y), src/structured.jl). fused_lasso(X, y, λ1, λ2)
# is D = [I; D₁] with λrows = [fill(λ1, p); fill(λ2, p - 1)], D₁ the first
# differences; trend_filter(y, λ; order) is D = difference_matrix(p, order + 1)
# with λrows = fill(λ, p - order - 1).
#
# With f(β) = ‖ỹ - X̃β‖²/(2n) (the intercept eliminated by centering), any u
# with |uₖ| ≤ λrowsₖ gives the lower bound -f*(-D'u) = min_β f(β) + u'Dβ, at
# z = (X̃'X̃/n)⁻¹(X̃'ỹ/n - D'u) from a Cholesky factor (X̃ of full column rank, else
# uncertifiable), evaluated as f(z) + u'Dz. u solves the box-constrained least
# squares min ‖D'u - X̃'r_c/n‖ (box_lsq), which is tight at the optimum, where
# D'u* = X̃'r_c/n = -∇f(β*).
function genlasso_parts(X, y, β0, β, D, λrows; intercept::Bool, maxiter::Integer = 500)
  d = cert_regression(X, y, β0, β)
  d isa String && return d
  (; y, β, n, p) = d
  X === nothing && p != n && return "a signal has one coefficient per observation"
  size(D, 2) == p && length(λrows) == size(D, 1) && cert_finite(D, λrows) && all(>=(0), λrows) ||
    return "D needs a column per coefficient, and λrows a finite nonnegative level per row of D"
  Dm = sparse(f64(D))
  λr = f64(λrows)
  fit = X === nothing ? β : d.X * β
  r = y .- d.β0 .- fit
  penalty = dot(λr, abs.(Dm * β))
  P = sum(abs2, r) / (2n) + penalty
  yc = intercept ? y .- mean(y) : y
  rc = intercept ? r .- mean(r) : r
  if X === nothing
    intercept && return "a signal with an intercept is not identifiable: the centered identity is singular"
    u = box_lsq(Dm, rc ./ n, λr; maxiter)
    z = yc .- n .* (transpose(Dm) * u)
    fitz = z
  else
    Xc = Matrix(d.X)
    intercept && (Xc .-= mean(Xc; dims = 1))
    factor = cholesky(Symmetric(transpose(Xc) * Xc ./ n); check = false)
    pivots = diag(factor.U)
    issuccess(factor) && minimum(pivots)^2 > p * eps() * maximum(pivots)^2 ||
      return "X̃ does not have full column rank"
    u = box_lsq(Dm, transpose(Xc) * rc ./ n, λr; maxiter)
    z = factor \ (transpose(Xc) * yc ./ n .- transpose(Dm) * u)
    fitz = Xc * z
  end
  Dz = Dm * z
  bound = sum(abs2, yc .- fitz) / (2n) + dot(u, Dz)
  scale = sum((abs.(y) .+ abs(d.β0) .+ abs.(fit)) .^ 2) / (2n) + penalty +
          sum((abs.(yc) .+ abs.(fitz)) .^ 2) / (2n) + dot(abs.(u), abs.(Dz))
  (; P, D = bound, u, scale, terms = max(n, p, size(Dm, 1)))
end

function genlasso_gap(X, y, β0, β, D, λrows; intercept::Bool, maxiter::Integer = 500)
  a = genlasso_parts(X, y, β0, β, D, λrows; intercept, maxiter)
  a isa String && return uncertifiable(a)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

######################################################################
#   SLOPE
######################################################################

# ‖y - β₀ - Xβ‖²/(2n) + ∑ⱼλⱼ|β̃|₍ⱼ₎, β̃ = s∘β with s the root mean squares of the
# columns centered with an intercept (a zero column scaled by one) when
# standardize_x (src/slope.jl), β̃ = β otherwise. The dual norm of the sorted ℓ1
# norm is maxₖ ∑_{i≤k} z₍ᵢ₎/∑_{i≤k} λᵢ, so θ = t r_c/n is feasible for
# t = min(1, minₖ ∑_{i≤k}λᵢ/∑_{i≤k}zᵢ), z = |(X̃'θ)./s| sorted decreasingly; the
# dual value is that of the lasso. λ₁ = 0 is least squares, with its KKT residual.
function slope_parts(X, y, β0, β, λseq; intercept::Bool, standardize_x::Bool)
  d = cert_regression(X, y, β0, β)
  d isa String && return d
  length(λseq) == d.p && cert_finite(λseq) && all(>=(0), λseq) && issorted(λseq; rev = true) ||
    return "λseq must be finite, nonnegative and nonincreasing, one level per column"
  (; X, y, β, n, p) = d
  λ = f64(λseq)
  s = standardize_x ? column_scales(X; intercept) : ones(p)
  penalty = dot(λ, sort(abs.(s .* β); rev = true))
  fit = X * β
  r = y .- d.β0 .- fit
  P = sum(abs2, r) / (2n) + penalty
  yc = intercept ? y .- mean(y) : y
  if λ[1] == 0
    kkt = max(cert_maxabs((transpose(X) * r) ./ (n .* s)), intercept ? abs(sum(r)) / n : 0.0)
    return (; P, kkt, g0 = cert_maxabs((transpose(X) * yc) ./ (n .* s)))
  end
  rc = intercept ? r .- mean(r) : r
  θ = rc ./ n
  t = box_scale(cumsum(sort(abs.((transpose(X) * θ) ./ s); rev = true)), cumsum(λ))
  D = t * dot(yc, θ) - t^2 * sum(abs2, rc) / (2n)
  scale = sum((abs.(y) .+ abs(d.β0) .+ abs.(fit)) .^ 2) / (2n) + penalty + t * dot(abs.(yc), abs.(θ)) +
          t^2 * sum(abs2, rc) / (2n)
  (; P, D, θ = t .* θ, t, s, scale, terms = max(n, p), kkt = nothing)
end

function slope_gap(X, y, β0, β, λseq; intercept::Bool, standardize_x::Bool)
  a = slope_parts(X, y, β0, β, λseq; intercept, standardize_x)
  a isa String && return uncertifiable(a)
  a.kkt === nothing || return relative_residual(a.kkt, a.g0, :kkt_residual)
  relative_gap(a.P, a.D; a.scale, a.terms)
end

######################################################################
#   Selection and inference
######################################################################

"""
    knockoff_feasibility(Sigma, s)

The largest violation of the equicorrelated knockoff constraints on the
correlation scale (src/knockoffs.jl): `0 ≤ sⱼ/Σⱼⱼ ≤ min(1, 2λmin(R))` and
`λmin(2R - diag(s./diag Σ)) ≥ 0`, `R` the correlation matrix of `Σ`; zero when
`s` is feasible.
"""
function knockoff_feasibility(Sigma, s)
  m = :feasibility
  p = size(Sigma, 1)
  size(Sigma, 2) == p && length(s) == p || return uncertifiable("Sigma must be square, with one s per column", m)
  cert_finite(Sigma, s) || return uncertifiable("nonfinite input", m)
  S, sv = f64(Sigma), f64(s)
  issymmetric(S) && all(>(0), diag(S)) || return uncertifiable("Sigma must be symmetric with a positive diagonal", m)
  scales = sqrt.(diag(S))
  R = Symmetric(S ./ (scales * transpose(scales)))
  ρ = sv ./ diag(S)
  top = min(1.0, 2eigmin(R))
  violation = max(cert_maxabs(max.(-ρ, 0.0)), cert_maxabs(max.(ρ .- top, 0.0)),
                  max(-eigmin(Symmetric(2 .* Matrix(R) .- Diagonal(ρ))), 0.0))
  cert_ok(violation, m)
end

"""
    equicorrelated_check(Sigma, s; safety = 1e-8)

The largest relative deviation `maxⱼ |sⱼ/Σⱼⱼ - a| / max(a, floatmin)` of `s`
from the equicorrelated diagonal `sⱼ = a Σⱼⱼ` of `equicorrelated_s`
(src/knockoffs.jl), `a = (1 - safety) min(1, 2λmin(R))` with `R` the
correlation matrix of `Σ` (and `a = 0` when that is negative): zero, to
rounding, exactly for that diagonal. Feasibility alone
(`knockoff_feasibility`) also accepts other diagonals, `s = 0` among them.
"""
function equicorrelated_check(Sigma, s; safety = 1e-8)
  m = :consistency
  p = size(Sigma, 1)
  size(Sigma, 2) == p && length(s) == p || return uncertifiable("Sigma must be square, with one s per column", m)
  cert_finite(Sigma, s, safety) && 0 <= safety < 1 ||
    return uncertifiable("nonfinite input, or safety outside [0, 1)", m)
  S, sv = f64(Sigma), f64(s)
  issymmetric(S) && all(>(0), diag(S)) || return uncertifiable("Sigma must be symmetric with a positive diagonal", m)
  scales = sqrt.(diag(S))
  a = max((1 - safety) * min(1.0, 2eigmin(Symmetric(S ./ (scales * transpose(scales))))), 0.0)
  cert_ok(cert_maxabs(sv ./ diag(S) .- a) / max(a, floatmin(Float64)), m)
end

"""
    ols_coefficient_error(X, y, support, β0, β; intercept)

The relative error `‖(β₀, β) - (b₀, b)‖∞ / ‖(b₀, b)‖∞` of the returned
coefficients against the least squares `(b₀, b_S)` on `[1 X_S]` (on `X_S`
without an intercept), solved by a pivoted QR, with `b` zero off the support;
the coefficients of `β` off the support count against it. A support whose
columns are rank deficient (a pivot at most `max(n, |S|) eps` times the
largest) is uncertifiable.
"""
function ols_coefficient_error(X, y, support, β0, β; intercept::Bool)
  m = :coefficient_error
  d = cert_regression(X, y, β0, β)
  d isa String && return uncertifiable(d, m)
  S = sort!(unique(collect(Int, support)))
  all(j -> 1 <= j <= d.p, S) || return uncertifiable("the support has indices outside the columns", m)
  A = intercept ? hcat(ones(d.n), Matrix(d.X[:, S])) : Matrix(d.X[:, S])
  b = zeros(size(A, 2))
  if size(A, 2) > 0
    size(A, 1) >= size(A, 2) || return uncertifiable("more coefficients than observations", m)
    F = qr(A, ColumnNorm())
    pivots = abs.(diag(F.R))
    minimum(pivots) > max(size(A)...) * eps() * maximum(pivots) ||
      return uncertifiable("the selected columns are rank deficient", m)
    b = F \ d.y
  end
  reference = zeros(d.p)
  reference[S] .= b[intercept+1:end]
  e = [d.β0 - (intercept ? b[1] : 0.0); d.β .- reference]
  magnitude = max(intercept ? abs(b[1]) : 0.0, cert_maxabs(reference))
  cert_ok(cert_maxabs(e) / (magnitude > 0 ? magnitude : 1.0), m)
end

# the knockoff threshold recomputed from its definition: the smallest positive
# |Wⱼ| with (offset + #{W ≤ -t})/max(1, #{W ≥ t}) ≤ q, or Inf
function knockoff_threshold_reference(W, q, offset)
  for t in sort!(unique(abs.(W[W .!= 0])))
    (offset + count(<=(-t), W)) / max(1, count(>=(t), W)) <= q && return t
  end
  Inf
end

"""
    knockoff_threshold_check(W, q, threshold, offset)

The knockoff (`offset = 0`) or knockoff+ (`offset = 1`) threshold recomputed
from `W`, against the returned `threshold`: zero when they are equal (both
infinite included), else `|t - threshold|/max(|t|, |threshold|)`, or one when
exactly one of them is infinite.
"""
function knockoff_threshold_check(W, q, threshold, offset)
  m = :consistency
  cert_finite(W, q) && !isnan(threshold) && 0 < q < 1 && offset in (0, 1) ||
    return uncertifiable("W and q must be finite, q in (0, 1), and offset 0 or 1", m)
  t = knockoff_threshold_reference(f64(W), Float64(q), offset)
  thr = Float64(threshold)
  t == thr && return cert_ok(0.0, m)
  isfinite(t) && isfinite(thr) && return cert_ok(abs(t - thr) / max(abs(t), abs(thr)), m)
  cert_ok(1.0, m; note = "recomputed threshold $t, returned $thr")
end

"""
    frequency_check(selections, frequencies)

`maxⱼ |mean(selections[j, :]) - frequencies[j]|`, `selections` a matrix of
selection indicators with a row per variable and a column per fit.
"""
function frequency_check(selections::AbstractMatrix, frequencies)
  m = :consistency
  size(selections, 1) == length(frequencies) && size(selections, 2) > 0 ||
    return uncertifiable("selections needs a row per frequency and at least one column", m)
  cert_finite(frequencies) || return uncertifiable("nonfinite frequencies", m)
  cert_ok(cert_maxabs(vec(mean(selections; dims = 2)) .- f64(frequencies)), m)
end
