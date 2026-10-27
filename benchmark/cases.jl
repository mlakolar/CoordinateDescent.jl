# The case table: a scenario of problems.jl, one public estimator call with fixed
# settings, the certificate of certificates.jl for its documented objective, and
# a target. Each case states its target and settings in its add_case! call;
# thresholds are relative gaps or residuals, and `record` (nonconvex and
# selection procedures only) records a valid certificate without a threshold.
#
# build(problem) prepares the inputs once, untimed: copies in the element type
# and storage of the case, penalty levels, loadings for the certificate, options
# and a solver generator. Each sample times call(prepared) on a fresh copy, and
# certify(problem, prepared, output) checks that output from the prepared inputs
# and the returned coefficients and intercepts only (and, for the adaptive
# procedures, the penalty parameters they return), never from the solver's own
# gaps, residuals or convergence flags.

struct CaseSpec
  meta::NamedTuple
  build::Function
  call::Function
  certify::Function
end

const CASE_GROUPS = ("lasso", "glm", "group", "multitask", "lowrank", "quantile", "structured",
                     "selection", "inference", "controls")
# The negative controls run only when HDM_GROUPS names them.
const DEFAULT_GROUPS = filter(!=("controls"), CASE_GROUPS)

# The measures of the certificates, as the catalog records them
const MEASURE = (gap = "relative_gap", kkt = "kkt_residual", fixed_point = "fixed_point_residual",
                 excess = "objective_excess", coefficients = "coefficient_error",
                 feasibility = "feasibility", consistency = "consistency")

# Random coordinate orders, knockoff draws and subsamples start from the same
# state in every sample; the default generator is reseeded with SOLVER_SEED too.
const SOLVER_SEED = 20261028
const HARNESS_SEED_SCHEME = "inputs: problems.jl scenario seeds by key and repetition; " *
  "solvers: Xoshiro from SHA-256 of solver, reset per sample; " *
  "order: Xoshiro from SHA-256 of measurement_order and repetition"

# A Xoshiro generator whose state is the SHA-256 digest of parts, the same on
# every Julia version
function harness_rng(parts...)
  digest = sha256(join(string.(parts), '/'))
  word(k) = foldl((h, b) -> (h << 8) | UInt64(b), digest[8k-7:8k]; init = UInt64(0))
  Xoshiro(word(1), word(2), word(3), word(4))
end
solver_rng() = harness_rng("solver")

# The data hash of a case: its scenario and the penalty levels (and other
# numbers) that the case derives from it
case_data_hash(problem, prepared) =
  data_hash(merge(problem, (; case_levels = Float64.(prepared.levels))))

function add_case!(catalog, group, estimator, variant; scenario, measure, target,
                   build, call, certify, suite = "quick")
  parts = (group, estimator, variant)
  any(part -> occursin(r"[/\t\n\r]", part), parts) && error("invalid case identifier")
  group in CASE_GROUPS || error("unknown group $group")
  suite in ("quick", "extended") || error("unknown suite $suite")
  target == "record" || target isa Float64 && isfinite(target) && target > 0 ||
    error("the target must be record or a positive threshold")
  meta = (; id = join(parts, '/'), group, scenario, estimator, variant, phase = "call", suite,
            inputs = "draw", measure, target)
  push!(catalog, CaseSpec(meta, build, call, certify))
end

######################################################################
#
#   Inputs and penalty levels
#
######################################################################

case64(A::AbstractMatrix) = Matrix{Float64}(A)
case64(v::AbstractVector) = Vector{Float64}(collect(v))
case64(x::Real) = Float64(x)
case64(::Nothing) = nothing

# The loadings ωⱼ of standardize_x, from the inputs as the estimator receives them
case_loadings(X; intercept = true, weights = nothing) =
  column_loadings(case64(X); intercept, weights = case64(weights))

# The response minus the offset, centered at its weighted mean with an intercept
function centered_response(y, w, o, intercept)
  r = o === nothing ? case64(y) : case64(y) .- case64(o)
  intercept && (r .-= dot(w, r) / sum(w))
  r
end

# The smallest λ with a zero lasso solution, maxⱼ |x̃ⱼ'W(y - o - ȳ)| / (ωⱼ ∑w); with
# an intercept, ∑ wᵢ(yᵢ - oᵢ - ȳ) = 0, so the columns need no centering. For a GLM
# without offset this is also λmax at the intercept-only fit, and with an offset a
# scale for λ.
function ls_lambda_max(X, y, ω; intercept = true, weights = nothing, offset = nothing)
  w = weights === nothing ? ones(length(y)) : case64(weights)
  r = centered_response(y, w, offset, intercept)
  maximum(abs.(case64(X)' * (w .* r)) ./ (sum(w) .* ω))
end

# λmax of the square-root lasso ‖y - o - β₀ - Xβ‖_W + λ∑ωⱼ|βⱼ|
function sqrt_lambda_max(X, y, ω; intercept = true, weights = nothing, offset = nothing)
  w = weights === nothing ? ones(length(y)) : case64(weights)
  r = centered_response(y, w, offset, intercept)
  maximum(abs.(case64(X)' * (w .* r)) ./ (sqrt(dot(w, r .^ 2)) .* ω))
end

case_centered(X) = case64(X) .- mean(case64(X); dims = 1)

# λmax of the group lasso: maxₖ ‖X̃_G'ỹ‖/(n wₖ), or with orthonormalized groups
# maxₖ ‖Q_G'ỹ‖/(√n wₖ), Q_G an orthonormal basis of the centered columns of G
function group_lambda_max(X, y, groups, weights; orthonormal)
  Xc, yc, n = case_centered(X), case64(y) .- mean(y), size(X, 1)
  maximum(enumerate(groups)) do (k, G)
    orthonormal ? norm(Matrix(qr(Xc[:, G]).Q)' * yc) / (sqrt(n) * weights[k]) :
                  norm(Xc[:, G]' * yc) / (n * weights[k])
  end
end

# The largest standardized score maxⱼ |x̃ⱼ'ỹ|/(n sⱼ); λα above it zeroes every
# group of the sparse group lasso, so it bounds that λmax by itself divided by α
standardized_score(X, y) = maximum(abs.(case_centered(X)' * (case64(y) .- mean(y))) ./
                                   (size(X, 1) .* case_loadings(X)))

# λmax of the multi-response lasso, maxₖ ‖x̃ₖ'Ỹ‖/(n ωₖ)
multi_lambda_max(X, Y, ω) =
  maximum(norm.(eachrow(case_centered(X)' * case_centered(Y))) ./ (size(X, 1) .* ω))

# The smallest factor a with a zero SLOPE solution at a * base, on columns
# centered and scaled to mean square one: maxₖ ∑_{i≤k} |g|₍ᵢ₎ / ∑_{i≤k} baseᵢ
function slope_factor_max(X, y, base)
  Z = case_centered(X)
  Z ./= sqrt.(mean(abs2, Z; dims = 1))
  g = sort!(abs.(Z' * (case64(y) .- mean(y))) ./ size(X, 1); rev = true)
  maximum(cumsum(g) ./ cumsum(base))
end

# A scale of the quantile-lasso level: maxⱼ |xⱼ'u|/(n ωⱼ) at the scores
# u = τ - 1{y < q_τ} of the intercept-only fit
function quantile_scale(X, y, ω, τ; offset = nothing)
  r = offset === nothing ? case64(y) : case64(y) .- case64(offset)
  u = τ .- (r .< quantile(r, τ))
  maximum(abs.(case64(X)' * u) ./ (length(r) .* ω))
end

# A decreasing grid of `levels` values from top to ratio * top, evenly spaced on
# the log scale
level_grid(top, ratio, levels) = [top * ratio^((k - 1) / (levels - 1)) for k in 1:levels]

# The noise level of a signal from its successive differences
signal_noise(y) = std(diff(case64(y))) / sqrt(2)

# The order-k differences of p coefficients at the locations x, as
# difference_matrix documents them: D[1] the successive differences
# β[j+1] - β[j], and D[k+1] = D[1] * diag(k ./ (x[j+k] - x[j])) * D[k]
function difference_operator(p, order; x = nothing)
  first_differences(m) = spdiagm(m - 1, m, 0 => fill(-1.0, m - 1), 1 => ones(m - 1))
  order == 0 && return sparse(1.0I, p, p)
  D = first_differences(p)
  for k in 1:order-1
    scale = x === nothing ? ones(p - k) : [k / (x[j+k] - x[j]) for j in 1:p-k]
    D = first_differences(p - k) * Diagonal(scale) * D
  end
  D
end

# The fused lasso λ1‖β‖₁ + λ2∑|βⱼ₊₁ - βⱼ| as a generalized lasso with a level per row
fused_operator(p, λ1, λ2) =
  ([sparse(1.0I, p, p); difference_operator(p, 1)], [fill(λ1, p); fill(λ2, p - 1)])

# A graph on a ring of p vertices with chords to the vertex seven steps ahead,
# as a signed incidence matrix (rows βⱼ - βᵢ for the edge (i, j))
function ring_graph(p)
  edges = [[(i, mod1(i + 1, p)) for i in 1:p]; [(i, mod1(i + 7, p)) for i in 1:2:p]]
  sparse(repeat(1:length(edges), inner = 2), [k for e in edges for k in e],
         repeat([-1.0, 1.0], length(edges)), length(edges), p)
end

######################################################################
#
#   Certificates of returned objects
#
######################################################################

# The worst certificate of the levels of a path: the first one that is not ok,
# or the largest value; incomplete when fewer levels than requested came back
function worst_certificate(certs, requested)
  if isempty(certs)
    return (; value = NaN, measure = :none, status = :incomplete,
              note = "returned 0 of $requested requested levels")
  end
  bad = findfirst(c -> c.status != :ok, certs)
  k = bad === nothing ? argmax([c.value for c in certs]) : bad
  worst = certs[k]
  note = "level $k of $(length(certs)): $(worst.note)"
  status = length(certs) == requested ? worst.status : :incomplete
  status == :incomplete && (note = "returned $(length(certs)) of $requested requested levels; " * note)
  (; value = worst.value, measure = worst.measure, status, note)
end

path_certificate(f, solutions, requested) =
  worst_certificate([f(k) for k in eachindex(solutions)], requested)

# The scaled lasso's gap is that of the square-root lasso at √W λ; its σ̂ must be
# the root mean square of the residuals of the returned fit. The value is the
# larger of the gap and the relative error of σ̂, so that a wrong σ̂ misses the
# target too.
function scaled_certificate(s, out)
  X, y, w = case64(s.X), case64(s.y), case64(s.weights)
  r = s.offset === nothing ? y : y .- case64(s.offset)
  W = w === nothing ? length(y) : sum(w)
  cert = sqrt_l1_gap(X, r, case64(out.intercept), case64(out.x), sqrt(W) * case64(s.λ), s.ω;
                     intercept = true, weights = w)
  e = r .- case64(out.intercept) .- X * case64(out.x)
  σ = sqrt((w === nothing ? sum(abs2, e) : dot(w, e .^ 2)) / W)
  σerror = abs(case64(out.σ) - σ) / max(σ, floatmin(Float64))
  cert.status == :ok && isfinite(σerror) || return cert
  merge(cert, (; value = max(cert.value, σerror), note = cert.note * "; σ̂ relative error $σerror"))
end

# The effect regression of double selection, y on [1 d X_S], at the returned
# estimate α̂: the intercept and the coefficients of the controls are the least
# squares of y - α̂d on [1 X_S], which are optimal exactly when α̂ is, and the
# certificate compares the completed point with an independent solve
function effect_certificate(s, out)
  X, y, d = case64(s.X), case64(s.y), case64(s.d)
  S, α = out.selected, case64(out.estimate)
  γ = [ones(length(y)) X[:, S]] \ (y .- α .* d)
  β = zeros(size(X, 2) + 1)
  β[1] = α
  β[S .+ 1] .= γ[2:end]
  ols_coefficient_error([d X], y, [1; S .+ 1], γ[1], β; intercept = true)
end

######################################################################
#
#   The catalog
#
######################################################################

function lasso_cases!(catalog)
  # Least-squares inputs with λ = ratio λmax for the objective (:lasso, :sqrt for
  # the square-root lasso, :scaled for the scaled lasso) and the certificate's loadings
  function ls_inputs(p; T = Float64, ratio = 0.1, objective = :lasso, sparse_x = false, weighted = false)
    X, y = Matrix{T}(p.X), Vector{T}(p.y)
    sparse_x && (X = sparse(X))
    weights = weighted ? Vector{T}(p.weights) : nothing
    offset = weighted ? Vector{T}(p.offset) : nothing
    ω = case_loadings(X; intercept = true, weights)
    W = weights === nothing ? size(X, 1) : sum(case64(weights))
    top = objective == :lasso ? ls_lambda_max(X, y, ω; weights, offset) :
      sqrt_lambda_max(X, y, ω; weights, offset) / (objective == :scaled ? sqrt(W) : 1)
    λ = T(ratio * top)
    (; X, y, weights, offset, λ, ω, rng = solver_rng(), levels = [case64(λ)])
  end
  ls_gap(s, out) = ls_l1_gap(case64(s.X), case64(s.y), case64(out.intercept), case64(out.x), case64(s.λ),
                             s.ω; intercept = true, weights = case64(s.weights), offset = case64(s.offset))
  lasso_call(s; kw...) = lasso(s.X, s.y, s.λ; intercept = true, standardize_x = true,
                               weights = s.weights, offset = s.offset, rng = s.rng, kw...)

  add_case!(catalog, "lasso", "lasso", "dense-standardized"; scenario = "ls-200x50",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> ls_inputs(p), call = s -> lasso_call(s), certify = (p, s, out) -> ls_gap(s, out))
  add_case!(catalog, "lasso", "lasso", "sparse-csc"; scenario = "ls-sparse-500x200", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> ls_inputs(p; sparse_x = true), call = s -> lasso_call(s),
    certify = (p, s, out) -> ls_gap(s, out))
  add_case!(catalog, "lasso", "lasso", "gram-tall"; scenario = "ls-tall-5000x40", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> ls_inputs(p), call = s -> lasso_call(s; gram = true),
    certify = (p, s, out) -> ls_gap(s, out))
  add_case!(catalog, "lasso", "lasso", "weights-offset"; scenario = "ls-weighted-200x50", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> ls_inputs(p; weighted = true), call = s -> lasso_call(s),
    certify = (p, s, out) -> ls_gap(s, out))
  for (variant, scenario) in (("wide", "ls-wide-100x400"), ("correlated", "ls-corr-300x100"),
                              ("large", "ls-2000x1000"))
    add_case!(catalog, "lasso", "lasso", variant; scenario, suite = "extended",
      measure = MEASURE.gap, target = 1e-6,
      build = p -> ls_inputs(p), call = s -> lasso_call(s), certify = (p, s, out) -> ls_gap(s, out))
  end
  # Float32 arithmetic limits the attainable relative gap to about 1e-6.
  add_case!(catalog, "lasso", "lasso", "float32"; scenario = "ls-200x50", suite = "extended",
    measure = MEASURE.gap, target = 1e-4,
    build = p -> ls_inputs(p; T = Float32), call = s -> lasso_call(s),
    certify = (p, s, out) -> ls_gap(s, out))

  # ‖y - β₀ - Xβ‖ + λ∑ωⱼ|βⱼ|
  add_case!(catalog, "lasso", "sqrt_lasso", "dense-standardized"; scenario = "ls-200x50",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> ls_inputs(p; ratio = 0.3, objective = :sqrt),
    call = s -> sqrt_lasso(s.X, s.y, s.λ; intercept = true, standardize_x = true, rng = s.rng),
    certify = (p, s, out) -> sqrt_l1_gap(case64(s.X), case64(s.y), case64(out.intercept), case64(out.x),
                                         case64(s.λ), s.ω; intercept = true))
  # ‖y - o - β₀ - Xβ‖_W/√W + λ∑ωⱼ|βⱼ| after the minimum over σ
  for (variant, scenario, weighted) in (("dense", "ls-200x50", false),
                                        ("weights-offset", "ls-weighted-200x50", true))
    add_case!(catalog, "lasso", "scaled_lasso", variant; scenario, suite = "extended",
      measure = MEASURE.gap, target = 1e-6,
      build = p -> ls_inputs(p; ratio = 0.3, objective = :scaled, weighted),
      call = s -> scaled_lasso(s.X, s.y, s.λ, Vector{eltype(s.y)}(s.ω); intercept = true,
                               weights = s.weights, offset = s.offset, rng = s.rng),
      certify = (p, s, out) -> scaled_certificate(s, out))
  end

  # Ten levels from λmax to λmax/20, with max_hat_s left infinite so that the path
  # is never cut short. gram = :auto solves a dense path with n ≥ p on the Gram
  # matrix, which has no duality gap: each level stops on opt_tol. On the
  # residuals (sparse or wide X), the default path gap of 1e-4 would stop each
  # level short of a 1e-6 relative gap, so those paths stop at a gap of 1e-8.
  function lasso_path(p; sparse_x = false)
    s = ls_inputs(p; sparse_x, ratio = 1)
    λpath = level_grid(case64(s.λ), 0.05, 10)
    gram = !sparse_x && size(s.X, 1) >= size(s.X, 2)
    merge(s, (; λpath, options = gram ? CDOptions() : CDOptions(gaptol = 1e-8), levels = λpath))
  end
  path_gap(s, out) = path_certificate(out.βpath, length(s.λpath)) do k
    ls_l1_gap(case64(s.X), case64(s.y), out.β0path[k], case64(out.βpath[k]), s.λpath[k], s.ω;
              intercept = true)
  end
  for (variant, scenario, suite, screening, sparse_x) in
      (("screening", "ls-200x50", "quick", true, false),
       ("no-screening", "ls-200x50", "extended", false, false),
       ("sparse-screening", "ls-sparse-500x200", "extended", true, true),
       ("wide-screening", "ls-wide-100x400", "extended", true, false))
    add_case!(catalog, "lasso", "LassoPath", variant; scenario, suite, measure = MEASURE.gap, target = 1e-6,
      build = p -> lasso_path(p; sparse_x),
      call = s -> LassoPath(s.X, s.y, s.λpath, s.options; intercept = true, standardize_x = true,
                            screening, rng = s.rng),
      certify = (p, s, out) -> path_gap(s, out))
  end

  # Least-squares stationarity of the refit on every distinct support of the path
  add_case!(catalog, "lasso", "refit_lasso_path", "intercept"; scenario = "ls-200x50", suite = "extended",
    measure = MEASURE.kkt, target = 1e-8,
    build = p -> begin
      s = lasso_path(p)
      path = LassoPath(s.X, s.y, s.λpath, s.options; intercept = true, standardize_x = true, rng = solver_rng())
      merge(s, (; path, supports = unique([findall(!iszero, β) for β in path.βpath])))
    end,
    call = s -> refit_lasso_path(s.path, s.X, s.y),
    certify = (p, s, out) -> begin
      found = [S for S in s.supports if haskey(out, S)]
      path_certificate(found, length(s.supports)) do k
        S, b = found[k], out[found[k]]
        β = zeros(size(s.X, 2))
        β[S] .= b[2:end]
        ls_refit_stationarity(case64(s.X), case64(s.y), b[1], β; intercept = true)
      end
    end)
end

function glm_cases!(catalog)
  function glm_inputs(p; T = Float64, ratio = 0.1, weighted = false)
    X, y = Matrix{T}(p.X), Vector{T}(p.y)
    weights = weighted ? Vector{T}(p.weights) : nothing
    offset = weighted ? Vector{T}(p.offset) : nothing
    ω = case_loadings(X; intercept = true, weights)
    # the scales of a penalty other than the ℓ1 norm on s∘β (one for a zero spread)
    scales = column_scales(case64(X); intercept = true, weights = case64(weights))
    λ = T(ratio * ls_lambda_max(X, y, ω; weights))
    (; X, y, weights, offset, λ, ω, scales, rng = solver_rng(), levels = [case64(λ)])
  end
  loss(family) = family == :logistic ? ProximalBase.Logistic() : ProximalBase.Poisson()
  glm_call(s, family; kw...) = glm_lasso(s.X, s.y, s.λ, loss(family); intercept = true, standardize_x = true,
                                         weights = s.weights, offset = s.offset, rng = s.rng, kw...)
  glm_gap(s, out, family; alpha = 1) =
    glm_l1_gap(family, case64(s.X), case64(s.y), case64(out.intercept), case64(out.x), case64(s.λ),
               alpha == 1 ? s.ω : s.scales; alpha, intercept = true, weights = case64(s.weights),
               offset = case64(s.offset))

  for (variant, family, scenario, suite, weighted) in
      (("logistic", :logistic, "logistic-300x60", "quick", false),
       ("poisson", :poisson, "poisson-300x40", "quick", false),
       ("logistic-weights-offset", :logistic, "logistic-weighted-300x60", "extended", true),
       ("logistic-large", :logistic, "logistic-2000x500", "extended", false))
    add_case!(catalog, "glm", "glm_lasso", variant; scenario, suite, measure = MEASURE.gap, target = 1e-6,
      build = p -> glm_inputs(p; weighted), call = s -> glm_call(s, family),
      certify = (p, s, out) -> glm_gap(s, out, family))
  end
  # λ(α∑|s∘β| + (1 - α)‖s∘β‖²/2) on the standardized coefficients s∘β
  add_case!(catalog, "glm", "glm_lasso", "logistic-enet"; scenario = "logistic-300x60", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> glm_inputs(p), call = s -> glm_call(s, :logistic; alpha = 0.5),
    certify = (p, s, out) -> glm_gap(s, out, :logistic; alpha = 0.5))
  # Float32 arithmetic limits the attainable relative gap to about 1e-6.
  add_case!(catalog, "glm", "glm_lasso", "logistic-float32"; scenario = "logistic-300x60", suite = "extended",
    measure = MEASURE.gap, target = 1e-4,
    build = p -> glm_inputs(p; T = Float32), call = s -> glm_call(s, :logistic),
    certify = (p, s, out) -> glm_gap(s, out, :logistic))
  # Stationarity only: MCP and SCAD have local solutions. With standardize_x the
  # exact coordinate steps are convex for a > 1 (MCP) and a > 2 (SCAD).
  for (penalty, a, template) in ((:mcp, 3.0, ProximalBase.MCP(1.0, 3.0)),
                                 (:scad, 3.7, ProximalBase.SCAD(1.0, 3.7)))
    add_case!(catalog, "glm", "glm_lasso", "logistic-$penalty"; scenario = "logistic-300x60",
      suite = "extended", measure = MEASURE.fixed_point, target = 1e-6,
      build = p -> glm_inputs(p), call = s -> glm_call(s, :logistic; penalty = template),
      certify = (p, s, out) -> nonconvex_stationarity(:logistic, penalty, case64(s.X), case64(s.y),
        case64(out.intercept), case64(out.x), case64(s.λ), a, s.scales; intercept = true))
  end

  # Eight levels from λmax to λmax/20, with devmax = 1 so that the deviance rule
  # never stops the path, and a path gap of 1e-8 instead of the default 1e-4
  add_case!(catalog, "glm", "GLMLassoPath", "logistic"; scenario = "logistic-300x60", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> begin
      s = glm_inputs(p; ratio = 1)
      λpath = level_grid(case64(s.λ), 0.05, 8)
      merge(s, (; λpath, options = CDOptions(gaptol = 1e-8), levels = λpath))
    end,
    call = s -> GLMLassoPath(s.X, s.y, s.λpath, ProximalBase.Logistic(), s.options; intercept = true,
                             standardize_x = true, devmax = 1, rng = s.rng),
    certify = (p, s, out) -> path_certificate(out.βpath, length(s.λpath)) do k
      glm_l1_gap(:logistic, case64(s.X), case64(s.y), out.β0path[k], case64(out.βpath[k]), s.λpath[k], s.ω;
                 intercept = true)
    end)
end

function group_cases!(catalog)
  function groups_inputs(p; orthonormal = true, ratio = 0.1, levels = 0)
    X, y, groups = case64(p.X), case64(p.y), [collect(Int, G) for G in p.groups]
    weights = sqrt.(length.(groups))
    top = group_lambda_max(X, y, groups, weights; orthonormal)
    λpath = levels == 0 ? [ratio * top] : level_grid(top, 0.05, levels)
    (; X, y, groups, weights, λ = λpath[1], λpath, options = CDOptions(gaptol = 1e-8), levels = λpath)
  end
  # The penalty λ∑wₖ‖X̃_G β_G‖₂/√n of :orthonormal, or λ∑wₖ‖β_G‖₂ of :none. Without
  # orthonormalization the block steps are not exact, and opt_tol = 1e-7 leaves
  # relative gaps up to about 1e-6, so that solve stops at a gap of 1e-8 instead.
  for (variant, standardize, suite, options) in (("orthonormal", :orthonormal, "quick", CDOptions()),
                                                 ("none", :none, "extended", CDOptions(gaptol = 1e-8)))
    orthonormal = standardize == :orthonormal
    add_case!(catalog, "group", "group_lasso", variant; scenario = "groups-200x60", suite,
      measure = MEASURE.gap, target = 1e-6,
      build = p -> groups_inputs(p; orthonormal),
      call = s -> group_lasso(s.X, s.y, s.λ, s.groups, options; intercept = true, standardize),
      certify = (p, s, out) -> group_gap(s.X, s.y, out.intercept, case64(out.x), s.λ, s.groups, s.weights;
                                         intercept = true, orthonormal))
  end
  # Eight levels from λmax to λmax/20; the default path gap of 1e-4 is replaced by
  # 1e-8 and max_groups is left infinite.
  add_case!(catalog, "group", "GroupLassoPath", "orthonormal"; scenario = "groups-200x60", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> groups_inputs(p; levels = 8),
    call = s -> GroupLassoPath(s.X, s.y, s.λpath, s.groups, s.options; intercept = true,
                               standardize = :orthonormal),
    certify = (p, s, out) -> path_certificate(out.βpath, length(s.λpath)) do k
      group_gap(s.X, s.y, out.β0path[k], case64(out.βpath[k]), s.λpath[k], s.groups, s.weights;
                intercept = true, orthonormal = true)
    end)

  # λ(α∑|s∘β| + (1 - α)∑wₖ‖(s∘β)_G‖₂) on standardized columns, at α = 0.5 so that
  # both terms bind; λ is a fraction of the bound max|g|/α on λmax
  α = 0.5
  function sgl_inputs(p; levels = 0)
    s = groups_inputs(p; orthonormal = false)
    top = standardized_score(s.X, s.y) / α
    λpath = levels == 0 ? [0.1 * top] : level_grid(top, 0.05, levels)
    merge(s, (; λ = λpath[1], λpath, rng = solver_rng(), levels = λpath))
  end
  add_case!(catalog, "group", "sparse_group_lasso", "standardized"; scenario = "groups-200x60",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> sgl_inputs(p),
    call = s -> sparse_group_lasso(s.X, s.y, s.λ, s.groups; alpha = α, intercept = true, standardize_x = true,
                                   rng = s.rng),
    certify = (p, s, out) -> sparse_group_gap(s.X, s.y, out.intercept, case64(out.x), s.λ, α, s.groups,
                                              s.weights; intercept = true, standardize_x = true))
  add_case!(catalog, "group", "SparseGroupLassoPath", "standardized"; scenario = "groups-200x60",
    suite = "extended", measure = MEASURE.gap, target = 1e-6,
    build = p -> sgl_inputs(p; levels = 8),
    call = s -> SparseGroupLassoPath(s.X, s.y, s.λpath, s.groups, s.options; alpha = α, intercept = true,
                                     standardize_x = true, rng = s.rng),
    certify = (p, s, out) -> path_certificate(out.βpath, length(s.λpath)) do k
      sparse_group_gap(s.X, s.y, out.β0path[k], case64(out.βpath[k]), s.λpath[k], α, s.groups, s.weights;
                       intercept = true, standardize_x = true)
    end)
end

function multitask_cases!(catalog)
  # ‖Y - 1β₀' - XB‖²_F/(2n) + λ∑ωₖ‖B[k, :]‖₂ with the loadings of standardize_x
  function multi_inputs(p; levels = 0)
    X, Y = case64(p.X), case64(p.Y)
    ω = case_loadings(X)
    top = multi_lambda_max(X, Y, ω)
    λpath = levels == 0 ? [0.1 * top] : level_grid(top, 0.05, levels)
    (; X, Y, ω, λ = λpath[1], λpath, levels = λpath)
  end
  add_case!(catalog, "multitask", "multi_lasso", "standardized"; scenario = "multi-200x40x5",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> multi_inputs(p),
    call = s -> multi_lasso(s.X, s.Y, s.λ; intercept = true, standardize_x = true),
    certify = (p, s, out) -> multi_gap(s.X, s.Y, out.intercept, Matrix(out.B), s.λ, s.ω; intercept = true))
  # Eight levels from λmax to λmax/20, with max_rows left infinite. The
  # multi-response model has no duality gap in ProximalSolvers, so each level
  # stops on opt_tol whatever gaptol.
  add_case!(catalog, "multitask", "MultiLassoPath", "screening"; scenario = "multi-200x40x5", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> multi_inputs(p; levels = 8),
    call = s -> MultiLassoPath(s.X, s.Y, s.λpath; intercept = true, standardize_x = true),
    certify = (p, s, out) -> path_certificate(out.Bpath, length(s.λpath)) do k
      multi_gap(s.X, s.Y, out.β0path[k], Matrix(out.Bpath[k]), s.λpath[k], s.ω; intercept = true)
    end)
end

function lowrank_cases!(catalog)
  rank = 3
  function lowrank_inputs(p; T = Float64, levels = 0)
    X, Y = Matrix{T}(p.X), Matrix{T}(p.Y)
    top = opnorm(case_centered(X)' * case_centered(Y)) / size(X, 1)
    λpath = levels == 0 ? [0.2 * top] : level_grid(top, 0.05, levels)
    (; X, Y, λ = λpath[1], λpath, levels = [Float64(rank); λpath])
  end
  # ‖Y - 1β₀' - XB‖²_F subject to rank(B) ≤ 3: the loss over the closed-form optimum
  # Float32 arithmetic limits the attainable relative excess to about 1e-6. B is
  # certified in its own element type, which sets the tolerance of its rank.
  for (variant, scenario, T, target, suite) in
      (("rank3", "lowrank-150x30x10", Float64, 1e-8, "quick"),
       ("rank3-wide", "lowrank-wide-60x100x10", Float64, 1e-8, "extended"),
       ("rank3-float32", "lowrank-150x30x10", Float32, 1e-4, "extended"))
    add_case!(catalog, "lowrank", "reduced_rank", variant; scenario, suite,
      measure = MEASURE.excess, target,
      build = p -> lowrank_inputs(p; T),
      call = s -> reduced_rank(s.X, s.Y, rank; intercept = true),
      certify = (p, s, out) -> rrr_excess(case64(s.X), case64(s.Y), out.intercept, out.B, rank; intercept = true))
  end
  # ‖Y - 1β₀' - XB‖²_F/(2n) + λ‖B‖_*
  nuclear_gap_of(s, out, λ) = nuclear_gap(s.X, s.Y, out.intercept, out.B, λ; intercept = true)
  for (variant, scenario, suite) in (("apg", "lowrank-150x30x10", "quick"),
                                     ("apg-wide", "lowrank-wide-60x100x10", "extended"))
    add_case!(catalog, "lowrank", "nuclear_regression", variant; scenario, suite,
      measure = MEASURE.gap, target = 1e-6,
      build = p -> lowrank_inputs(p),
      call = s -> nuclear_regression(s.X, s.Y, s.λ; intercept = true),
      certify = (p, s, out) -> nuclear_gap_of(s, out, s.λ))
  end
  add_case!(catalog, "lowrank", "LowRankPath", "nuclear"; scenario = "lowrank-150x30x10", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> lowrank_inputs(p; levels = 6),
    call = s -> LowRankPath(s.X, s.Y; method = :nuclear, λpath = s.λpath, intercept = true),
    certify = (p, s, out) -> path_certificate(out.solutions, length(s.λpath)) do k
      nuclear_gap_of(s, out.solutions[k], s.λpath[k])
    end)
  # Nonconvex: the loss excess over the rank-3 optimum, the price of row sparsity,
  # is recorded without a threshold.
  add_case!(catalog, "lowrank", "sparse_reduced_rank", "rank3"; scenario = "lowrank-150x30x10", suite = "extended",
    measure = MEASURE.excess, target = "record",
    build = p -> begin
      s = lowrank_inputs(p)
      λ = 0.1 * maximum(norm.(eachrow(case_centered(s.X)' * case_centered(s.Y)))) / size(s.X, 1)
      merge(s, (; λ, levels = [Float64(rank), λ]))
    end,
    call = s -> sparse_reduced_rank(s.X, s.Y, rank, s.λ; intercept = true),
    certify = (p, s, out) -> rrr_excess(s.X, s.Y, out.intercept, out.B, rank; intercept = true))
end

function quantile_cases!(catalog)
  # mean ρ_τ(y - o - β₀ - Xβ) + λ∑wⱼ|βⱼ|, the penalty factors one times the
  # loadings of standardize_x
  function quantile_inputs(p; τ = 0.5, offset = false, levels = 0)
    X, y = case64(p.X), case64(p.y)
    o = offset ? case64(p.offset) : nothing
    ω = case_loadings(X)
    top = quantile_scale(X, y, ω, τ; offset = o)
    λpath = levels == 0 ? [0.1 * top] : level_grid(top, 0.05, levels)
    (; X, y, offset = o, τ, ω, λ = λpath[1], λpath, rng = solver_rng(), levels = [τ; λpath])
  end
  exact_gap(s, out, λ) = quantile_exact_gap(s.X, s.y, out.intercept, out.β, λ, s.ω, s.τ; intercept = true,
                                            offset = s.offset)
  for (variant, scenario, τ, offset, suite) in
      (("exact-median", "quantile-200x30", 0.5, false, "quick"),
       ("exact-tau25-offset", "ls-weighted-200x50", 0.25, true, "extended"),
       ("exact-large", "quantile-1000x100", 0.5, false, "extended"))
    add_case!(catalog, "quantile", "quantile_lasso", variant; scenario, suite,
      measure = MEASURE.gap, target = 1e-6,
      build = p -> quantile_inputs(p; τ, offset),
      call = s -> quantile_lasso(s.X, s.y, s.λ; tau = s.τ, method = :exact, intercept = true, offset = s.offset),
      certify = (p, s, out) -> exact_gap(s, out, s.λ))
  end
  # The convolution-smoothed loss at bandwidth h. The default tol = 2e-5 stops the
  # smoothed fit near a relative residual of 2e-5, so it is tightened to 1e-8.
  h = 0.2
  add_case!(catalog, "quantile", "quantile_lasso", "smooth-median"; scenario = "quantile-200x30",
    measure = MEASURE.fixed_point, target = 1e-6,
    build = p -> quantile_inputs(p),
    call = s -> quantile_lasso(s.X, s.y, s.λ; tau = s.τ, method = :smooth, bandwidth = h, tol = 1e-8,
                               intercept = true, rng = s.rng),
    certify = (p, s, out) -> quantile_smooth_residual(s.X, s.y, out.intercept, out.β, s.λ, s.ω, s.τ, h;
                                                      intercept = true))
  add_case!(catalog, "quantile", "QuantileLassoPath", "exact-median"; scenario = "quantile-200x30",
    suite = "extended", measure = MEASURE.gap, target = 1e-6,
    build = p -> quantile_inputs(p; levels = 6),
    call = s -> QuantileLassoPath(s.X, s.y, s.λpath; tau = s.τ, method = :exact, intercept = true),
    certify = (p, s, out) -> path_certificate(out.solutions, length(s.λpath)) do k
      exact_gap(s, out.solutions[k], s.λpath[k])
    end)
end

function structured_cases!(catalog)
  # Signals: ‖y - β‖²/(2n) + ∑ₖ λₖ|(Dβ)ₖ|, λ a multiple of σ̂/n
  function signal_inputs(p; scale)
    y = case64(p.y)
    x = haskey(p, :x) ? case64(p.x) : nothing
    λ = scale * signal_noise(y) / length(y)
    (; y, x, λ, levels = [λ])
  end
  signal_gap(s, out, D, λrows) = genlasso_gap(nothing, s.y, 0.0, out.x, D, λrows; intercept = false)
  for (variant, scenario, suite) in (("signal", "signal-200", "quick"), ("signal-long", "signal-5000", "extended"))
    add_case!(catalog, "structured", "fused_lasso", variant; scenario, suite,
      measure = MEASURE.gap, target = 1e-6,
      build = p -> begin
        s = signal_inputs(p; scale = 3)
        λ1 = s.λ / 10
        merge(s, (; λ1, levels = [λ1, s.λ]))
      end,
      call = s -> fused_lasso(s.y, s.λ1, s.λ),
      certify = (p, s, out) -> signal_gap(s, out, fused_operator(length(s.y), s.λ1, s.λ)...))
  end
  # ADMM stops on residuals (the linear signal on a gap of 0.3 reltol), which
  # leaves relative gaps near 1e-6 at the default tolerances.
  add_case!(catalog, "structured", "trend_filter", "signal-linear"; scenario = "trend-200",
    measure = MEASURE.gap, target = 1e-5,
    build = p -> signal_inputs(p; scale = 20),
    call = s -> trend_filter(s.y, s.λ; order = 1, x = s.x),
    certify = (p, s, out) -> signal_gap(s, out, difference_operator(length(s.y), 2; x = s.x),
                                        fill(s.λ, length(s.y) - 2)))
  add_case!(catalog, "structured", "generalized_lasso", "graph-signal"; scenario = "signal-200",
    suite = "extended", measure = MEASURE.gap, target = 1e-5,
    build = p -> merge(signal_inputs(p; scale = 3), (; D = ring_graph(length(p.y)))),
    call = s -> generalized_lasso(s.y, s.D, s.λ),
    certify = (p, s, out) -> signal_gap(s, out, s.D, fill(s.λ, size(s.D, 1))))
  add_case!(catalog, "structured", "trend_filter_path", "signal-linear"; scenario = "trend-200",
    suite = "extended", measure = MEASURE.gap, target = 1e-5,
    build = p -> begin
      s = signal_inputs(p; scale = 20)
      λpath = level_grid(10 * s.λ, 0.05, 6)
      merge(s, (; λpath, levels = λpath))
    end,
    call = s -> trend_filter_path(s.y, s.λpath; order = 1, x = s.x),
    certify = (p, s, out) -> path_certificate(out.solutions, length(s.λpath)) do k
      signal_gap(s, out.solutions[k], difference_operator(length(s.y), 2; x = s.x),
                 fill(s.λpath[k], length(s.y) - 2))
    end)

  # Regression with n > p, an intercept and levels relative to max|x̃ⱼ'ỹ|/n
  function regression_inputs(p; levels = 0)
    X, y = case64(p.X), case64(p.y)
    top = maximum(abs.(case_centered(X)' * (y .- mean(y)))) / size(X, 1)
    λpath = levels == 0 ? [0.05 * top] : level_grid(0.5 * top, 0.05, levels)
    (; X, y, λ = λpath[1], λpath, levels = λpath)
  end
  regression_gap(s, out, D, λrows) = genlasso_gap(s.X, s.y, out.intercept, out.x, D, λrows; intercept = true)
  add_case!(catalog, "structured", "fused_lasso", "regression"; scenario = "fusedreg-150x60",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> regression_inputs(p),
    call = s -> fused_lasso(s.X, s.y, s.λ, s.λ; intercept = true),
    certify = (p, s, out) -> regression_gap(s, out, fused_operator(size(s.X, 2), s.λ, s.λ)...))
  add_case!(catalog, "structured", "trend_filter", "regression-linear"; scenario = "fusedreg-150x60",
    suite = "extended", measure = MEASURE.gap, target = 1e-5,
    build = p -> regression_inputs(p),
    call = s -> trend_filter(s.X, s.y, s.λ; order = 1, intercept = true),
    certify = (p, s, out) -> regression_gap(s, out, difference_operator(size(s.X, 2), 2),
                                            fill(s.λ, size(s.X, 2) - 2)))
  # λ1 = λ2 = λ/2 at each level, l1_ratio = 0.5
  add_case!(catalog, "structured", "fused_lasso_path", "regression"; scenario = "fusedreg-150x60",
    suite = "extended", measure = MEASURE.gap, target = 1e-6,
    build = p -> regression_inputs(p; levels = 6),
    call = s -> fused_lasso_path(s.X, s.y, s.λpath; l1_ratio = 0.5, intercept = true),
    certify = (p, s, out) -> path_certificate(out.solutions, length(s.λpath)) do k
      λ = s.λpath[k] / 2
      regression_gap(s, out.solutions[k], fused_operator(size(s.X, 2), λ, λ)...)
    end)
end

function selection_cases!(catalog)
  # ‖y - β₀ - Xβ‖²/(2n) + ∑ⱼ λⱼ|β|₍ⱼ₎ on columns centered and scaled to mean square
  # one; the sequence is a linear decrease from 2 to 1 times a factor
  function slope_inputs(p; ratio = 0.2, levels = 0)
    X, y = case64(p.X), case64(p.y)
    base = collect(range(2.0, 1.0; length = size(X, 2)))
    top = slope_factor_max(X, y, base)
    factors = levels == 0 ? [ratio * top] : level_grid(top, 0.05, levels)
    (; X, y, base, λseq = factors[1] .* base, factors, levels = [factors; base])
  end
  slope_gap_of(s, β0, β, λseq) = slope_gap(s.X, s.y, β0, β, λseq; intercept = true, standardize_x = true)
  add_case!(catalog, "selection", "slope", "standardized"; scenario = "slope-150x60",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> slope_inputs(p),
    call = s -> slope(s.X, s.y, s.λseq; intercept = true, standardize_x = true),
    certify = (p, s, out) -> slope_gap_of(s, out.intercept, out.x, s.λseq))
  add_case!(catalog, "selection", "SLOPEPath", "screening"; scenario = "slope-150x60", suite = "extended",
    measure = MEASURE.gap, target = 1e-6,
    build = p -> slope_inputs(p; levels = 8),
    call = s -> SLOPEPath(s.X, s.y, s.factors; lambda = s.base, screening = true, intercept = true,
                          standardize_x = true),
    certify = (p, s, out) -> path_certificate(out.βpath, length(s.factors)) do k
      slope_gap_of(s, out.β0path[k], out.βpath[k], s.factors[k] .* s.base)
    end)

  function knockoff_inputs(p)
    X, y, Sigma = case64(p.X), case64(p.y), case64(p.Sigma)
    mu = haskey(p, :mu) ? case64(p.mu) : zeros(size(X, 2))
    λ = 0.1 * maximum(abs.(X' * y)) / size(X, 1)
    (; X, y, Sigma, mu, λ, q = 0.1, rng = solver_rng(), levels = [λ])
  end
  # The equicorrelated diagonal and the knockoffs drawn with it; the diagonal must
  # be feasible on the correlation scale.
  add_case!(catalog, "selection", "equicorrelated_s", "gaussian"; scenario = "knockoff-300x50", suite = "extended",
    measure = MEASURE.feasibility, target = 1e-10,
    build = p -> knockoff_inputs(p),
    call = s -> begin
      d = equicorrelated_s(s.Sigma)
      (; s = d, Xk = gaussian_knockoffs(s.X, s.mu, s.Sigma; rng = s.rng, s = d))
    end,
    certify = (p, s, out) -> knockoff_feasibility(s.Sigma, out.s))
  # Selection procedures: the selections are recorded; their post-processing (the
  # knockoff+ threshold of the returned W) is checked exactly.
  add_case!(catalog, "selection", "knockoff_filter", "gaussian"; scenario = "knockoff-300x50",
    measure = MEASURE.consistency, target = "record",
    build = p -> knockoff_inputs(p),
    call = s -> knockoff_filter(s.X, s.y, s.λ; method = :gaussian, mu = s.mu, Sigma = s.Sigma, q = s.q,
                                rng = s.rng),
    certify = (p, s, out) -> knockoff_threshold_check(case64(out.W), s.q, case64(out.threshold), 1))
  # Ten complementary pairs of lasso fits at a tenth of λmax, at most ten
  # variables each; the frequencies must be the means of the returned selections.
  add_case!(catalog, "selection", "stability_selection", "lasso"; scenario = "ls-200x50", suite = "extended",
    measure = MEASURE.consistency, target = "record",
    build = p -> begin
      X, y = case64(p.X), case64(p.y)
      λ = 0.1 * ls_lambda_max(X, y, case_loadings(X))
      (; X, y, λ, rng = solver_rng(), levels = [λ])
    end,
    call = s -> stability_selection(s.X, s.y, s.λ; q = 10, npairs = 10, intercept = true, standardize_x = true,
                                    rng = s.rng),
    certify = (p, s, out) -> frequency_check(out.selections, out.frequencies))
end

function inference_cases!(catalog)
  hdm_inputs(p) = (; X = case64(p.X), y = case64(p.y), d = case64(p.d), rng = solver_rng(), levels = Float64[])
  # The final nuisance lasso, ‖y - β₀ - Xβ‖²/(2n) + λ0∑Γⱼ|βⱼ| on the scale of
  # theory_lambda, at the level and loadings of that solve as the result returns
  # them; certification is conditional on these data-driven parameters.
  nuisance_gap(s, fit, response) = ls_l1_gap(s.X, response, fit.lasso_intercept, case64(fit.lasso),
                                             fit.lasso_λ0, case64(fit.lasso_loadings); intercept = true)
  for (variant, post, suite) in (("post", true, "quick"), ("no-post", false, "extended"))
    add_case!(catalog, "inference", "rlasso", variant; scenario = "hdm-200x80", suite,
      measure = MEASURE.gap, target = 1e-6,
      build = p -> hdm_inputs(p),
      call = s -> rlasso(s.X, s.y; post, intercept = true, rng = s.rng),
      certify = (p, s, out) -> nuisance_gap(s, out, s.y))
  end
  # Least squares of y on [1 X_S] against an independent QR solve, for the support S
  # of the coefficients that generated the data (fixed by the draw, not by the
  # package, so that revisions refit the same columns)
  add_case!(catalog, "inference", "post_lasso", "true-support"; scenario = "hdm-200x80", suite = "extended",
    measure = MEASURE.coefficients, target = 1e-8,
    build = p -> begin
      s = hdm_inputs(p)
      support = findall(!iszero, p.truth)
      merge(s, (; support, levels = Float64.(support)))
    end,
    call = s -> post_lasso(s.X, s.y, s.support; intercept = true),
    certify = (p, s, out) -> ols_coefficient_error(s.X, s.y, out.selected, out.intercept, case64(out.x);
                                                   intercept = true))
  add_case!(catalog, "inference", "double_selection", "hdm"; scenario = "hdm-200x80",
    measure = MEASURE.coefficients, target = 1e-8,
    build = p -> hdm_inputs(p),
    call = s -> double_selection(s.y, s.d, s.X; rng = s.rng),
    certify = (p, s, out) -> effect_certificate(s, out))
end

function control_cases!(catalog)
  # One pass of coordinate descent cannot reach a 1e-8 gap: this case must end
  # target_missed, with no timing-eligible sample. The warning of every
  # unconverged call is silenced.
  add_case!(catalog, "controls", "lasso", "max-iter-1"; scenario = "ls-200x50",
    measure = MEASURE.gap, target = 1e-8,
    build = p -> begin
      X, y = case64(p.X), case64(p.y)
      ω = case_loadings(X)
      λ = 0.1 * ls_lambda_max(X, y, ω)
      (; X, y, λ, ω, rng = solver_rng(), levels = [λ])
    end,
    call = s -> Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
      lasso(s.X, s.y, s.λ; intercept = true, standardize_x = true, max_iter = 1, rng = s.rng)
    end,
    certify = (p, s, out) -> ls_l1_gap(s.X, s.y, out.intercept, case64(out.x), s.λ, s.ω; intercept = true))
end

function all_cases()
  catalog = CaseSpec[]
  lasso_cases!(catalog)
  glm_cases!(catalog)
  group_cases!(catalog)
  multitask_cases!(catalog)
  lowrank_cases!(catalog)
  quantile_cases!(catalog)
  structured_cases!(catalog)
  selection_cases!(catalog)
  inference_cases!(catalog)
  control_cases!(catalog)
  ids = [c.meta.id for c in catalog]
  allunique(ids) || error("duplicate case ids")
  unknown = setdiff(unique(c.meta.scenario for c in catalog), SCENARIOS)
  isempty(unknown) || error("scenarios missing from problems.jl: $(join(unknown, ", "))")
  catalog
end

# The cases of the suite (quick cases also run in extended), the groups and the
# id patterns (none for all), sorted by id
function build_catalog(suite, groups, patterns = Regex[])
  suite in ("quick", "extended") || throw(ArgumentError("HDM_SUITE must be quick or extended"))
  all(in(CASE_GROUPS), groups) || throw(ArgumentError("unknown HDM_GROUPS entry"))
  selected = filter(all_cases()) do c
    c.meta.group in groups && (suite == "extended" || c.meta.suite == "quick") &&
      (isempty(patterns) || any(p -> occursin(p, c.meta.id), patterns))
  end
  isempty(selected) && error("no case matches HDM_SUITE, HDM_GROUPS and HDM_CASES")
  sort!(selected; by = c -> c.meta.id)
end
