# Controls for the certificates: analytic problems with closed-form values,
# designed problems whose optimum is known by construction (the data are built
# from a primal point and a dual point that satisfy the optimality conditions),
# designed nonoptimal points, weak duality and dual feasibility at random
# points, zero solutions at λ ≥ λmax, nonfinite inputs, and the reproducibility
# of the scenarios. Needs only the standard library, Test and HiGHS; the last
# testset compares with HighDimensionalModels when it can be loaded.

using Test, LinearAlgebra, Random, SparseArrays, Statistics

include(joinpath(@__DIR__, "problems.jl"))
include(joinpath(@__DIR__, "certificates.jl"))

const TIGHT = 1e-11       # a gap or residual at a solution known in closed form

# divided differences of order `order` on the locations x, and first differences,
# written independently of the package
function difference_matrix_ref(n, order, x)
  D = Matrix(1.0I, n, n)
  for k in 1:order
    m = size(D, 1) - 1
    E = zeros(m, m + 1)
    for i in 1:m
      E[i, i], E[i, i+1] = -1.0, 1.0
    end
    k > 1 && (D = Diagonal([(k - 1) / (x[i+k-1] - x[i]) for i in 1:m+1]) * D)
    D = E * D
  end
  D
end
fused_rows(p) = sparse([1:p-1; 1:p-1], [1:p-1; 2:p], [-ones(p - 1); ones(p - 1)], p - 1, p)

ok(c) = c.status === :ok
bad(c) = c.status === :uncertifiable && isnan(c.value)

# θ of least norm with X'θ = v, and 1'θ = 0 with an intercept
function dual_with(X, v; intercept::Bool)
  A = intercept ? vcat(transpose(ones(size(X, 1))), transpose(Matrix(X))) : transpose(Matrix(X))
  pinv(A) * (intercept ? [0.0; v] : v)
end

# a subgradient of ∑ⱼ(c1ⱼ|βⱼ| + c2ⱼβⱼ²/2): c1ⱼ sign(βⱼ) + c2ⱼβⱼ on the support,
# strictly inside [-c1ⱼ, c1ⱼ] off it
l1_subgradient(rng, β, c1, c2 = zero(c1)) =
  [β[j] != 0 ? c1[j] * sign(β[j]) + c2[j] * β[j] : c1[j] * (rand(rng) - 0.5) for j in eachindex(β)]

function sparse_vector(rng, p, k)
  β = zeros(p)
  β[draw_indices(rng, p, k)] .= draw_signs(rng, k) .* (0.5 .+ rand(rng, k))
  β
end

# a lasso whose solution is (β₀, β) by construction: θ = w∘r/∑w with 1'θ = 0
# and X'θ ∈ λω∘∂|β|, so y = o + β₀ + Xβ + ∑w θ./w
function designed_lasso(rng; n = 40, p = 8, intercept = true, weighted = true, λ = 0.1, zero_loading = false)
  X = randn(rng, n, p) .* transpose(0.5 .+ rand(rng, p)) .+ transpose(randn(rng, p))
  w = weighted ? 0.5 .+ rand(rng, n) : ones(n)
  o = weighted ? 0.3 .* randn(rng, n) : zeros(n)
  ω = column_loadings(X; intercept, weights = w)
  zero_loading && (ω[1] = 0)
  β = sparse_vector(rng, p, 3)
  β0 = intercept ? 0.7 : 0.0
  θ = dual_with(X, l1_subgradient(rng, β, λ .* ω); intercept)
  y = o .+ β0 .+ X * β .+ sum(w) .* θ ./ w
  (; X, y, w, o, ω, β, β0, λ, θ)
end

@testset "Relative gap, residuals and scaling" begin
  @test relative_gap(1.0, 1.0).value == 0 && ok(relative_gap(1.0, 1.0))
  @test relative_gap(2.0, 1.5).value == 0.25
  @test relative_gap(0.0, 0.0).value == 0
  @test relative_gap(1.0, nextfloat(1.0)).value == 0
  @test bad(relative_gap(1.0, 1.001))
  @test occursin("weak duality", relative_gap(1.0, 1.001).note)
  @test bad(relative_gap(Inf, 0.0)) && bad(relative_gap(1.0, NaN))
  @test ok(relative_gap(1.0, 1.0 + 1e-12; scale = 1.0, terms = 10^6))
  @test relative_residual(0.5, 0.25, :kkt_residual).value == 0.5
  @test relative_residual(0.5, 4.0, :kkt_residual).value == 0.125
  @test bad(relative_residual(NaN, 1.0, :kkt_residual))
  @test box_scale([2.0, -0.5, 0.0], [1.0, 1.0, 0.0]) == 0.5
  @test box_scale([0.5], [1.0]) == 1
  @test decreasing_root(c -> 3.0 - c, 0.0) == 3.0
  @test decreasing_root(c -> -c^3 - c + 10, 100.0) ≈ 2.0
  @test decreasing_root(c -> 1.0, 0.0) === nothing
  @test all(m -> m in CERTIFICATE_MEASURES, (:relative_gap, :kkt_residual, :fixed_point_residual,
                                              :objective_excess, :coefficient_error, :feasibility, :consistency))
end

@testset "Loadings" begin
  X = [1.0 0.0; 3.0 0.0; 5.0 2.0]
  @test column_loadings(X; intercept = true) ≈ [sqrt(8 / 3), sqrt(8 / 9)]
  @test column_loadings(X; intercept = false) ≈ [sqrt(35 / 3), sqrt(4 / 3)]
  # integer weights are replicated rows
  w = [1.0, 2.0, 3.0]
  Xr = X[[1, 2, 2, 3, 3, 3], :]
  @test column_loadings(X; intercept = true, weights = w) ≈ column_loadings(Xr; intercept = true)
  @test column_loadings(sparse(X); intercept = true, weights = w) ≈ column_loadings(X; intercept = true, weights = w)
  @test column_scales([1.0 2.0; 1.0 4.0]; intercept = true) == [1.0, 1.0]
  @test column_loadings([1.0 2.0; 1.0 4.0]; intercept = true) == [0.0, 1.0]
end

@testset "Least squares lasso" begin
  rng = scenario_rng("certificate-controls-lasso", 1)
  # X = I, n = 2: the loss is ‖y - β‖²/4, whose lasso solution is S(y, 2λ)
  X, y = Matrix(1.0I, 2, 2), [2.0, -1.0]
  @test ls_l1_gap(X, y, 0.0, [1.0, 0.0], 0.5, ones(2); intercept = false).value ≤ 1e-15
  # β = 0: P = 5/4, θ = y/2, t = 1/2, D = 1/2·5/2 - 1/4·5/4 = 15/16
  a = ls_l1_parts(X, y, 0.0, zeros(2), 0.5, ones(2); intercept = false)
  @test a.P == 1.25 && a.D == 0.9375 && a.t == 0.5
  @test ls_l1_gap(X, y, 0.0, zeros(2), 0.5, ones(2); intercept = false).value == 0.25
  # a zero loading leaves β₁ free: the KKT residual |g₁| = |(y₁ - β₁)/2| = 1/2
  c = ls_l1_gap(X, y, 0.0, [1.0, 0.0], 0.5, [0.0, 1.0]; intercept = false)
  @test c.measure === :kkt_residual && c.value == 0.5
  @test ls_l1_gap(X, y, 0.0, [2.0, 0.0], 0.5, [0.0, 1.0]; intercept = false).value == 0
  for intercept in (true, false), weighted in (true, false)
    d = designed_lasso(rng; intercept, weighted)
    c = ls_l1_gap(d.X, d.y, d.β0, d.β, d.λ, d.ω; intercept, weights = d.w, offset = d.o)
    @test ok(c) && c.measure === :relative_gap && c.value ≤ TIGHT
    @test ls_l1_gap(sparse(d.X), d.y, d.β0, d.β, d.λ, d.ω; intercept, weights = d.w, offset = d.o).value ≤ TIGHT
    a = ls_l1_parts(d.X, d.y, d.β0, d.β, d.λ, d.ω; intercept, weights = d.w, offset = d.o)
    @test a.θ ≈ d.θ atol = 1e-12
    # weak duality and dual feasibility at random points; every bound is below
    # the known optimum
    for _ in 1:5
      β, β0 = d.β .+ 0.3 .* randn(rng, length(d.β)), d.β0 + randn(rng)
      a = ls_l1_parts(d.X, d.y, β0, β, d.λ, d.ω; intercept, weights = d.w, offset = d.o)
      @test ok(ls_l1_gap(d.X, d.y, β0, β, d.λ, d.ω; intercept, weights = d.w, offset = d.o))
      @test all(abs.(transpose(d.X) * a.θ) .<= d.λ .* d.ω .* (1 + 1e-12))
      @test !intercept || abs(sum(a.θ)) ≤ 1e-14
      @test a.D ≤ a.P && a.D ≤ ls_l1_parts(d.X, d.y, d.β0, d.β, d.λ, d.ω; intercept, weights = d.w, offset = d.o).P + 1e-13
    end
    zl = designed_lasso(rng; intercept, weighted, zero_loading = true)
    c = ls_l1_gap(zl.X, zl.y, zl.β0, zl.β, zl.λ, zl.ω; intercept, weights = zl.w, offset = zl.o)
    @test c.measure === :kkt_residual && c.value ≤ TIGHT
    @test ls_l1_gap(zl.X, zl.y, zl.β0, 0.5 .* zl.β, zl.λ, zl.ω; intercept, weights = zl.w, offset = zl.o).value > 1e-3
  end
  # integer weights are replicated rows; an offset is a shift of y
  d = designed_lasso(rng; weighted = false)
  w = Float64.(rand(1:3, 40))
  rows = reduce(vcat, [fill(i, Int(w[i])) for i in 1:40])
  β = d.β .+ 0.1
  @test ls_l1_gap(d.X, d.y, 0.5, β, d.λ, d.ω; intercept = true, weights = w).value ≈
        ls_l1_gap(d.X[rows, :], d.y[rows], 0.5, β, d.λ, d.ω; intercept = true).value rtol = 1e-10
  o = randn(rng, 40)
  @test ls_l1_gap(d.X, d.y, 0.5, β, d.λ, d.ω; intercept = true, offset = o).value ≈
        ls_l1_gap(d.X, d.y .- o, 0.5, β, d.λ, d.ω; intercept = true).value rtol = 1e-12
  # at λ ≥ λmax the solution is zero, with the weighted mean as intercept
  X, y, w = d.X, d.y, 0.5 .+ rand(rng, 40)
  ω = column_loadings(X; intercept = true, weights = w)
  ỹ = y .- dot(w, y) / sum(w)
  λmax = maximum(abs.(transpose(X) * (w .* ỹ)) ./ (sum(w) .* ω))
  @test ls_l1_gap(X, y, dot(w, y) / sum(w), zeros(8), λmax, ω; intercept = true, weights = w).value ≤ TIGHT
  @test ls_l1_gap(X, y, dot(w, y) / sum(w), zeros(8), 0.9λmax, ω; intercept = true, weights = w).value > 1e-4
end

@testset "Square-root lasso" begin
  rng = scenario_rng("certificate-controls-sqrt", 1)
  X, y = Matrix(1.0I, 2, 2), [3.0, 4.0]
  # β = 0 is optimal for λ ≥ ‖X'y‖∞/‖y‖ = 0.8: P = D = 5
  @test sqrt_l1_gap(X, y, 0.0, zeros(2), 2.0, ones(2); intercept = false).value == 0
  # λ = 0.5: t = 0.5/0.8, D = 5t = 3.125
  @test sqrt_l1_gap(X, y, 0.0, zeros(2), 0.5, ones(2); intercept = false).value ≈ 0.375
  @test bad(sqrt_l1_gap(X, y, 0.0, y, 0.5, ones(2); intercept = false))
  @test bad(sqrt_l1_gap(X, y, 0.0, y .- 1e-12, 0.5, ones(2); intercept = false))
  for intercept in (true, false), weighted in (true, false)
    # optimal by construction: θ = w∘r_c/‖r_c‖_w with ∑θ²/w = 1, 1'θ = 0 and
    # X'θ ∈ λ′ω∘∂|β|, so r = ρθ./w for any ρ > 0; λ′ is set by normalizing θ
    n, p = 40, 6
    X = randn(rng, n, p) .+ transpose(randn(rng, p))
    w = weighted ? 0.5 .+ rand(rng, n) : ones(n)
    ω = column_loadings(X; intercept, weights = w)
    β, β0 = sparse_vector(rng, p, 2), intercept ? -0.4 : 0.0
    θ = dual_with(X, l1_subgradient(rng, β, ω); intercept)
    κ = sqrt(sum(θ .^ 2 ./ w))
    θ ./= κ
    λ′ = 1 / κ
    y = β0 .+ X * β .+ 2.0 .* θ ./ w
    c = sqrt_l1_gap(X, y, β0, β, λ′, ω; intercept, weights = w)
    @test ok(c) && c.value ≤ TIGHT
    for _ in 1:4
      b = β .+ 0.2 .* randn(rng, p)
      a = sqrt_l1_parts(X, y, β0 + 0.1, b, λ′, ω; intercept, weights = w)
      @test sum(a.θ .^ 2 ./ w) ≤ 1 + 1e-12 && all(abs.(transpose(X) * a.θ) .<= λ′ .* ω .* (1 + 1e-12))
      @test a.D ≤ sqrt_l1_parts(X, y, β0, β, λ′, ω; intercept, weights = w).P + 1e-13
    end
  end
end

@testset "Refit stationarity and least-squares coefficients" begin
  rng = scenario_rng("certificate-controls-ols", 1)
  n, p = 50, 10
  X = randn(rng, n, p)
  S = [2, 5, 7]
  A = hcat(ones(n), X[:, S])
  e = (I - A * pinv(A)) * randn(rng, n)
  c = [0.3, 1.0, -2.0, 0.5]
  y = A * c .+ e
  β = zeros(p)
  β[S] .= c[2:end]
  @test ls_refit_stationarity(X, y, c[1], β; intercept = true).value ≤ 1e-14
  @test ols_coefficient_error(X, y, S, c[1], β; intercept = true).value ≤ 1e-13
  βp = copy(β)
  βp[5] += 0.02
  @test ols_coefficient_error(X, y, S, c[1], βp; intercept = true).value ≈ 0.01 rtol = 1e-9
  βp = copy(β)
  βp[1] = 0.04
  @test ols_coefficient_error(X, y, S, c[1], βp; intercept = true).value ≈ 0.02 rtol = 1e-9
  @test ls_refit_stationarity(X, y, c[1] + 0.1, β; intercept = true).value > 1e-3
  # without an intercept, the refit is on X_S alone
  y0 = X[:, S] * c[2:end] .+ (I - X[:, S] * pinv(X[:, S])) * randn(rng, n)
  @test ls_refit_stationarity(X, y0, 0.0, β; intercept = false).value ≤ 1e-14
  @test ols_coefficient_error(X, y0, S, 0.0, β; intercept = false).value ≤ 1e-13
  Xd = hcat(X, X[:, 2])
  @test bad(ols_coefficient_error(Xd, y, [2, 11], 0.0, zeros(11); intercept = true))
  @test ols_coefficient_error(X, y, Int[], mean(y), zeros(p); intercept = true).value ≤ 1e-14
end

# a penalized GLM whose solution is (β₀, β) by construction: u = w∘(y - μ)/∑w
# with 1'u = 0 and X'u ∈ ∂G(β), so y = μ + ∑w u./w; u is linear in λ, which is
# chosen so that y stays inside the domain (proportions in (0, 1) for logistic
# regression)
function designed_glm(rng, family; n = 60, p = 6, intercept = true, alpha = 1.0, zero_loading = false)
  X = randn(rng, n, p) .* transpose(0.5 .+ rand(rng, p)) .+ transpose(0.5 .* randn(rng, p))
  w, o = 0.5 .+ rand(rng, n), 0.2 .* randn(rng, n)
  ω = alpha == 1 ? column_loadings(X; intercept, weights = w) : column_scales(X; intercept, weights = w)
  zero_loading && (ω[1] = 0)
  β = 0.4 .* sparse_vector(rng, p, 2)
  β0 = intercept ? 0.2 : 0.0
  μ = cert_mean.(family, β0 .+ X * β .+ o)
  u1 = dual_with(X, l1_subgradient(rng, β, alpha .* ω, (1 - alpha) .* ω .^ 2); intercept)
  margin = family === :logistic ? min.(μ, 1 .- μ) : μ
  λ = min(0.05, 0.5 * minimum(margin ./ abs.(sum(w) .* u1 ./ w)))
  u = λ .* u1
  (; X, y = μ .+ sum(w) .* u ./ w, w, o, ω, β, β0, λ, u, alpha)
end

@testset "Generalized linear models" begin
  rng = scenario_rng("certificate-controls-glm", 1)
  # one observation, y = 1, x = 1: λ = 1/4 at β = log 3 is optimal
  X1, y1 = ones(1, 1), [1.0]
  optimum = log(4.0) - 0.75log(3.0)
  a = glm_parts(:logistic, X1, y1, 0.0, [log(3.0)], 0.25, [1.0]; intercept = false)
  @test a.P ≈ optimum && a.D ≈ optimum
  @test glm_l1_gap(:logistic, X1, y1, 0.0, [0.0], 0.25, [1.0]; intercept = false).value ≈ 1 - optimum / log(2.0)
  # Poisson, y = 2, x = 1: the optimum of exp(β) - 2β + λ|β| is β = log(2 - λ)
  a = glm_parts(:poisson, X1, [2.0], 0.0, [log(1.5)], 0.5, [1.0]; intercept = false)
  @test a.P ≈ 1.5 - 1.5log(1.5) && a.D ≈ a.P
  for family in (:logistic, :poisson), intercept in (true, false), alpha in (1.0, 0.5)
    d = designed_glm(rng, family; intercept, alpha)
    @test cert_glm_domain(family, d.y) && (family === :poisson || all(<(1), d.y))
    kw = (; alpha, intercept, weights = d.w, offset = d.o)
    c = glm_l1_gap(family, d.X, d.y, d.β0, d.β, d.λ, d.ω; kw...)
    @test ok(c) && c.measure === :relative_gap && c.value ≤ TIGHT
    a = glm_parts(family, d.X, d.y, d.β0, d.β, d.λ, d.ω; kw...)
    @test a.u ≈ d.u atol = 1e-12
    Pstar = a.P
    for _ in 1:4
      β, β0 = d.β .+ 0.2 .* randn(rng, length(d.β)), d.β0 + 0.3randn(rng)
      a = glm_parts(family, d.X, d.y, β0, β, d.λ, d.ω; kw...)
      @test ok(glm_l1_gap(family, d.X, d.y, β0, β, d.λ, d.ω; kw...))
      # the intercept constraint and the polar of the ℓ1 part
      @test !intercept || abs(sum(a.u)) ≤ 1e-14
      alpha == 1 && @test all(abs.(transpose(d.X) * a.u) .<= d.λ .* d.ω .* (1 + 1e-12))
      @test a.D ≤ Pstar + 1e-13 && a.D ≤ a.P
    end
    zl = designed_glm(rng, family; intercept, alpha, zero_loading = true)
    c = glm_l1_gap(family, zl.X, zl.y, zl.β0, zl.β, zl.λ, zl.ω; alpha, intercept, weights = zl.w, offset = zl.o)
    @test c.measure === :kkt_residual && c.value ≤ TIGHT
  end
  # at λ ≥ λmax the solution is zero with the null intercept
  d = designed_glm(rng, :logistic)
  c0 = glm_intercept(:logistic, d.o, d.y, d.w, 0.0)
  g = transpose(d.X) * (d.w .* (cert_mean.(:logistic, c0 .+ d.o) .- d.y)) ./ sum(d.w)
  λmax = maximum(abs.(g) ./ d.ω)
  kw = (; intercept = true, weights = d.w, offset = d.o)
  @test glm_l1_gap(:logistic, d.X, d.y, c0, zeros(6), λmax, d.ω; kw...).value ≤ TIGHT
  @test glm_l1_gap(:logistic, d.X, d.y, c0, zeros(6), 0.8λmax, d.ω; kw...).value > 1e-5
  # responses all at a bound have no intercept optimum; others are outside the domain
  @test bad(glm_l1_gap(:logistic, d.X, ones(60), 0.0, zeros(6), 0.1, d.ω; intercept = true))
  @test bad(glm_l1_gap(:poisson, d.X, zeros(60), 0.0, zeros(6), 0.1, d.ω; intercept = true))
  @test bad(glm_l1_gap(:logistic, d.X, fill(2.0, 60), 0.0, zeros(6), 0.1, d.ω; intercept = false))
  @test bad(glm_l1_gap(:binomial, d.X, d.y, 0.0, zeros(6), 0.1, d.ω; intercept = false))
end

@testset "MCP and SCAD stationarity" begin
  rng = scenario_rng("certificate-controls-nonconvex", 1)
  @test cert_mcp_prox(1.0, 0.5, 3.0, 1.0) ≈ 0.75
  @test cert_mcp_prox(0.4, 0.5, 3.0, 1.0) == 0 && cert_mcp_prox(2.0, 0.5, 3.0, 1.0) == 2.0
  @test cert_scad_prox(1.5, 0.5, 4.0, 1.0) == 1.25
  @test cert_scad_prox(0.8, 0.5, 4.0, 1.0) ≈ 0.3
  mcp′(t, λ, a) = sign(t) * max(λ - abs(t) / a, 0.0)
  scad′(t, λ, a) = sign(t) * (abs(t) <= λ ? λ : max(a * λ - abs(t), 0.0) / (a - 1))
  for (penalty, a, deriv) in ((:mcp, 3.0, mcp′), (:scad, 3.7, scad′)), family in (:gaussian, :logistic),
      intercept in (true, false)
    # stationary by construction: X'θ = ω∘pen′(ω∘β) on the support, |·| < λω off it
    n, p = 60, 6
    λ = family === :gaussian ? 0.1 : 0.02
    X = randn(rng, n, p) .+ transpose(0.3 .* randn(rng, p))
    w, o = 0.5 .+ rand(rng, n), 0.2 .* randn(rng, n)
    ω = column_scales(X; intercept, weights = w)
    β = zeros(p)
    # b = ω∘β below λ, between λ and aλ, and beyond aλ
    β[1], β[2], β[3] = 0.5λ / ω[1], -2λ / ω[2], (a * λ + 0.5) / ω[3]
    b = ω .* β
    v = [b[j] != 0 ? ω[j] * deriv(b[j], λ, a) : ω[j] * λ * (rand(rng) - 0.5) for j in 1:p]
    u = dual_with(X, v; intercept)
    β0 = intercept ? 0.1 : 0.0
    η = β0 .+ X * β .+ o
    y = family === :gaussian ? η .+ sum(w) .* u ./ w : cert_mean.(:logistic, η) .+ sum(w) .* u ./ w
    family === :logistic && @test all(t -> 0 < t < 1, y)
    kw = (; intercept, weights = w, offset = o)
    c = nonconvex_stationarity(family, penalty, X, y, β0, β, λ, a, ω; kw...)
    @test ok(c) && c.measure === :fixed_point_residual && c.value ≤ TIGHT
    @test nonconvex_stationarity(family, penalty, X, y, β0, zeros(p), λ, a, ω; kw...).value > 1e-4
  end
  X, y = randn(rng, 20, 3), randn(rng, 20)
  @test bad(nonconvex_stationarity(:gaussian, :scad, X, y, 0.0, zeros(3), 0.1, 2.0, ones(3); intercept = false))
  @test bad(nonconvex_stationarity(:gaussian, :mcp, X, y, 0.0, zeros(3), 0.1, 3.0, [1.0, 0.0, 1.0]; intercept = false))
  @test bad(nonconvex_stationarity(:gaussian, :lasso, X, y, 0.0, zeros(3), 0.1, 3.0, ones(3); intercept = false))
end

@testset "Group lasso" begin
  rng = scenario_rng("certificate-controls-group", 1)
  # X = I, n = 2, one group: ‖y - β‖²/4 + λ‖β‖ at β = (1 - 2λ/‖y‖)y
  X, y = Matrix(1.0I, 2, 2), [3.0, 4.0]
  @test group_gap(X, y, 0.0, 0.8 .* y, 0.5, [[1, 2]], [1.0]; intercept = false, orthonormal = false).value ≤ 1e-15
  # β = 0: P = 25/4, θ = y/2 of norm 5/2, t = 1/5, D = (1/5)(25/2) - (1/25)(25/4) = 9/4
  @test group_gap(X, y, 0.0, zeros(2), 0.5, [[1, 2]], [1.0]; intercept = false, orthonormal = false).value ≈ 0.64
  n, p = 50, 12
  groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10], [11, 12]]
  for orthonormal in (true, false), intercept in (true, false)
    X = randn(rng, n, p) .+ transpose(randn(rng, p))
    orthonormal && (X[:, 6] .= X[:, 4] .- 2 .* X[:, 5])     # a rank-deficient group
    wg, λ = sqrt.(length.(groups)), 0.05
    β = zeros(p)
    β[groups[2]] .= randn(rng, 3)
    β[groups[3]] .= randn(rng, 4)
    xbar = intercept ? vec(mean(X; dims = 1)) : zeros(p)
    v = zeros(p)
    for (k, G) in enumerate(groups)
      XcG = X[:, G] .- transpose(xbar[G])
      if any(!iszero, β[G])
        fitG = XcG * β[G]
        v[G] .= orthonormal ? λ * wg[k] .* (transpose(XcG) * fitG) ./ (sqrt(n) * norm(fitG)) :
                              λ * wg[k] .* β[G] ./ norm(β[G])
      else
        e = randn(rng, orthonormal ? n : length(G))
        e .*= 0.5 / norm(e)
        v[G] .= orthonormal ? λ * wg[k] .* (transpose(XcG) * e) ./ sqrt(n) : λ * wg[k] .* e
      end
    end
    θ = dual_with(X, v; intercept)
    β0 = intercept ? 1.5 : 0.0
    y = β0 .+ X * β .+ n .* θ
    kw = (; intercept, orthonormal)
    c = group_gap(X, y, β0, β, λ, groups, wg; kw...)
    @test ok(c) && c.value ≤ TIGHT
    Pstar = group_parts(X, y, β0, β, λ, groups, wg; kw...).P
    for _ in 1:4
      b = β .+ 0.2 .* randn(rng, p)
      a = group_parts(X, y, β0 - 0.2, b, λ, groups, wg; kw...)
      for (k, G) in enumerate(groups)
        norms = orthonormal ? sqrt(n) * norm(transpose(a.blocks[k].Q) * a.θ) : norm(transpose(X[:, G]) * a.θ)
        @test norms ≤ λ * wg[k] * (1 + 1e-12)
      end
      @test a.D ≤ Pstar + 1e-13 && a.D ≤ a.P
    end
    # a zero group weight: the KKT residual of the whole problem
    wz = copy(wg)
    wz[1] = 0
    vz = copy(v)
    vz[groups[1]] .= 0
    θz = dual_with(X, vz; intercept)
    yz = β0 .+ X * β .+ n .* θz
    c = group_gap(X, yz, β0, β, λ, groups, wz; kw...)
    @test c.measure === :kkt_residual && c.value ≤ TIGHT
    @test group_gap(X, yz, β0, 0.5 .* β, λ, groups, wz; kw...).value > 1e-4
  end
  # at λ ≥ λmax the solution is zero (here without orthonormalization)
  X, y = randn(rng, n, p), randn(rng, n)
  yc = y .- mean(y)
  λmax = maximum(norm(transpose(X[:, G]) * yc) / (n * sqrt(length(G))) for G in groups)
  @test group_gap(X, y, mean(y), zeros(p), λmax, groups, sqrt.(length.(groups)); intercept = true,
                  orthonormal = false).value ≤ TIGHT
  @test bad(group_gap(X, y, 0.0, zeros(p), 0.1, [[1, 2], [2, 3]], [1.0, 1.0]; intercept = true, orthonormal = true))
  @test bad(group_gap(X, y, 0.0, zeros(p), 0.1, groups[1:3], ones(3); intercept = true, orthonormal = true))
end

@testset "Sparse group lasso" begin
  rng = scenario_rng("certificate-controls-sparse-group", 1)
  # one column: ‖y - β‖²/2 + λ(α|β| + (1 - α)|β|) is soft thresholding at λ
  @test sparse_group_gap(ones(1, 1), [2.0], 0.0, [1.0], 1.0, 0.5, [[1]], [1.0];
                         intercept = false, standardize_x = false).value ≤ 1e-15
  # β = 0 at y = 2: P = 2, θ = 2, t = 1/2, D = 1/2·4 - 1/4·2 = 3/2
  @test sparse_group_gap(ones(1, 1), [2.0], 0.0, [0.0], 1.0, 0.5, [[1]], [1.0];
                         intercept = false, standardize_x = false).value ≈ 0.25
  n, p = 50, 9
  groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
  for standardize_x in (true, false), intercept in (true, false)
    X = randn(rng, n, p) .* transpose(0.5 .+ rand(rng, p)) .+ transpose(randn(rng, p))
    wg, λ, α = sqrt.(length.(groups)), 0.1, 0.4
    s = standardize_x ? column_scales(X; intercept) : ones(p)
    b = zeros(p)
    b[[1, 3]] .= [0.8, -1.2]
    b[[4, 5, 6]] .= [0.5, 1.0, -0.3]
    g = zeros(p)
    for (k, G) in enumerate(groups)
      nb = norm(b[G])
      ξ = [b[j] != 0 ? sign(b[j]) : rand(rng) - 0.5 for j in G]
      e = randn(rng, length(G))
      g[G] .= λ * α .* ξ .+ λ * (1 - α) * wg[k] .* (nb > 0 ? b[G] ./ nb : 0.5 .* e ./ norm(e))
    end
    θ = dual_with(X, s .* g; intercept)
    β0 = intercept ? -0.5 : 0.0
    β = b ./ s
    y = β0 .+ X * β .+ n .* θ
    kw = (; intercept, standardize_x)
    c = sparse_group_gap(X, y, β0, β, λ, α, groups, wg; kw...)
    @test ok(c) && c.value ≤ TIGHT
    Pstar = sparse_group_parts(X, y, β0, β, λ, α, groups, wg; kw...).P
    for _ in 1:4
      a = sparse_group_parts(X, y, β0, β .+ 0.3 .* randn(rng, p), λ, α, groups, wg; kw...)
      z = (transpose(X) * a.θ) ./ a.s
      @test all(norm(cert_soft.(z[G], λ * α)) ≤ λ * (1 - α) * wg[k] * (1 + 1e-12) for (k, G) in enumerate(groups))
      @test a.D ≤ Pstar + 1e-13 && a.D ≤ a.P
    end
    c = sparse_group_gap(X, y, β0, β, 0.0, α, groups, wg; kw...)
    @test c.measure === :kkt_residual && c.value > 0
  end
end

@testset "Multiple responses, reduced rank and nuclear norm" begin
  rng = scenario_rng("certificate-controls-multi", 1)
  n, p, q = 40, 6, 3
  for intercept in (true, false)
    X = randn(rng, n, p) .+ transpose(randn(rng, p))
    ω, λ = column_loadings(X; intercept), 0.1
    B = zeros(p, q)
    B[[2, 5], :] .= randn(rng, 2, q)
    G = zeros(p, q)
    for k in 1:p
      e = randn(rng, q)
      G[k, :] .= λ * ω[k] .* (any(!iszero, B[k, :]) ? B[k, :] ./ norm(B[k, :]) : 0.5 .* e ./ norm(e))
    end
    Θ = reduce(hcat, [dual_with(X, G[:, j]; intercept) for j in 1:q])
    β0 = intercept ? randn(rng, q) : zeros(q)
    Y = transpose(β0) .+ X * B .+ n .* Θ
    c = multi_gap(X, Y, β0, B, λ, ω; intercept)
    @test ok(c) && c.value ≤ TIGHT
    Pstar = multi_parts(X, Y, β0, B, λ, ω; intercept).P
    for _ in 1:3
      a = multi_parts(X, Y, β0, B .+ 0.2 .* randn(rng, p, q), λ, ω; intercept)
      @test all(norm((transpose(X) * a.Θ)[k, :]) ≤ λ * ω[k] * (1 + 1e-12) for k in 1:p)
      @test a.D ≤ Pstar + 1e-13
    end
    # the nuclear norm: X̃'Θ = λ(UV' + W), W orthogonal to U and V with ‖W‖ = 1/2
    F = Matrix(qr(randn(rng, p, p)).Q)
    H = Matrix(qr(randn(rng, q, q)).Q)
    U, V = F[:, 1:2], H[:, 1:2]
    Bn = U * Diagonal([2.0, 0.7]) * transpose(V)
    Gn = λ .* (U * transpose(V) .+ 0.5 .* F[:, 3] * transpose(H[:, 3]))
    Θn = reduce(hcat, [dual_with(X, Gn[:, j]; intercept) for j in 1:q])
    Yn = transpose(β0) .+ X * Bn .+ n .* Θn
    c = nuclear_gap(X, Yn, β0, Bn, λ; intercept)
    @test ok(c) && c.value ≤ TIGHT
    Pstar = nuclear_parts(X, Yn, β0, Bn, λ; intercept).P
    for _ in 1:3
      a = nuclear_parts(X, Yn, β0, Bn .+ 0.2 .* randn(rng, p, q), λ; intercept)
      @test opnorm(transpose(X) * a.Θ) ≤ λ * (1 + 1e-12) && a.D ≤ Pstar + 1e-13
    end
    # reduced rank: Y = 1β₀' + XB + E with E orthogonal to [1 X] has L* = ‖E‖²
    # at the true rank, and L* = ‖E‖² + σ₂(X̃B)² at rank one
    Br = randn(rng, p, 2) * randn(rng, 2, q)
    A = intercept ? hcat(ones(n), X) : X
    E = (I - A * pinv(A)) * randn(rng, n, q)
    Yr = transpose(β0) .+ X * Br .+ E
    @test rrr_excess(X, Yr, β0, Br, 2; intercept).value ≤ TIGHT
    Xc = intercept ? X .- mean(X; dims = 1) : X
    Yc = intercept ? Yr .- mean(Yr; dims = 1) : Yr
    @test rrr_excess(X, Yr, vec(mean(Yr; dims = 1)) .* intercept, zeros(p, q), 2; intercept).value ≈
          sum(abs2, Xc * Br) / sum(abs2, E) rtol = 1e-10
    S = svd(Xc * Br)
    B1 = pinv(Xc) * (S.S[1] .* S.U[:, 1] * transpose(S.V[:, 1]))
    c = rrr_excess(X, Yr, intercept ? vec(mean(Yr; dims = 1) .- mean(X; dims = 1) * B1) : zeros(q), B1, 1; intercept)
    @test ok(c) && c.value ≤ TIGHT
    @test bad(rrr_excess(X, Yr, β0, Br, 1; intercept))
    @test occursin("rank", rrr_excess(X, Yr, β0, Br, 1; intercept).note)
  end
  # a Float32 fit of rank 3: B keeps singular values at the Float32 rounding
  # level, which may lower the loss below the rank-3 optimum by that order
  a = scenario("lowrank-150x30x10", 1)
  X32, Y32 = Float32.(a.X), Float32.(a.Y)
  mx, my = mean(X32; dims = 1), mean(Y32; dims = 1)
  F = svd(X32 .- mx)
  H = svd(transpose(F.U) * (Y32 .- my))
  B32 = (F.V * (H.U[:, 1:3] ./ F.S)) * (H.S[1:3] .* H.Vt[1:3, :])
  @test eltype(B32) === Float32
  c = rrr_excess(X32, Y32, vec(my .- mx * B32), B32, 3; intercept = true)
  @test ok(c) && c.value ≤ 1e-4
  @test bad(rrr_excess(X32, Y32, vec(my .- mx * B32), B32, 2; intercept = true))
  # a Float32 B of numerical rank 3 whose fourth singular value, at the Float32
  # rounding level, lowers the loss below the rank-3 optimum by more than
  # Float64 rounding: the precision of the inputs sets the tolerance
  Xd, Yd = Float64.(X32), Float64.(Y32)
  mxd, myd = mean(Xd; dims = 1), mean(Yd; dims = 1)
  Fd = svd(Xd .- mxd)
  Hd = svd(transpose(Fd.U) * (Yd .- myd))
  component(k) = (Fd.V * (Hd.U[:, k] ./ Fd.S)) * (Hd.S[k] .* transpose(Hd.Vt[k, :]))
  B3 = component(1) .+ component(2) .+ component(3)
  Bε = Float32.(B3 .+ (5e-6 * opnorm(B3) / opnorm(component(4))) .* component(4))
  loss(Bm) = sum(abs2, (Yd .- myd) .- (Xd .- mxd) * Bm)
  @test loss(Float64.(Bε)) < loss(B3) * (1 - 1e-10)
  c = rrr_excess(X32, Y32, vec(Float32.(myd .- mxd * Float64.(Bε))), Bε, 3; intercept = true)
  @test ok(c) && c.value ≤ 1e-6
  # a wide design fits Y exactly at full rank: L* = 0, and the floor of the
  # denominator keeps the excess of the least-norm fit at rounding level
  Xw, Yw = randn(rng, 10, 20), randn(rng, 10, 3)
  Bw = pinv(Xw .- mean(Xw; dims = 1)) * (Yw .- mean(Yw; dims = 1))
  c = rrr_excess(Xw, Yw, vec(mean(Yw; dims = 1) .- mean(Xw; dims = 1) * Bw), Bw, 3; intercept = true)
  @test ok(c) && c.value ≤ 1e-6
end

# The primal LP of quantile regression, solved by HiGHS: variables β⁺, β⁻ ≥ 0,
# β₀ free, r⁺, r⁻ ≥ 0, with β₀ + X(β⁺ - β⁻) + r⁺ - r⁻ = y.
function quantile_primal(X, y, λ, w, τ; intercept)
  n, p = size(X)
  A = sparse(hcat(X, -X, ones(n, intercept), Matrix(1.0I, n, n), -Matrix(1.0I, n, n)))
  ncol = size(A, 2)
  cost = [λ .* w; λ .* w; zeros(intercept); fill(τ / n, n); fill((1 - τ) / n, n)]
  lower = [zeros(2p); fill(-Inf, intercept); zeros(2n)]
  upper = fill(Inf, ncol)
  h = HiGHS.Highs_create()
  col = zeros(ncol)
  try
    HiGHS.Highs_setBoolOptionValue(h, "output_flag", 0)
    HiGHS.Highs_setIntOptionValue(h, "threads", HIGHS_THREADS)
    HiGHS.Highs_setDoubleOptionValue(h, "primal_feasibility_tolerance", 1e-10)
    HiGHS.Highs_setDoubleOptionValue(h, "dual_feasibility_tolerance", 1e-10)
    HiGHS.Highs_passLp(h, ncol, n, nnz(A), HiGHS.kHighsMatrixFormatColwise, HiGHS.kHighsObjSenseMinimize, 0.0,
                       cost, lower, upper, y, y, Cint.(A.colptr[1:end-1] .- 1), Cint.(rowvals(A) .- 1), nonzeros(A))
    HiGHS.Highs_run(h)
    @assert HiGHS.Highs_getModelStatus(h) == HiGHS.kHighsModelStatusOptimal
    HiGHS.Highs_getSolution(h, col, zeros(ncol), zeros(n), zeros(n))
  finally
    HiGHS.Highs_destroy(h)
  end
  (; β = col[1:p] .- col[p+1:2p], β0 = intercept ? col[2p+1] : 0.0)
end

@testset "Exact quantile regression" begin
  rng = scenario_rng("certificate-controls-quantile", 1)
  # intercept only: the median 3 of 1:5 has P = mean|y - 3|/2 = 0.6 = D
  Xq, yq = reshape([1.0, -1.0, 0.0, 2.0, -2.0], 5, 1), collect(1.0:5.0)
  a = quantile_parts(Xq, yq, 3.0, [0.0], 10.0, [1.0], 0.5; intercept = true)
  @test a.P ≈ 0.6 && a.D ≈ 0.6 && a.Z == [3]
  @test quantile_exact_gap(Xq, yq, 2.0, [0.0], 10.0, [1.0], 0.5; intercept = true).value > 0.1
  for intercept in (true, false), τ in (0.3, 0.5)
    n, p = 60, 5
    X = randn(rng, n, p) .+ transpose(randn(rng, p))
    y = 1.0 .+ X * [1.0, 0.0, -0.5, 0.0, 0.0] .+ laplace(rng, n)
    w, λ = column_loadings(X; intercept), 0.05
    o = 0.1 .* randn(rng, n)
    s = quantile_primal(X, y .- o, λ, w, τ; intercept)
    c = quantile_exact_gap(X, y, s.β0, s.β, λ, w, τ; intercept, offset = o)
    @test ok(c) && c.value ≤ 1e-9 && isempty(c.note)
    Pstar = quantile_parts(X, y, s.β0, s.β, λ, w, τ; intercept, offset = o).P
    for _ in 1:4
      a = quantile_parts(X, y, s.β0 + 0.1randn(rng), s.β .+ 0.1 .* randn(rng, p), λ, w, τ; intercept, offset = o)
      @test all(t -> τ - 1 <= t <= τ, a.a)
      @test !intercept || abs(sum(a.a)) ≤ 1e-13
      @test all(abs.(transpose(X) * a.a) ./ n .<= λ .* w .* (1 + 1e-12))
      @test a.D ≤ Pstar + 1e-13 && a.D ≤ a.P
    end
  end
  # λ ≥ λmax: zero slopes at the τ-quantile, whose dual coordinate balances ∑a = 0
  n, τ = 41, 0.3
  X, y = randn(rng, n, 4), randn(rng, n)
  k = ceil(Int, n * τ)
  q = sort(y)[k]
  a = [t > q ? τ : t < q ? τ - 1 : 0.0 for t in y]
  a[findfirst(==(q), y)] = -sum(a)
  λmax = maximum(abs.(transpose(X) * a) ./ n)
  @test quantile_exact_gap(X, y, q, zeros(4), λmax, ones(4), τ; intercept = true).value ≤ TIGHT
  @test bad(quantile_exact_gap(X, y, q, zeros(4), λmax, [1.0, 0.0, 1.0, 1.0], τ; intercept = true))
  @test bad(quantile_exact_gap(X, y, q, zeros(4), λmax, ones(4), 1.0; intercept = true))
end

@testset "Smoothed quantile regression" begin
  rng = scenario_rng("certificate-controls-smooth", 1)
  τ, h = 0.3, 0.4
  # the closed form against quadrature of ρ_τ(u - v) against the kernel
  kernel(v) = abs(v) < 1 ? 0.75 * (1 - v^2) : 0.0
  quadrature(u) = (m = 20_000; sum(cert_check(u - h * v, τ) * kernel(v) for v in range(-1 + 1 / m, 1 - 1 / m; length = m)) * 2 / m)
  for u in (-0.5, -0.4, -0.3, -0.05, 0.0, 0.1, 0.39, 0.6)
    @test cert_conv_check(u, τ, h) ≈ quadrature(u) atol = 1e-7
    δ = 1e-6
    @test cert_conv_score(u, τ, h) ≈ (cert_conv_check(u + δ, τ, h) - cert_conv_check(u - δ, τ, h)) / 2δ atol = 1e-6
  end
  @test cert_conv_score(0.0, τ, h) ≈ τ - 0.5
  @test cert_conv_score(h, τ, h) == τ && cert_conv_score(-h, τ, h) == τ - 1
  # stationary by construction: the scores ψ(r) of the residuals have
  # X'ψ/n ∈ λw∘∂|β| (and 1'ψ = 0 with an intercept); r inverts the score
  for intercept in (true, false)
    n, p, λ = 80, 4, 0.02
    X = randn(rng, n, p) .+ transpose(0.5 .* randn(rng, p))
    w = column_loadings(X; intercept)
    β = [0.6, 0.0, -0.4, 0.0]
    ψ = dual_with(X, n .* l1_subgradient(rng, β, λ .* w); intercept)
    @test all(t -> τ - 1 < t < τ, ψ)
    r = [decreasing_root(v -> s - cert_conv_score(v, τ, h), 0.0) for s in ψ]
    @test all(i -> cert_conv_score(r[i], τ, h) ≈ ψ[i], 1:n)
    β0 = intercept ? 0.5 : 0.0
    y = β0 .+ X * β .+ r
    c = quantile_smooth_residual(X, y, β0, β, λ, w, τ, h; intercept)
    @test ok(c) && c.measure === :fixed_point_residual && c.value ≤ TIGHT
    @test quantile_smooth_residual(X, y, β0, zeros(p), λ, w, τ, h; intercept).value > 1e-4
  end
  @test bad(quantile_smooth_residual(ones(3, 1), ones(3), 0.0, [0.0], 0.1, [1.0], 0.3, 0.0; intercept = false))
end

@testset "Generalized lasso" begin
  rng = scenario_rng("certificate-controls-genlasso", 1)
  # the two-point fused lasso, y = (0.5, -0.5), whose optimum is β = 0
  y2, D2 = [0.5, -0.5], [1.0 0.0; 0.0 1.0; -1.0 1.0]
  for (λ1, λ2) in ((0.1, 1.0), (0.05, 0.5)), X in (nothing, Matrix(1.0I, 2, 2))
    λr = [λ1, λ1, λ2]
    @test genlasso_gap(X, y2, 0.0, zeros(2), D2, λr; intercept = false).value ≤ 1e-15
    a = genlasso_parts(X, y2, 0.0, zeros(2), D2, λr; intercept = false)
    @test all(abs.(a.u) .<= λr) && a.D ≈ 0.125
  end
  # with the loss ‖y - β‖²/(2n) the first levels leave the least-norm dual inside
  # the box; at the second, the least-norm dual clipped to the box is not tight
  g = y2 ./ 2
  for (λ1, λ2, tight) in ((0.1, 1.0, true), (0.05, 0.5, false))
    λr = [λ1, λ1, λ2]
    u = clamp.(D2 * ((transpose(D2) * D2) \ g), -λr, λr)
    z = 2 .* (y2 ./ 2 .- transpose(D2) * u)
    bound = sum(abs2, y2 .- z) / 4 + dot(u, D2 * z)
    @test (0.125 - bound ≤ 1e-15) == tight
  end
  @test genlasso_gap(nothing, y2, 0.0, [0.2, -0.2], D2, [0.1, 0.1, 1.0]; intercept = false).value > 1e-2
  # designed optima: a piecewise-linear signal under second differences, and
  # stacked fused rows [I; D₁] in regression, with and without an intercept
  # (200 irregular locations, as in trend-200, make DD' ill-conditioned: its
  # smallest eigenvalue is near 1e-7, so an inexact dual solve shows here)
  n = 200
  x = cumsum(0.5 .+ rand(rng, n))
  Dt = sparse(difference_matrix_ref(n, 2, x))
  βt = [max(t - x[60], 0.0) - 2max(t - x[130], 0.0) for t in x]
  for (X, β, D, intercept) in ((nothing, βt, Dt, false),
                               (randn(rng, 80, 10), [0, 0, 1, 1, 1, 0, 0, -1, -1, 0.0], [sparse(1.0I, 10, 10); fused_rows(10)], false),
                               (randn(rng, 80, 10), [0, 0, 1, 1, 1, 0, 0, -1, -1, 0.0], [sparse(1.0I, 10, 10); fused_rows(10)], true))
    m = size(D, 1)
    # the signal's dual is smooth off the kinks, as trend-filtering duals are,
    # which weights the directions of the smallest eigenvalues of DD'
    X === nothing && @test eigmin(Symmetric(Matrix(D * transpose(D)))) < 1e-5
    λr = X === nothing ? fill(0.02, m) : 0.02 .* (0.5 .+ rand(rng, m))
    Dβ = D * β
    interior(k) = X === nothing ? 0.6 * λr[k] * cos(π * k / m) : 0.5 * λr[k] * (2rand(rng) - 1)
    u = [abs(Dβ[k]) > 1e-12 ? λr[k] * sign(Dβ[k]) : interior(k) for k in 1:m]
    nobs = X === nothing ? n : size(X, 1)
    if X === nothing
      y = β .+ nobs .* (transpose(D) * u)
      β0 = 0.0
    else
      Xc = intercept ? X .- mean(X; dims = 1) : X
      β0 = intercept ? 0.3 : 0.0
      y = β0 .+ X * β .+ nobs .* (Xc * ((transpose(Xc) * Xc) \ (transpose(D) * u)))
    end
    c = genlasso_gap(X, y, β0, β, D, λr; intercept)
    @test ok(c) && c.value ≤ TIGHT
    Pstar = genlasso_parts(X, y, β0, β, D, λr; intercept).P
    for _ in 1:3
      a = genlasso_parts(X, y, β0, β .+ 0.05 .* randn(rng, length(β)), D, λr; intercept)
      @test all(abs.(a.u) .<= λr) && a.D ≤ Pstar + 1e-13 && a.D ≤ a.P
    end
  end
  Xr = randn(rng, 20, 4)
  Xr[:, 4] .= Xr[:, 1] .+ Xr[:, 2]
  @test bad(genlasso_gap(Xr, randn(rng, 20), 0.0, zeros(4), fused_rows(4), fill(0.1, 3); intercept = false))
  @test bad(genlasso_gap(nothing, randn(rng, 5), 0.0, zeros(5), fused_rows(5), fill(0.1, 4); intercept = true))
end

@testset "SLOPE" begin
  rng = scenario_rng("certificate-controls-slope", 1)
  # X = I, n = 2: ‖y - β‖²/4 + λ₁|β|₍₁₎ + λ₂|β|₍₂₎ at y = (3, 1), λ = (1/2, 1/4):
  # β = (2, 1/2) has gradient -(y - β)/2 = -(1/2, 1/4) = -λ in sorted order
  X, y, λ = Matrix(1.0I, 2, 2), [3.0, 1.0], [0.5, 0.25]
  @test slope_gap(X, y, 0.0, [2.0, 0.5], λ; intercept = false, standardize_x = false).value ≤ 1e-15
  # β = 0: θ = y/2, cumulative z = (3/2, 2) against (1/2, 3/4): t = 1/3
  a = slope_parts(X, y, 0.0, zeros(2), λ; intercept = false, standardize_x = false)
  @test a.t ≈ 1 / 3 && a.D ≈ (1 / 3) * 5 - (1 / 9) * 10 / 4
  n, p = 50, 8
  for standardize_x in (true, false), intercept in (true, false)
    X = randn(rng, n, p) .* transpose(0.5 .+ rand(rng, p)) .+ transpose(randn(rng, p))
    λ = 0.1 .* collect(range(1.0, 0.3; length = p))
    s = standardize_x ? column_scales(X; intercept) : ones(p)
    b = zeros(p)
    b[[2, 5, 7]] .= [1.5, -0.8, 0.3]          # distinct magnitudes take λ₁, λ₂, λ₃
    order = sortperm(abs.(b); rev = true)
    v = zeros(p)
    for (i, j) in enumerate(order)
      v[j] = i <= 3 ? λ[i] * sign(b[j]) : 0.5 * λ[i] * draw_signs(rng, 1)[1]
    end
    θ = dual_with(X, s .* v; intercept)
    β0 = intercept ? 0.4 : 0.0
    β = b ./ s
    y = β0 .+ X * β .+ n .* θ
    kw = (; intercept, standardize_x)
    c = slope_gap(X, y, β0, β, λ; kw...)
    @test ok(c) && c.value ≤ TIGHT
    Pstar = slope_parts(X, y, β0, β, λ; kw...).P
    for _ in 1:3
      a = slope_parts(X, y, β0, β .+ 0.2 .* randn(rng, p), λ; kw...)
      z = sort(abs.((transpose(X) * a.θ) ./ a.s); rev = true)
      @test all(cumsum(z) .<= cumsum(λ) .* (1 + 1e-12)) && a.D ≤ Pstar + 1e-13
    end
  end
  @test bad(slope_gap(X, y, 0.0, zeros(2), [0.25, 0.5]; intercept = false, standardize_x = false))
end

@testset "Knockoffs, thresholds and frequencies" begin
  rng = scenario_rng("certificate-controls-selection", 1)
  p = 6
  R = [0.5^abs(i - j) for i in 1:p, j in 1:p]
  d = 0.5 .+ rand(rng, p)
  Σ = Matrix(Symmetric(d .* R .* transpose(d)))
  top = min(1.0, 2eigmin(Symmetric(R)))
  @test knockoff_feasibility(Σ, (1 - 1e-8) * top .* diag(Σ)).value ≤ 1e-14
  @test knockoff_feasibility(Σ, zeros(p)).value == 0
  @test knockoff_feasibility(Σ, 1.1 * top .* diag(Σ)).value ≈ 0.1top rtol = 1e-6
  @test knockoff_feasibility(Σ, [-0.1Σ[1, 1]; zeros(p - 1)]).value ≈ 0.1
  @test bad(knockoff_feasibility(Σ[:, 1:5], zeros(5)))
  # feasibility accepts s = 0 and any smaller diagonal; the equicorrelated
  # check accepts only s = (1 - safety) min(1, 2λmin(R)) diag(Σ)
  equi = (1 - 1e-8) * top .* diag(Σ)
  @test equicorrelated_check(Σ, equi).value ≤ 1e-14
  @test equicorrelated_check(Σ, zeros(p)).value ≈ 1
  @test equicorrelated_check(Σ, 0.9 .* equi).value ≈ 0.1 rtol = 1e-12
  @test equicorrelated_check(Σ, top .* diag(Σ); safety = 0).value ≤ 1e-14
  @test equicorrelated_check(Σ, top .* diag(Σ)).value ≈ 1e-8 rtol = 1e-6
  @test equicorrelated_check(Σ, equi).measure === :consistency
  @test bad(equicorrelated_check(Σ[:, 1:5], zeros(5)))
  @test bad(equicorrelated_check(Σ, equi; safety = 1.0))
  # candidates 0.5, 1, 2, 3 have #{W ≤ -t} = 2, 1, 0, 0 and #{W ≥ t} = 4, 4, 3, 1
  W = [3.0, -1.0, 2.0, 2.0, -0.5, 1.0, 0.0]
  @test knockoff_threshold_reference(W, 0.5, 1) == 1.0     # (1 + 1)/4 ≤ 0.5
  @test knockoff_threshold_reference(W, 0.4, 1) == 2.0     # (1 + 0)/3 ≤ 0.4
  @test knockoff_threshold_reference(W, 0.5, 0) == 0.5     # 2/4 ≤ 0.5
  @test knockoff_threshold_reference(W, 0.1, 1) == Inf
  @test knockoff_threshold_check(W, 0.5, 1.0, 1).value == 0
  @test knockoff_threshold_check(W, 0.1, Inf, 1).value == 0
  @test knockoff_threshold_check(W, 0.5, 2.0, 1).value == 0.5
  @test knockoff_threshold_check(W, 0.5, Inf, 1).value == 1
  @test bad(knockoff_threshold_check([NaN, 1.0], 0.5, 1.0, 1))
  S = rand(rng, 5, 8) .< 0.4
  @test frequency_check(S, vec(sum(S; dims = 2)) ./ 8).value == 0
  @test frequency_check(S, vec(sum(S; dims = 2)) ./ 8 .+ [0.0, 0.25, 0.0, 0.0, 0.0]).value == 0.25
  @test bad(frequency_check(S, zeros(4)))
end

@testset "Nonfinite inputs are uncertifiable" begin
  rng = scenario_rng("certificate-controls-nonfinite", 1)
  X, y, Y = randn(rng, 20, 4), rand(rng, 20), randn(rng, 20, 2)
  groups, D = [[1, 2], [3, 4]], fused_rows(4)
  for v in (NaN, Inf)
    β = [v, 0.0, 0.0, 0.0]
    Xv = copy(X)
    Xv[3, 2] = v
    for (b, Xb, yb) in ((β, X, y), (zeros(4), Xv, y), (zeros(4), X, [v; y[2:end]]))
      @test bad(ls_l1_gap(Xb, yb, 0.0, b, 0.1, ones(4); intercept = true))
      @test bad(ls_l1_gap(Xb, yb, 0.0, b, 0.1, [0.0, 1, 1, 1]; intercept = true))
      @test bad(sqrt_l1_gap(Xb, yb, 0.0, b, 0.1, ones(4); intercept = true))
      @test bad(ls_refit_stationarity(Xb, yb, 0.0, b; intercept = true))
      @test bad(glm_l1_gap(:logistic, Xb, yb, 0.0, b, 0.1, ones(4); intercept = true))
      @test bad(glm_l1_gap(:poisson, Xb, yb, 0.0, b, 0.1, ones(4); alpha = 0.5, intercept = true))
      @test bad(nonconvex_stationarity(:gaussian, :mcp, Xb, yb, 0.0, b, 0.1, 3.0, ones(4); intercept = true))
      @test bad(group_gap(Xb, yb, 0.0, b, 0.1, groups, ones(2); intercept = true, orthonormal = true))
      @test bad(sparse_group_gap(Xb, yb, 0.0, b, 0.1, 0.5, groups, ones(2); intercept = true, standardize_x = true))
      @test bad(quantile_exact_gap(Xb, yb, 0.0, b, 0.1, ones(4), 0.5; intercept = true))
      @test bad(quantile_smooth_residual(Xb, yb, 0.0, b, 0.1, ones(4), 0.5, 0.2; intercept = true))
      @test bad(genlasso_gap(Xb, yb, 0.0, b, D, fill(0.1, 3); intercept = false))
      @test bad(slope_gap(Xb, yb, 0.0, b, fill(0.1, 4); intercept = true, standardize_x = true))
      @test bad(ols_coefficient_error(Xb, yb, [1, 2], 0.0, b; intercept = true))
    end
    @test bad(ls_l1_gap(X, y, v, zeros(4), 0.1, ones(4); intercept = true))
    @test bad(ls_l1_gap(X, y, 0.0, zeros(4), v, ones(4); intercept = true))
    @test bad(ls_l1_gap(X, y, 0.0, zeros(4), 0.1, ones(4); intercept = true, weights = [v; ones(19)]))
    @test bad(ls_l1_gap(X, y, 0.0, zeros(4), 0.1, ones(4); intercept = true, offset = [v; zeros(19)]))
    @test bad(genlasso_gap(nothing, y, 0.0, [v; zeros(19)], fused_rows(20), fill(0.1, 19); intercept = false))
    B = zeros(4, 2)
    B[1, 1] = v
    @test bad(multi_gap(X, Y, zeros(2), B, 0.1, ones(4); intercept = true))
    @test bad(nuclear_gap(X, Y, zeros(2), B, 0.1; intercept = true))
    @test bad(rrr_excess(X, Y, zeros(2), B, 1; intercept = true))
    @test bad(knockoff_feasibility([1.0 v; v 1.0], [0.1, 0.1]))
    @test bad(equicorrelated_check([1.0 v; v 1.0], [0.1, 0.1]))
    @test bad(equicorrelated_check([1.0 0.5; 0.5 1.0], [v, 0.1]))
    @test bad(frequency_check(trues(2, 3), [v, 0.5]))
  end
end

@testset "Scenarios" begin
  @test rand(scenario_rng("ls-200x50", 1), UInt64) == rand(Xoshiro(0x91bae84adbad28da), UInt64)
  # the same hashes on Julia 1.10 and 1.13, aarch64 and x86_64
  pinned = ("ls-200x50" => "ad8f0a23c3dd3989b99e6c74233b1aaafec8720b647d2099b9bb20de96aa2a38",
            "logistic-weighted-300x60" => "3d8bfd57eeb6e53fde9a315177c9f9e05bdce0e1773b9baadeba90ec546bea71",
            "poisson-300x40" => "8cc187c2b38451acd49fa8f61b57bff7f9a8f51cd0f01697fc9451a543e7b14f",
            "signal-200" => "f03e0a44af6c7b91e4089a52b790d9d65f7cb5886d8ae530be4f4002c7026d34",
            "knockoff-300x50" => "cbe952518fb416ac30317c8df0941a643262a2bc98c32999c847c30578a0c002")
  for (key, hash) in pinned
    @test data_hash(scenario(key, 1)) == hash
  end
  required = ["ls-200x50", "ls-sparse-500x200", "ls-tall-5000x40", "ls-wide-100x400", "ls-corr-300x100",
              "ls-weighted-200x50", "logistic-300x60", "poisson-300x40", "groups-200x60", "multi-200x40x5",
              "lowrank-150x30x10", "lowrank-wide-60x100x10", "quantile-200x30", "signal-200", "fusedreg-150x60",
              "trend-200", "slope-150x60", "knockoff-300x50", "hdm-200x80"]
  @test all(in(SCENARIOS), required) && allunique(SCENARIOS)
  for key in SCENARIOS
    a, b, c = scenario(key, 1), scenario(key, 1), scenario(key, 2)
    @test keys(a) == keys(b) && all(getproperty(a, k) == getproperty(b, k) for k in keys(a))
    @test data_hash(a) == data_hash(b) && occursin(r"^[0-9a-f]{64}$", data_hash(a))
    response(t) = haskey(t, :y) ? t.y : t.Y
    @test data_hash(a) != data_hash(c) && response(a) != response(c)
    @test all(v -> !(v isa AbstractArray{<:AbstractFloat}) || eltype(v) === Float64, values(a))
    @test all(v -> cert_finite(v), (v for v in values(a) if !(v isa Vector{Vector{Int}})))
  end
  @test_throws ArgumentError scenario("ls-1x1", 1)
  @test_throws ArgumentError scenario("ls-200x50", 0)
  a = scenario("ls-sparse-500x200", 1)
  @test a.X isa SparseMatrixCSC && 0.03 < nnz(a.X) / length(a.X) < 0.07
  a = scenario("knockoff-300x50", 1)
  @test issymmetric(a.Sigma) && isposdef(a.Sigma) && !all(≈(1), diag(a.Sigma)) && size(a.X) == (300, 50)
  a = scenario("groups-200x60", 1)
  @test length(a.groups) == 12 && all(==(5), length.(a.groups)) && cert_partition(a.groups, 60)
  a = scenario("ls-weighted-200x50", 1)
  @test all(>(0), a.weights) && length(a.offset) == 200
  @test haskey(scenario("logistic-weighted-300x60", 1), :weights)
  @test all(v -> v in (0.0, 1.0), scenario("logistic-300x60", 1).y)
  @test all(v -> v >= 0 && isinteger(v), scenario("poisson-300x40", 1).y)
  a = scenario("fusedreg-150x60", 1)
  @test rank(a.X) == 60
  @test issorted(scenario("trend-200", 1).x; lt = <=) && haskey(scenario("hdm-200x80", 1), :d)
  # hashes cover element types, sizes, sparsity and field names
  @test data_hash((; x = [1.0, 2.0])) != data_hash((; x = Float32[1.0, 2.0]))
  @test data_hash((; x = [1.0, 2.0])) != data_hash((; x = reshape([1.0, 2.0], 1, 2)))
  @test data_hash((; X = sparse([1.0 0.0; 0.0 2.0]))) != data_hash((; X = [1.0 0.0; 0.0 2.0]))
  @test data_hash((; a = 1, b = [1.0])) == data_hash((; b = [1.0], a = 1))
  @test data_hash((; a = 1)) != data_hash((; b = 1))
end

if Base.find_package("ProximalBase") !== nothing
  # secondary: the convolution loss, written independently above, against
  # ProximalBase's ConvolutionCheck, whose callable takes (η, y) with u = y - η
  import ProximalBase
  @testset "Secondary agreement with ProximalBase's convolution loss" begin
    for τ in (0.2, 0.5, 0.8), h in (0.1, 1.0)
      ℓ = ProximalBase.ConvolutionCheck(τ, h)
      @test ProximalBase.curvature(ℓ) ≈ 3 / (4h)
      for u in (-2h, -h, -0.7h, -0.2h, 0.0, 0.3h, 0.99h, h, 1.5h)
        @test cert_conv_check(u, τ, h) ≈ ℓ(0.0, u) atol = 1e-14
        @test cert_conv_score(u, τ, h) ≈ -ProximalBase.deriv(ℓ, 0.0, u) atol = 1e-14
        @test cert_conv_check(u, τ, h) ≈ ℓ(1.5, u + 1.5) atol = 1e-13
      end
    end
  end
end

if Base.find_package("HighDimensionalModels") !== nothing && get(ENV, "HDM_CERTIFICATES_PACKAGE", "1") != "0"
  # secondary: tight package fits against the certificates, and the certificates
  # against the diagnostics that the package and its solvers report
  # (HDM_CERTIFICATES_PACKAGE=0 skips)
  import HighDimensionalModels as HDM
  import ProximalSolvers as PS
  @testset "Secondary agreement with HighDimensionalModels at tight solutions" begin
    a = scenario("ls-weighted-200x50", 1)
    ω = column_loadings(a.X; intercept = true, weights = a.weights)
    s = HDM.lasso(a.X, a.y, 0.05; intercept = true, standardize_x = true, weights = a.weights, offset = a.offset,
                  opt_tol = 1e-12, rng = Xoshiro(1))
    @test ls_l1_gap(a.X, a.y, s.intercept, Vector(s.x), 0.05, ω; intercept = true, weights = a.weights,
                    offset = a.offset).value ≤ 1e-9
    a = scenario("groups-200x60", 2)
    s = HDM.group_lasso(a.X, a.y, 0.05, a.groups, HDM.CDOptions(opt_tol = 1e-12))
    @test group_gap(a.X, a.y, s.intercept, Vector(s.x), 0.05, a.groups, s.groupweights; intercept = true,
                    orthonormal = true).value ≤ 1e-9
    # linear trend filtering on irregular locations, whose D D' is ill-conditioned
    a = scenario("trend-200", 1)
    D = HDM.difference_matrix(length(a.y), 2; x = a.x)
    s = HDM.trend_filter(a.y, 0.0561; order = 1, x = a.x, abstol = 1e-12, reltol = 1e-12, max_iter = 100_000)
    @test genlasso_gap(nothing, a.y, 0.0, s.x, D, fill(0.0561, size(D, 1)); intercept = false).value ≤ 1e-9
    # the lasso and logistic gaps against ProximalSolvers' duality_gap on the
    # model the package fits (the coordinate-descent model with an intercept)
    a = scenario("ls-200x50", 1)
    n, p = size(a.X)
    ω = column_loadings(a.X; intercept = true)
    s = HDM.lasso(a.X, a.y, 0.05; intercept = true, standardize_x = true, opt_tol = 1e-12, rng = Xoshiro(1))
    own = PS.duality_gap(PS.cdmodel((1 / n) * ProximalBase.LeastSquaresLoss(a.y, a.X), s.x; intercept = true),
                         ProximalBase.NormL1(0.05, ω), s.x)
    mine = ls_l1_parts(a.X, a.y, s.intercept, Vector(s.x), 0.05, ω; intercept = true)
    @test mine.P ≈ own.primal rtol = 1e-10
    @test abs(own.gap) / abs(own.primal) ≤ 1e-8 && (mine.P - mine.D) / mine.P ≤ 1e-8
    a = scenario("logistic-weighted-300x60", 1)
    ω = column_loadings(a.X; intercept = true, weights = a.weights)
    s = HDM.glm_lasso(a.X, a.y, 0.02, ProximalBase.Logistic(), HDM.CDOptions(opt_tol = 1e-12); weights = a.weights,
                      offset = a.offset, newton_tol = 1e-14, rng = Xoshiro(1))
    loss = inv(sum(a.weights)) * ProximalBase.GLMLoss(ProximalBase.Logistic(), a.y, a.X; weights = a.weights,
                                                      offset = a.offset)
    own = PS.duality_gap(PS.cdmodel(loss, s.x; intercept = true), ProximalBase.NormL1(0.02, ω), s.x)
    mine = glm_parts(:logistic, a.X, a.y, s.intercept, Vector(s.x), 0.02, ω; intercept = true, weights = a.weights,
                     offset = a.offset)
    @test mine.P ≈ own.primal rtol = 1e-10
    @test abs(own.gap) / abs(own.primal) ≤ 1e-8 && (mine.P - mine.D) / mine.P ≤ 1e-8
    # exact quantile regression against the objective and the gaps that
    # QuantileLassoSolution reports; the certificate LPs above have already run,
    # so this also checks that they leave HiGHS's thread scheduler usable
    a = scenario("quantile-200x30", 1)
    w = column_loadings(a.X; intercept = true)
    s = HDM.quantile_lasso(a.X, a.y, 0.02; tau = 0.3)
    mine = quantile_parts(a.X, a.y, s.intercept, s.β, 0.02, w, 0.3; intercept = true)
    @test s.converged
    @test mine.P ≈ s.objective rtol = 1e-12
    @test abs(s.duality_gap) / s.objective ≤ 1e-6 && s.certificate_gap / s.objective ≤ 1e-6
    @test (mine.P - mine.D) / mine.P ≤ 1e-8
    # the equicorrelated diagonal of the package
    a = scenario("knockoff-300x50", 1)
    @test equicorrelated_check(a.Sigma, HDM.equicorrelated_s(a.Sigma)).value ≤ 1e-12
    @test knockoff_feasibility(a.Sigma, HDM.equicorrelated_s(a.Sigma)).value == 0
  end
end
