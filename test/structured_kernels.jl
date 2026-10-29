module StructuredKernelsTest
using Test, Random, LinearAlgebra, SparseArrays, HighDimensionalModels, ProximalBase
const HM = HighDimensionalModels

function check_factor(H, width, b)
  T, n = eltype(H), length(b)
  saved = copy(b)
  F = HM._structured_band_factor(sparse(H), width)
  @test size(F.lower) == (width, n)
  @test length(F.pivots) == n
  L = Matrix{T}(I, n, n)
  for j in 1:n, k in 1:min(width, n-j)
    L[j+k, j] = F.lower[k, j]
  end
  @test L * Diagonal(F.pivots) * L' ≈ Matrix(H) rtol=16eps(T)
  x = HM._structured_solve!(zeros(T, n), F, b)
  @test x ≈ Matrix(H) \ b rtol=16eps(T)*cond(Matrix(H))
  @test norm(H*x-b) <= 64eps(T) * (norm(H)*norm(x) + norm(b))
  @test isequal(b, saved)
end

@testset "banded LDL pivots and solves" begin
  rng = Xoshiro(98101)
  for T in (Float32, Float64), n in (1, 2, 3, 4, 5, 31), width in 0:min(3,n-1),
      irregular in (false, true), rho in (1e-3, 1, 1e3)
    x = irregular ? cumsum(T.(.8 .+ rand(rng, n))) : nothing
    K = T.(difference_matrix(n, width; x))
    for masked in (false, true)
      mask = ones(T, n)
      masked && (mask[cld(n,2)] = 0)
      H = Diagonal(mask) + T(rho) * (K'K)
      check_factor(H, width, randn(rng, T, n))
    end
  end
  for T in (Float32, Float64), width in 1:3, n in 1:width
    L = Matrix{T}(I, n, n) + tril(T(.1) * randn(rng, T, n, n), -1)
    H = L * Diagonal(T.(1:n)) * L'
    check_factor(H, width, randn(rng, T, n))
  end
  for T in (Float32, Float64), width in 0:3, n in (1, 5)
    scale = T === Float64 ? T(1e-320) : T(1e-40)
    H = spdiagm(0 => scale .* T.(1:n))
    b = H * ones(T, n)
    saved = copy(b)
    F = HM._structured_band_factor(H, width)
    @test HM._structured_solve!(zeros(T, n), F, b) == ones(T, n)
    @test b == saved
  end
  message = ArgumentError("trend-filtering normal equations must be positive definite.")
  for T in (Float32, Float64)
    @test_throws message HM._structured_band_factor(sparse(T[1 2; 2 1]), 1)
    # The first division overflows before the next pivot can be inspected.
    tiny = nextfloat(zero(T))
    @test_throws message HM._structured_band_factor(sparse(T[tiny 1; 1 2]), 1)
    for value in (zero(T), -one(T), T(Inf), T(NaN))
      @test_throws message HM._structured_band_factor(spdiagm(0 => [value]), 0)
    end
  end
end

kernel_norm(x) = HM._structured_norm(HM._structured_sumsq(x), x)

@testset "norm sums and scaled fallbacks" begin
  rng = Xoshiro(98102)
  for T in (Float32, Float64), n in (31, 100_000), scale in (1, 1e-10, 1e10)
    x = T(scale) .* randn(rng, T, n)
    # Long BLAS reductions can lose more than two ulps themselves.
    expected = n > 31 ? T(norm(BigFloat.(x))) : norm(x)
    actual = kernel_norm(x)
    @test actual isa T
    @test abs(actual - expected) <= 2eps(expected)
  end
  for T in (Float32, Float64)
    vectors = [zeros(T, 3), T[floatmax(T)], T[floatmax(T), floatmax(T)],
      T[Inf], T[NaN], T[Inf, NaN], T[NaN, Inf]]
    if T === Float64
      append!(vectors, [[1e-160, -2e-160], [1e-200, -2e-200], [1e200, -2e200]])
    end
    for x in vectors
      @test isequal(kernel_norm(x), norm(x))
      scratch = fill(T(NaN), length(x))
      @test isequal(HM._structured_norm(HM._structured_sumsq(x), scratch, copyto!, x), norm(x))
    end
  end
  for T in (Float16, BigFloat), x in (T[1, -2, 3], zeros(T, 3), T[Inf, NaN])
    @test isequal(kernel_norm(x), norm(x))
    @test isequal(HM._structured_norm(0.0, similar(x), copyto!, x), norm(x))
  end
end

# Keep the original passes and proximal copy independent of the fused loop.
function split_reference(A, b, K, g, ρ; max_iter, abstol, reltol, factor,
                         start = nothing, relative_gap = nothing, products = K)
  split_reference(A, b, K, g, ρ, max_iter, abstol, reltol, factor, start, relative_gap, products)
end

@noinline function split_reference(A, b, K, g, ρ, max_iter, abstol, reltol, factor, start, relative_gap, products)
  @nospecialize A K g factor products
  T, p, m = eltype(b), size(A, 2), size(K, 1)
  xy = HM._structured_tproduct(A, b)
  β = start === nothing ? zeros(T, p) : copy(start.β)
  z = start === nothing ? HM._split_product!(zeros(T, m), products, β) : copy(start.z)
  u = start === nothing ? zeros(T, m) : (start.ρ / ρ) .* start.u
  kb, v, zold = zeros(T, m), zeros(T, m), zeros(T, m)
  rhs, dual, ktu = zeros(T, p), zeros(T, p), zeros(T, p)
  energy = relative_gap === nothing ? zero(T) : sum(abs2, b)
  ok, it, pr, dr = false, 0, T(Inf), T(Inf)
  for iter in 1:max_iter
    @. v = z - u
    HM._split_tproduct!(rhs, products, v)
    @. rhs = xy + ρ * rhs
    HM._structured_solve!(β, factor, rhs)
    HM._split_product!(kb, products, β)
    copyto!(zold, z)
    @. v = kb + u
    prox!(z, g, v, inv(ρ))
    @. u = u + kb - z
    @. v = kb - z
    pr = kernel_norm(v)
    @. v = z - zold
    HM._split_tproduct!(dual, products, v)
    dr = ρ * kernel_norm(dual)
    HM._split_tproduct!(ktu, products, u)
    it = iter
    if relative_gap === nothing
      if pr <= sqrt(m) * abstol + reltol * max(kernel_norm(kb), kernel_norm(z)) &&
         dr <= sqrt(p) * abstol + reltol * ρ * kernel_norm(ktu)
        ok = true
        break
      end
    else
      ep = sqrt(m) * abstol + reltol * max(kernel_norm(kb), kernel_norm(z))
      ed = sqrt(p) * abstol + reltol * ρ * kernel_norm(ktu)
      if pr <= 5ep && dr <= 5ed && (iter == 1 || iter % 10 == 0 || iter == max_iter)
        gap, objective = HM._structured_signal_gap!(dual, v, products, β, b, kb, u, ρ, g.λ)
        nearzero = objective <= 100eps(T) * energy && pr <= ep && dr <= ed
        if isfinite(gap) && isfinite(objective) && isfinite(energy) &&
           (gap <= relative_gap * objective || nearzero)
          ok = true
          break
        end
      end
    end
  end
  β, it, ok, pr, dr, (; β, z, u, ρ)
end

same_bits(x::AbstractVector, y::AbstractVector) = reinterpret(UInt8, x) == reinterpret(UInt8, y)
same_bits(x::T, y::T) where {T<:Union{Float32, Float64}} =
  T === Float64 ? reinterpret(UInt64, x) == reinterpret(UInt64, y) : reinterpret(UInt32, x) == reinterpret(UInt32, y)
# BigFloat has no fixed bit layout; equal values, signed zeros and precision stand in.
same_bits(x::BigFloat, y::BigFloat) = isequal(x, y) && precision(x) == precision(y)
same_bits(x::AbstractVector{BigFloat}, y::AbstractVector{BigFloat}) =
  length(x) == length(y) && all(same_bits(a, b) for (a, b) in zip(x, y))

function check_split(A, b, K, g, rho, products; max_iter = 5, abstol = 0, reltol = 0,
                     start = nothing, relative_gap = nothing, bandwidth = nothing)
  factor = HM._split_factor(HM._structured_gram(A), K, rho; bandwidth)
  saved = start === nothing ? nothing : (copy(start.β), copy(start.z), copy(start.u))
  options = (; max_iter, abstol, reltol, factor, products, start, relative_gap)
  expected = split_reference(A, b, K, g, rho; options...)
  actual = HM._split_fit(A, b, K, g, rho; options...)
  check_split_result(actual, expected, start, saved)
  actual
end

@noinline function check_split_result(actual, expected, start, saved)
  @test same_bits(actual[1], expected[1])
  @test actual[2:3] == expected[2:3]
  @test same_bits(actual[4], expected[4])
  @test same_bits(actual[5], expected[5])
  for key in (:β, :z, :u, :ρ)
    @test same_bits(getproperty(actual[6], key), getproperty(expected[6], key))
  end
  @test actual[1] === actual[6].β
  if start !== nothing
    for (key, old) in zip((:β, :z, :u), saved)
      @test same_bits(getproperty(start, key), old)
      @test getproperty(actual[6], key) !== getproperty(start, key)
    end
  end
  nothing
end

@testset "fused ADMM equals the unfused passes" begin
  rng = Xoshiro(98103)
  for T in (Float32, Float64), order in 1:3, irregular in (false, true)
    n = 9
    x = irregular ? cumsum(T.(.8 .+ rand(rng, n))) : nothing
    K = T.(difference_matrix(n, order; x))
    A, b = HM._SignalDesign{T}(n), randn(rng, T, n)
    P = HM._trend_products(K, order)
    order <= 2 && @test P isa HM._TrendProducts{T, order+1}
    for products in (P, K, Matrix(K)), g in (NormL1(T(.2)), TotalVariation1D(T(.2)))
      result = check_split(A, b, K, g, T(.7), products; bandwidth = order)
      @test result[2] == 5
      @test !result[3]
    end
    for count in (0, 1, 2, 6)
      result = check_split(A, b, K, TotalVariation1D(T(.2)), T(.7), P;
        max_iter = count, bandwidth = order)
      @test result[2] == count
      @test !result[3]
      if count == 0
        @test iszero(result[6].z)
        @test result[4] == result[5] == T(Inf)
      end
    end
    # Warm starts rescale only the copied dual, including on a zero-step fit.
    for g in (NormL1(T(.2)), TotalVariation1D(T(.2)))
      start = check_split(A, b, K, g, T(.7), P; bandwidth = order)[6]
      for count in (0, 5, 6)
        result = check_split(A, b, K, g, T(1.4), P; start, max_iter = count, bandwidth = order)
        if count == 0
          @test same_bits(result[6].z, start.z)
          @test same_bits(result[6].u, (start.ρ / T(1.4)) .* start.u)
        end
      end
    end
  end
  # Exercise the general dense and sparse normal-equation factors as well.
  for T in (Float32, Float64), sparse_design in (false, true)
    A = sparse_design ? spdiagm(0 => ones(T, 7)) : Matrix{T}(I, 7, 7)
    K, b = T.(difference_matrix(7, 2)), randn(rng, T, 7)
    for products in (K, Matrix(K))
      check_split(A, b, K, NormL1(T(.2)), T(.7), products)
    end
  end
  # Minimal stencils exercise both ends when there is no interior row.
  for T in (Float32, Float64), order in 1:2, n in order+1:order+2
    K = T.(difference_matrix(n, order))
    check_split(HM._SignalDesign{T}(n), randn(rng, T, n), K, TotalVariation1D(T(.2)),
      T(.7), HM._trend_products(K, order); bandwidth = order)
  end
  for scale in (1e-160, 1e-200, 1e200, Inf, NaN)
    K = difference_matrix(5)
    check_split(HM._SignalDesign{Float64}(5), scale .* [.1, -.2, .3, -.4, .5], K,
      NormL1(.2), .7, HM._trend_products(K, 1); bandwidth = 1)
  end
  # Cover boundary groups, complete sum blocks and incomplete final blocks.
  for T in (Float32, Float64), order in 1:2, irregular in (false, true),
      n in (8, 16, 256, 257, 263, 264, 512, 513)
    x = irregular ? cumsum(T.(.8 .+ rand(rng, n))) : nothing
    K = T.(difference_matrix(n, order; x))
    A, b = HM._SignalDesign{T}(n), randn(rng, T, n)
    for products in (HM._trend_products(K, order), K, Matrix(K))
      check_split(A, b, K, TotalVariation1D(T(.2)), T(.7), products;
        max_iter = 2, bandwidth = order)
    end
  end
end

@testset "width-one stencils equal the unfused passes" begin
  rng = Xoshiro(98107)
  for T in (Float32, Float64), n in (1, 2, 8, 9, 263, 264), irregular in (false, true)
    K = irregular ? spdiagm(0 => T.(.5 .+ rand(rng, n))) : T.(difference_matrix(n, 0))
    A, P, rho = HM._SignalDesign{T}(n), HM._trend_products(K, 0), T(.7)
    @test P isa HM._TrendProducts{T, 1}
    xy, z, zold, u, β = (randn(rng, T, n) for _ in 1:5)
    rhs, v = zeros(T, n), zeros(T, n)
    @. v = z - u
    HM._split_tproduct!(rhs, P, v)
    @. rhs = xy + rho * rhs
    @test same_bits(HM._split_rhs!(fill(T(NaN), n), P, xy, rho, z, u, fill(T(NaN), n)), rhs)
    kb = HM._split_product!(zeros(T, n), P, β)
    @. v = kb + u
    fused_kb, fused_v = fill(T(NaN), n), fill(T(NaN), n)
    HM._split_primal!(fused_kb, fused_v, P, β, u)
    @test same_bits(fused_kb, kb)
    @test same_bits(fused_v, v)
    dual, ktu = zeros(T, n), zeros(T, n)
    @. v = z - zold
    HM._split_tproduct!(dual, P, v)
    HM._split_tproduct!(ktu, P, u)
    sums = HM._split_dual_sums!(fill(T(NaN), n), fill(T(NaN), n), fill(T(NaN), n), P, z, zold, u)
    @test same_bits(sums[1], HM._structured_sumsq(dual))
    @test same_bits(sums[2], HM._structured_sumsq(ktu))
    for g in (NormL1(T(.2)), TotalVariation1D(T(.2)))
      start = check_split(A, xy, K, g, rho, P; bandwidth = 0)[6]
      check_split(A, xy, K, g, T(1.4), P; start, bandwidth = 0)
    end
    result = check_split(A, xy, K, TotalVariation1D(T(.2)), rho, P;
      max_iter = 1000, abstol = 1e-5, reltol = 1e-4, bandwidth = 0)
    @test result[3]
  end
end

@testset "BigFloat trend fits use materialized norms" begin
  rng = Xoshiro(98108)
  n, abstol, reltol = 20, 1e-9, 1e-8
  K64, b64 = difference_matrix(n, 1), randn(rng, n)
  fits = map((Float64, BigFloat)) do T
    K, b, rho = T.(K64), T.(b64), T(.7)
    A, P = HM._SignalDesign{T}(n), HM._trend_products(K, 1)
    @test P isa HM._TrendProducts{T, 2}
    @test HM._split_factor(HM._structured_gram(A), K, rho; bandwidth = 1) isa HM._StructuredBandFactor{T, 1}
    options = (; max_iter = 2000, abstol, reltol, bandwidth = 1)
    cold = check_split(A, b, K, TotalVariation1D(T(1)), rho, P; options...)
    warm = check_split(A, b, K, TotalVariation1D(T(.4)), 2rho, P; options..., start = cold[6])
    @test cold[3] && warm[3]
    cold, warm
  end
  # Fused sums of other element types are zero, so every norm is materialized.
  big = BigFloat.(b64)
  @test HM._split_residuals!(zero(big), big, reverse(big)) === (0.0, 0.0, 0.0)
  # Allow the Float64 stopping tolerance in case the two runs stop at different iterations.
  for (single, extended) in zip(fits...)
    @test norm(Float64.(extended[1]) - single[1]) <= sqrt(n) * abstol + reltol * norm(single[1])
  end
end

@testset "gap schedule, near-zero objectives and residual stopping" begin
  rng = Xoshiro(98104)
  for T in (Float32, Float64), order in 1:2, irregular in (false, true)
    n = 9
    x = irregular ? cumsum(T.(.8 .+ rand(rng, n))) : nothing
    K = T.(difference_matrix(n, order; x))
    A, b = HM._SignalDesign{T}(n), randn(rng, T, n)
    P, rho, g = HM._trend_products(K, order), T(.01), TotalVariation1D(zero(T))
    first = check_split(A, b, K, g, rho, P; max_iter = 1, bandwidth = order)
    atol = first[5] / (10sqrt(n))
    for (limit, stop) in ((12, 10), (7, 7), (1, 1))
      result = check_split(A, b, K, g, rho, P; max_iter = limit, bandwidth = order,
        abstol = limit == 1 ? 100 : atol, relative_gap = T(2))
      @test result[2] == stop
      @test result[3]
    end
    exhausted = check_split(A, b, K, TotalVariation1D(T(.2)), rho, P;
      max_iter = 6, bandwidth = order, relative_gap = zero(T))
    @test exhausted[2] == 6
    @test !exhausted[3]
    residual = check_split(A, b, K, g, rho, P; abstol = 100, bandwidth = order)
    @test residual[2] == 1 && residual[3]
    for nearzero in (zeros(T, n), fill(eps(T), n))
      result = check_split(A, nearzero, K, TotalVariation1D(one(T)), rho, P;
        abstol = 1e-6, reltol = 1e-6, relative_gap = zero(T), bandwidth = order)
      @test all(isfinite, result[1])
      iszero(nearzero) && @test result[2] == 1 && result[3]
    end
  end
end

@testset "trend fit and path degrees of freedom" begin
  rng = Xoshiro(98105)
  for T in (Float32, Float64), order in 1:2, irregular in (false, true)
    n, rho, df_tol = 9, T(.7), 1e-4
    y = randn(rng, T, n)
    x = irregular ? cumsum(T.(.8 .+ rand(rng, n))) : nothing
    D, K = difference_matrix(n, order+1; x), difference_matrix(n, order; x)
    x === nothing || (K = Diagonal(T[order / (x[j+order]-x[j]) for j in 1:n-order]) * K)
    K = T.(K)
    A, P = HM._SignalDesign{T}(n), HM._trend_products(K, order)
    abstol, reltol = T === Float32 ? (1e-6, 1e-5) : (1e-9, 1e-8)
    relative_gap = T === Float64 && order == 1 ? T(3reltol/10) : nothing
    factor = HM._split_factor(HM._structured_gram(A), K, rho; bandwidth = order)
    options = (; max_iter = 2000, abstol, reltol, rho, df_tol, order, x)
    levels = T[.06, .02]
    path = trend_filter_path(y, levels; options...)
    state = nothing
    for (j, level) in enumerate(levels)
      g = TotalVariation1D(T(n*level))
      reference = split_reference(A, y, K, g, rho; max_iter = 2000, abstol, reltol,
        factor, products = P, start = state, relative_gap)
      state = reference[6]
      @test path.solutions[j].converged && reference[3]
      @test path.solutions[j].df == HM._structured_df(A, D, reference[1], df_tol; trend_order = order)
      single = trend_filter(y, level; options...)
      cold = split_reference(A, y, K, g, rho; max_iter = 2000, abstol, reltol,
        factor, products = P, relative_gap)
      @test single.converged && cold[3]
      @test single.df == HM._structured_df(A, D, cold[1], df_tol; trend_order = order)
    end
  end
end

@testset "Float32 masked trend CV agrees with explicit selection" begin
  rng = Xoshiro(98106)
  n, folds, levels = 12, repeat(1:3, 4), Float32[.02, .005, 0]
  y = randn(rng, Float32, n)
  design = spdiagm(0 => ones(Float32, n))
  for order in 0:2, irregular in (false, true)
    x = irregular ? cumsum(.8 .+ rand(rng, n)) : nothing
    options = (; order, x, folds, threads = false, max_iter = 3000, abstol = 3e-6, reltol = 1e-5)
    actual = cv_trend_filter(y, levels; options...)
    expected = cv_trend_filter(design, y, levels; options...)
    @test actual.cvm ≈ expected.cvm atol=3e-4 rtol=3e-4
    @test actual.cvse ≈ expected.cvse atol=3e-4 rtol=3e-4
    @test actual.λmin == expected.λmin
    @test actual.λ1se == expected.λ1se
    @test coef(actual.path) ≈ coef(expected.path) atol=3e-4 rtol=3e-4
  end
end
end
