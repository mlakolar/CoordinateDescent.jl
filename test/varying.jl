module VaryingTest

using CoordinateDescent, KernelSmoothing
using LinearAlgebra, Random, SparseArrays, Statistics, Test

# the weighted least squares fit of y on the columns Xᵢⱼ (zᵢ - z₀)ʳ, with
# weights wᵢ k((zᵢ - z₀)/h)/h, by dense algebra: the coefficients θⱼᵣ, estimates of
# βⱼ⁽ʳ⁾(z₀)/r!, and the design and weights of the observations in the window
function wls_at(X, z, y, z0, k, h, q; w = ones(length(y)))
  v = [k((z[i] - z0) / h) / h * w[i] for i in eachindex(z)]
  keep = v .> 0
  Z = reduce(hcat, [X[:, j] .* (z .- z0) .^ r for j in axes(X, 2) for r in 0:q])[keep, :]
  sv = sqrt.(v[keep])
  θ = (sv .* Z) \ (sv .* y[keep])
  θ, Z, v[keep], findall(keep)
end

@testset "varying-coefficient models" begin
  rng = Xoshiro(81)
  n = 300
  z = rand(rng, n)
  X = [ones(n) randn(rng, n) randn(rng, n)]
  β1(t) = 1 + 2t; β2(t) = sin(2π * t); β3(t) = 0.0
  y = X[:, 1] .* β1.(z) .+ X[:, 2] .* β2.(z) .+ 0.2 .* randn(rng, n)
  zgrid = [0.6, 0.2, 0.4, 0.8]                  # sorted by the fit
  epa = EpanechnikovKernel()

  @testset "the local fits equal weighted least squares" begin
    for (k, h, q) in ((epa, 0.2, 1), (GaussianKernel(), 0.08, 2), (UniformKernel(), 0.25, 0))
      fit = varyingCoefficient(X, z, y, zgrid; kernel = k, bandwidth = h, degree = q)
      @test fit.zgrid == sort(zgrid) && fit.degree == q && fit.bandwidth == h && all(==(3(q + 1)), fit.rank)
      for (g, z0) in enumerate(fit.zgrid)
        θ, = wls_at(X, z, y, z0, k, h, q)
        for r in 0:q
          @test coef(fit; derivative = r)[:, g] ≈ factorial(r) .* θ[(1:3) .* (q + 1) .- q .+ r] rtol = 1e-8
        end
      end
    end
    # a GaussianKernel's bandwidth is a standard deviation
    k = GaussianKernel()
    @test k(0.5) / k(0.0) ≈ exp(-0.125)
    # any function of the scaled distance is a kernel
    a = varyingCoefficient(X, z, y, zgrid; kernel = u -> abs(u) < 1 ? 0.75 * (1 - u^2) : 0.0, bandwidth = 0.2)
    b = varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0.2)
    @test coef(a) ≈ coef(b) && coef(a; derivative = 1) ≈ coef(b; derivative = 1)
  end

  @testset "polynomial coefficient functions are reproduced" begin
    yp = X[:, 1] .* (1 .+ 2 .* z) .+ X[:, 2] .* z .^ 2
    fit = varyingCoefficient(X, z, yp, zgrid; kernel = epa, bandwidth = 0.3, degree = 2, se = true)
    g = fit.zgrid
    @test coef(fit) ≈ [(1 .+ 2 .* g)'; (g .^ 2)'; zeros(4)'] atol = 1e-9
    @test coef(fit; derivative = 1) ≈ [fill(2.0, 4)'; (2 .* g)'; zeros(4)'] atol = 1e-8
    @test coef(fit; derivative = 2) ≈ [zeros(4)'; fill(2.0, 4)'; zeros(4)'] atol = 1e-7
    @test maximum(abs, residuals(fit)) < 1e-9
    @test_throws ArgumentError coef(fit; derivative = 3)
  end

  @testset "standard errors, intervals, fitted values and prediction" begin
    fit = varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0.2, se = true)
    @test nobs(fit) == n
    # the fitted value at zᵢ is that of the local fit at zᵢ
    for i in (1, 50, 200)
      θ, = wls_at(X, z, y, z[i], epa, 0.2, 1)
      @test fitted(fit)[i] ≈ dot(X[i, :], θ[[1, 3, 5]])
    end
    @test residuals(fit) ≈ y .- fitted(fit)
    # the sandwich at one point, from the residuals of the local fits
    θ, Z, v, idx = wls_at(X, z, y, fit.zgrid[2], epa, 0.2, 1)
    A = inv(Z' * (v .* Z))
    V = A * (Z' * ((v .^ 2 .* residuals(fit)[idx] .^ 2) .* Z)) * A
    @test stderror(fit)[:, 2] ≈ sqrt.(diag(V)[[1, 3, 5]]) rtol = 1e-8
    @test stderror(fit; derivative = 1)[:, 2] ≈ sqrt.(diag(V)[[2, 4, 6]]) rtol = 1e-6
    lo, hi = confint(fit)
    @test lo ≈ coef(fit) .- 1.959963984540054 .* stderror(fit) && hi ≈ coef(fit) .+ 1.959963984540054 .* stderror(fit)
    @test_throws ArgumentError confint(fit; level = 1.5)
    # predict interpolates linearly between the points, and is constant outside them
    B = coef(fit)
    Xn = randn(Xoshiro(2), 4, 3)
    zn = [0.2, 0.3, 0.05, 0.95]
    @test predict(fit, Xn, zn) ≈ [dot(Xn[1, :], B[:, 1]), dot(Xn[2, :], (B[:, 1] + B[:, 2]) / 2),
                                  dot(Xn[3, :], B[:, 1]), dot(Xn[4, :], B[:, 4])]
    @test_throws DimensionMismatch predict(fit, Xn[:, 1:2], zn)
    @test_throws DimensionMismatch predict(fit, Xn, zn[1:3])
    nose = varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0.2)        # se = false, the default
    @test coef(nose) == coef(fit)
    @test_throws ArgumentError stderror(nose)
    @test_throws ArgumentError fitted(nose)
    @test_throws ArgumentError residuals(nose)
  end

  @testset "observation weights and windows without enough data" begin
    w = rand(Xoshiro(3), 1:3, n)
    rows = reduce(vcat, [fill(i, w[i]) for i in 1:n])
    a = varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0.2, weights = w, se = true)
    b = varyingCoefficient(X[rows, :], z[rows], y[rows], zgrid; kernel = epa, bandwidth = 0.2, se = true)
    @test coef(a) ≈ coef(b) && stderror(a) ≈ stderror(b)
    # a window with fewer observations than coefficients: the fit of minimum norm, no standard errors
    zs = [0.0; 0.5 .+ 0.01 .* (1:20)]
    Xs = [ones(21) randn(Xoshiro(4), 21)]
    ys = randn(Xoshiro(5), 21)
    f = varyingCoefficient(Xs, zs, ys, [0.0, 0.6]; kernel = epa, bandwidth = 0.05, se = true)
    @test f.rank[1] < 4 && f.rank[2] == 4
    @test all(isnan, stderror(f)[:, 1]) && all(isfinite, stderror(f)[:, 2])
  end

  @testset "leave-one-out cross-validation of the bandwidth" begin
    m = 120
    Xs, zs, ys = X[1:m, :], z[1:m], y[1:m]
    hs = [0.1, 0.2, 0.4]
    cv = varyingCoefficientCV(Xs, zs, ys, hs; kernel = epa)
    for (k, h) in enumerate(hs)
      err = map(1:m) do i
        keep = setdiff(1:m, i)
        θ, = wls_at(Xs[keep, :], zs[keep], ys[keep], zs[i], epa, h, 1)
        abs2(ys[i] - dot(Xs[i, :], θ[[1, 3, 5]]))
      end
      @test cv.cvm[k] ≈ mean(err) rtol = 1e-8
    end
    @test cv.bandwidth == hs[argmin(cv.cvm)] && cv.bandwidths == hs
    # too small a bandwidth leaves observations alone in their windows
    @test varyingCoefficientCV(Xs, zs, ys, [1e-4]; kernel = epa).cvm == [Inf]
    # with weights, the weighted mean of the leave-one-out errors
    w = rand(Xoshiro(6), 1:3, m)
    cvw = varyingCoefficientCV(Xs, zs, ys, [0.2]; kernel = epa, weights = w)
    err = map(1:m) do i
      keep = setdiff(1:m, i)
      θ, = wls_at(Xs[keep, :], zs[keep], ys[keep], zs[i], epa, 0.2, 1; w = w[keep])
      abs2(ys[i] - dot(Xs[i, :], θ[[1, 3, 5]]))
    end
    @test cvw.cvm[1] ≈ sum(w .* err) / sum(w) rtol = 1e-8
    @test_throws ArgumentError varyingCoefficientCV(Xs, zs, ys, Float64[]; kernel = epa)
  end

  @testset "the varying-coefficient lasso" begin
    Xl = [X randn(Xoshiro(7), n, 5)]                # five variables that do not matter
    for penalty in (:group, :lasso)
      fit = varyingCoefficientLasso(Xl, z, y, zgrid, 0.1, CDOptions(; optTol = 1e-12); kernel = epa, bandwidth = 0.2,
                                    penalty)
      @test fit.penalty == penalty && size(fit.selected) == (8, 4) && nobs(fit) == n
      @test all(fit.selected[1:2, :]) && !any(fit.selected[3:8, :])          # β₁ and β₂ in, the rest out
      # a variable is selected when its level or its slope is not zero
      @test fit.selected == ((coef(fit) .!= 0) .| (coef(fit; derivative = 1) .!= 0))
      # the refit is the unpenalized local fit on the selected variables
      for (g, z0) in enumerate(fit.zgrid)
        S = findall(fit.selected[:, g])
        θ, = wls_at(Xl[:, S], z, y, z0, epa, 0.2, 1)
        @test coef(fit; refit = true)[S, g] ≈ θ[(1:length(S)) .* 2 .- 1] rtol = 1e-8
        @test all(iszero, coef(fit; refit = true)[setdiff(1:8, S), g])
      end
      @test predict(fit, Xl, z; refit = true) ≈ [dot(Xl[i, :], CoordinateDescent._interpolate(coef(fit; refit = true), fit.zgrid, z[i])) for i in 1:n]
    end
    # the group penalty at one point: the group lasso on the locally weighted, scaled design
    z0 = 0.4
    v = [epa((z[i] - z0) / 0.2) / 0.2 for i in 1:n]
    keep = findall(v .> 0)
    Zl = reduce(hcat, [Xl[keep, j] .* ((z[keep] .- z0) ./ 0.2) .^ r for j in 1:8 for r in 0:1])
    r = sqrt.(length(keep) .* v[keep] ./ sum(v[keep]))
    gl = groupLasso(r .* Zl, r .* y[keep], 0.05, [[2j - 1, 2j] for j in 1:8], CDOptions(; optTol = 1e-12);
                    intercept = false, groupweights = fill(sqrt(2), 8))
    fit = varyingCoefficientLasso(Xl, z, y, [z0], 0.05, CDOptions(; optTol = 1e-12); kernel = epa, bandwidth = 0.2)
    @test coef(fit)[:, 1] ≈ Vector(gl.x)[1:2:end] atol = 1e-8
    @test coef(fit; derivative = 1)[:, 1] ≈ Vector(gl.x)[2:2:end] ./ 0.2 atol = 1e-7
    # without refit
    nr = varyingCoefficientLasso(Xl, z, y, zgrid, 0.05; kernel = epa, bandwidth = 0.2, refit = false)
    @test_throws ArgumentError coef(nr; refit = true)
    # a large λ selects nothing
    @test !any(varyingCoefficientLasso(Xl, z, y, zgrid, 10.0; kernel = epa, bandwidth = 0.2).selected)
  end

  @testset "leave-one-out cross-validation of the lasso" begin
    m = 60
    Xl = [X[1:m, :] randn(Xoshiro(8), m, 2)]
    zs, ys = z[1:m], y[1:m]
    cv = varyingCoefficientLassoCV(Xl, zs, ys, [0.25, 0.5], 0.05; kernel = epa)
    for (k, h) in enumerate([0.25, 0.5])
      err = map(1:m) do i
        keep = setdiff(1:m, i)
        f = varyingCoefficientLasso(Xl[keep, :], zs[keep], ys[keep], [zs[i]], 0.05; kernel = epa, bandwidth = h)
        abs2(ys[i] - dot(Xl[i, :], coef(f; refit = true)[:, 1]))
      end
      @test cv.cvm[k] ≈ mean(err) rtol = 1e-6
    end
    nr = varyingCoefficientLassoCV(Xl, zs, ys, [0.5], 0.05; kernel = epa, refit = false, penalty = :lasso)
    @test isfinite(nr.cvm[1])
  end

  @testset "errors and conversions" begin
    @test coef(varyingCoefficient(round.(Int, 10 .* X), z, round.(Int, 10 .* y), zgrid; kernel = epa, bandwidth = 0.3)) isa Matrix{Float64}
    @test_throws ArgumentError varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0)
    @test_throws ArgumentError varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0.2, degree = -1)
    @test_throws DimensionMismatch varyingCoefficient(X, z[1:end-1], y, zgrid; kernel = epa, bandwidth = 0.2)
    @test_throws DimensionMismatch varyingCoefficient(X, z, y[1:end-1], zgrid; kernel = epa, bandwidth = 0.2)
    @test_throws DimensionMismatch varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0.2, weights = ones(n - 1))
    @test_throws ArgumentError varyingCoefficient(X, z, y, zgrid; kernel = epa, bandwidth = 0.2, weights = -ones(n))
    zn = copy(z); zn[1] = NaN
    @test_throws ArgumentError varyingCoefficient(X, zn, y, zgrid; kernel = epa, bandwidth = 0.2)
    @test_throws ArgumentError varyingCoefficientLasso(X, z, y, zgrid, 0.1; kernel = epa, bandwidth = 0.2, penalty = :scad)
    @test_throws ArgumentError varyingCoefficientLasso(X, z, y, zgrid, -0.1; kernel = epa, bandwidth = 0.2)
    @test_throws ArgumentError varyingCoefficientLassoCV(X, z, y, Float64[], 0.1; kernel = epa)
    @test_throws ArgumentError varyingCoefficientLassoCV(X, z, y, [0.2], 0.1; kernel = epa, penalty = :scad)
  end
end

end
