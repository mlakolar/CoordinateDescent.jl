using Random, LinearAlgebra, Statistics, SparseArrays

@testset "Debiasing and decorrelated inference" begin
  rng = Xoshiro(7701)
  X = randn(rng, 100, 8) .* transpose(collect(1.0:8.0)) .+ 2
  y = 1.5 .+ X[:, 1] .- 0.7X[:, 3] .+ randn(rng, 100)
  opt = CDOptions(; opt_tol = 1e-10)
  P = nodewise_lasso(X, 0.08; options = opt, threads = false, rng = Xoshiro(1))
  Xc = X .- mean(X; dims = 1)
  @test all(P.converged)
  @test diag(P.theta * (Xc'Xc / 100)) ≈ ones(8) atol=1e-9
  @test P.scores ≈ Xc * P.theta'
  # Compare every row to an ordinary lasso on the other standardized columns.
  Z = Xc ./ transpose(P.scales)
  for j in 1:8
    others = setdiff(1:8, j)
    f = lasso(Z[:, others], Z[:, j], 0.08, opt)
    v = zeros(8); v[j] = 1; v[others] = -Vector(f.x)
    tau = dot(Z[:, j], Z * v) / 100
    @test Vector(P.theta[j, :]) ≈ v ./ (tau .* P.scales .* P.scales[j]) atol=1e-7
  end

  highdim = randn(Xoshiro(7710), 30, 50)
  hq = debiasing_matrix(highdim; method = :qp, mu = 0.4, targets = [1, 10], options = opt)
  @test all(hq.converged)
  @test maximum(hq.error) <= 0.4 + 1e-7
  duplicated = hcat(X[:, 1], X[:, 1])
  @test_throws ArgumentError debiasing_matrix(duplicated; method = :qp, mu = 0.1,
                                             threads = false, options = CDOptions(; max_iter = 20))
  for gram in (false, true)
    Q = nodewise_lasso(X, 0.08; options = opt, gram, lambda_path = [0.3, 0.15, 0.08], rng = Xoshiro(2))
    @test Q.theta ≈ P.theta atol=1e-7
    @test Q.iterations != P.iterations
  end
  subset = nodewise_lasso(sparse(X), 0.08; targets = [7, 2], options = opt)
  @test subset.theta ≈ P.theta[[7, 2], :] atol=1e-7
  @test nodewise_lasso(X, 0.08; max_gram_bytes = 0, options = opt).theta ≈ P.theta atol=1e-7

  r = debiased_lasso(X, y, 0.1; sigma = 1, projection = P, options = opt)
  @test r.converged
  @test coef(r) ≈ r.initial + P.theta * Xc' * r.residuals / 100
  @test stderror(r) ≈ sqrt.(diag(vcov(r)))
  @test size(confint(r)) == (8, 2)
  @test nobs(r) == 100
  @test all(confint(r)[:, 1] .< confint(r)[:, 2])
  rr = debiased_lasso(X, y, 0.1; sigma = 1, projection = P, robust = true, options = opt)
  @test maximum(abs, mean(rr.influence; dims = 1)) < 1e-12
  @test all(isfinite, vcov(rr))
  @test rr.variance == :hc0
  hc3 = debiased_lasso(X, y, 0.1; projection=P, variance=:hc3, options=opt)
  selected = hcat(ones(size(X,1)), X[:,findall(!iszero,hc3.initial)])
  hat = vec(sum((selected / (selected'selected)) .* selected; dims=2))
  direct = P.scores .* (hc3.residuals ./ (1 .- hat))
  direct .-= mean(direct; dims=1)
  @test hc3.variance == :hc3 && hc3.robust
  @test coef(hc3) ≈ coef(r)
  @test hc3.influence ≈ direct atol=1e-10
  @test vcov(hc3) ≈ direct'direct / size(X,1)^2 atol=1e-10
  hc1_32 = HighDimensionalModels._inference_influence(Float32.(X),Float32.(r.residuals),
    Float32.(P.scores),Float32.(r.initial),true,1f0,:hc1)
  @test eltype(hc1_32) == Float32
  @test_throws ArgumentError debiased_lasso(X,y,.1; robust=false,variance=:hc3)
  @test_throws ArgumentError debiased_lasso(X,y,.1; variance=:unknown)
  @test_throws ArgumentError debiased_lasso(X,Float64.(y .> median(y)),.1;family=:binomial,variance=:hc3)
  @test debiased_lasso(X, y; options = opt).converged

  # The out-of-box Gaussian policy must reproduce the source-frozen recipe.
  auto = debiased_lasso(X,y;threads=false,rng=Xoshiro(7720))
  frozen = debiased_lasso(X,y;threads=false,nodewise_lambda=:ldpe,
    ldpe_unpenalized=4,ldpe_kappa=.5,variance=:hc3,
    options=CDOptions(;opt_tol=1e-8,gaptol=0),rng=Xoshiro(7720))
  @test auto.variance == :hc3 && auto.projection.tuning == :ldpe
  @test auto.initial ≈ frozen.initial atol=1e-10
  @test auto.projection.theta ≈ frozen.projection.theta atol=1e-10
  @test coef(auto) ≈ coef(frozen) atol=1e-10
  @test stderror(auto) ≈ stderror(frozen) atol=1e-10

  # Orthogonal design: the debiasing removes soft-threshold bias exactly.
  O = Matrix(qr(randn(rng, 40, 5)).Q)[:, 1:5] .* sqrt(40)
  b = randn(rng, 40)
  ortho = debiased_lasso(O, b, 0.2; intercept = false, sigma = 1, nodewise_lambda = 0.05, options = opt)
  @test coef(ortho) ≈ O'b / 40 atol=1e-8
  qp = debiasing_matrix(O; method = :qp, mu = 0.1, intercept = false, options = opt)
  @test Matrix(qp.theta) ≈ 0.9Matrix(I, 5, 5) atol=1e-8
  @test maximum(qp.error) <= 0.1 + 1e-8
  # On a full rank Gram matrix, the dual solution certifies the primal QP.
  q = debiasing_matrix(X; method = :qp, mu = 0.12, options = opt)
  G = Z'Z / 100
  for j in 1:8
    v = Vector(q.theta[j, :]) .* q.scales .* q.scales[j]
    @test maximum(abs, G*v - Matrix(I, 8, 8)[:, j]) <= 0.12 + 1e-7
    @test abs(dot(v, G*v) - v[j] + 0.12sum(abs, v)) < 1e-7
  end

  E = randn(rng, 100, 99)
  boot = multiplier_bootstrap(r; multipliers = E)
  @test boot.draws ≈ (r.influence'E) ./ (100 .* r.se)
  @test all(boot.adjusted_pvalues .>= boot.pvalues)
  step = multiplier_bootstrap(r; multipliers = E, stepdown = true)
  @test all(step.adjusted_pvalues .<= boot.adjusted_pvalues .+ 1e-12)
  @test step.intervals == boot.intervals
  @test all(multiplier_bootstrap(r; B = 1).intervals[:, 2] .== Inf)
  # Bootstrap refits can be reproduced from explicitly supplied errors.
  errors = (r.residuals .- mean(r.residuals))[rand(rng, 1:100, 100, 9)]
  bs = bootstrap_lasso(X, y, r; resamples = errors, options = opt, rng = Xoshiro(8))
  bs2 = bootstrap_lasso(X, y, r; resamples = errors, options = opt, rng = Xoshiro(8))
  @test bs.draws == bs2.draws
  @test all(bs.adjusted_pvalues .>= bs.pvalues)
  b3 = bootstrap_lasso(X,y,hc3;resamples=errors[:,1:1],options=opt,rng=Xoshiro(7722))
  bootfit = lasso(X,X*hc3.initial .+ hc3.intercept .+ errors[:,1],hc3.lambda,opt;
                  intercept=true,standardize_x=true,rng=Xoshiro(7722))
  bootbeta = Vector(bootfit.x)
  bootM = hcat(ones(size(X,1)),X[:,findall(!iszero,bootbeta)])
  boothat = vec(sum((bootM / (bootM'bootM)) .* bootM;dims=2))
  bootinf = P.scores .* (bootfit.residuals ./ (1 .- boothat))
  bootinf .-= mean(bootinf;dims=1)
  bootse = vec(sqrt.(sum(abs2,bootinf;dims=1))) / size(X,1)
  bootest = bootbeta + P.scores'bootfit.residuals / size(X,1)
  @test b3.draws[:,1] ≈ (bootest - hc3.initial) ./ bootse atol=1e-8
  @test_throws ArgumentError bootstrap_lasso(X, y .+ 1, r; resamples = errors)

  # At zero penalty, the Gaussian score reduces to the classical restricted score.
  sc = decorrelated_score(X, y, 2, 0.0; family = :gaussian, sigma = 1, options = opt)
  W = [ones(100) X[:, setdiff(1:8, 2)]]
  z = X[:, 2] - W * (W \ X[:, 2])
  res = y - W * (W \ y)
  @test sc.statistic ≈ -dot(z, res) / norm(z) atol=1e-6
  @test sc.converged
  for family in (:binomial, :poisson)
    R = randn(rng, 120, 4) ./ 2
    yy = family == :binomial ? Float64.(rand(rng, 120) .< 0.5) : Float64.(rand(rng, 0:3, 120))
    d = debiased_lasso(R, yy, 0.06; family, nodewise_lambda = 0.1, options = opt)
    @test d.converged
    @test all(isfinite, coef(d)) && all(>(0), stderror(d))
    s = decorrelated_score(R, yy, 1, 0.06; family, null = 0.2, options = opt)
    @test s.converged && 0 <= s.pvalue <= 1
    @test s.information > 0
    @test_throws ArgumentError bootstrap_lasso(R, yy, d; B = 2)
  end
  one = decorrelated_score(X[:, 1:1], y, 1, 0.1; family = :gaussian, sigma = 1)
  @test isfinite(one.statistic)
  @test_throws ArgumentError nodewise_lasso(ones(10, 2))
  @test_throws ArgumentError nodewise_lasso(X, -0.1)
  @test_throws ArgumentError nodewise_lasso(X; targets = [1, 1])
  @test_throws ArgumentError nodewise_lasso(X; lambda_path = [0.1, 0.2])
  @test_throws ArgumentError debiasing_matrix(X; method = :qp, mu = 1)
  @test_throws ArgumentError debiasing_matrix(X; method = :qp, gram = false)
  @test_throws ArgumentError debiased_lasso(X, y, 0.1; sigma = -1)
  @test_throws DimensionMismatch debiased_lasso(X, y[1:5])
  @test_throws ArgumentError multiplier_bootstrap(r; B = 0)
  @test_throws DimensionMismatch multiplier_bootstrap(r; multipliers = ones(3, 5))
  @test_throws ArgumentError confint(r; level = 1)
  @test_throws ArgumentError decorrelated_score(X, y, 9, 0.1)
end
