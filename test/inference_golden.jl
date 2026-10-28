using Random, LinearAlgebra, Statistics

@testset "inference golden data" begin
  readcsv = name -> begin
    path = joinpath(@__DIR__, "data", "inference", name)
    rows = [parse.(Float64, split(line, ',')) for line in readlines(path)[2:end]]
    reduce(vcat, transpose.(rows))
  end
  X = readcsv("X.csv"); y = vec(readcsv("y.csv")); ref = readcsv("hdi.csv")
  opt = CDOptions(; opt_tol = 1e-11, max_iter = 100000)
  P = nodewise_lasso(X, 0.1; standardize_x = false, options = opt, rng = Xoshiro(77))
  raw = readcsv("Z-raw.csv")
  normalized = raw ./ mean(raw .* X; dims = 1)
  @test P.scores ≈ normalized atol=2e-6
  r = debiased_lasso(X, y, 0.08; initial = ref[:, 1], sigma = 1,
                     standardize_x = false, projection = P, options = opt)
  @test coef(r) ≈ ref[:, 2] atol=2e-6
  @test stderror(r) ≈ ref[:, 3] atol=2e-6
  @test r.pvalues ≈ ref[:, 4] atol=2e-6
  q = debiasing_matrix(X; method = :qp, mu = 0.12, standardize_x = false, options = opt)
  @test Matrix(q.theta) ≈ readcsv("jm-theta.csv") atol=2e-6
  bs = bootstrap_lasso(X, y, r; resamples = readcsv("errors.csv"), options = opt, rng = Xoshiro(78))
  @test bs.draws ≈ readcsv("hdi-bootstrap-draws.csv") ./ ref[:, 3] atol=2e-5
  @test bs.null_draws ≈ readcsv("hdi-bootstrap-null-draws.csv") ./ ref[:, 3] atol=2e-5
  bref = readcsv("hdi-bootstrap.csv")
  @test bs.equal_tail_pvalues ≈ bref[:, 3] atol=1e-12
  @test bs.adjusted_pvalues ≈ bref[:, 4] atol=1e-12
  yy = readcsv("glm-y.csv")
  for (k, family) in enumerate((:binomial, :poisson))
    mle = readcsv("$(family)-mle.csv")
    fit = debiased_lasso(X, yy[:, k], 0.0; family, nodewise_lambda = 0.0, options = opt)
    @test fit.converged
    @test coef(fit) ≈ mle[:, 1] atol=2e-5
    @test stderror(fit) ≈ mle[:, 2] atol=2e-5
    score = decorrelated_score(X, yy[:, k], 2, 0.0; family, null = 0.25, options = opt)
    @test [score.score, score.information, score.statistic] ≈ vec(readcsv("$(family)-score.csv")) atol=2e-5
  end
end
