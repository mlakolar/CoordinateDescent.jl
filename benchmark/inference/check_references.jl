using CoordinateDescent, DelimitedFiles, Test, LinearAlgebra, Statistics, Random
using CovarianceSelection
readcsv(name) = readdlm(joinpath(ARGS[1],name),',',Float64;skipstart=1)
X = readcsv("X.csv"); y = vec(readcsv("y.csv")); ref = readcsv("hdi.csv")
options = CDOptions(;opt_tol=1e-11,max_iter=100000)
@testset "external inference references" begin
  P = nodewise_lasso(X,.1;intercept=true,standardize_x=false,options)
  @test all(P.converged)
  @test P.scores ≈ readcsv("Z.csv") atol=2e-6
  r = debiased_lasso(X,y,.08;initial=ref[:,1],sigma=1,standardize_x=false,projection=P,options)
  @test coef(r) ≈ ref[:,2] atol=2e-6
  @test stderror(r) ≈ ref[:,3] atol=2e-6
  @test r.pvalues ≈ ref[:,4] atol=2e-6
  jm = debiasing_matrix(X;method=:qp,mu=.12,standardize_x=false,options)
  @test Matrix(jm.theta) ≈ readcsv("jm-theta.csv") atol=2e-6
  # With fusion zero, CovarianceSelection's two neighborhood problems decouple.
  for j in (1,5,12)
    others = setdiff(1:12,j)
    opt = CovarianceSelection.ADMMOptions(;maxiter=100000,abstol=1e-10,reltol=1e-10)
    a,b = fusedNeighborhoodSelection(X[:,others],X[:,j],X[:,others],X[:,j],.1,0.;options=opt)
    inferred = -Vector(P.theta[j,others]) ./ P.theta[j,j]
    @test inferred ≈ a atol=2e-6
    @test inferred ≈ b atol=2e-6
  end
  boot = bootstrap_lasso(X,y,r;resamples=readcsv("errors.csv"),options,rng=Xoshiro(77))
  # hdi exports unstudentized centered draws; divide by its observed SE.
  @test boot.draws ≈ readcsv("hdi-bootstrap-draws.csv") ./ ref[:,3] atol=2e-5
  @test boot.null_draws ≈ readcsv("hdi-bootstrap-null-draws.csv") ./ ref[:,3] atol=2e-5
  bref = readcsv("hdi-bootstrap.csv")
  @test boot.equal_tail_pvalues ≈ bref[:,3] atol=1e-12
  @test boot.adjusted_pvalues ≈ bref[:,4] atol=1e-12
  glmy = readcsv("glm-y.csv")
  for (k,family) in enumerate((:binomial,:poisson))
    yy = glmy[:,k]
    mle = readcsv("$(family)-mle.csv")
    f = debiased_lasso(X,yy,0.;family,nodewise_lambda=0.,options)
    @test f.converged
    @test coef(f) ≈ mle[:,1] atol=2e-5
    @test stderror(f) ≈ mle[:,2] atol=2e-5
    s = decorrelated_score(X,yy,2,0.;family,null=.25,options)
    sr = vec(readcsv("$(family)-score.csv"))
    @test [s.score,s.information,s.statistic] ≈ sr atol=2e-5
  end
end
