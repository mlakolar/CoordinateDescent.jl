module QuantileTuningTests
using HighDimensionalModels, HiGHS, Random, Statistics, LinearAlgebra, SparseArrays, Test
const HDM = HighDimensionalModels

@testset "fixed-quantile pivotal calibration" begin
  n,p = 79,5
  X = randn(Xoshiro(41),n,p) .+ (1:p)'
  U = rand(Xoshiro(13),n,64)
  w = [.7,1.,1.2,.8,2.]
  for tau in (.1,.5,.8), intercept in (false,true), sp in (false,true)
    A = sp ? sparse(X) : X
    Z = intercept ? X .- mean(X;dims=1) : X
    expected = [maximum(abs(sum(Z[i,j]*(tau-(U[i,b]<tau)) for i in 1:n))/(n*w[j]) for j in 1:p) for b in axes(U,2)]
    @test HDM._quantile_score_maxima(A,w,tau,intercept,U) ≈ expected
    λ = quantile_lambda(A;tau,intercept,standardize_x=false,penalty_factors=w,nsim=64,rng=Xoshiro(13))
    @test λ ≈ 2quantile(expected,.9)
    @test quantile_lambda(A;tau,intercept,standardize_x=false,penalty_factors=w,nsim=64,c=1.,gamma=.2,rng=Xoshiro(13)) ≈ quantile(expected,.8)
  end
  a = quantile_lambda(X;nsim=150,rng=Xoshiro(6))
  @test a == quantile_lambda(X;nsim=150,rng=Xoshiro(6))
  @test a ≈ quantile_lambda(X .* (1:p)';nsim=150,rng=Xoshiro(6))
  @test a ≈ quantile_lambda(X .+ 20;nsim=150,rng=Xoshiro(6))
  @test a ≈ quantile_lambda(sparse(X);nsim=150,rng=Xoshiro(6))
  @test quantile_lambda(round.(Int,X);nsim=64,rng=Xoshiro(3)) == quantile_lambda(Float64.(round.(Int,X));nsim=64,rng=Xoshiro(3))
  @test quantile_lambda(X;nsim=150,c=3.,rng=Xoshiro(6)) ≈ 1.5a
  for kw in ((tau=0.,),(tau=NaN,),(c=Inf,),(c=-1.,),(gamma=1.,),(gamma=NaN,),(nsim=1,),
             (penalty_factors=zeros(p),),(penalty_factors=ones(p+1),))
    @test_throws Exception quantile_lambda(X;kw...)
  end
  @test_throws ArgumentError quantile_lambda(ones(12,2))
  @test_throws ArgumentError quantile_lambda(zeros(0,2))
  @test_throws ArgumentError quantile_lambda(fill(NaN,5,2))
end

@testset "quantile check-loss cross-validation" begin
  n,p = 83,5
  X = randn(Xoshiro(71),n,p) .* (1:p)' .+ (1:p)'
  y = .8 .+ X*[1.,-.5,0,0,0] .+ randn(Xoshiro(5),n)
  offset = .2 .* sin.(1:n)
  folds = [(i%3)+1 for i in 1:n]
  ls = [.3,.15,.05]
  certified = HighDimensionalModels.ProximalSolvers._hasconjugate(HighDimensionalModels.ProximalBase.ConvolutionCheck(.5,.2))
  for method in (certified ? (:exact,:smooth) : (:exact,)), tau in (.2,.5,.8)
    cv = CVQuantileLassoPath(X,y,ls;folds,tau,method,offset,threads=false,rng=Xoshiro(19),tol=2e-5)
    expected = zeros(3,3)
    for k in 1:3
      tr,te = folds .!= k, folds .== k
      fp = QuantileLassoPath(X[tr,:],y[tr],ls;method,tau,offset=offset[tr],tol=2e-5,rng=Xoshiro(9))
      @test all(s.converged for s in fp.solutions)
      # Deliberately use the scalar check loss even for convolution fits.
      for j in 1:3
        r = y[te] .- predict(fp[j],X[te,:];offset=offset[te])
        expected[k,j] = mean(max.(tau.*r,(tau-1).*r))
      end
    end
    sizes = [count(==(k),folds) for k in 1:3]
    means = vec(sum(expected .* sizes;dims=1)) ./ n
    ses = [sqrt(sum(sizes.*(expected[:,j].-means[j]).^2)/n/2) for j in 1:3]
    @test cv.fold_errors ≈ expected atol=2e-5
    @test cv.cvm ≈ means atol=2e-5
    @test cv.cvse ≈ ses atol=2e-5
    @test cv.λmin == ls[argmin(cv.cvm)]
    @test cv.λ1se >= cv.λmin
    @test coef(cv) == coef(cv.path,cv.λ1se)
    @test dual_scores(cv) == dual_scores(cv.path[findfirst(==(cv.λ1se),ls)])
    @test dual_scores(cv;λ=:λmin) == dual_scores(cv.path[findfirst(==(cv.λmin),ls)])
    b = coef(cv;λ=:λmin)
    @test predict(cv,X;λ=:λmin,offset) ≈ b[1] .+ X*b[2:end] .+ offset
    @test nobs(cv) == n
    @test_throws ArgumentError coef(cv;λ=.123)
    @test_throws ArgumentError dual_scores(cv;λ=:other)
  end
  if !certified
    # Older solvers can return inaccurate smooth folds. Their API contract is
    # to exclude those levels, not to certify them or average fewer folds.
    for tau in (.2,.5,.8)
      legacy = CVQuantileLassoPath(X,y,ls;folds,tau,method=:smooth,offset,
                                  threads=false,rng=Xoshiro(19),tol=2e-5)
      for j in eachindex(ls)
        eligible = legacy.path[j].converged && all(isfinite,legacy.fold_errors[:,j])
        @test isfinite(legacy.cvm[j]) == eligible
        @test eligible || isnan(legacy.cvse[j])
      end
      @test all(isfinite,dual_scores(legacy))
    end
  end
  a = CVQuantileLassoPath(X,y,ls;folds,threads=false,rng=Xoshiro(2))
  b = CVQuantileLassoPath(sparse(X),y,ls;folds,threads=true,rng=Xoshiro(2))
  @test a.cvm ≈ b.cvm && a.folds == b.folds && a.λ1se == b.λ1se
  c = fit(CVQuantileLassoPath,X,y; nλ=5,nfolds=3,threads=false,rng=Xoshiro(7))
  @test length(c.λpath)==5 && size(c.fold_errors)==(3,5)
  @test isfinite(c.λmin) && isfinite(c.λ1se)
  # Every fold must contribute; never reward a penalty for a missing hard fold.
  errors = [.5 .2 .1; .4 NaN .2; .6 .3 .3]
  means,ses,jmin,j1se = HDM._quantile_cvsummary(a.path,errors,[28.,28.,27.])
  @test isinf(means[2]) && isnan(ses[2])
  @test jmin==3 && j1se in (1,3)
  invalid = CVQuantileLassoPath(a.path,a.λpath,means,ses,a.nnz,a.λmin,a.λ1se,a.folds,errors)
  @test_throws ArgumentError coef(invalid;λ=ls[2])
  @test_throws ErrorException HDM._quantile_cvsummary(a.path,fill(NaN,3,3),[28.,28.,27.])
  @test_throws ErrorException CVQuantileLassoPath(X,y,ls;folds,time_limit=1e-12,threads=false)
  for fl in (ones(Int,n),fill(0,n),[0;folds[2:end]],fill(2,n),ones(Int,n-1))
    @test_throws Exception CVQuantileLassoPath(X,y,ls;folds=fl)
  end
  @test_throws DimensionMismatch CVQuantileLassoPath(X,y[2:end],ls)
  @test_throws ArgumentError CVQuantileLassoPath(zeros(0,p),Float64[],ls)
  @test_throws ArgumentError CVQuantileLassoPath(X,y,[.1,.2])
end
end
