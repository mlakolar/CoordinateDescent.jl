using Test, Random, LinearAlgebra, SparseArrays, StatsAPI

@testset "Gaussian SLOPE calibration" begin
  # Independent normal quantiles and the two-term published recurrence.
  b1, b2 = 1.959963984540054, 1.6448536269514722
  @test slope_gaussian(1;n=100)[1] ≈ 1.6448536269514722 / 10
  @test slope_gaussian(2;n=100) ≈ [b1,b2*sqrt(1+b1^2/98)] ./ 10
  @test slope_gaussian(2;n=100,intercept=true) ≈ [b1,b2*sqrt(1+b1^2/97)] ./ 10
  for n in [2,3,10,100], p in [1,2,10,200], q in [1e-12,0.1,1-1e-12], intercept in [false,true]
    n-Int(intercept)==1 && p>1 && continue
    a = slope_gaussian(p;n,q,intercept)
    @test length(a)==p && all(isfinite,a) && all(>=(0),a) && issorted(a;rev=true)
    @test slope_gaussian(p;n,q,intercept,sigma=3) ≈ 3a
    @test all(iszero,slope_gaussian(p;n,q,intercept,sigma=0))
    @test all(a .>= slope_bh(p;n,q) .- 10eps(Float64))
  end
  @test slope_gaussian(5;n=2) == fill(slope_bh(5;n=2)[1],5)
  for call in [()->slope_gaussian(0;n=100), ()->slope_gaussian(2;n=0),
               ()->slope_gaussian(2;n=1), ()->slope_gaussian(1;n=1,intercept=true),
               ()->slope_gaussian(2;n=100,q=0), ()->slope_gaussian(2;n=100,q=1),
               ()->slope_gaussian(2;n=100,q=NaN), ()->slope_gaussian(2;n=100,sigma=-1),
               ()->slope_gaussian(2;n=100,sigma=Inf)]
    @test_throws ArgumentError call()
  end

  rng = Xoshiro(10701)
  for T in [Float32,Float64], intercept in [false,true], standardize_x in [false,true]
    X = T.(randn(rng,70,12)); X[:,end] .= 2
    intercept && (X[:,1] .+= T(1000))
    y = T.(randn(rng,70)) .+ X[:,2] .- T(.5).*X[:,3]
    original_X, original_y = copy(X),copy(y)
    tol = T==Float32 ? 2e-4 : 1e-9
    weights = slope_gaussian(12;n=70,intercept)
    a = slope(X,y;calibration=:gaussian,intercept,standardize_x,tol)
    b = slope(X,y,weights;intercept,standardize_x,tol)
    c = slope(sparse(X),y;calibration=:gaussian,intercept,standardize_x,tol)
    @test a.converged && b.converged && c.converged
    @test a.x == b.x && a.intercept == b.intercept
    @test a.lambda == T.(weights)
    @test c.x ≈ a.x rtol=10tol atol=10tol
    @test a.calibration.method==:gaussian && b.calibration.method==:user
    @test a.calibration.sigma==1 && a.calibration.noise==:supplied_sigma
    @test a.calibration.effective_n==70-Int(intercept)
    @test a.calibration.standardize_x==standardize_x
    @test X==original_X && y==original_y
    @test eltype(a.x)==T
    @test coef(a) == (intercept ? [a.intercept;a.x] : a.x)
    @test predict(a,X) ≈ predict(b,X)
    factors = [1.,.8,.6]
    pa = SLOPEPath(X,y,factors;calibration=:gaussian,intercept,standardize_x,tol)
    pb = SLOPEPath(X,y,factors;lambda=weights,intercept,standardize_x,tol)
    @test all(pa.converged) && all(pb.converged)
    @test coef(pa)==coef(pb) && pa.lambda==pb.lambda
    @test pa.calibration==a.calibration && pb.calibration.method==:user
  end
  X,y = randn(rng,50,5),randn(rng,50)
  bh = slope(X,y)
  @test bh.lambda==slope_bh(5;n=50) && bh.calibration.method==:bh
  supplied = fill(.2,5)
  a = slope(X,y;lambda=supplied,calibration=:gaussian,q=NaN,sigma=NaN)
  @test a.lambda==supplied && a.calibration.method==:user
  @test a.calibration.q===nothing && a.calibration.sigma===nothing
  p = SLOPEPath(X,y,[1.,.5];lambda=supplied,calibration=:gaussian,q=NaN,sigma=NaN)
  @test p.lambda==supplied && p.calibration.method==:user
  @test_throws ArgumentError slope(X,y;calibration=:user)
  @test_throws ArgumentError SLOPEPath(X,y;calibration=:user)
  @test_throws ArgumentError slope(X,y;lambda=supplied,calibration=:unknown)
  @test_throws ArgumentError SLOPEPath(X,y;lambda=supplied,calibration=:unknown)
  @test_throws ArgumentError SLOPEPath(X,y;calibration=:gaussian,sigma=0)
  z = slope(X,y;calibration=:gaussian,sigma=0)
  @test all(iszero,z.lambda) && z.calibration.sigma==0
  # Legacy positional construction retains accessors and identifies missing metadata.
  old = SLOPESolution(a.x,a.intercept,a.hasintercept,a.residuals,a.lambda,a.iterations,a.converged,a.kkt_residual)
  @test coef(old)==coef(a) && old.calibration.method==:unspecified
  oldpath = SLOPEPath(p.λpath,p.lambda,p.βpath,p.β0path,p.intercept,p.nobs,p.converged,p.iterations,p.kkt_residuals,p.working_sizes,p.kkt_repairs)
  @test coef(oldpath)==coef(p) && oldpath.calibration.method==:unspecified
end
