module SparseSLOPETests
using CoordinateDescent, ProximalBase, ProximalSolvers, LinearAlgebra, Random, SparseArrays, Statistics, Test
const CD = CoordinateDescent

@testset "Sparse SLOPE transformations and products" begin
  rng = Xoshiro(8201)
  X = sprand(rng, 70, 22, 0.15)
  X[:,1] .= 2.0                       # constant and zero columns
  X[:,2] .= 0.0
  X[1,3] = 1.0
  X[1,3] = 0.0                       # explicitly stored zero
  X[:,4] .*= 20
  saved = copy(X)
  y = randn(rng,70) .+ 3
  for intercept in (false,true), standardize_x in (false,true)
    Z,v,xb,yb,sc = CD._slope_data(X,y,intercept,standardize_x)
    D,vd,xbd,ybd,sd = CD._slope_data(Matrix(X),y,intercept,standardize_x)
    @test Z.X isa SparseMatrixCSC
    @test nnz(Z.X) == nnz(X)
    @test sc ≈ sd
    @test xb ≈ xbd
    @test v ≈ vd
    @test yb ≈ ybd
    # Sparse and dense reductions distribute roundoff differently. Supplying
    # atol alone disables isapprox's default relative tolerance on Julia 1.10.
    @test Matrix(Z) ≈ D rtol=100eps(Float64) atol=1e-14
    @test Z[:,[4,2,5]].X isa SparseMatrixCSC
    @test Matrix(Z[:,[4,2,5]]) ≈ D[:,[4,2,5]] rtol=100eps(Float64) atol=1e-14
    b,r = randn(rng,22),randn(rng,70)
    a = randn(rng,70); expected = 0.7D*b - 0.3a
    mul!(a,Z,b,0.7,-0.3)
    @test a ≈ expected atol=1e-12
    g = randn(rng,22); expected = 0.7D'*r - 0.3g
    mul!(g,Z',r,0.7,-0.3)
    @test g ≈ expected atol=1e-12
    @test Z*b ≈ D*b
    @test transpose(Z)*r ≈ D'*r
    @test dot(Z*b,r) ≈ dot(b,Z'*r)
    @test X == saved && nonzeros(X) == nonzeros(saved)
    # The loss uses mul! directly, without materializing the centered matrix.
    f = LeastSquaresLoss(v,Z)
    g = zeros(22)
    ProximalBase.gradient!(g,f,b)
    @test g ≈ D'*(D*b-v)
    @test f(b) ≈ sum(abs2,D*b-v)/2
  end
  @test_throws DimensionMismatch CD._slope_data(X,y[2:end],true,true)
  bad = copy(X); bad[2,3] = Inf
  @test_throws ArgumentError slope(bad,y,ones(22))
  @test_throws ArgumentError slope(X,fill(NaN,70),ones(22))
  @test_throws ArgumentError slope(spzeros(0,22),Float64[],ones(22))
end

@testset "Sparse SLOPE fit, path and full certificate" begin
  rng = Xoshiro(8202)
  X = sprand(rng,100,30,0.2)
  X[:,1] .= 0.7
  X[:,2] .= 0.0
  y = 1.5 .+ X[:,3]*3 - X[:,7]*2 + 0.2randn(rng,100)
  λ = slope_bh(30;n=100)
  for intercept in (false,true), standardize_x in (false,true)
    a = slope(X,y,λ;intercept,standardize_x,tol=1e-8)
    b = slope(Matrix(X),y,λ;intercept,standardize_x,tol=1e-8)
    @test a.converged && b.converged
    @test predict(a,X) ≈ predict(b,X) atol=2e-6
    @test a.x ≈ b.x atol=2e-6
    @test a.intercept ≈ b.intercept atol=2e-6
    @test residuals(a) ≈ y-predict(a,X)
    path = SLOPEPath(X,y;nλ=7,λmin_ratio=0.1,intercept,standardize_x,tol=1e-8)
    dense = SLOPEPath(Matrix(X),y,path.λpath;lambda=path.lambda,intercept,standardize_x,tol=1e-8)
    full = SLOPEPath(X,y,path.λpath;lambda=path.lambda,screening=false,intercept,standardize_x,tol=1e-8)
    @test all(path.converged) && all(dense.converged) && all(full.converged)
    @test predict(path,X) ≈ predict(dense,X) atol=3e-6
    @test predict(path,X) ≈ predict(full,X) atol=3e-6
    Z,v,_,_,s = CD._slope_data(X,y,intercept,standardize_x)
    scale = max(1.0,norm(Z'*v/length(v),Inf))
    for j in eachindex(path.λpath)
      b = path.βpath[j].*s
      grad = Z'*(Z*b-v)/length(v)
      trial = similar(b)
      prox!(trial,ProximalBase.SLOPE(path.λpath[j]*path.lambda),b-grad,1.0)
      r = norm(b-trial,Inf)
      @test r/scale <= 1e-8
    end
  end
  Z,v,_,_,_ = CD._slope_data(spdiagm(0 => fill(2.0,4)),[3.,2.,1.,0.],false,false)
  b = zeros(4)
  result = ProximalSolvers.solve!(b,(1/4)*LeastSquaresLoss(v,Z),
    ProximalBase.SLOPE([0.4,0.3,0.2,0.1]),ProximalSolvers.SLOPEAPG(;maxiter=500,tol=1e-9);candidates=Int[])
  @test result.converged && result.kkt_repairs >= 1
  @test b ≈ [1.1,0.7,0.3,0.] atol=1e-7
  zero_path = SLOPEPath(X,zeros(100);nλ=1,intercept=true,standardize_x=true)
  @test all(zero_path.converged) && all(iszero,coef(zero_path))
  @test slope(Float32.(X),Float32.(y),Float32.(λ);intercept=true,standardize_x=true,tol=1e-4).converged
  Xi = SparseMatrixCSC{Float64,Int32}(X)
  @test slope(Xi,y,λ;intercept=true,standardize_x=true).converged
  # Rank-deficient designs have potentially nonunique coefficients.
  A = hcat(X[:,3:8],X[:,3:8])
  a = slope(A,y,fill(0.01,12);tol=1e-8)
  b = slope(Matrix(A),y,fill(0.01,12);tol=1e-8)
  objective(s) = deviance(s)/200 + 0.01sum(abs,s.x)
  @test a.converged && b.converged
  @test predict(a,A) ≈ predict(b,A) atol=2e-5
  @test objective(a) ≈ objective(b) atol=1e-9
end

@testset "Sparse SLOPE storage and loss restriction" begin
  X = sparse(collect(1:10000),collect(1:10000),ones(10000),10000,20000)
  Z,v,_,_,_ = CD._slope_data(X,zeros(10000),true,true)
  @test Z.X isa SparseMatrixCSC && nnz(Z.X) == 10000
  # The dense transformed matrix would occupy 1.6 GB. This checks owned
  # representation, rather than a compiler-dependent allocation count.
  @test Base.summarysize(Z) < 2_000_000
  f = ProximalBase.restrict(LeastSquaresLoss(v,Z),collect(1:10))
  @test f.X.X isa SparseMatrixCSC && size(f.X) == (10000,10)
  @test Base.summarysize(f) < 1_000_000
end
end
