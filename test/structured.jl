module StructuredTest
using HighDimensionalModels, Test, LinearAlgebra, SparseArrays, Random, Statistics

# Exhaustive dual active sets give an independent exact reference for small
# problems, including redundant rows of D. No optimizer is reused here.
function dual_reference(X, y, D, λ)
  n, p = size(X)
  H = X'X / n
  h = X'y / n
  G = inv(H)
  Q, c = D * G * D', D * G * h
  m = size(D, 1)
  for signs in Iterators.product(ntuple(_ -> (-1, 0, 1), m)...)
    B, F = findall(!iszero, collect(signs)), findall(iszero, collect(signs))
    u = λ .* collect(signs)
    if !isempty(F)
      u[F] = pinv(Q[F, F]) * (c[F] - Q[F, B] * u[B])
    end
    g = Q*u - c
    maximum(abs, u; init = 0.) <= λ + 1e-8 || continue
    all(i -> abs(g[i]) <= 1e-8, F) || continue
    all(i -> signs[i] * g[i] <= 1e-8, B) || continue
    return G * (h - D'u)
  end
  error("no feasible dual active set")
end

@testset "difference and graph matrices" begin
  @test Matrix(difference_matrix(4)) == [-1 1 0 0; 0 -1 1 0; 0 0 -1 1]
  @test difference_matrix(6, 2) * collect(1.:6) ≈ zeros(4)
  x = [0., 0.2, 0.8, 1.9, 3., 5.]
  @test difference_matrix(6, 2; x) * x ≈ zeros(4) atol=1e-14
  @test difference_matrix(6, 3; x) * x.^2 ≈ zeros(3) atol=1e-13
  @test Matrix(graph_difference(3, [(1,2),(3,2)]; weights=[2.,3.])) == [-2 2 0; 0 3 -3]
  @test size(graph_difference(3, [])) == (0, 3)
  @test_throws ArgumentError difference_matrix(3, 3)
  @test_throws ArgumentError difference_matrix(3; x=[0.,0,1])
  @test_throws DimensionMismatch difference_matrix(3; x=[0.,1])
  @test_throws ArgumentError graph_difference(3, [(1,4)])
  @test_throws ArgumentError graph_difference(3, [(1,1)])
  @test_throws ArgumentError graph_difference(3, [(1,2)]; weights=[-1])
end

@testset "banded signal normal equations" begin
  rng = Xoshiro(830)
  for T in (Float32, Float64), order in 1:3, irregular in (false, true), rho in (T(.1), T(10))
    p = 31
    x = irregular ? cumsum(T.(.5 .+ rand(rng, p))) : nothing
    K = difference_matrix(p, order; x)
    gram = Diagonal(ones(T, p))
    F = HighDimensionalModels._split_factor(gram, K, rho; bandwidth=order)
    H = Matrix(gram) + rho * Matrix(K)' * Matrix(K)
    b, answer = T.(randn(rng, p)), zeros(T, p)
    original = copy(b)
    HighDimensionalModels._structured_solve!(answer, F, b)
    @test H * answer ≈ b rtol=200eps(T)
    @test answer ≈ H \ b rtol=200eps(T)
    @test b == original
  end
end

@testset "fused regression and signal fits" begin
  rng = Xoshiro(80)
  X, y = randn(rng, 35, 4), randn(rng, 35)
  D = [Matrix{Float64}(I, 4, 4); Matrix(difference_matrix(4))]
  s = fused_lasso(X, y, .03, .06; abstol=1e-10, reltol=1e-9)
  ref = dual_reference(X, y, [.5Matrix{Float64}(I,4,4); Matrix(difference_matrix(4))], .06)
  @test s.converged
  @test coef(s) ≈ ref atol=1e-7
  @test s.objective ≈ sum(abs2, y-X*s.x)/70 + .03sum(abs,s.x) + .06sum(abs,diff(s.x))
  @test predict(s, X) + residuals(s) ≈ y
  @test fitted(s) ≈ predict(s, X)
  @test nobs(s) == 35
  @test deviance(s) ≈ sum(abs2, residuals(s))
  @test fused_lasso(sparse(X), y, .03, .06).x ≈ s.x atol=1e-5
  @test fused_lasso(X, y, 0., 0.; abstol=1e-10, reltol=1e-9).x ≈ X\y atol=1e-7
  @test !fused_lasso(X, y, .03, .06; max_iter=0).converged
  signal = [0.,0,2,2,0,0]
  direct = fused_lasso(signal, .02, .03)
  ref = generalized_lasso(signal, [.02Matrix{Float64}(I,6,6); .03Matrix(difference_matrix(6))], 1.;
                           abstol=1e-10, reltol=1e-9)
  @test direct.iterations == 1
  @test direct.x ≈ ref.x atol=1e-7
  @test dof(direct) == 1  # only the middle, nonzero fused block is free
  @test dof(fused_lasso(signal, 1., 1.)) == 0
  @test fused_lasso(Float32.(signal), 0, .03).x isa Vector{Float32}
  @test_throws ArgumentError fused_lasso(X, y, -.1, .2)
  @test_throws ArgumentError fused_lasso(X, y, NaN, .2)
  @test_throws ArgumentError fused_lasso(X, fill(NaN,35), .1, .2)
  @test_throws ArgumentError fused_lasso(X, y, .1, .2; reltol=-1)
  shifted = fused_lasso(X .+ 2, y .+ 5, .03, .06; intercept=true, abstol=1e-10, reltol=1e-9)
  centered = fused_lasso(X .- mean(X;dims=1), y .- mean(y), .03, .06; abstol=1e-10, reltol=1e-9)
  @test shifted.x ≈ centered.x atol=1e-7
  @test mean(residuals(shifted)) ≈ 0 atol=1e-12
  @test length(coef(shifted)) == 5
  # Acceleration must retain the same optimum on a poorly conditioned design.
  correlated = copy(X)
  correlated[:,2] .= X[:,1] .+ .02 .* X[:,2]
  exact = dual_reference(correlated, y, [.5Matrix{Float64}(I,4,4); Matrix(difference_matrix(4))], .06)
  fast = fused_lasso(correlated, y, .03, .06; abstol=1e-10, reltol=1e-9)
  @test fast.converged
  @test fast.x ≈ exact atol=1e-7
  # Paths reuse centered normal equations, but must retain the original
  # objective and intercept. Check against exhaustive dual active sets.
  levels = [.12, .03]
  path = fused_lasso_path(X .+ 2, y .+ 5, levels; l1_ratio=.25, intercept=true,
                         abstol=1e-10, reltol=1e-9)
  Xc, yc = X .- mean(X;dims=1), y .- mean(y)
  for (λ, fit) in zip(levels, path.solutions)
    expected = dual_reference(Xc,yc,[.25Matrix{Float64}(I,4,4); .75Matrix(difference_matrix(4))],λ)
    @test fit.converged
    @test fit.x ≈ expected atol=1e-7
    @test fit.intercept ≈ mean(y .+ 5) - dot(vec(mean(X .+ 2;dims=1)), expected) atol=1e-7
  end
  # Fusion identifies an equal split of the one identifiable coefficient sum
  # even when the design has duplicate columns.
  a = collect(range(-1,1;length=40)); yy = 1.5a .+ .1sin.(1:40)
  duplicated = fused_lasso_path([a a], yy, levels; l1_ratio=.25, abstol=1e-10,reltol=1e-9)
  for (λ, fit) in zip(levels, duplicated.solutions)
    expected = max(0, (dot(a,yy)-length(a)*.25λ)/sum(abs2,a))/2
    @test fit.x ≈ fill(expected,2) atol=1e-7
  end
end

@testset "generalized lasso, graphs, paths and df" begin
  rng = Xoshiro(81)
  X, y = randn(rng,30,5), randn(rng,30)
  D = graph_difference(5, [(1,2),(2,3),(3,1),(3,4),(4,5),(1,2)])
  for λ in (.01, .1, 1.)
    s = generalized_lasso(X, y, D, λ; abstol=1e-10, reltol=1e-9)
    @test s.converged
    @test s.x ≈ dual_reference(X,y,Matrix(D),λ) atol=1e-7
    @test s.objective ≈ sum(abs2,y-X*s.x)/60 + λ*sum(abs,D*s.x)
    inactive = findall(v -> abs(v) < 1e-5, D*s.x)
    @test dof(s) == 5-rank(Matrix(D[inactive,:]))
  end
  @test generalized_lasso(X,y,D,0).x ≈ X\y
  signal = [0.,0,1,1,3,3]
  G = graph_difference(6, [(1,2),(2,3),(3,1),(4,5),(5,6)])
  @test dof(generalized_lasso(signal,G,10.; abstol=1e-10,reltol=1e-9)) == 2
  # Unobserved coefficient directions are identified by the penalty, not a ridge.
  R = sparse([1,2,3],[1,3,5],ones(3),3,6)
  s = generalized_lasso(R, [0.,1,2], difference_matrix(6), .1)
  @test s.converged
  @test dof(s) <= 3
  @test_throws ArgumentError generalized_lasso(zeros(3,6), ones(3), difference_matrix(6), .1)
  levels = [.3,.1,.02,0.]
  p = GeneralizedLassoPath(X,y,D,levels; rho=1., abstol=1e-10,reltol=1e-9)
  @test all(s -> s.converged, p.solutions)
  @test coef(p,.1) ≈ generalized_lasso(X,y,D,.1;abstol=1e-10,reltol=1e-9).x atol=1e-7
  @test predict(p,X) + residuals(p) ≈ repeat(y,1,4)
  @test length(aic(p)) == length(bic(p)) == 4
  @test bic(p)[2] ≈ 30log(deviance(p)[2]/30) + log(30)*dof(p)[2]
  @test_throws ArgumentError coef(p,.05)
  @test_throws ArgumentError GeneralizedLassoPath(X,y,D,[.1,.2])
  @test_throws ArgumentError GeneralizedLassoPath(X,y,D,[.1,.1])
  @test_throws ArgumentError GeneralizedLassoPath(X,y,D,Float64[])
  @test_throws DimensionMismatch generalized_lasso(X,y,ones(2,4),.1)
  @test_throws ArgumentError generalized_lasso(X,y,D,.1;rho=0.)
  @test generalized_lasso(Float32.(X),Float32.(y),D,.1).x isa Vector{Float32}
  onelevel = GeneralizedLassoPath(X,y,D,[.1])
  @test size(coef(onelevel)) == (5,1)
  @test size(predict(onelevel,X)) == (30,1)
end

@testset "trend filtering and irregular locations" begin
  rng = Xoshiro(82)
  y = randn(rng, 16)
  for order in 0:3, x in (nothing, cumsum(1 .+ rand(rng,16)))
    λ = .01
    s = trend_filter(y,λ;order,x,abstol=1e-10,reltol=1e-9)
    D = difference_matrix(16,order+1;x)
    r = generalized_lasso(y,D,λ;abstol=1e-10,reltol=1e-9,max_iter=100_000)
    @test s.converged
    @test r.converged
    @test s.x ≈ r.x atol=2e-6
    @test s.objective ≈ r.objective atol=1e-8
    @test dof(s) == dof(r)
    p = trend_filter_path(y,[.1,.01,0.];order,x,abstol=1e-9,reltol=1e-8)
    @test coef(p,.01) ≈ s.x atol=2e-6
    @test coef(p,0.) == y
  end
  @test trend_filter(y,.1;order=0).iterations == 1
  @test_throws ArgumentError trend_filter(y,.1;order=15)
  @test_throws ArgumentError trend_filter(y,.1;order=0,rho=0.)
  @test_throws ArgumentError trend_filter_path(y,[.1];order=0,rho=0.)
  @test !trend_filter(y,.1;max_iter=0).converged
  @test all(isnan, aic(trend_filter_path(y,[.1,.01];max_iter=0)))
end

@testset "signal objective certificates" begin
  for seed in 1:3
    rng = Xoshiro(8400 + seed)
    # The dual identity holds for every K, although production gap stopping
    # remains calibrated only for order-one Float64 signals.
    for n in (5, 17), order in 1:3, alpha in (.1, 10.), irregular in (false, true)
      x = irregular ? cumsum(.8 .+ .4rand(rng,n)) : nothing
      K = difference_matrix(n,order;x)
      x === nothing || (K = Diagonal([order/(x[j+order]-x[j]) for j in 1:n-order]) * K)
      D1 = Matrix(difference_matrix(n-order))
      beta, y, u = randn(rng,n), randn(rng,n), randn(rng,n-order)
      rho = .7
      gap, objective = HighDimensionalModels._structured_signal_gap!(
        zeros(n), zeros(n-order), K, beta, y, K*beta, u, rho, alpha)
      q = clamp.(-cumsum(rho*u)[1:end-1], -alpha, alpha)
      v = transpose(Matrix(K)) * transpose(D1) * q
      primal = sum(abs2,beta-y)/2 + alpha*sum(abs,D1*Matrix(K)*beta)
      dual = dot(y,v) - sum(abs2,v)/2
      @test maximum(abs,q) <= alpha
      @test objective ≈ primal rtol=2e-14
      @test gap ≈ primal-dual rtol=2e-14
      @test gap >= 0
    end
    y = 1 .+ sin.(1:5) .+ .2randn(rng,5)
    for x in (nothing, cumsum(.8 .+ .4rand(rng,5))), alpha in (.1, 10.)
      D = difference_matrix(5,2;x)
      lambda = alpha/5
      expected = dual_reference(Matrix{Float64}(I,5,5), y, Matrix(D), lambda)
      optimum = sum(abs2,expected-y)/10 + lambda*sum(abs,D*expected)
      fit = trend_filter(y,lambda;order=1,x)
      @test fit.converged
      @test norm(fit.x-expected)/norm(expected) <= 1e-4
      @test abs(fit.objective-optimum)/optimum <= 1e-6
      path = trend_filter_path(y,[2.,.02,0.];order=1,x)
      @test all(s.converged for s in path.solutions)
      @test coef(path,lambda) ≈ expected rtol=1e-4 atol=1e-6
    end
  end
  for y in (zeros(20), fill(3.,20))
    fit = trend_filter(y,.1;order=1)
    @test fit.converged
    @test fit.x ≈ y atol=1e-12
    @test fit.iterations == 1
  end
  y = 1 .+ sin.(1:15)
  absolute = trend_filter(y,.1;order=1,reltol=0,abstol=1e-8)
  @test absolute.primal_residual <= sqrt(14)*1e-8
  @test absolute.dual_residual <= sqrt(15)*1e-8
  single = trend_filter(Float32.(y),.1;order=1)
  @test single.x isa Vector{Float32}
  @test single.converged
end

@testset "default structured stopping accuracy" begin
  for seed in 1:3
    rng = Xoshiro(9700 + seed)
    X = randn(rng,60,4)
    for j in 2:4
      X[:,j] .= .95 .* X[:,j-1] .+ sqrt(1-.95^2) .* X[:,j]
    end
    y = X*[.5,-.5,.5,0.] + .1randn(rng,60)
    levels = [.1,.03,.003]
    G = graph_difference(4,[(1,2),(2,3),(3,4),(4,1),(1,2)])
    F = [.25Matrix{Float64}(I,4,4); .75Matrix(difference_matrix(4))]
    for (path,D) in ((GeneralizedLassoPath(X,y,G,levels),Matrix(G)),
                     (fused_lasso_path(X,y,levels;l1_ratio=.25),F))
      for (lambda,fit) in zip(levels,path.solutions)
        exact = dual_reference(X,y,D,lambda)
        optimum = sum(abs2,y-X*exact)/120 + lambda*sum(abs,D*exact)
        @test fit.converged
        @test norm(fit.x-exact)/max(norm(exact),eps()) <= 1e-4
        @test abs(fit.objective-optimum)/max(abs(optimum),eps()) <= 1e-6
      end
    end
    for order in 2:3, x in (nothing,cumsum(.8 .+ .4rand(rng,6)))
      signal = 1 .+ sin.(1:6) .+ .1randn(rng,6)
      D = Matrix(difference_matrix(6,order+1;x))
      path = trend_filter_path(signal,[10.,1.,.1]./6;order,x)
      for (lambda,fit) in zip(path.λpath,path.solutions)
        exact = dual_reference(Matrix{Float64}(I,6,6),signal,D,lambda)
        optimum = sum(abs2,signal-exact)/12 + lambda*sum(abs,D*exact)
        @test fit.converged
        @test norm(fit.x-exact)/norm(exact) <= 1e-4
        @test abs(fit.objective-optimum)/max(abs(optimum),eps()) <= 1e-6
      end
    end
    # Explicit tolerances still control both direct and cached regression.
    for path in (false,true)
      fit = path ? only(fused_lasso_path(X,y,[.03];l1_ratio=.25,
        abstol=1e-8,reltol=0).solutions) :
        fused_lasso(X,y,.0075,.0225;abstol=1e-8,reltol=0)
      @test fit.converged
      @test fit.dual_residual <= sqrt(4)*1e-8
    end
    # Returned-point verification also works at single precision, whose
    # defaults deliberately retain the original residual scale.
    single = fused_lasso(Float32.(X),Float32.(y),.0075,.0225)
    @test single.converged
    @test single.x isa Vector{Float32}
  end
end

@testset "CV has a common grid and respects training folds" begin
  rng = Xoshiro(83)
  X, y = randn(rng,36,5), randn(rng,36)
  D, levels, fl = difference_matrix(5), [.2,.08,.02], repeat(1:3,12)
  cv = CVGeneralizedLassoPath(X,y,D,levels;folds=fl,threads=false,abstol=1e-9,reltol=1e-8)
  E = zeros(3,3)
  for k in 1:3
    train, test = findall(!=(k),fl), findall(==(k),fl)
    p = GeneralizedLassoPath(X[train,:],y[train],D,levels;abstol=1e-9,reltol=1e-8)
    E[k,:] = vec(mean(abs2, predict(p,X[test,:]) .- y[test]; dims=1))
  end
  @test cv.cvm ≈ vec(mean(E;dims=1))
  @test coef(cv;λ=:λmin) == coef(cv.path,cv.λmin)
  @test predict(cv,X) ≈ predict(cv.path,X,cv.λ1se)
  @test cv.λ1se >= cv.λmin
  @test CVGeneralizedLassoPath(X,y,D,levels;folds=fl,threads=true).cvm ≈ cv.cvm atol=1e-5
  @test cv_fused_lasso(X,y,levels;folds=fl,threads=false).path.family == :fused
  ys = [0.2sin(i/3)+.02randn(rng) for i in 1:24]
  tc = cv_trend_filter(ys,[.01,.003];nfolds=3,rng=Xoshiro(1),threads=false)
  @test all(isfinite,tc.cvm)
  @test tc.path.family == :trend
  @test_throws ArgumentError CVGeneralizedLassoPath(X,y,D,levels;folds=zeros(Int,36))
  @test_throws ErrorException CVGeneralizedLassoPath(X,y,D,levels;max_iter=0,threads=false)
end

@testset "structured signal interfaces and safeguards" begin
  identity_design = HighDimensionalModels._SignalDesign{Float64}(6)
  @test Matrix(identity_design) == Matrix{Float64}(I,6,6)
  @test_throws BoundsError identity_design[0,1]
  @test_throws BoundsError identity_design[1,7]

  y = [1.,1.2,-.5,-.3,.8,.6]
  D, levels = difference_matrix(6), [.1,.03]
  path = GeneralizedLassoPath(y,D,levels)
  @test all(s -> s.converged,path.solutions)
  @test fitted(path) ≈ coef(path)
  @test fitted(path) + residuals(path) ≈ repeat(y,1,2)
  @test coef(fit(GeneralizedLassoPath,y,D,levels)) ≈ coef(path)
  fused_path = fused_lasso_path(y,levels;l1_ratio=.25)
  for (j,lambda) in enumerate(levels)
    exact = fused_lasso(y,.25lambda,.75lambda)
    @test fused_path.solutions[j].x ≈ exact.x
  end
  unpenalized = trend_filter(y,0.;order=2)
  @test unpenalized.converged
  @test unpenalized.x ≈ y
  @test dof(unpenalized) == length(y)

  # Unequal row weights are not graph incidence constraints. Degrees of
  # freedom must fall back to the actual nullspace instead of union-find.
  weighted = reshape([1.,2.],1,2)
  signal = [1.,2.]
  solution = generalized_lasso(signal,weighted,1.;abstol=1e-10,reltol=1e-10)
  @test solution.converged
  @test solution.x ≈ dual_reference(Matrix{Float64}(I,2,2),signal,weighted,1.) atol=1e-8
  @test norm(weighted*solution.x) <= 1e-8
  @test dof(solution) == 1

  # A deliberately perturbed normal-equation cache gives a false stationary
  # point. The original-design residual must reject it and resume iteration.
  X = Matrix{Float64}(I,3,3)
  response = [2.,-1.,.5]
  gram = X'X/3
  cached = fused_lasso(X,response,.03,.04;_gram=gram,_xy=zeros(3),
    abstol=1e-10,reltol=0)
  penalty = [.03Matrix{Float64}(I,3,3); .04Matrix(difference_matrix(3))]
  @test cached.converged
  @test cached.iterations > 1
  @test cached.dual_residual <= sqrt(3)*1e-10
  @test cached.x ≈ dual_reference(X,response,penalty,1.) atol=1e-8

  folds = repeat(1:3,2)
  cv = CVGeneralizedLassoPath(y,D,levels;folds,threads=false)
  explicit = CVGeneralizedLassoPath(Matrix(identity_design),y,D,levels;folds,threads=false)
  via_fit = fit(CVGeneralizedLassoPath,y,D,levels;folds,threads=false)
  @test cv.cvm ≈ explicit.cvm
  @test cv.cvse ≈ explicit.cvse
  @test via_fit.cvm ≈ cv.cvm
  @test coef(via_fit) ≈ coef(cv)
  fused_cv = cv_fused_lasso(y,levels;folds,threads=false)
  explicit_fused = cv_fused_lasso(Matrix(identity_design),y,levels;folds,threads=false)
  @test fused_cv.cvm ≈ explicit_fused.cvm
  @test fused_cv.cvse ≈ explicit_fused.cvse
end

@testset "component degrees of freedom against dense null spaces" begin
  HDM = HighDimensionalModels
  dense_df(A, D, b, tol) = begin
    inactive = findall(v -> abs(v) <= tol, D*b)
    N = isempty(inactive) ? Matrix{eltype(b)}(I,length(b),length(b)) : nullspace(Matrix(D[inactive,:]))
    size(N,2) == 0 ? 0 : rank(A*N)
  end
  rng = Xoshiro(99026)
  for T in (Float32,Float64), rep in 1:40
    p = rand(rng,3:22)
    edges = [(rand(rng,1:p),rand(rng,1:p)) for _ in 1:2p]
    filter!(e -> e[1] != e[2], edges)
    D = T.(graph_difference(p,edges;weights=.25 .+ rand(rng,length(edges))))
    pins = spdiagm(0 => T.(rand(rng,p) .> .6))
    rep % 2 == 0 && (D = [D;pins])
    b = T.(rand(rng,(-1.,0.,1.),p))
    A = T.(randn(rng,p+5,p))
    rep % 3 == 0 && (A[:,end] .= A[:,1])
    rep % 4 == 0 && (A = A[1:div(p,2),:])
    rep % 5 == 0 && (A = sparse(A))
    workspace = HDM._StructuredDFWorkspace(A,D)
    for tol in (T(0),T(.01),T(10))
      @test HDM._structured_df(A,D,b,tol) == dense_df(A,D,b,tol)
      @test HDM._structured_df(A,D,b,tol;workspace) == dense_df(A,D,b,tol)
      @test HDM._structured_df(A,D,copy(b),tol;workspace) == dense_df(A,D,b,tol)
    end
  end
  # Independent pin/cycle/repeated-edge fixture, cancellation, and rank guards.
  D = [graph_difference(5,[(1,2),(2,3),(3,1),(1,2),(4,5)];weights=[1.,2,3,1,4]); sparse([1],[4],[2.],1,5)]
  for A in (randn(rng,9,5),randn(rng,2,5),zeros(3,5)), b in (zeros(5),ones(5),collect(1.:5))
    @test HDM._structured_df(A,D,b,1e-4) == dense_df(A,D,b,1e-4)
  end
  A = [1. -1.;2. -2.]
  D = difference_matrix(2)
  @test HDM._structured_df(A,D,ones(2),0.) == dense_df(A,D,ones(2),0.)
  # Julia rank uses min(m,n)*eps, even for a very tall design. Preserve a
  # singular value lying between the min- and max-dimension cutoffs.
  A = zeros(50,3); A[1,1] = 1; A[2,2] = 5eps(Float64)
  D = spzeros(0,3)
  ws = HDM._StructuredDFWorkspace(A,D)
  @test ws.design_rank == rank(A) == 2
  @test HDM._structured_df(A,D,zeros(3),0.;workspace=ws) == dense_df(A,D,zeros(3),0.)
  D = graph_difference(3,[(1,2),(2,3)];weights=[1.,1e-20])
  @test HDM._incidence_components(sparse(D'),[1,2];check_rank=true) === nothing
  @test HDM._structured_df(Matrix{Float64}(I,3,3),D,zeros(3),0.) == dense_df(Matrix{Float64}(I,3,3),D,zeros(3),0.)
  # Generic weighted contrasts and trend constraints keep the SVD fallback.
  for D in (sparse([1. 2. 0.;0. 1. -1.]),difference_matrix(8,2))
    A,b = randn(rng,12,size(D,2)),zeros(size(D,2))
    @test HDM._structured_df(A,D,b,0.) == dense_df(A,D,b,0.)
  end
  X,y = randn(rng,60,10),randn(rng,60)
  for ratio in (0.,.5,1.), intercept in (false,true)
    path = fused_lasso_path(X,y,[.5,.1,.01,0.];l1_ratio=ratio,intercept)
    Ac = intercept ? X .- mean(X;dims=1) : X
    for (lambda,fit) in zip(path.λpath,path.solutions)
      D = ratio < 1 && lambda > 0 ? difference_matrix(10) : spzeros(0,10)
      ratio > 0 && lambda > 0 && (D = [D;spdiagm(0=>ones(10))])
      @test dof(fit) == dense_df(Ac,D,fit.x,1e-4) + intercept
    end
  end
end

end
