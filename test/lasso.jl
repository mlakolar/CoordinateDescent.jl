
##############################################
#
#  Lasso
#
##############################################

facts("lasso") do

  context("zero") do
    n = 100
    p = 10

    X = randn(n, p)
    Y = X * ones(p) + 0.1 * randn(n)
    Xy = X' * Y / n

    lambda = maximum(abs.(Xy)) + 0.1
    out = lasso(X, Y, lambda)
    @fact out.x --> SparseIterate(p)
  end

  context("non-zero") do
    for i=1:NUMBER_REPEAT
      n = 100
      p = 10
      s = 5

      X = randn(n, p)
      Y = X[:,1:s] * ones(s) + 0.1 * randn(n)

      λ = fill(0.3, p)
      beta = lasso(X, Y, 1., λ, CDOptions(;optTol=1e-12))

      f = CDQuadraticLoss(X'X/n, -X'Y/n)
      g = ProximalBase.ProxL1(1., λ)
      x1 = SparseIterate( p )
      coordinateDescent!(x1, f, g, CDOptions(;optTol=1e-12))
      @fact beta.x --> roughly(x1; atol=1e-5)

      @fact (maximum(abs.(X'*(Y - X*beta.x) / n)) - 0.3) / 0.3 --> roughly(0.; atol=1e-5)
    end
  end

  context("different interfaces") do
    n = 500
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    λ = 0.1
    x1 = lasso(X, Y, λ)
    x2 = lasso(X, Y, λ, ones(p))

    @fact full(x1.x) --> roughly(full(x2.x); atol=1e-5)
  end

end

facts("cd lasso") do

  for i=1:NUMBER_REPEAT
    n = 200
    p = 50
    s = 10

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + 0.1 * randn(n)

    g = ProximalBase.ProxL1(0.2)
    f1 = CDQuadraticLoss(X'X/n, -X'Y/n)
    f2 = CDLeastSquaresLoss(Y, X)

    x1 = SparseIterate(p)
    x2 = SparseIterate(p)
    coordinateDescent!(x1, f1, g, CDOptions(;optTol=1e-12))
    coordinateDescent!(x2, f2, g, CDOptions(;optTol=1e-12))

    @fact maximum(abs.(x1 - x2)) --> roughly(0.; atol=1e-5)
    @fact (maximum(abs.(X'*(Y - X*x1) / n)) - 0.2) / 0.2 --> roughly(0.; atol=1e-5)
    @fact (maximum(abs.(X'*(Y - X*x2) / n)) - 0.2) / 0.2 --> roughly(0.; atol=1e-5)
  end

end


facts("cd sqrt-lasso") do

  context("kkt") do
    for i=1:NUMBER_REPEAT
      n = 100
      p = 50
      s = 5

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      λ = 2.8
      g = ProximalBase.ProxL1(λ)
      f = CDSqrtLassoLoss(Y, X)

      x1 = SparseIterate(p)
      coordinateDescent!(x1, f, g, CDOptions(;maxIter=5000, optTol=1e-8))

      @fact max(0, maximum(abs.(X'*(Y - X*x1) / vecnorm(Y - X*x1))) - λ) / λ  --> roughly(0.; atol=1e-3)
    end
  end

  context("interfaces") do
    for i=1:NUMBER_REPEAT
      n = 500
      p = 500
      s = 50

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      opt1 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=false)
      opt2 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=true, randomize=true)
      opt3 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=false)
      opt4 = CDOptions(;maxIter=5000, optTol=1e-8, warmStart=false, randomize=true)

      x1 = convert(SparseIterate, sprand(p, 0.6))
      x2 = convert(SparseIterate, sprand(p, 0.6))
      x3 = convert(SparseIterate, sprand(p, 0.6))
      x4 = convert(SparseIterate, sprand(p, 0.6))

      λ = 1.5
      g = ProximalBase.ProxL1(λ)
      f = CDSqrtLassoLoss(Y, X)

      coordinateDescent!(x1, f, g, opt1)
      coordinateDescent!(x2, f, g, opt2)
      coordinateDescent!(x3, f, g, opt3)
      coordinateDescent!(x4, f, g, opt4)

      @fact full(x1) --> roughly(full(x2); atol=1e-5)
      @fact full(x3) --> roughly(full(x2); atol=1e-5)
      @fact full(x4) --> roughly(full(x2); atol=1e-5)

      y1 = sqrtLasso(X, Y, λ, opt1, standardizeX=false)
      y2 = sqrtLasso(X, Y, λ, opt2, standardizeX=false)
      y3 = sqrtLasso(X, Y, λ, opt3, standardizeX=false)
      y4 = sqrtLasso(X, Y, λ, opt4, standardizeX=false)

      @fact full(y1.x) --> roughly(full(x2); atol=1e-5)
      @fact full(y2.x) --> roughly(full(x2); atol=1e-5)
      @fact full(y3.x) --> roughly(full(x2); atol=1e-5)
      @fact full(y4.x) --> roughly(full(x2); atol=1e-5)

      z1 = sqrtLasso(X, Y, λ, ones(p), opt1)
      z2 = sqrtLasso(X, Y, λ, ones(p), opt2)
      z3 = sqrtLasso(X, Y, λ, ones(p), opt3)
      z4 = sqrtLasso(X, Y, λ, ones(p), opt4)

      @fact full(z1.x) --> roughly(full(x2); atol=1e-5)
      @fact full(z2.x) --> roughly(full(x2); atol=1e-5)
      @fact full(z3.x) --> roughly(full(x2); atol=1e-5)
      @fact full(z4.x) --> roughly(full(x2); atol=1e-5)

    end
  end

end


facts("scaled lasso") do

  for i=1:NUMBER_REPEAT
    n = 1000
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    λ = rand() / 5.

    opt1 = IterLassoOptions(;maxIter=100, optTol=1e-8, optionsCD=CDOptions(;maxIter=5000, optTol=1e-8))
    opt2 = IterLassoOptions(;maxIter=100, optTol=1e-8, initProcedure=:InitStd, σinit=2., optionsCD=CDOptions(;maxIter=5000, optTol=1e-8))

    x1 = SparseIterate(p)
    x2 = SparseIterate(p)
    sol1 = scaledLasso!(x1, X, Y, λ, ones(p), opt1)
    sol2 = scaledLasso!(x2, X, Y, λ, ones(p), opt2)

    σ1 = get(sol1.σ)
    σ2 = get(sol2.σ)

    @fact max.((maximum(abs.(X'*(Y - X*x1) / n)) - λ*σ1), 0.) / (σ1*λ) --> roughly(0.; atol=1e-4)
    @fact max.((maximum(abs.(X'*(Y - X*x2) / n)) - λ*σ2), 0.) / (σ2*λ) --> roughly(0.; atol=1e-4)
    @fact full(x1) --> roughly(full(x2); atol=1e-4)
  end


end



facts("lasso path") do

  context("standardizeX = false") do
    n = 1000
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    λ1 = 0.3
    λ2 = 0.1
    opt = CDOptions(;maxIter=5000, optTol=1e-8)

    x1 = lasso(X, Y, λ1)
    x2 = lasso(X, Y, λ2)

    λpath = [λ1, λ2]
    path = LassoPath(X, Y, λpath, opt; standardizeX=false)

    @fact typeof(path) == LassoPath{Float64} --> true
    @fact full(path.βpath[1]) --> roughly(full(x1.x); atol=1e-5)
    @fact full(path.βpath[2]) --> roughly(full(x2.x); atol=1e-5)

    S1 = find(x1.x)
    S2 = find(x2.x)
    rf = refitLassoPath(path, X, Y)

    @fact rf[S1] --> roughly(X[:,S1] \ Y; atol=1e-5)
    @fact rf[S2] --> roughly(X[:,S2] \ Y; atol=1e-5)
  end

  context("standardizeX = true") do
    n = 1000
    p = 500
    s = 50

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    loadingX = Array{Float64}(p)
    _stdX!(loadingX, X)

    λ1 = 0.3
    λ2 = 0.1
    opt = CDOptions(;maxIter=5000, optTol=1e-8)

    x1 = lasso(X, Y, λ1, loadingX, opt)
    x2 = lasso(X, Y, λ2, loadingX, opt)

    λpath = [λ1, λ2]
    path = LassoPath(X, Y, λpath, opt)

    @fact typeof(path) == LassoPath{Float64} --> true
    @fact full(path.βpath[1]) --> roughly(full(x1.x); atol=1e-5)
    @fact full(path.βpath[2]) --> roughly(full(x2.x); atol=1e-5)

    S1 = find(x1.x)
    S2 = find(x2.x)
    rf = refitLassoPath(path, X, Y)

    @fact rf[S1] --> roughly(X[:,S1] \ Y; atol=1e-5)
    @fact rf[S2] --> roughly(X[:,S2] \ Y; atol=1e-5)
  end


end
