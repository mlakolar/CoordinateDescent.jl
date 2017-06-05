
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

    lambda = fill(maximum(abs.(Xy)) + 0.1, p)
    beta = lasso(X, Y, lambda)
    @fact beta --> spzeros(p)
  end

  context("non-zero") do
    for i=1:100
      n = 100
      p = 10
      s = 5

      X = randn(n, p)
      Y = X[:,1:s] * ones(s) + 0.1 * randn(n)

      lambda = fill(0.3, p)
      beta = lasso(X, Y, lambda, CDOptions(;optTol=1e-12))

      f = CDQuadraticLoss(X'X/n, -X'Y/n)
      g = ProximalBase.AProxL1(1., lambda)
      x1 = SparseIterate(p)
      coordinateDescent!(x1, f, g, CDOptions(;optTol=1e-12))
      @fact beta --> roughly(x1; atol=1e-5)

      @fact (maximum(abs.(X'*(Y - X*beta) / n)) - 0.3) / 0.3 --> roughly(0.; atol=1e-5)
    end
  end

end

facts("cd lasso") do

  for i=1:100
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

  srand(123)
  for i=1:100
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

    @fact (maximum(abs.(X'*(Y - X*x1) / vecnorm(Y - X*x1))) - λ) / λ  --> roughly(0.; atol=1e-3)

    x2 = Convex.Variable(p)
    prob = Convex.minimize(Convex.vecnorm(Y-X*x2) + λ * vecnorm(x2, 1))
    Convex.solve!(prob)

    @fact (maximum(abs.(X'*(Y - X*x2.value) / vecnorm(Y - X*x2.value))) - λ) / λ --> roughly(0.; atol=3e-3)
    @fact maximum(abs.(x1 - x2.value)) --> roughly(0.; atol=2e-3)
  end


end


facts("scaled lasso") do

  srand(223)
  for i=1:100
    n = 1000
    p = 500
    s = 5

    X = randn(n, p)
    β = randn(s)
    Y = X[:,1:s] * β + randn(n)

    λ = rand() / 5.

    x, σh = scaledLasso(X, Y, λ, ones(p),
                          ScaledLassoOptions(;maxIter=100,
                           optTol=1e-12,
                           optionsCD=CDOptions(;maxIter=5000, optTol=1e-8)))

    @fact max.((maximum(abs.(X'*(Y - X*x) / n)) - λ*σh), 0.) / (σh*λ) --> roughly(0.; atol=1e-5)
  end


end
