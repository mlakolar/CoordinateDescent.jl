import HD
reload("HD")

function f1()
  srand(123)
  n = 100
  p = 10

  X = randn(n, p)
  Y = X * ones(p) + 0.1 * randn(n)

  lambda = fill(3., p)
  # lambda = [3., 0.3]
  HD.lasso_raw(X, Y, lambda)
end

function f2()
  srand(123)
  n = 100
  p = 10

  X = randn(n, p)
  Y = X * ones(p) + 0.1 * randn(n)

  lambda = fill(3., p)
  # lambda = [3., 0.3]

  f = HD.CDLeastSquaresLoss(Y,X)
  HD.coordinateDescent(f, lambda)
end

function g()
  @time o1 = f1()
  @time o2 = f2()
  maximum(abs.(o1-o2))
end
