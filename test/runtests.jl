using FactCheck

using HD
using Convex, Mosek

solver = MosekSolver(LOG=0)
set_default_solver(solver)


facts("prox l2 norm") do
  context("shrink to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)
    lambda = norm(x) + 0.1
    prox_l2!(hat_x, x, lambda)
    @fact hat_x => roughly(zeros(Float64, p))
    @fact hat_x => zeros(Float64, p)

    # compare to Mosek
    theta = Variable(p)
    prob = minimize(sum_squares(theta - x) / 2 + lambda * norm(theta, 2))
    solve!(prob)
    @fact hat_x - vec(theta.value) => roughly(zeros(Float64, p); atol=5e-4)
  end

  context("non-zero") do
    # compare to Mosek output
    p = 10
    x = randn(p)
    hat_x = randn(p)
    lambda = 0.3
    prox_l2!(hat_x, x, lambda)

    theta = Variable(p)
    prob = minimize(sum_squares(theta - x) / 2 + lambda * norm(theta, 2))
    solve!(prob)
    @fact hat_x - vec(theta.value) => roughly(zeros(Float64, p); atol=5e-4)
  end
end

facts("minimize one group") do
  n = 100
  p = 10

  X = randn(n, p)
  Y = X * ones(p) + 0.1 * randn(n)

  XX = X' * X / n
  Xy = X' * Y / n

  context("non-zero") do
    # compare to Mosek
    lambda = 0.3
    theta = Variable(p);
    prob = minimize(sum_squares(Y-X*theta) / (2*n) + lambda * norm(theta, 2))
    solve!(prob)

    beta = zeros(p)
    HD.minimize_one_group_raw!(beta, X, Y, lambda)
    @fact beta => roughly(vec(theta.value); atol=5e-4)

    beta = zeros(p)
    HD.minimize_one_group!(beta, XX, Xy, lambda)
    @fact beta => roughly(vec(theta.value); atol=5e-4)
  end

  context("zero") do
    # compare to Mosek
    lambda = norm(Xy) + 0.1
    theta = Variable(p);
    prob = minimize(sum_squares(Y-X*theta) / (2*n) + lambda * norm(theta, 2))
    solve!(prob)

    beta = zeros(p)
    HD.minimize_one_group_raw!(beta, X, Y, lambda)
    @fact beta => roughly(vec(theta.value); atol=5e-4)

    beta = zeros(p)
    HD.minimize_one_group!(beta, XX, Xy, lambda)
    @fact beta => roughly(vec(theta.value); atol=5e-4)

    beta = zeros(p)
    HD.minimize_one_group!(beta, XX, Xy, lambda)
    @fact beta => zeros(Float64, p)
  end

end

facts("SoftThreshold") do

  @fact SoftThreshold(5., 3.) => 2.
  @fact SoftThreshold(-2., 3.) => 0.
  @fact SoftThreshold(-1.,1.) => 0.
  @fact SoftThreshold(-1.,0.9) => roughly(-0.1, 1e-5)

end


FactCheck.exitstatus()