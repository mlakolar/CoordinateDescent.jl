using FactCheck

using HD
using Convex
using Gurobi
# using GLPKMathProgInterface
# using SCS

solver = GurobiSolver()
#solver = SCSSolver(verbose=0)
set_default_solver(solver)

##############################################
#
#  Lasso
#
##############################################

facts("minimize_active_set") do

  context("one coordinate") do
    n = 100
    p = 1
    X = randn(n, p)
    Y = X * ones(p) + 0.1 * randn(n)
    XX = X'*X / n
    Xy = X'*Y / n

    lambda = [maximum(abs(Xy)) + 0.1]

    beta = sparsevec([1], [eps()], 1)
    HD.minimize_active_set!(beta, XX, Xy, lambda)
    @fact beta[1] --> 0.

    beta = sparsevec([1], [1.], 1)
    HD.minimize_active_set!(beta, XX, Xy, lambda)
    @fact beta[1] --> 0.

    lambda = [0.2]
    beta = sparsevec([1], [eps()], 1)
    HD.minimize_active_set!(beta, XX, Xy, lambda)
    @fact beta[1] --> roughly(SoftThreshold(Xy[1]/XX[1,1], lambda[1]/XX[1,1]))

    lambda = [0.2]
    HD.minimize_active_set!(beta, XX, Xy, lambda)
    @fact beta[1] --> roughly(SoftThreshold(Xy[1]/XX[1,1], lambda[1]/XX[1,1]))
  end

  context("more coordinates") do
    n = 100
    p = 10
    X = randn(n, p)
    Y = X * ones(p) + 0.1 * randn(n)
    XX = X'*X / n
    Xy = X'*Y / n

    lambda = fill(maximum(abs(Xy)) + 0.1, p)
    beta = sparsevec(collect(1:10), fill(1., 10), 10)
    HD.minimize_active_set!(beta, XX, Xy, lambda)
    @fact vec(full(beta)) --> roughly(zeros(Float64, p))

    lambda = fill(0.3, p)
    beta = sparsevec(collect(1:10), fill(eps(), p), p)
    HD.minimize_active_set!(beta, XX, Xy, lambda)

    theta = Variable(p)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + dot(lambda, abs(theta)))
    solve!(prob)
    @fact maximum(abs(vec(beta) - vec(theta.value))) --> roughly(0; atol=5e-4)
  end
end

facts("add_violating_index") do
  n = 100
  p = 10
  X = randn(n, p)
  Y = X * ones(p) + 0.1 * randn(n)
  XX = X'*X / n
  Xy = X'*Y / n

  beta = spzeros(Float64, p)
  lambda = fill(maximum(abs(Xy)) + 0.1, p)
  ind = HD.add_violating_index!(beta, XX, Xy, lambda)

  @fact ind --> 0
  @fact length(beta.nzval) --> 0

  beta = spzeros(Float64, p)
  lambda = fill(0.1, p)
  ind = HD.add_violating_index!(beta, XX, Xy, lambda)

  @fact ind --> indmax(abs(Xy))
  @fact length(beta.nzval) --> 1
  @fact beta[ind] --> roughly(eps())

end


facts("lasso") do

  context("zero") do
    n = 100
    p = 10

    X = randn(n, p)
    Y = X * ones(p) + 0.1 * randn(n)
    XX = X' * X / n
    Xy = X' * Y / n

    beta = spzeros(p)
    lambda = fill(maximum(abs(Xy)) + 0.1, p)
    HD.lasso_raw!(beta, X, Y, lambda)
    @fact beta --> spzeros(p)


    beta = sparsevec(collect(1:5), randn(5), p)
    HD.lasso_raw!(beta, X, Y, lambda)
    @fact beta --> spzeros(p)
  end

  context("non-zero") do
    n = 100
    p = 10
    s = 5

    X = randn(n, p)
    Y = X[:,1:s] * ones(s) + 0.1 * randn(n)
    XX = X' * X / n
    Xy = X' * Y / n

    beta = spzeros(p)
    lambda = fill(0.3, p)
    HD.lasso_raw!(beta, X, Y, lambda)

    theta = Variable(p)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + dot(lambda, abs(theta)))
    solve!(prob)
    @fact maximum(abs(vec(beta) - vec(theta.value))) --> roughly(0; atol=5e-4)

  end

end


##############################################
#
#  Group Lasso Functionallity
#
##############################################

facts("prox l2 norm") do
  context("shrink to zero") do
    p = 10
    x = randn(p)
    hat_x = randn(p)
    lambda = norm(x) + 0.1
    prox_l2!(hat_x, x, lambda)
    @fact hat_x --> roughly(zeros(Float64, p))
    @fact hat_x --> zeros(Float64, p)

    # compare to Mosek
    theta = Variable(p)
    prob = minimize(sumsquares(theta - x) / 2 + lambda * norm(theta, 2))
    solve!(prob)
    @fact hat_x - vec(theta.value) --> roughly(zeros(Float64, p); atol=5e-4)
  end

  context("non-zero") do
    # compare to Mosek output
    p = 10
    x = randn(p)
    hat_x = randn(p)
    lambda = 0.3
    prox_l2!(hat_x, x, lambda)

    theta = Variable(p)
    prob = minimize(sumsquares(theta - x) / 2 + lambda * norm(theta, 2))
    solve!(prob)
    @fact hat_x - vec(theta.value) --> roughly(zeros(Float64, p); atol=5e-4)
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
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda * norm(theta, 2))
    solve!(prob)

    beta = zeros(p)
    HD.minimize_one_group_raw!(beta, X, Y, lambda)
    @fact beta --> roughly(vec(theta.value); atol=5e-4)

    beta = zeros(p)
    HD.minimize_one_group!(beta, XX, Xy, lambda)
    @fact beta --> roughly(vec(theta.value); atol=5e-4)
  end

  context("zero") do
    # compare to Mosek
    lambda = norm(Xy) + 0.1
    theta = Variable(p);
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda * norm(theta, 2))
    solve!(prob)

    beta = zeros(p)
    HD.minimize_one_group_raw!(beta, X, Y, lambda)
    @fact beta --> roughly(vec(theta.value); atol=5e-4)

    beta = zeros(p)
    HD.minimize_one_group!(beta, XX, Xy, lambda)
    @fact beta --> roughly(vec(theta.value); atol=5e-4)

    beta = zeros(p)
    HD.minimize_one_group!(beta, XX, Xy, lambda)
    @fact beta --> zeros(Float64, p)
  end

end

facts("compute_group_residual") do
  n = 100
  p = 10
  X = randn(n, p)
  Y = X * ones(p) + 0.1 * randn(n)
  XX = X' * X / n
  Xy = X' * Y / n

  groups=Array(Array{Int64, 1}, 2)
  groups[1] = 1:5
  groups[2] = 6:10
  active_set=[1,2]

  res = zeros(Float64, 5)
  k = 1
  HD.compute_group_residual!(res, XX, Xy, zeros(p), groups, active_set, k)
  @fact res --> roughly(Xy[groups[k]])

  res = zeros(Float64, 5)
  beta = [ones(5); zeros(5)]
  k = 1
  HD.compute_group_residual!(res, XX, Xy, beta, groups, active_set, k)
  @fact res --> roughly(Xy[groups[k]])

  res = zeros(Float64, 5)
  beta = [ones(5); zeros(5)]
  k = 2
  HD.compute_group_residual!(res, XX, Xy, beta, groups, active_set, k)
  @fact res --> roughly(-XX[groups[k],groups[1]]*beta[groups[1]]+Xy[groups[k]])

  res = zeros(Float64, 5)
  beta = [ones(5); ones(5)]
  k = 2
  HD.compute_group_residual!(res, XX, Xy, beta, groups, active_set, k)
  @fact res --> roughly(-XX[groups[k],groups[1]]*beta[groups[1]]+Xy[groups[k]])

  res = zeros(Float64, 5)
  beta = [zeros(5); ones(5)]
  k = 1
  HD.compute_group_residual!(res, XX, Xy, beta, groups, active_set, k)
  @fact res --> roughly(-XX[groups[k],groups[2]]*beta[groups[2]]+Xy[groups[k]])
end


facts("minimize_active_groups") do
  n = 100
  p = 10
  X = randn(n, p)
  Y = X * ones(p) + 0.1 * randn(n)
  XX = X' * X / n
  Xy = X' * Y / n

  context("one active set") do
    groups=Array(Array{Int64, 1}, 2)
    groups[1] = collect(1:5)
    groups[2] = collect(6:10)
    active_set=[1]

    lambda = maximum(map((x)->norm(Xy[x]), groups)) + 0.1
    lambda = [lambda, lambda]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)
    @fact beta --> zeros(p)

    lambda = [0.3, 0.6]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)

    # compare to Mosek
    theta = Variable(5)
    prob = minimize(sumsquares(Y-X[:,1:5]*theta) / (2*n) + lambda[1] * norm(theta, 2))
    solve!(prob)

    @fact beta[1:5] - vec(theta.value) --> roughly(zeros(Float64, 5), 1e-3)

    active_set = [2]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)

    # compare to Mosek
    theta = Variable(5)
    prob = minimize(sumsquares(Y-X[:,6:10]*theta) / (2*n) + lambda[2] * norm(theta, 2))
    solve!(prob)

    @fact beta[6:10] - vec(theta.value) --> roughly(zeros(Float64, 5), 1e-3)
  end

  context("two active sets") do
    groups=Array(Array{Int64, 1}, 2)
    groups[1] = collect(1:5)
    groups[2] = collect(6:10)
    active_set=[1, 2]

    lambda_max = maximum(map((x)->norm(Xy[x]), groups)) + 0.1
    lambda = [lambda_max, lambda_max]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)
    @fact beta --> zeros(p)

    lambda = [0.3, lambda_max]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)

    # compare to Mosek
    theta = Variable(10)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda[1] * norm(theta[1:5], 2) + lambda[2] * norm(theta[6:10], 2))
    solve!(prob)

    @fact beta - vec(theta.value) --> roughly(zeros(Float64, 10), 1e-3)

    lambda = [lambda_max, 0.6]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)

    # compare to Mosek
    theta = Variable(10)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda[1] * norm(theta[1:5], 2) + lambda[2] * norm(theta[6:10], 2))
    solve!(prob)

    @fact beta - vec(theta.value) --> roughly(zeros(Float64, 10), 1e-3)

    lambda = [0.3, 0.6]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)

    # compare to Mosek
    theta = Variable(10)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda[1] * norm(theta[1:5], 2) + lambda[2] * norm(theta[6:10], 2))
    solve!(prob)

    @fact beta - vec(theta.value) --> roughly(zeros(Float64, 10), 1e-3)

    active_set = [2]
    beta = zeros(p)
    HD.minimize_active_groups!(beta, XX, Xy, groups, active_set, lambda)

    # compare to Mosek
    theta = Variable(5)
    prob = minimize(sumsquares(Y-X[:,6:10]*theta) / (2*n) + lambda[2] * norm(theta, 2))
    solve!(prob)

    @fact beta[6:10] - vec(theta.value) --> roughly(zeros(Float64, 5), 1e-3)
  end
end

facts("group lasso") do

  context("two groups") do
    n = 100
    p = 10
    X = randn(n, p)
    Y = X * ones(p) + 0.1 * randn(n)
    XX = X' * X / n
    Xy = X' * Y / n

    groups=Array(Array{Int64, 1}, 2)
    groups[1] = collect(1:5)
    groups[2] = collect(6:10)

    lambda_max = maximum(map((x)->norm(Xy[x]), groups)) + 0.1
    lambda = [lambda_max, lambda_max]
    beta = zeros(p)
    HD.group_lasso!(beta, XX, Xy, groups, lambda)
    @fact beta --> zeros(p)

    lambda = [0.3, lambda_max]
    beta = zeros(p)
    HD.group_lasso!(beta, XX, Xy, groups, lambda)

    # compare to Mosek
    theta = Variable(10)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda[1] * norm(theta[1:5], 2) + lambda[2] * norm(theta[6:10], 2))
    solve!(prob)

    @fact beta - vec(theta.value) --> roughly(zeros(Float64, 10), 1e-3)

    lambda = [lambda_max, 0.6]
    beta = zeros(p)
    HD.group_lasso!(beta, XX, Xy, groups, lambda)

    # compare to Mosek
    theta = Variable(10)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda[1] * norm(theta[1:5], 2) + lambda[2] * norm(theta[6:10], 2))
    solve!(prob)

    @fact beta - vec(theta.value) --> roughly(zeros(Float64, 10), 1e-3)

    lambda = [0.3, 0.6]
    beta = zeros(p)
    HD.group_lasso!(beta, XX, Xy, groups, lambda)

    # compare to Mosek
    theta = Variable(10)
    prob = minimize(sumsquares(Y-X*theta) / (2*n) + lambda[1] * norm(theta[1:5], 2) + lambda[2] * norm(theta[6:10], 2))
    solve!(prob)

    @fact beta - vec(theta.value) --> roughly(zeros(Float64, 10), 1e-3)
  end

  context("more groups") do
    n = 400
    p = 1000
    X = randn(n, p)
    Y = X[:,1:50] * ones(50) + 0.1 * randn(n)
    XX = X' * X / n
    Xy = X' * Y / n

    numG = div(p, 5)
    groups=Array(Array{Int64, 1}, numG)
    for i=1:numG
      groups[i] = (i-1)*5+1:i*5
    end
    lambda = 0.4 .* ones(numG)

    beta = zeros(p)
    HD.group_lasso!(beta, XX, Xy, groups, lambda)

    theta = Variable(p)
    normT = norm(theta[1:5])
    for i=2:numG
      normT = normT + norm(theta[(i-1)*5+1:i*5])
    end
    normT = 0.4 * normT

    prob = minimize(sumsquares(Y-X*theta) / (2*n) + normT)
    solve!(prob)
    @fact beta - vec(theta.value) --> roughly(zeros(Float64, p), 1e-3)

    beta = zeros(p)
    HD.group_lasso_raw!(beta, X, Y, groups, lambda)
    @fact beta - vec(theta.value) --> roughly(zeros(Float64, p), 1e-3)
  end
end

##############################################
#
#  Other
#
##############################################

facts("SoftThreshold") do

  @fact SoftThreshold(5., 3.) --> 2.
  @fact SoftThreshold(-2., 3.) --> 0.
  @fact SoftThreshold(-1.,1.) --> 0.
  @fact SoftThreshold(-1.,0.9) --> roughly(-0.1, 1e-5)

end

FactCheck.exitstatus()
