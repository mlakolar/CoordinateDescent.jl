# Simultaneously sparse and low-rank multivariate regression
#     julia --project=examples examples/14_sparse_nuclear.jl
using HighDimensionalModels, LinearAlgebra, Random, Statistics, Printf

rng = Xoshiro(92)
uniform(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)

# --- A sparse rank-two coefficient matrix ---
n, p, q = 60, 10, 6
U = zeros(p, 2)
U[1:2, 1] = [1, 0.7]
U[3:4, 2] = [1, -0.8]
V = [1.0 0; 0.7 0; 0.5 0; 0 1; 0 0.8; 0 0.6]
B = U * V'
β0 = [1.0, -0.5, 0, 2, 1, -1]
X = uniform(rng, n, p)
Y = X*B .+ β0' + 0.4uniform(rng, n, q)
Xnew = uniform(rng, 200, p)
Ynew = Xnew*B .+ β0' + 0.4uniform(rng, 200, q)
folds = repeat(1:3, 20)

# --- Tune both levels on a small grid ---
# :entry penalizes absolute entries; :row selects predictors across responses.
cv = CVPenaltyGrid(sparse_nuclear_regression, X, Y; sparsity = :row,
  nλ = (3, 3), λminratio = (0.03, 0.03), folds, threads = false, tol = 1e-8)
s = cv.grid.solutions[cv.index]
println("Selected (rank, sparse) penalties: ", cv.λmin)
println("Relative gap: ", s.gap, "; rank: ", s.rank, "; rank tolerance: ", s.rank_tol)

# --- Single-penalty baselines, tuned on the same folds ---
cvn = CVLowRankPath(X, Y; method = :nuclear, λpath = cv.grid.λ1, folds, tol = 1e-9)
sn = cvn.path.solutions[cvn.index]
options = CDOptions(; opt_tol = 1e-10, gaptol = 0)
errors = zeros(length(cv.grid.λ2))
for k in 1:3
  tr, te = findall(!=(k), folds), findall(==(k), folds)
  for (j, λ) in enumerate(cv.grid.λ2)
    solution = multi_lasso(X[tr, :], Y[tr, :], λ, options; standardize_x = false)
    @assert solution.converged
    errors[j] += sum(abs2, Y[te, :] - predict(solution, X[te, :])) / length(Y)
  end
end
sm = multi_lasso(X, Y, cv.grid.λ2[argmin(errors)], options; standardize_x = false)
@assert s.converged && sn.converged && sm.converged
println("method                 test MSE    coefficient error    rows    rank")
for (name, solution) in (("sparse nuclear", s), ("nuclear", sn), ("row lasso", sm))
  coefficients = Matrix(solution.B)
  rows = count(row -> any(!iszero, row), eachrow(coefficients))
  numerical_rank = rank(coefficients; rtol = 1e-6)
  @printf("%-22s %8.4f %20.4f %7d %7d\n", name,
    mean(abs2, Ynew - predict(solution, Xnew)), norm(coefficients-B), rows, numerical_rank)
end
# This one draw illustrates the APIs; it does not establish a method ranking.
