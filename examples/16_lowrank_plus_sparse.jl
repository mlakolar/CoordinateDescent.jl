# Low-rank-plus-sparse multivariate regression
#     julia --project=examples examples/16_lowrank_plus_sparse.jl
using HighDimensionalModels, LinearAlgebra, Random, Statistics

rng = Xoshiro(91)
uniform(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)

# --- A diffuse rank-one component and sparse departures ---
n, p, q = 48, 8, 4
L = uniform(rng, p, 1)*uniform(rng, 1, q)
S = zeros(p, q)
S[1, 1], S[3, 2], S[8, 4] = 2, -1.5, 1.2
B = L+S
β0 = [1.0, -0.5, 0, 2]
X = uniform(rng, n, p)
Y = X*B .+ β0' + 0.3uniform(rng, n, q)
Xnew = uniform(rng, 200, p)
Ynew = Xnew*B .+ β0' + 0.3uniform(rng, 200, q)
folds = repeat(1:3, 16)

# --- Tune the additive and single-matrix models on the same folds ---
# :row permits sparse predictor departures and matches the row-lasso baseline.
cv = CVPenaltyGrid(lowrank_plus_sparse, X, Y; sparsity = :row,
  nλ = (3, 3), λminratio = (0.03, 0.03), folds, threads = false, tol = 1e-8)
s = cv.grid.solutions[cv.index]
cvs = CVPenaltyGrid(sparse_nuclear_regression, X, Y; sparsity = :row,
  λ1 = cv.grid.λ1, λ2 = cv.grid.λ2, folds, threads = false, tol = 1e-8)
ss = cvs.grid.solutions[cvs.index]
println("Selected additive (rank, sparse) penalties: ", cv.λmin)
println("Relative gap: ", s.gap, "; rank of L: ", s.lowrank_rank,
  "; nonzero entries of S: ", s.sparse_nnz)
println("Component errors on this draw: L = ", norm(s.L-L), "; S = ", norm(s.S-S))

# --- Single-penalty baselines ---
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
@assert s.converged && ss.converged && sn.converged && sm.converged
println("method                 test MSE    coefficient error    rows    rank")
for (name, solution) in (("low rank plus sparse", s), ("sparse nuclear", ss),
                         ("nuclear", sn), ("row lasso", sm))
  coefficients = Matrix(solution.B)
  rows = count(row -> any(!iszero, row), eachrow(coefficients))
  numerical_rank = rank(coefficients; rtol = 1e-6)
  mse = round(mean(abs2, Ynew-predict(solution, Xnew)); digits = 4)
  error = round(norm(coefficients-B); digits = 4)
  println(rpad(name, 22), lpad(mse, 9), lpad(error, 21),
    lpad(rows, 8), lpad(numerical_rank, 8))
end
# This one draw illustrates the APIs; it does not establish a method ranking.
# Prediction accuracy and a small gap do not imply component recovery.
