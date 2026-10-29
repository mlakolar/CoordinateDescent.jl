# Multitask row penalties with shared and task-specific supports
#
# Compare row ℓ1/ℓ2, row ℓ1/ℓ∞ and multitask sparse group by held-out error,
# selected rows and selected entries. A selected row can have zero entries.
#
#     julia --project=examples examples/13_multitask_penalties.jl

using HighDimensionalModels
using Random, Statistics

rng = Xoshiro(13)
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)
p, q = 30, 3
B = zeros(p, q)
B[1:3, :] .= [1.0, -0.8, 0.6]
for t in 1:q
  B[3 + t, t] = 1.2
end
β0 = [1.0, -0.5, 2.0]
penalties = (("row l1/l2", :l2, 0.0), ("row l1/linf", :linf, 0.0), ("sparse group", :l2, 0.5))
options = CDOptions(; opt_tol = 1e-7)

# The first three rows matter in every task, and rows 4, 5 and 6 in one task
# each. Cross-validation refits all preprocessing on the training rows.
println("== One design shared by the responses")
X, Xnew = centered_rand(rng, 90, p), centered_rand(rng, 150, p)
Y = X * B .+ β0' .+ 0.5 .* centered_rand(rng, 90, q)
Ynew = Xnew * B .+ β0' .+ 0.5 .* centered_rand(rng, 150, q)
folds = repeat(1:3, 30)
println("penalty          test MSE   selected rows   selected entries")
for (name, row_norm, alpha) in penalties
  cv = CVMultiLassoPath(X, Y, options; folds, row_norm, alpha, nλ = 18,
                        λminratio = 0.03, rng = Xoshiro(1), threads = false)
  fitted = Matrix(coef(cv))[2:end, :]
  rows = findall(j -> any(!iszero, fitted[j, :]), 1:p)
  test_error = mean(abs2, Ynew - predict(cv, Xnew))
  println(rpad(name, 14), "  ", round(test_error; digits = 3), "       ",
          length(rows), "               ", count(!iszero, fitted))
  println("  rows ", rows, "; true rows 1:6")
  println("  entries by task ", [findall(!iszero, fitted[:, t]) for t in 1:q])
end

# Different sample sizes, centers and scales in each task, with the same
# coefficient supports. Equal task weights count each task's MSE once.
println("\n== A design of its own for each task")
ns = [60, 90, 120]
Xs = [centered_rand(rng, n, p) .* (0.7 .+ rand(rng, p))' .+ t / 4 for (t, n) in enumerate(ns)]
ys = [β0[t] .+ Xs[t] * B[:, t] .+ 0.5 .* centered_rand(rng, ns[t]) for t in 1:q]
Xtest = [centered_rand(rng, 150, p) for _ in 1:q]
ytest = [β0[t] .+ Xtest[t] * B[:, t] .+ 0.5 .* centered_rand(rng, 150) for t in 1:q]
folds_task = [repeat(1:3, n ÷ 3) for n in ns]
println("penalty          test MSE   selected rows   selected entries")
for (name, row_norm, alpha) in penalties
  cv = CVMultitaskLassoPath(Xs, ys, options; folds = folds_task, row_norm, alpha,
                            nλ = 18, λminratio = 0.03, rng = Xoshiro(1), threads = false)
  fitted = Matrix(coef(cv))[2:end, :]
  rows = findall(j -> any(!iszero, fitted[j, :]), 1:p)
  predictions = predict(cv, Xtest)
  test_error = mean(mean(abs2, ytest[t] - predictions[t]) for t in 1:q)
  println(rpad(name, 14), "  ", round(test_error; digits = 3), "       ",
          length(rows), "               ", count(!iszero, fitted))
  println("  rows ", rows, "; true rows 1:6")
  println("  entries by task ", [findall(!iszero, fitted[:, t]) for t in 1:q])
end
