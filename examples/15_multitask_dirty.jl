# Dirty multitask regression with shared and task-specific supports
#
# Compare held-out error, selected rows and selected entries. Each joint method
# has 25 tuning candidates; each separate lasso has 25 levels on the same folds.
# The returned components are a canonical split, not identified latent features.
#
#     julia --project=examples examples/15_multitask_dirty.jl

using HighDimensionalModels
using Random, Statistics

rng = Xoshiro(15)
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)
p, q = 18, 3
B = zeros(p, q)
B[1:3, :] .= [1.0, -0.8, 0.6]
B[2, :] .*= [0.5, 1.0, 1.5]
for t in 1:q
  B[3 + t, t] = 1.2
end
β0 = [1.0, -0.5, 2.0]
options = CDOptions(; opt_tol = 1e-7)

function report(name, fitted, predictions, ytest)
  rows = findall(j -> any(!iszero, fitted[j, :]), axes(fitted, 1))
  error = mean(mean(abs2, ytest[t] - predictions[t]) for t in eachindex(ytest))
  println(rpad(name, 16), round(error; digits = 3), "       ", length(rows),
          "               ", count(!iszero, fitted))
  println("  rows ", rows, "; true rows 1:6")
end

function compare(Xs, ys, Xtest, ytest; shared)
  q = length(ys)
  folds = [repeat(1:3, length(y) ÷ 3) for y in ys]
  data = shared ? (Xs[1], hcat(ys...)) : (Xs, ys)
  cvfolds = shared ? folds[1] : folds
  println("method          test MSE   selected rows   selected entries")
  for shared_norm in (:linf, :l2)
    cv = CVDirtyMultitaskPath(data..., options; folds = cvfolds, shared_norm, nλ = 5,
                              λminratio = 0.03, rng = Xoshiro(1), threads = false)
    report("dirty $shared_norm", Matrix(cv.solution.B), predict(cv, Xtest), ytest)
    println("  selected ratio ", round(cv.ratio; digits = 3), "; shared rows in canonical split ",
            cv.solution.sharedrows)
  end
  for row_norm in (:linf, :l2)
    constructor = shared ? CVMultiLassoPath : CVMultitaskLassoPath
    cv = constructor(data..., options; folds = cvfolds, row_norm, nλ = 25,
                      λminratio = 0.03, rng = Xoshiro(1), threads = false)
    fitted = Matrix(coef(cv; λ = :λmin))[2:end, :]
    predictions = shared ? collect(eachcol(predict(cv, Xtest[1]; λ = :λmin))) : predict(cv, Xtest; λ = :λmin)
    report("row $row_norm", fitted, predictions, ytest)
  end
  separate = [CVLassoPath(Xs[t], ys[t], options; folds = folds[t], nλ = 25, λminratio = 0.03,
                          intercept = true, standardize_x = true, rng = Xoshiro(1), threads = false) for t in 1:q]
  fitted = hcat([Vector(coef(cv; λ = :λmin))[2:end] for cv in separate]...)
  predictions = [predict(separate[t], Xtest[t]; λ = :λmin) for t in 1:q]
  report("separate lasso", fitted, predictions, ytest)
end

println("== One design shared by the responses")
X, Xnew = centered_rand(rng, 75, p), centered_rand(rng, 120, p)
Y = X * B .+ β0' .+ 0.5 .* centered_rand(rng, 75, q)
Ynew = Xnew * B .+ β0' .+ 0.5 .* centered_rand(rng, 120, q)
compare(fill(X, q), collect(eachcol(Y)), fill(Xnew, q), collect(eachcol(Ynew)); shared = true)

println("\n== A design of its own for each task")
ns = [54, 72, 90]
Xs = [centered_rand(rng, n, p) .* (0.7 .+ rand(rng, p))' .+ t / 4 for (t, n) in enumerate(ns)]
ys = [β0[t] .+ Xs[t] * B[:, t] .+ 0.5 .* centered_rand(rng, ns[t]) for t in 1:q]
Xtest = [centered_rand(rng, 120, p) for _ in 1:q]
ytest = [β0[t] .+ Xtest[t] * B[:, t] .+ 0.5 .* centered_rand(rng, 120) for t in 1:q]
compare(Xs, ys, Xtest, ytest; shared = false)
