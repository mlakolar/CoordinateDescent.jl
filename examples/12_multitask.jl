# Multitask regression with task-specific designs
#
#     minimize  ∑ₜ cₜ ‖yₜ - β₀ₜ - Xₜbₜ‖² / (2nₜ) + λ ∑ⱼ ωⱼ ‖(s₁ⱼb₁ⱼ, …, s_qⱼb_qⱼ)‖₂
#
# for q regressions yₜ = β₀ₜ + Xₜbₜ + eₜ on the same p predictors, each with its
# own observations and sample size nₜ, with multitask_lasso,
# MultitaskLassoPath and CVMultitaskLassoPath. The penalty on the rows of
# B = [b₁ ⋯ b_q] selects a predictor for all the tasks or for none, sₜⱼ is the
# standard deviation of column j within task t, and the task weights are
# cₜ = 1 by default. A predictor that matters in every task is found from the
# observations of all of them, while each task keeps coefficients of its own.
#
#     julia --project=examples examples/12_multitask.jl

using HighDimensionalModels
using Printf, Random, Statistics

rng = Xoshiro(12)

# q = 3 tasks with nₜ = 60, 100 and 150 observations of p = 500 predictors. The
# first s = 10 predictors matter in every task, with coefficients of their own
# in each, between 0.5 and 1.5 in absolute value, and the intercepts differ.
q, p, s = 3, 500, 10
ns = [60, 100, 150]
B = zeros(p, q)
B[1:s, :] = (0.5 .+ rand(rng, s, q)) .* rand(rng, (-1, 1), s, q)
β₀ = [1.0, -0.5, 2.0]
Xs = [randn(rng, n, p) for n in ns]
ys = [β₀[t] .+ Xs[t] * B[:, t] + randn(rng, ns[t]) for t in 1:q]

# 1000 new observations of each task; the noise variance, 1, is the smallest
# prediction error
Xnew = [randn(rng, 1000, p) for _ in 1:q]
ynew = [β₀[t] .+ Xnew[t] * B[:, t] + randn(rng, 1000) for t in 1:q]

truesel(rows) = count(<=(s), rows)
falsesel(rows) = count(>(s), rows)

println("== Cross-validation within each task")

# Five folds drawn within each task, so that every fold holds out part of every
# task; each training fold centers, standardizes and fits on its own
# observations. The default task weights cₜ = 1 count the mean squared error of
# each task once, whatever its size; task_weights = :observations counts each
# observation once instead.
cv = CVMultitaskLassoPath(Xs, ys; nfolds = 5, rng = Xoshiro(1))
j = findfirst(==(cv.λ1se), cv.λpath)
B̂ = coef(cv)[2:end, :]                  # the intercepts are in the first row
rows = findall(k -> any(!iszero, view(B̂, k, :)), 1:p)
@printf("λ1se = %.4f, level %d of %d: %d rows selected\n", cv.λ1se, j, length(cv.λpath), length(rows))
println("selected rows ", rows, ", true support 1:$s")
println("held-out error of each task at λ1se (cvm_task) ", round.(cv.cvm_task[:, j]; digits = 2))

println("\n== Each task on its own")

# CVLassoPath on each task alone, with five folds, an intercept and
# standardized columns, against the multitask fit: the selected predictors, the
# cross-validated error at λ1se, and the error on the new observations, which
# predict(cv, Xnew) gives as a vector with one prediction vector per task. The
# multitask fit selects the same rows in every task, and the smaller tasks gain
# the most from the observations of the others.
ŷ = predict(cv, Xnew)
println("             multitask lasso                  lasso on each task")
println("task   nₜ   true  false  CV error  test error   true  false  CV error  test error")
for t in 1:q
  cvt = CVLassoPath(Xs[t], ys[t]; nfolds = 5, intercept = true, rng = Xoshiro(1))
  jt = findfirst(==(cvt.λ1se), cvt.λpath)
  sel = findall(!iszero, coef(cvt)[2:end])
  @printf("%4d  %4d   %4d  %5d  %8.2f  %10.2f   %4d  %5d  %8.2f  %10.2f\n", t, ns[t],
          truesel(rows), falsesel(rows), cv.cvm_task[t, j], mean(abs2, ynew[t] - ŷ[t]),
          truesel(sel), falsesel(sel), cvt.cvm[jt], mean(abs2, ynew[t] - predict(cvt, Xnew[t])))
end
