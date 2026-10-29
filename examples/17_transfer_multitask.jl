# Joint-sparse transfer from source domains to a target
#
# One draw with a small target and two larger sources: one source shares the
# target's coefficients, the other differs in two rows. TMTL(Fused),
# TMTL(Debiased), a target-only group lasso and a pooled fit with domain
# intercepts are tuned on the same target-only folds. The pooled fit uses very
# large contrast weights, which force common slopes. One draw does not rank
# the methods; transfer can also hurt when the sources differ more.
#
#     julia --project=examples examples/17_transfer_multitask.jl

using HighDimensionalModels
using LinearAlgebra, Random, Statistics

rng = Xoshiro(17)
centered_rand(rng, dims...) = sqrt(3) .* (2 .* rand(rng, dims...) .- 1)
p, K = 30, 4
n = [40, 150, 150]
B0 = zeros(p, K)
B0[1:4, :] .= [1.0, -0.8, 0.6, 0.5]
B0[2, :] .*= [0.5, 1.0, 1.5, 1.0]
shift = zeros(p, K)
shift[3, :] .= -0.6
shift[7, :] .= 0.5
blocks = [B0, B0, B0 .+ shift]
means = [[1.0, -0.5, 2.0, 0.0], [3.0, 1.0, -1.0, 0.5], [-2.0, 0.0, 1.0, 1.0]]
X = [centered_rand(rng, n[l], p) for l in 1:3]
Y = [X[l] * blocks[l] .+ means[l]' .+ 0.5 .* centered_rand(rng, n[l], K) for l in 1:3]
target, sources = (X[1], Y[1]), [(X[2], Y[2]), (X[3], Y[3])]
Xtest = centered_rand(rng, 500, p)
Ytest = Xtest * B0 .+ means[1]'

folds = repeat(1:5, n[1] ÷ 5)
options = CDOptions(; opt_tol = 1e-7)
grid = (; folds, nλ = 20, λminratio = 0.02, threads = false, rng = Xoshiro(1))

function report(name, W, prediction)
  error = norm(W - B0) / sqrt(K)
  rows = count(j -> any(!iszero, W[j, :]), axes(W, 1))
  println(rpad(name, 18), rpad(round(error; digits = 3), 18), rpad(round(mean(abs2, prediction - Ytest); digits = 3), 18),
          rows)
end

println("== Tuned fits on one draw")
cv = CVTransferMultitaskPath(target, sources, options; grid..., nλ̃ = 20)
pooled = CVTransferMultitaskPath(target, sources, options; grid..., families = ([1e6, 1e6],), nλ̃ = 1)
alone = CVMultiLassoPath(X[1], Y[1], options; folds, nλ = 20, λminratio = 0.02, standardize_x = false,
                         threads = false, rng = Xoshiro(1))
println("method            ‖W - B⁰‖/√K       noiseless MSE     nonzero rows")
report("TMTL(Fused)", cv.fused.W, predict(cv.fused, Xtest))
report("TMTL(Debiased)", cv.solution.W, predict(cv, Xtest))
report("target only", Matrix(coef(alone; λ = :λmin))[2:end, :], predict(alone, Xtest; λ = :λmin))
report("pooled", pooled.fused.W, predict(pooled.fused, Xtest))
println("selected family ", cv.family, ", λ₀ = ", round(cv.λ₀; sigdigits = 3), ", λ̃ = ", round(cv.λ̃; sigdigits = 3))
println("contrast rows of source 2: ", findall(j -> any(!iszero, cv.fused.contrasts[2][j, :]), 1:p), "; shifted rows 3, 7")

println("\n== Diagnostics of the selected fit")
sol = cv.solution
println("fused: objective ", round(sol.objective; sigdigits = 6), ", relative gap ", sol.gap, ", converged ", sol.converged)
println("correction: objective ", round(sol.correction_objective; sigdigits = 6), ", relative gap ",
        sol.correction_gap, ", converged ", sol.correction_converged)
admm = transfer_multitask(target, sources, cv.λ₀; contrast_weights = cv.family, solver = :admm,
                          admm = (; abstol = 1e-10, reltol = 1e-10))
println("ADMM at the selected level: ", admm.iterations, " iterations, objective ",
        round(admm.objective; sigdigits = 6), ", primal and dual residuals ",
        round(admm.primal_residual; sigdigits = 2), ", ", round(admm.dual_residual; sigdigits = 2))
