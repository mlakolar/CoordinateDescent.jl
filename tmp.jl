using JuMP
using Mosek
using HD
reload("HD")

solver = MosekSolver(LOG=1,OPTIMIZER=Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)
solver = MosekSolver(LOG=1)


p = 10
n = 100
X = randn(n, p)
tX = hcat(ones(n), X)
Y = X * ones(p) + 0.1 * randn(n)

qr_problem = HD.QR(solver, tX, Y)
lambda = vcat(fill(0., p+1))
@time hBeta, hXi = HD.solve!(qr_problem, lambda, 0.5)


for k = hBeta
  @show "$(k[1]) -> $(k[2])"
end


