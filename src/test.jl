using Convex
using Mosek
using HD
reload("HD")


# ## compare to mosek

solver = MosekSolver(LOG=0)
set_default_solver(solver);

n = 500
p = 500
s = 10

rho = 0.5
covMat = rho.^abs([1:p]*ones(p)' - ones(p)*[1:p]')

X = randn(n, p) * sqrtm(covMat)
true_beta = vcat( (1.+rand(s)) .* (2.*rand(0:1,s).-1), zeros(p-s))
Y = X * true_beta + 0.1 .* randn(n)

lambda = 0.3
theta = Variable(p)
prob = minimize(sum_squares(Y-X*theta) / (2*n) + lambda * norm(theta, 1))
@time solve!(prob)

solShoot = spzeros(p,1)
@time HD.lasso_raw!(solShoot, X, Y, lambda * ones(p))

maximum(abs(solShoot - theta.value))

beta = spzeros(Float64, 10, 1)
length(find(beta))
beta[1] = eps()
length(find(beta))

reload("FactCheck")

