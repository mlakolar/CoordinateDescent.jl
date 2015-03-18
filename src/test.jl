SoftThreshold(5, 3)
SoftThreshold(-2, 3)
SoftThreshold(-1,1)
SoftThreshold(-1,0.9)

active_set = Array(Integer, 0)

## compare to mosek
using Convex
using Mosek
using HD
push!(LOAD_PATH, "/home/mkolar/projects/juliaPkg")
reload("HD")

solver = MosekSolver(LOG=1)
set_default_solver(solver);

n = 500
p = 5000
s = 10

rho = 0.5
covMat = rho.^abs([1:p]*ones(p)' - ones(p)*[1:p]')

X = randn(n, p) * sqrtm(covMat)
true_beta = vcat( (1.+rand(s)) .* (2.*rand(0:1,s).-1), zeros(p-s))
Y = X * true_beta + 0.1 .* randn(n)

lambda = 0.3
theta = Variable(p);
prob = minimize(sum_squares(Y-X*theta) / n + lambda * norm(theta, 1))
solve!(prob)
theta.value

@time solShoot = HD.LassoActiveShooting(X, Y, lambda * ones(p, p))

maximum(abs(solShoot - theta.value))

dot(vec(X[:,3]), Y) / norm(X[:, 3])^2 - lambda / norm(X[:, 3])^2 * n
dot(vec(X[:,3]), Y)

