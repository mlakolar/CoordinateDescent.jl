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

a = [1,2,3]
b = [1, 0, 0]
a += b

a = {[1:3],[1:10],[1:4]}
maximum(map(length, a))

F(i,j) = (i-1)*3+j
A = [ F(i,1) for i=1:3]


a = [1:10]
b = [1:3]
b[1:3] = a[5:7]
b[1] = 0
a

## test group lasso
using Convex
using Mosek
import HD
reload("HD")

n = 1000
p = 100
X = randn(n, p)
Y = X * ones(p) + 0.1 * randn(n)

solver = MosekSolver(LOG=0)
set_default_solver(solver);
lambda = 0.3
theta = Variable(p);
prob = minimize(sum_squares(Y-X*theta) / (2*n) + lambda * norm(theta, 2))
@time solve!(prob)
mBeta = theta.value


XtX = X' * X / n
S0 = - At_mul_B(X, Y) / n
beta = zeros(p)
@time for rep=1:1000
  fill!(beta, 0.)
  HD.minimize_one_group_1!(beta, X, Y, lambda)
end
beta = zeros(p)
@time for rep=1:1000
  fill!(beta, 0.)
  HD.minimize_one_group!(beta, X, Y, lambda)
end

@show maximum(abs(beta-mBeta))

## two groups
n = 100
p = 20
X = randn(n, p)
Y = X * ones(p) + 0.1 * randn(n)

lambda = 0.3
theta = Variable(p);
prob = minimize(sum_squares(Y-X*theta) / (2*n) + lambda * norm(theta[1:10], 2) + lambda * norm(theta[11:20], 2))
@time solve!(prob)
mBeta = theta.value


#function updateGroupBeta!(beta, XtX, Xy, groups, active_set, lambda; maxIter=1000, optTol=1e-7)
XtX = X' * X / n
Xy = X' * Y / n
beta = zeros(p)
groups = {[1:10],[11:20]}
active_set = [1,2]
lambdaG = [lambda, lambda]
HD.updateGroupBeta!(beta, XtX, Xy, groups, active_set, lambdaG)

maximum(abs(beta - mBeta))

