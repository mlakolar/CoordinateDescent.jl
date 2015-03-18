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

n = 100
p = 10
X = randn(n, p)
Y = X * ones(p) + 0.1 * randn(n)

solver = MosekSolver(LOG=1)
set_default_solver(solver);
lambda = 0.3
theta = Variable(p);
prob = minimize(sum_squares(Y-X*theta) / (2*n) + lambda * norm(theta, 2))
@time solve!(prob)
mBeta = theta.value


beta = zeros(p)
XtX = X' * X / n
S0 = - At_mul_B(X, Y) / n
@time HD.updateOneGroup!(beta, XtX, S0, [1:10], lambda)
@time HD.updateOneGroupSimple!(beta, X, Y, lambda)
@time beta

XtX*mBeta + S0 - lambda .* mBeta ./ norm(mBeta, 2)

(norm(Y-X*mBeta)^2 / (2*n) + lambda * norm(mBeta, 2)) - (norm(Y-X*beta)^2 / (2*n) + lambda * norm(beta, 2))
norm(Y-X*mBeta)^2 / (2*n) + lambda * norm(mBeta, 2)
norm([1,2])^2

