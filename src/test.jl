# ## compare to mosek
# using Convex
# using Mosek
# using HD
# push!(LOAD_PATH, "/home/mkolar/projects/juliaPkg")
# reload("HD")

# solver = MosekSolver(LOG=1)
# set_default_solver(solver);

# n = 500
# p = 5000
# s = 10

# rho = 0.5
# covMat = rho.^abs([1:p]*ones(p)' - ones(p)*[1:p]')

# X = randn(n, p) * sqrtm(covMat)
# true_beta = vcat( (1.+rand(s)) .* (2.*rand(0:1,s).-1), zeros(p-s))
# Y = X * true_beta + 0.1 .* randn(n)

# lambda = 0.3
# theta = Variable(p);
# prob = minimize(sum_squares(Y-X*theta) / n + lambda * norm(theta, 1))
# solve!(prob)
# theta.value

# @time solShoot = HD.LassoActiveShooting(X, Y, lambda * ones(p, p))

# maximum(abs(solShoot - theta.value))


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
HD.minimize_active_groups!(beta, XtX, Xy, groups, active_set, lambdaG)

reload("FactCheck")


