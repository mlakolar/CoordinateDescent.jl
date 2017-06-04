import ProximalBase
import HD


reload("ProximalBase")
reload("HD")

n = 3000
p = 5000
s = 100

srand(123)
X = randn(n, p)
Y = X[:,1:s] * (randn(s) .* (1. .+ rand(s))) + 6. * randn(n)

stdX = std(X, 1)[:]

options = HD.ScaledLassoOptions(;optTol=1e-3, maxIter=50)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time HD.scaledLasso!(x, X, Y, λ, stdX, options)


@show σinit = HD.findInitSigma(X, Y, 30)
options = HD.ScaledLassoOptions(;optTol=1e-2, maxIter=10, σinit=σinit)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time HD.scaledLasso!(x, X, Y, λ, stdX, options)



λ = 0.001

options = HD.CDOptions(;warmStart=true)
x = ProximalBase.SparseIterate(p)
f = HD.CDLeastSquaresLoss(Y,X)
g = ProximalBase.ProxL1(λ)
@time HD.coordinateDescent!(x, f, g, options)

options = HD.CDOptions(;warmStart=false)
x = ProximalBase.SparseIterate(p)
f = HD.CDLeastSquaresLoss(Y,X)
g = ProximalBase.ProxL1(λ)
@time HD.coordinateDescent!(x, f, g, options)

options = HD.CDOptions(;warmStart=true)
x = ProximalBase.SparseIterate(p)
f = HD.CDLeastSquaresLoss(Y,X)
g = ProximalBase.AProxL1(λ, stdX)
@time HD.coordinateDescent!(x, f, g, options)

options = HD.CDOptions(;warmStart=false, numSteps=100)
x = ProximalBase.SparseIterate(p)
f = HD.CDLeastSquaresLoss(Y,X)
g = ProximalBase.AProxL1(λ, stdX)
@time HD.coordinateDescent!(x, f, g, options)


options = HD.CDOptions(;warmStart=false, numSteps=100)
x = ProximalBase.SparseIterate(p)
f = HD.CDLeastSquaresLoss(Y,X)
g = ProximalBase.AProxL1(λ, stdX)
@time HD.coordinateDescent!(x, f, g, options)
