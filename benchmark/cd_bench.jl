import ProximalBase
import CoordinateDescent


reload("ProximalBase")
reload("CoordinateDescent")

n = 3000
p = 5000
s = 100

srand(123)
X = randn(n, p)
Y = X[:,1:s] * (randn(s) .* (1. .+ rand(s))) + 6. * randn(n)

stdX = std(X, 1)[:]

options = CoordinateDescent.ScaledLassoOptions(;optTol=1e-3, maxIter=50)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time CoordinateDescent.scaledLasso!(x, X, Y, λ, stdX, options)


@show σinit = CoordinateDescent.findInitSigma(X, Y, 30)
options = CoordinateDescent.ScaledLassoOptions(;optTol=1e-2, maxIter=10, σinit=σinit)
x = ProximalBase.SparseIterate(p)
λ = sqrt(2. * log(p) / n)
@time CoordinateDescent.scaledLasso!(x, X, Y, λ, stdX, options)



λ = 0.001

options = CoordinateDescent.CDOptions(;warmStart=true)
x = ProximalBase.SparseIterate(p)
f = CoordinateDescent.CDLeastSquaresLoss(Y,X)
g = ProximalBase.ProxL1(λ)
@time CoordinateDescent.coordinateDescent!(x, f, g, options)

options = CoordinateDescent.CDOptions(;warmStart=false)
x = ProximalBase.SparseIterate(p)
f = CoordinateDescent.CDLeastSquaresLoss(Y,X)
g = ProximalBase.ProxL1(λ)
@time CoordinateDescent.coordinateDescent!(x, f, g, options)

options = CoordinateDescent.CDOptions(;warmStart=true)
x = ProximalBase.SparseIterate(p)
f = CoordinateDescent.CDLeastSquaresLoss(Y,X)
g = ProximalBase.ProxL1(λ, stdX)
@time CoordinateDescent.coordinateDescent!(x, f, g, options)

options = CoordinateDescent.CDOptions(;warmStart=false, numSteps=100)
x = ProximalBase.SparseIterate(p)
f = CoordinateDescent.CDLeastSquaresLoss(Y,X)
g = ProximalBase.ProxL1(λ, stdX)
@time CoordinateDescent.coordinateDescent!(x, f, g, options)


options = CoordinateDescent.CDOptions(;warmStart=false, numSteps=100)
x = ProximalBase.SparseIterate(p)
f = CoordinateDescent.CDLeastSquaresLoss(Y,X)
g = ProximalBase.ProxL1(λ, stdX)
@time CoordinateDescent.coordinateDescent!(x, f, g, options)
