# Fused regression, graphs and trend filtering
#     julia --project=examples examples/11_structured.jl
using CoordinateDescent, Random, LinearAlgebra, Statistics
import KernelSmoothing
rng = Xoshiro(80)

# --- Fused regression on ordered coefficients ---
X = randn(rng, 80, 8)
beta = [0.,0,1,1,1,-.5,-.5,0]
y = X*beta + .3randn(rng,80)
levels = [.2,.08,.03,.01]
path = fused_lasso_path(X,y,levels;l1_ratio=.25)
@assert all(s -> s.converged, path.solutions)
println("Fused degrees of freedom: ", dof(path))
cv = cv_fused_lasso(X,y,levels;l1_ratio=.25,nfolds=4,rng=Xoshiro(1),threads=false)
println("CV level: ", cv.λmin)
D = graph_difference(8, [(j,j+1) for j in 1:7])
gpath = GeneralizedLassoPath(X,y,D,levels)
println("Graph BIC: ", bic(gpath))

# --- Compare trend filtering and local linear regression on the same data ---
x = collect(1.:80)
truth = @. .02*x + .03*max(x-25,0) - .06*max(x-55,0)
signal = truth + .1randn(rng,80)
tpath = trend_filter_path(signal,[.03,.01,.003,.001];order=1)
valid = findall(isfinite,bic(tpath))
j = valid[argmin(bic(tpath)[valid])]
tf = coef(tpath,tpath.λpath[j])
lp = KernelSmoothing.localpolynomial(x,signal,x;bandwidth=8.,degree=1)
println("Trend filtering MSE on this signal: ", mean(abs2,tf-truth))
println("Local linear MSE on this signal: ", mean(abs2,lp-truth))
@assert all(isfinite,tf) && all(isfinite,lp)
