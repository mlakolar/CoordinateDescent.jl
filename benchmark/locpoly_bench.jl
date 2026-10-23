# Local polynomial regression in CoordinateDescent against the alternatives it
# was chosen over. Each part checks that the implementations agree and reports
# their median times:
#
#   1. locpoly, least squares on the √w-weighted expanded design by QR, against
#      the normal equations formed from the expanded design X̃ (alt1) and
#      accumulated without forming X̃, solved by Cholesky (alt)
#   2. _expand_Xt_w_X!, which accumulates X̃' diag(w) X̃ without forming X̃,
#      against a naive loop over the entries and against BLAS on the formed X̃
#   3. the weighted lasso at one point z0: CDWeightedLSLoss against
#      CDLeastSquaresLoss on the √w-scaled data
#   4. locpolyl1, which updates one CDWeightedLSLoss in place along the grid and
#      warm starts from the solution at the previous grid point, against a
#      CDLeastSquaresLoss on the √w-scaled data built at every grid point and
#      solved from zero
#
# From the root of the repository:
#
#   julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
#   julia --project=benchmark benchmark/locpoly_bench.jl

using CoordinateDescent
using ProximalBase
using BenchmarkTools
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics

# one BLAS thread, so that the times compare the algorithms and not the threading
BLAS.set_num_threads(1)
const SECONDS = 3          # time budget of each benchmark

Random.seed!(1)


############################################################
#
# data
#
############################################################

# varying coefficients βⱼ(z) = sin(bⱼ z), with bⱼ drawn from {2, 4, 6, 8};
# column i of betaMat is β(Z[i])
function genData(n, p)
  X = randn(n, p)
  Z = rand(n)
  ε = 0.1 * randn(n)
  Y = zeros(n)
  betaMat = zeros(p, n)

  randBeta = [rand([2, 4, 6, 8]) for j=1:p]
  for i=1:n
    betaMat[:, i] = sin.(randBeta * Z[i])
    Y[i] = dot(betaMat[:, i], X[i, :]) + ε[i]
  end
  Y, X, Z, betaMat
end

# constant coefficients, drawn from [0.5, 1.5]
function genDataIID(n, p)
  X = randn(n, p)
  Z = rand(n)
  ε = 0.1 * randn(n)
  beta = 0.5 .+ rand(p)
  Y = X * beta + ε
  Y, X, Z, beta
end


############################################################
#
# alternative implementations
#
############################################################

# local polynomial fit at z0 from the normal equations of the expanded design
function _locpoly_alt1!(
  eX::Matrix{T}, w::Vector{T},
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int, kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  w .= evaluate.(Ref(kernel), z, Ref(z0))
  CoordinateDescent._expand_X!(eX, X, z, z0, degree)
  Xt_w = eX' * Diagonal(w)
  (Xt_w * eX) \ (Xt_w * y)
end

function locpoly_alt1(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int,
  kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  n, p = size(X)
  ep = p * (degree + 1)
  out = Matrix{T}(undef, ep, length(zgrid))
  w = Vector{T}(undef, n)
  eX = Matrix{T}(undef, n, ep)
  for (ind, z0) in enumerate(zgrid)
    out[:, ind] = _locpoly_alt1!(eX, w, X, z, y, z0, degree, kernel)
  end
  out
end

# local polynomial fit at z0 from the normal equations accumulated without
# forming the expanded design, solved by Cholesky
function _locpoly_alt!(
  Xt_w_X::Matrix{T}, Xt_w_Y::Vector{T}, w::Vector{T},
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int, kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  w .= evaluate.(Ref(kernel), z, Ref(z0))
  CoordinateDescent._expand_Xt_w_X!(Xt_w_X, w, X, z, z0, degree)
  CoordinateDescent._expand_Xt_w_Y!(Xt_w_Y, w, X, z, y, z0, degree)
  cholesky!(Symmetric(Xt_w_X)) \ Xt_w_Y
end

function locpoly_alt(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int,
  kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  n, p = size(X)
  ep = p * (degree + 1)
  out = Matrix{T}(undef, ep, length(zgrid))
  w = Vector{T}(undef, n)
  Xt_w_X = Matrix{T}(undef, ep, ep)
  Xt_w_Y = Vector{T}(undef, ep)
  for (ind, z0) in enumerate(zgrid)
    out[:, ind] = _locpoly_alt!(Xt_w_X, Xt_w_Y, w, X, z, y, z0, degree, kernel)
  end
  out
end

# X̃' diag(w) X̃ entry by entry, recomputing (zᵢ - z0)^(jj + kk) for every entry
function _expand_Xt_w_X_alt!(
  Xt_w_X::Matrix{T},
  w::Vector{T}, X::Matrix{T},
  z::Vector{T}, z0::T, degree::Int) where {T <: AbstractFloat}

  n, p = size(X)
  fill!(Xt_w_X, zero(T))
  @inbounds for j=1:p, jj=0:degree
    col = (j-1)*(degree+1) + 1 + jj
    for k=j:p
      krange = k != j ? (0:degree) : (jj:degree)
      for kk=krange
        row = (k-1)*(degree+1) + 1 + kk
        for i=1:n
          Xt_w_X[row, col] += X[i, j] * (z[i] - z0)^(jj+kk) * X[i, k] * w[i]
        end
        if row != col
          Xt_w_X[col, row] = Xt_w_X[row, col]
        end
      end
    end
  end
  Xt_w_X
end

# X̃' diag(w) X̃ with BLAS, forming X̃ in eX and diag(w) X̃ in wX
function _expand_Xt_w_X_blas!(
  Xt_w_X::Matrix{T}, eX::Matrix{T}, wX::Matrix{T},
  w::Vector{T}, X::Matrix{T},
  z::Vector{T}, z0::T, degree::Int) where {T <: AbstractFloat}

  CoordinateDescent._expand_X!(eX, X, z, z0, degree)
  wX .= w .* eX
  mul!(Xt_w_X, transpose(eX), wX)
end

# locpolyl1 without the in-place updates: at every grid point, the √w-scaled
# expanded design is formed, a CDLeastSquaresLoss is built on it, and the lasso
# is solved from zero
function locpolyl1_alt(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int,
  kernel::SmoothingKernel{T},
  λ0::T,
  options::CDOptions=CDOptions()) where {T <: AbstractFloat}

  opt = CDOptions(; maxIter=options.maxIter, optTol=options.optTol, randomize=options.randomize,
                  warmStart=true, numSteps=options.numSteps, relativeTol=options.relativeTol)

  n, p = size(X)
  ep = p * (degree + 1)
  out = zeros(T, ep, length(zgrid))
  w = Vector{T}(undef, n)
  eX = Matrix{T}(undef, n, ep)
  stdX = Vector{T}(undef, ep)
  for (ind, z0) in enumerate(zgrid)
    w .= evaluate.(Ref(kernel), z, Ref(z0))
    CoordinateDescent._expand_X!(eX, X, z, z0, degree)
    CoordinateDescent._stdX!(stdX, w, eX)       # sqrt(∑ᵢ wᵢ X̃ᵢₖ² / n), as in locpolyl1

    sw = sqrt.(w)
    f = CDLeastSquaresLoss(sw .* y, sw .* eX)
    β = SparseIterate(T, ep)
    coordinateDescent!(β, f, NormL1(λ0, stdX), opt)
    out[:, ind] = Vector(β)
  end
  out
end


############################################################
#
# reporting
#
############################################################

# median time in milliseconds
medianms(b) = median(b).time / 1e6

maxdiff(a, b) = maximum(abs.(a .- b))

# rows of (method, benchmark, largest difference from the first row's result)
function report(title, rows)
  println("\n", title)
  @printf("  %-52s %12s %12s\n", "method", "median", "max |Δ|")
  for (name, b, d) in rows
    @printf("  %-52s %9.3f ms %12s\n", name, medianms(b), d === nothing ? "reference" : @sprintf("%.1e", d))
  end
end

println("Julia ", VERSION, ", ", BLAS.get_num_threads(), " BLAS thread, ",
        Threads.nthreads(), " Julia thread; times are medians over ", SECONDS, " s")


############################################################
#
# 1. locpoly
#
############################################################

n, p, degree = 1000, 10, 2
gk = GaussianKernel(0.2)
zgrid = collect(range(0.01, 0.99; length=50))
Y, X, Z, _ = genData(n, p)

o1 = locpoly(X, Z, Y, zgrid, degree, gk)
o2 = locpoly_alt1(X, Z, Y, zgrid, degree, gk)
o3 = locpoly_alt(X, Z, Y, zgrid, degree, gk)

b1 = @benchmark locpoly($X, $Z, $Y, $zgrid, $degree, $gk) seconds=SECONDS
b2 = @benchmark locpoly_alt1($X, $Z, $Y, $zgrid, $degree, $gk) seconds=SECONDS
b3 = @benchmark locpoly_alt($X, $Z, $Y, $zgrid, $degree, $gk) seconds=SECONDS

report("1. locpoly at $(length(zgrid)) grid points (n = $n, p = $p, degree = $degree, $(p * (degree + 1)) coefficients)",
       [("locpoly: QR on the √w-weighted expanded design", b1, nothing),
        ("alt1: normal equations from the expanded design", b2, maxdiff(o2, o1)),
        ("alt: normal equations without it, Cholesky", b3, maxdiff(o3, o1))])

# with constant coefficients, the zeroth-order coefficients estimate β at every
# z0, up to the estimation error
Yc, Xc, Zc, βc = genDataIID(n, p)
Bc = locpoly(Xc, Zc, Yc, zgrid, degree, gk)
@printf("  constant coefficients (genDataIID, noise sd 0.1): max |β̂ⱼ(z0) - βⱼ| = %.1e over the grid\n",
        maximum(abs.(Bc[1:(degree+1):end, :] .- βc)))


############################################################
#
# 2. X̃' diag(w) X̃
#
############################################################

z0 = 0.5
ep = p * (degree + 1)
w = evaluate.(Ref(gk), Z, Ref(z0))
A1, A2, A3 = zeros(ep, ep), zeros(ep, ep), zeros(ep, ep)
eX, wX = zeros(n, ep), zeros(n, ep)

CoordinateDescent._expand_Xt_w_X!(A1, w, X, Z, z0, degree)
_expand_Xt_w_X_alt!(A2, w, X, Z, z0, degree)
_expand_Xt_w_X_blas!(A3, eX, wX, w, X, Z, z0, degree)

b1 = @benchmark CoordinateDescent._expand_Xt_w_X!($A1, $w, $X, $Z, $z0, $degree) seconds=SECONDS
b2 = @benchmark _expand_Xt_w_X_alt!($A2, $w, $X, $Z, $z0, $degree) seconds=SECONDS
b3 = @benchmark _expand_Xt_w_X_blas!($A3, $eX, $wX, $w, $X, $Z, $z0, $degree) seconds=SECONDS

report("2. X̃' diag(w) X̃ at z0 = $z0 ($ep × $ep, n = $n)",
       [("_expand_Xt_w_X!: accumulated without forming X̃", b1, nothing),
        ("naive loop over the entries", b2, maxdiff(A2, A1)),
        ("BLAS on the formed X̃, including forming it", b3, maxdiff(A3, A1))])


############################################################
#
# 3. weighted lasso at one point
#
############################################################

# s varying coefficients and p - s irrelevant variables
n, s, p, degree = 500, 2, 50, 2
ep = p * (degree + 1)
Y, Xs, Z, _ = genData(n, s)
X = [Xs randn(n, p - s)]
z0 = 0.5
λ = 0.05

eX = zeros(n, ep)
CoordinateDescent._expand_X!(eX, X, Z, z0, degree)
w = evaluate.(Ref(gk), Z, Ref(z0))
sw = sqrt.(w)
stdX = CoordinateDescent._stdX!(zeros(ep), w, eX)

g = NormL1(λ, stdX)
f1 = CDWeightedLSLoss(Y, eX, w)
f2 = CDLeastSquaresLoss(sw .* Y, sw .* eX)
opt = CDOptions(; optTol=1e-8, warmStart=false)

x1, x2 = SparseIterate(ep), SparseIterate(ep)
coordinateDescent!(x1, f1, g, opt)
coordinateDescent!(x2, f2, g, opt)

b1 = @benchmark coordinateDescent!(x, $f1, $g, $opt) setup=(x = SparseIterate($ep)) evals=1 seconds=SECONDS
b2 = @benchmark coordinateDescent!(x, $f2, $g, $opt) setup=(x = SparseIterate($ep)) evals=1 seconds=SECONDS

report("3. weighted lasso at z0 = $z0 (n = $n, $ep coefficients, λ = $λ, $(nnz(x1)) nonzero), path from λmax",
       [("CDWeightedLSLoss", b1, nothing),
        ("CDLeastSquaresLoss on the √w-scaled data", b2, maxdiff(Vector(x2), Vector(x1)))])


############################################################
#
# 4. locpolyl1
#
############################################################

zgrid = collect(range(0.025, 0.975; length=20))
λ0 = 0.05
opt = CDOptions(; optTol=1e-8)

B, _ = locpolyl1(X, Z, Y, zgrid, degree, gk, λ0, false, opt)
Balt = locpolyl1_alt(X, Z, Y, zgrid, degree, gk, λ0, opt)

b1 = @benchmark locpolyl1($X, $Z, $Y, $zgrid, $degree, $gk, $λ0, false, $opt) seconds=SECONDS
b2 = @benchmark locpolyl1_alt($X, $Z, $Y, $zgrid, $degree, $gk, $λ0, $opt) seconds=SECONDS

report("4. locpolyl1 at $(length(zgrid)) grid points (n = $n, $ep coefficients, λ0 = $λ0, " *
       "$(round(Int, nnz(B) / length(zgrid))) nonzero per point on average)",
       [("locpolyl1: one CDWeightedLSLoss, warm starts", b1, nothing),
        ("a CDLeastSquaresLoss per grid point, from zero", b2, maxdiff(Balt, Matrix(B)))])
