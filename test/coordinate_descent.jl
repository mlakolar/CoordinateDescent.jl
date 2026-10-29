module HighDimensionalModelsTest

const  NUMBER_REPEAT = 1

using HighDimensionalModels
using Test
using ProximalBase
using ProximalSolvers
using Random
using LinearAlgebra
using SparseArrays
using Statistics

Random.seed!(1)

# test |Y - β|_2^2 + λ⋅|β|_1
@testset "Small NormL1" begin
  Y = [1., 1.5]

  λ = 1.2
  g = NormL1(λ)
  f = CDQuadraticLoss(Matrix(1.0I, 2, 2), -Y)

  opt = CDOptions(;max_iter=100, opt_tol=1e-8, warm_start=true, randomize=false, gaptol = 0)
  x = SparseIterate(2)
  coordinate_descent!(x, f, g, opt)

  @test Vector(x) ≈ Vector([0., 0.3])
end


# check that the warm start and non warm start produce the same result
@testset "NormL1" begin

    for i=1:NUMBER_REPEAT
      n = 500
      p = 50
      s = 5

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      λ = 0.02
      g = NormL1(λ)
      f = CDLeastSquaresLoss(Y, X)

      opt1 = CDOptions(;max_iter=5000, opt_tol=1e-12, warm_start=true, randomize=false, gaptol = 0)
      opt2 = CDOptions(;max_iter=5000, opt_tol=1e-12, warm_start=true, randomize=true, gaptol = 0)
      opt3 = CDOptions(;max_iter=5000, opt_tol=1e-12, warm_start=false, randomize=false, gaptol = 0)
      opt4 = CDOptions(;max_iter=5000, opt_tol=1e-12, warm_start=false, randomize=true, gaptol = 0)

      x1 = SparseIterate(sprand(p, 0.6))
      x2 = SparseIterate(sprand(p, 0.6))
      x3 = SparseIterate(sprand(p, 0.6))
      x4 = SparseIterate(sprand(p, 0.6))

      coordinate_descent!(x1, f, g, opt1)
      coordinate_descent!(x2, f, g, opt2)
      coordinate_descent!(x3, f, g, opt3)
      coordinate_descent!(x4, f, g, opt4)

      @test Vector(x1) ≈ Vector(x2) atol=1e-5
      @test Vector(x3) ≈ Vector(x2) atol=1e-5
      @test Vector(x4) ≈ Vector(x2) atol=1e-5
    end
end

@testset "weighted NormL1" begin

    for i=1:NUMBER_REPEAT
      n = 500
      p = 50
      s = 10

      X = randn(n, p)
      β = randn(s)
      Y = X[:,1:s] * β + randn(n)

      λ = 0.01
      g = NormL1(λ, rand(p))
      f = CDLeastSquaresLoss(Y, X)

      opt1 = CDOptions(;max_iter=5000, opt_tol=1e-8, warm_start=true, randomize=false, gaptol = 0)
      opt2 = CDOptions(;max_iter=5000, opt_tol=1e-8, warm_start=true, randomize=true, gaptol = 0)
      opt3 = CDOptions(;max_iter=5000, opt_tol=1e-8, warm_start=false, randomize=false, gaptol = 0)
      opt4 = CDOptions(;max_iter=5000, opt_tol=1e-8, warm_start=false, randomize=true, gaptol = 0)

      x1 = SparseIterate(sprand(p, 0.6))
      x2 = SparseIterate(sprand(p, 0.6))
      x3 = SparseIterate(sprand(p, 0.6))
      x4 = SparseIterate(sprand(p, 0.6))

      coordinate_descent!(x1, f, g, opt1)
      coordinate_descent!(x2, f, g, opt2)
      coordinate_descent!(x3, f, g, opt3)
      coordinate_descent!(x4, f, g, opt4)

      @test Vector(x1) ≈ Vector(x2) atol=1e-5
      @test Vector(x3) ≈ Vector(x2) atol=1e-5
      @test Vector(x4) ≈ Vector(x2) atol=1e-5
    end
end

# path following must end at the target λ, not one step short of it
@testset "path following reaches λ" begin
  n, p, s = 500, 50, 5

  for seed in 1:5
    Random.seed!(seed)
    X = randn(n, p)
    Y = X[:,1:s] * randn(s) + randn(n)

    λ = 0.02
    for g in (NormL1(λ), NormL1(1., fill(λ, p)))
      f = CDLeastSquaresLoss(Y, X)
      x = SparseIterate(p)
      coordinate_descent!(x, f, g, CDOptions(;max_iter=5000, opt_tol=1e-12, warm_start=false, gaptol = 0))
      @test maximum(abs.(X'*(Y - X*x) / n)) ≈ λ atol=1e-8
    end
  end
end

@testset "ZeroFunction" begin
  n, p = 200, 10
  X = randn(n, p)
  Y = X * randn(p) + randn(n)

  for warm_start in (true, false)
    opt = CDOptions(;max_iter=5000, opt_tol=1e-12, warm_start=warm_start, gaptol = 0)

    x = SparseIterate(p)
    coordinate_descent!(x, CDLeastSquaresLoss(Y, X), ZeroFunction(), opt)
    @test Vector(x) ≈ X \ Y atol=1e-6

    x = SparseIterate(p)
    coordinate_descent!(x, CDQuadraticLoss(X'X/n, -X'Y/n), ZeroFunction(), opt)
    @test Vector(x) ≈ X \ Y atol=1e-6
  end
end

@testset "weighted least squares loss" begin
  n, p, s = 500, 50, 5
  X = randn(n, p)
  Y = X[:,1:s] * randn(s) + randn(n)
  w = 2 * rand(n)
  sw = sqrt.(w)

  g = NormL1(0.05)
  f1 = CDLeastSquaresLoss(sw .* Y, sw .* X)
  f2 = CDWeightedLSLoss(Y, X, w)

  for warm_start in (true, false), randomize in (true, false)
    opt = CDOptions(;max_iter=5000, opt_tol=1e-12, warm_start=warm_start, randomize=randomize, gaptol = 0)
    x1 = SparseIterate(sprand(p, 0.6))
    x2 = SparseIterate(sprand(p, 0.6))
    coordinate_descent!(x1, f1, g, opt)
    coordinate_descent!(x2, f2, g, opt)
    @test Vector(x1) ≈ Vector(x2) atol=1e-6
  end
end

# the constraints lo ≤ xₖ ≤ hi, a separable penalty whose minimizers exclude zero
struct BoxPenalty <: ProximalBase.AbstractFunction
  lo::Float64
  hi::Float64
end
ProximalBase.isseparable(::Type{BoxPenalty}) = true
ProximalBase.isproximable(::Type{BoxPenalty}) = true
ProximalBase.prox_entry(g::BoxPenalty, v, i, γ) = clamp(v, g.lo, g.hi)
(g::BoxPenalty)(x) = all(v -> g.lo <= v <= g.hi, x) ? 0.0 : Inf

@testset "zero and constant columns" begin
  n, p = 100, 6
  X = randn(n, p)
  Y = X * randn(p) + randn(n)
  opt = CDOptions(;max_iter=5000, opt_tol=1e-12, gaptol = 0)
  keep = [1, 2, 4, 5, 6]

  # without a penalty, a zero column gets a zero coefficient instead of NaN
  Xz = copy(X); Xz[:, 3] .= 0
  for f in (CDLeastSquaresLoss(Y, Xz), CDWeightedLSLoss(Y, Xz, ones(n)), CDQuadraticLoss(Xz'Xz/n, -Xz'Y/n))
    x = SparseIterate(p)
    coordinate_descent!(x, f, ZeroFunction(), opt)
    @test x[3] == 0
    @test Vector(x)[keep] ≈ X[:, keep] \ Y atol=1e-6
  end
  x = SparseIterate(ones(p))                      # also from a nonzero start
  coordinate_descent!(x, CDLeastSquaresLoss(Y, Xz), ZeroFunction(), opt)
  @test x[3] == 0

  # with another penalty, it goes to the minimizer of the penalty nearest zero
  for f in (CDLeastSquaresLoss(Y, Xz), CDWeightedLSLoss(Y, Xz, ones(n)), CDQuadraticLoss(Xz'Xz/n, -Xz'Y/n))
    x = SparseIterate(p)
    coordinate_descent!(x, f, BoxPenalty(1., 2.), opt)
    @test x[3] == 1
    @test all(v -> 1 <= v <= 2, Vector(x))
  end

  # with an intercept, a constant column is a zero column, also when its mean
  # is not exact in floating point
  for c in (0.1, 0.3, 1.0)
    Xc = copy(X); Xc[:, 3] .= c
    f = CDLeastSquaresLoss(Y, Xc; intercept=true)
    x = SparseIterate(p)
    coordinate_descent!(x, f, ZeroFunction(), opt)
    @test x[3] == 0
    @test [ProximalSolvers.intercept(f, x); Vector(x)[keep]] ≈ [ones(n) X[:, keep]] \ Y atol=1e-6

    sol = lasso(Xc, Y, 0.01, opt; intercept=true, standardize_x=true)
    @test sol.penalty.w[3] == 0                   # its standard deviation is zero
    @test sol.x[3] == 0
  end
end

@testset "negative penalties" begin
  # rejected by ProximalBase when the penalty is built
  p = 6
  @test_throws ArgumentError NormL1(-0.1)
  @test_throws ArgumentError NormL1(NaN)
  @test_throws ArgumentError NormL1(0.1, [-1.; ones(p - 1)])
  @test_throws ArgumentError GroupNormL2(0.1, [1:3, 4:6], [1., -1.])
end

@testset "Float32" begin
  n, p, s = 200, 20, 5
  X = randn(Float32, n, p)
  Y = X[:,1:s] * ones(Float32, s) + 0.1f0 * randn(Float32, n)

  g = NormL1(0.1f0)
  for f in (CDLeastSquaresLoss(Y, X), CDQuadraticLoss(X'X/n, -X'Y/n))
    x = SparseIterate(Float32, p)
    coordinate_descent!(x, f, g, CDOptions(;opt_tol=1e-6))
    @test eltype(x) == Float32
    @test maximum(abs.(X'*(Y - X*x) / n)) ≈ 0.1f0 atol=1e-4
  end
end

@testset "argument checks" begin
  n, p = 20, 5
  X = randn(n, p)
  Y = randn(n)

  @test_throws DimensionMismatch CDLeastSquaresLoss(randn(n+1), X)
  @test_throws DimensionMismatch CDWeightedLSLoss(Y, X, rand(n+1))
  @test_throws DimensionMismatch CDSqrtLassoLoss(randn(n+1), X)
  @test_throws ArgumentError CDQuadraticLoss(randn(p, p) + triu(ones(p, p)), randn(p))
  @test_throws DimensionMismatch CDQuadraticLoss(Matrix(1.0I, p, p), randn(p+1))
  # a dense matrix is used as it is, as in 0.6
  A = X'X / n
  @test CDQuadraticLoss(A, -X'Y / n).A === A

  f = CDLeastSquaresLoss(Y, X)
  @test_throws DimensionMismatch coordinate_descent!(SparseIterate(p+1), f, NormL1(0.1))
  @test_throws DimensionMismatch coordinate_descent!(SparseIterate(p+1), f, ZeroFunction())
  @test_throws DimensionMismatch coordinate_descent!(SparseIterate(p), f, NormL1(0.1, ones(p+1)))
end

# a user-defined loss f(x) = ‖x - c‖² / 2 through the documented interface;
# with NormL1 its minimizer is the soft-thresholded c
struct SeparableLoss <: CoordinateDifferentiableFunction
  c::Vector{Float64}
end
HighDimensionalModels.initialize!(::SeparableLoss, x) = nothing
HighDimensionalModels.num_coordinates(f::SeparableLoss) = length(f.c)
HighDimensionalModels.gradient(f::SeparableLoss, x, k) = coordinates(x)[k] - f.c[k]
function HighDimensionalModels.descend_coordinate!(f::SeparableLoss, g::NormL1, x, k)
  cx = coordinates(x)
  oldVal = cx[k]
  cx[k] = f.c[k]
  prox_coordinate!(x, g, k, 1.) - oldVal
end

@testset "user-defined loss" begin
  c = [3., -0.5, 0.2, -2.]
  for warm_start in (true, false)
    x = SparseIterate(4)
    coordinate_descent!(x, SeparableLoss(c), NormL1(1.), CDOptions(;warm_start=warm_start))
    @test Vector(x) ≈ [2., 0., 0., -1.]
    @test nnz(x) == 2
  end

  # other iterates are not screened along the path of warm_start = false
  c = [3., -0.5, 0.2, -2., 1.5, 0.7]
  for warm_start in (true, false)
    x = SymmetricSparseIterate(3)
    coordinate_descent!(x, SeparableLoss(c), NormL1(1.), CDOptions(;warm_start=warm_start))
    @test collect(coordinates(x)) ≈ [2., 0., 0., -1., 0.5, 0.]
  end

  # with penalty weights (a p × p matrix on a symmetric iterate), the path
  # starts from max |cₖ| / wₖ over the penalized coordinates, 2 / 0.5 here
  w = [1., 0., 2., 0.5, 1., 1.]
  W = zeros(3, 3)
  for k in 1:6
    r, c2 = tril_cartesian(3, k)
    W[r, c2] = W[c2, r] = w[k]
  end
  x = SymmetricSparseIterate(3)
  @test lambda_max(HighDimensionalModels._LossModel(SeparableLoss(c), x), NormL1(1., W), x) == 4.
  for warm_start in (true, false)
    x = SymmetricSparseIterate(3)
    coordinate_descent!(x, SeparableLoss(c), NormL1(1., W), CDOptions(;warm_start=warm_start))
    @test collect(coordinates(x)) ≈ [2., -0.5, 0., -1.5, 0.5, 0.]
  end

  # a dimension that does not match the iterate
  @test_throws DimensionMismatch coordinate_descent!(SparseIterate(5), SeparableLoss(c[1:4]), NormL1(1.))

  # a missing function of the interface is a MethodError, which lists the
  # methods that exist
  struct IncompleteLoss <: CoordinateDifferentiableFunction end
  @test_throws MethodError HighDimensionalModels.initialize!(IncompleteLoss(), SparseIterate(2))
  @test_throws MethodError HighDimensionalModels.gradient(IncompleteLoss(), SparseIterate(2), 1)
  @test_throws MethodError HighDimensionalModels.num_coordinates(IncompleteLoss())
  @test_throws MethodError HighDimensionalModels.descend_coordinate!(IncompleteLoss(), ZeroFunction(), SparseIterate(2), 1)
  @test_throws MethodError coordinate_descent!(SparseIterate(2), IncompleteLoss(), NormL1(1.))
end

# a user-defined penalty, the elastic net λ (α ‖x‖₁ + (1 - α) ‖x‖² / 2), through
# its proximal operator on one coordinate
struct ElasticNetPenalty <: ProximalBase.AbstractFunction
  λ::Float64
  α::Float64
end
ProximalBase.isseparable(::Type{ElasticNetPenalty}) = true
ProximalBase.isproximable(::Type{ElasticNetPenalty}) = true
ProximalBase.isconvex(::Type{ElasticNetPenalty}) = true
ProximalBase.prox_entry(g::ElasticNetPenalty, v, i, γ) = shrink(v, γ * g.λ * g.α) / (1 + γ * g.λ * (1 - g.α))
(g::ElasticNetPenalty)(x) = g.λ * (g.α * norm(x, 1) + (1 - g.α) * sum(abs2, x) / 2)

@testset "user-defined penalty" begin
  n, p, s = 100, 20, 5
  X = randn(n, p)
  Y = X[:, 1:s] * ones(s) + randn(n)
  λ, α = 0.1, 0.5
  g = ElasticNetPenalty(λ, α)

  # the elastic net is the lasso on the data augmented with √(nλ(1 - α)) I,
  # whose loss divides by n + p instead of n
  Xa = [X; Matrix(sqrt(n * λ * (1 - α)) * I, p, p)]
  Ya = [Y; zeros(p)]
  xa = SparseIterate(p)
  coordinate_descent!(xa, CDLeastSquaresLoss(Ya, Xa), NormL1(n / (n + p) * λ * α), CDOptions(;opt_tol=1e-12, gaptol = 0))
  @test 0 < nnz(xa) < p

  for f in (CDLeastSquaresLoss(Y, X), CDWeightedLSLoss(Y, X, ones(n)), CDQuadraticLoss(X'X / n, -X'Y / n)),
      warm_start in (true, false), randomize in (true, false)
    x = SparseIterate(sprand(p, 0.5))
    r = solve!(x, f, g, CDOptions(;opt_tol=1e-12, warm_start=warm_start, randomize=randomize, gaptol = 0))
    @test r.converged
    @test Vector(x) ≈ Vector(xa) atol=1e-8
    @test nnz(x) == nnz(xa)                  # no stored zeros
  end

  x = zeros(p)
  coordinate_descent!(x, CDLeastSquaresLoss(Y, X), g, CDOptions(;opt_tol=1e-12, gaptol = 0))
  @test x ≈ Vector(xa) atol=1e-8

  # the square-root lasso loss takes the ℓ1 penalty only, and the group lasso
  # penalty needs an AtomIterate
  @test_throws ArgumentError coordinate_descent!(SparseIterate(p), CDSqrtLassoLoss(Y, X), g)
  @test_throws ArgumentError coordinate_descent!(SparseIterate(p), CDLeastSquaresLoss(Y, X), GroupNormL2(0.1, [1:10, 11:20]))
end

@testset "max_iter" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X * randn(p) + randn(n)
  f = CDLeastSquaresLoss(Y, X)
  g = NormL1(0.01)

  # a single pass is one sweep of coordinate updates
  x1 = SparseIterate(p)
  info = @test_logs (:warn, r"did not converge in maxpasses = 1 passes") solve!(x1, f, g, CDOptions(;max_iter=1, randomize=false))
  x2, r = zeros(p), copy(Y)
  for k=1:p
    a, b = sum(abs2, X[:, k]), dot(X[:, k], r)
    new = shrink(x2[k] + b / a, 0.01 * n / a)
    r .-= X[:, k] .* (new - x2[k])
    x2[k] = new
  end
  @test Vector(x1) ≈ x2
  @test info.passes == 1
  @test !info.converged
end

@testset "convergence info" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X * randn(p) + randn(n)
  g = NormL1(0.05)

  # the response scales of the two losses (see CDOptions)
  scale_ls = norm(Y) / sqrt(n)
  scale_q = maximum(abs.(X'Y / n) ./ sqrt.(vec(sum(abs2, X; dims=1)) / n))
  for (f, scale) in ((CDLeastSquaresLoss(Y, X), scale_ls), (CDQuadraticLoss(X'X/n, -X'Y/n), scale_q)),
      warm_start in (true, false)
    opt = CDOptions(;opt_tol=1e-10, warm_start=warm_start, gaptol = 0)

    x = SparseIterate(p)
    info = @test_logs solve!(x, f, g, opt)
    @test info.converged
    @test info.passes >= 2          # a pass that moves x and one that confirms it
    @test info.maxchange < info.tolerance
    @test info.tolerance ≈ 1e-10 * scale
    @test maximum(abs.(X'*(Y - X*x) / n)) ≈ 0.05 atol=1e-8

    # stopped by max_iter: reported and warned about; with warm_start = false the
    # passes of the num_steps path steps are counted too (the first step, at
    # λmax, stops after one pass)
    opt = CDOptions(;max_iter=2, opt_tol=1e-10, warm_start=warm_start, gaptol = 0)
    x = SparseIterate(p)
    info = @test_logs (:warn, r"did not converge") solve!(x, f, g, opt)
    @test !info.converged
    if warm_start
      @test info.passes == 2
    else
      @test opt.num_steps < info.passes <= 2 * (opt.num_steps + 1)
    end
    @test info.maxchange > info.tolerance
  end

  # ZeroFunction
  x = SparseIterate(p)
  info = @test_logs solve!(x, CDLeastSquaresLoss(Y, X), ZeroFunction(), CDOptions(;opt_tol=1e-10, gaptol = 0))
  @test info.converged
  @test Vector(x) ≈ X \ Y atol=1e-6


end

@testset "default options" begin
  opt = CDOptions()
  @test opt.max_iter == 100_000
  @test opt.opt_tol == 1e-7
  @test opt.randomize
  @test opt.warm_start
  @test opt.relative_tol
  @test opt.anderson == 0
  @test opt.momentum === nothing
end

@testset "scale-free stopping rule" begin
  # the lasso with y and λ scaled by c: with powers of two the iterates scale
  # exactly, so the relative rule stops after the same number of passes
  n, p = 100, 50
  X = randn(n, p)
  Y = X[:,1:5] * ones(5) + randn(n)
  λ = 0.05
  opt = CDOptions(;randomize=false)
  ref = lasso(X, Y, λ, opt)
  for c in (2.0^-10, 2.0^10)
    sol = lasso(X, c .* Y, c * λ, opt)
    @test sol.iterations == ref.iterations
    @test Vector(sol.x) ≈ c .* Vector(ref.x)
  end
  # the absolute rule depends on c
  opt = CDOptions(;randomize=false, relative_tol=false)
  @test lasso(X, 2.0^-10 .* Y, 2.0^-10 * λ, opt).iterations < lasso(X, 2.0^10 .* Y, 2.0^10 * λ, opt).iterations

  # a zero response falls back to the absolute rule, and zero is found at once
  x = SparseIterate(p)
  info = solve!(x, CDLeastSquaresLoss(zeros(n), X), NormL1(0.1), CDOptions())
  @test info.converged
  @test info.passes == 1
  @test info.tolerance == CDOptions().opt_tol
  @test nnz(x) == 0
end

@testset "option conversions" begin
  opt = CDOptions(;max_iter=1e5, opt_tol=1, num_steps=10.0)
  @test opt.max_iter === 100_000
  @test opt.opt_tol === 1.0
  @test opt.num_steps === 10
  @test_throws InexactError CDOptions(;max_iter=1.5)

  opt = IterLassoOptions(;max_iter=10.0, opt_tol=1, sinit=3.0, σinit=2)
  @test opt.max_iter === 10
  @test opt.opt_tol === 1.0
  @test opt.sinit === 3
  @test opt.σinit === 2.0
end

@testset "zero penalty weights" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X[:,1:3] * ones(3) + 0.1 * randn(n)
  f = CDLeastSquaresLoss(Y, X)

  # the path of warm_start = false starts from the largest λ0 at which the
  # penalized coordinates are zero; an unpenalized coordinate made it Inf
  w = ones(p)
  w[1] = 0.
  @test lambda_max(f, NormL1(1., w), SparseIterate(p)) ≈ maximum(abs.(X[:,2:end]'Y / n))

  for warm_start in (true, false)
    x = SparseIterate(p)
    info = solve!(x, f, NormL1(0.1, w), CDOptions(;opt_tol=1e-12, warm_start=warm_start, gaptol = 0))
    @test info.converged
    grad = X'*(Y - X*x) / n
    @test abs(grad[1]) < 1e-8                     # unpenalized
    @test maximum(abs.(grad[2:end])) ≈ 0.1 atol=1e-8
  end

  # no penalty at all: the path is skipped and the least squares fit is returned
  for g in (NormL1(1., zeros(p)), NormL1(0.))
    x = SparseIterate(p)
    coordinate_descent!(x, f, g, CDOptions(;opt_tol=1e-12, warm_start=false, gaptol = 0))
    @test Vector(x) ≈ X \ Y atol=1e-6
  end
end

@testset "dense starting point" begin
  n, p = 100, 20
  X = randn(n, p)
  Y = X * randn(p) + randn(n)
  f = CDLeastSquaresLoss(Y, X)
  g = NormL1(0.05)
  opt = CDOptions(;opt_tol=1e-12, gaptol = 0)

  x = SparseIterate(p)
  coordinate_descent!(x, f, g, opt)
  for x0 in (zeros(p), randn(p))
    xd = copy(x0)
    @test coordinate_descent!(xd, f, g, opt).minimizer === xd
    @test xd ≈ Vector(x) atol=1e-8
  end
end

@testset "residuals and stored entries" begin
  # the residuals of the least squares losses, and no stored zeros
  n, p = 100, 30
  X = randn(n, p)
  Y = X[:,1:5] * ones(5) + randn(n)
  w = 2 * rand(n)

  for (f, g) in ((CDLeastSquaresLoss(Y, X), NormL1(0.1)),
                 (CDWeightedLSLoss(Y, X, w), NormL1(0.1)),
                 (CDSqrtLassoLoss(Y, X), NormL1(2.5)))
    x = SparseIterate(p)
    coordinate_descent!(x, f, g, CDOptions(;opt_tol=1e-10, gaptol = 0))
    @test 0 < nnz(x) < p
    @test nnz(x) == count(!iszero, Vector(x))     # no stored zeros
    @test ProximalSolvers.residuals(f) ≈ Y - X*x
  end
end

@testset "data changed in place" begin
  # the weights and the design may change in place between solves; the cached
  # column norms must follow
  n, p = 100, 20
  X = randn(n, p)
  Y = X[:,1:3] * ones(3) + randn(n)
  w = rand(n)
  g = NormL1(0.05)
  opt = CDOptions(;opt_tol=1e-12, gaptol = 0)

  f = CDWeightedLSLoss(Y, X, w)
  x = SparseIterate(p)
  coordinate_descent!(x, f, g, opt)
  w .= 2 .* rand(n)
  X .*= 1.5
  coordinate_descent!(x, f, g, opt)

  x2 = SparseIterate(p)
  coordinate_descent!(x2, CDWeightedLSLoss(Y, copy(X), copy(w)), g, opt)
  @test Vector(x) ≈ Vector(x2) atol=1e-8
end


@testset "group lasso" begin
  # groups of sizes 1 to 5, four of them in the model; f(x) = ‖y - Xx‖² / (2n)
  sizes = repeat(1:5, 8)
  groups = [(e - s + 1):e for (s, e) in zip(sizes, cumsum(sizes))]
  n, p = 100, sum(sizes)
  X = randn(n, p)
  β = zeros(p)
  model = [2, 7, 13, 30]
  for k in model
    β[groups[k]] = randn(sizes[k]) .+ 2 .* sign.(randn(sizes[k]))
  end
  y = X * β + 0.5 * randn(n)
  w = sqrt.(sizes)
  λmax = maximum(norm(X[:, G]' * y / n) / w[k] for (k, G) in enumerate(groups))   # x = 0 is optimal for λ ≥ λmax
  λ = 0.2 * λmax
  g = GroupNormL2(λ, groups)
  opt = CDOptions(; opt_tol = 1e-10, gaptol = 0)

  # largest violation of the optimality conditions at x with gradient grad:
  # grad[G] = -λw x[G] / ‖x[G]‖ for a nonzero group, ‖grad[G]‖ ≤ λw for a zero one
  function kkt_residual(x, grad, λw)
    r = 0.
    for (k, G) in enumerate(groups)
      nx = norm(x[G])
      r = max(r, iszero(nx) ? norm(grad[G]) - λw[k] : norm(grad[G] + λw[k] * x[G] / nx))
    end
    r
  end
  ls_gradient(x) = -X' * (y - X * x) / n
  nonzero_groups(x) = findall(G -> !iszero(x[G]), groups)
  active_groups(x) = sort!(collect(activecoordinates(x)))

  x_ref = Vector(coordinate_descent!(AtomIterate(p, groups), CDLeastSquaresLoss(y, X), g, opt).minimizer)

  @testset "least squares" begin
    @test kkt_residual(x_ref, ls_gradient(x_ref), λ * w) < 1e-8
    @test issubset(model, nonzero_groups(x_ref))

    # with and without the path from λmax, in both orders
    for warm_start in (true, false), randomize in (true, false)
      x = AtomIterate(p, groups)
      info = solve!(x, CDLeastSquaresLoss(y, X), g, CDOptions(; opt_tol = 1e-10, warm_start, randomize, gaptol = 0))
      @test info.converged
      @test active_groups(x) == nonzero_groups(Vector(x))     # no zero group stays active
      @test Vector(x) ≈ x_ref atol = 1e-7
    end

    # warm started from a solution, one full pass confirms it
    x = AtomIterate(x_ref, groups)
    info = solve!(x, CDLeastSquaresLoss(y, X), g, opt)
    @test info.passes == 1
    @test Vector(x) ≈ x_ref atol = 1e-8
  end

  @testset "quadratic loss" begin
    x = AtomIterate(p, groups)
    coordinate_descent!(x, CDQuadraticLoss(X'X / n, -X'y / n), g, CDOptions(; opt_tol = 1e-10, warm_start = false, gaptol = 0))
    @test Vector(x) ≈ x_ref atol = 1e-7
  end

  @testset "intercept" begin
    # the intercept fit is the fit to the centered data
    y1 = y .+ 3
    Xc, yc = X .- mean(X; dims = 1), y1 .- mean(y1)
    x1, x2 = AtomIterate(p, groups), AtomIterate(p, groups)
    coordinate_descent!(x1, CDLeastSquaresLoss(y1, X; intercept = true), g, opt)
    coordinate_descent!(x2, CDLeastSquaresLoss(yc, Xc), g, opt)
    @test Vector(x1) ≈ Vector(x2) atol = 1e-7
    @test kkt_residual(Vector(x1), -Xc' * (yc - Xc * Vector(x1)) / n, λ * w) < 1e-8
  end

  @testset "groups of one coordinate give the lasso" begin
    # the same steps, passes and screening as for NormL1
    singletons = [[i] for i in 1:p]
    for warm_start in (true, false)
      o = CDOptions(; opt_tol = 1e-10, warm_start, randomize = false, gaptol = 0)
      xa, xs = AtomIterate(p, singletons), SparseIterate(p)
      ia = solve!(xa, CDLeastSquaresLoss(y, X), GroupNormL2(λ, singletons, ones(p)), o)
      is = solve!(xs, CDLeastSquaresLoss(y, X), NormL1(λ), o)
      @test Vector(xa) ≈ Vector(xs) atol = 1e-12
      @test ia.passes == is.passes
    end
  end

  @testset "zero solution" begin
    # λ ≥ λmax: x stays at zero. At λmax itself the group that attains it is
    # on the boundary, and the gradient computed in another order can exceed
    # λmax in the last bit, so the test takes λmax (1 + √eps), as LassoPath does
    for warm_start in (true, false)
      x = AtomIterate(p, groups)
      coordinate_descent!(x, CDLeastSquaresLoss(y, X), GroupNormL2(λmax * (1 + sqrt(eps())), groups), CDOptions(; warm_start))
      @test nactive(x) == 0
    end

    # just below λmax, the group that attains λmax enters
    x = AtomIterate(p, groups)
    coordinate_descent!(x, CDLeastSquaresLoss(y, X), GroupNormL2(0.99 * λmax, groups), CDOptions(; warm_start = false))
    kmax = argmax([norm(X[:, G]' * y) / w[k] for (k, G) in enumerate(groups)])
    @test active_groups(x) == [kmax]
  end

  @testset "unpenalized group" begin
    # a group with zero weight is not penalized and is left out of λmax
    w0 = copy(w)
    w0[1] = 0
    g0 = GroupNormL2(λ, groups, w0)
    for warm_start in (true, false)
      x = AtomIterate(p, groups)
      coordinate_descent!(x, CDLeastSquaresLoss(y, X), g0, CDOptions(; opt_tol = 1e-10, warm_start, gaptol = 0))
      xv = Vector(x)
      @test 1 in nonzero_groups(xv)
      @test kkt_residual(xv, ls_gradient(xv), λ * w0) < 1e-8
    end
  end

  @testset "zero columns" begin
    # a zero column in a group of the model stays at zero, and a group of zero
    # columns stays zero, though its curvature is zero
    i, k0 = groups[13][1], 5
    X0 = copy(X)
    X0[:, i] .= 0
    X0[:, groups[k0]] .= 0
    for f in (CDLeastSquaresLoss(y, X0), CDQuadraticLoss(X0'X0 / n, -X0'y / n))
      x = AtomIterate(p, groups)
      coordinate_descent!(x, f, g, CDOptions(; opt_tol = 1e-10, warm_start = false, gaptol = 0))
      xv = Vector(x)
      @test xv[i] == 0
      @test 13 in nonzero_groups(xv)
      @test !(k0 in nonzero_groups(xv))
      @test !any(isnan, xv)
      @test kkt_residual(xv, -X0' * (y - X0 * xv) / n, λ * w) < 1e-8
    end
  end

  @testset "Float32" begin
    X32, y32 = Float32.(X), Float32.(y)
    g32 = GroupNormL2(Float32(λ), groups)
    x = AtomIterate(Float32, p, groups)
    coordinate_descent!(x, CDLeastSquaresLoss(y32, X32), g32, CDOptions(; opt_tol = 1e-6, warm_start = false))
    @test eltype(x) == Float32
    @test Vector(x) ≈ x_ref atol = 1e-3
  end

  @testset "dense starting point" begin
    x = zeros(p)
    @test coordinate_descent!(x, CDLeastSquaresLoss(y, X), g, opt).minimizer === x
    @test x ≈ x_ref atol = 1e-8
  end

  @testset "argument checks" begin
    # the groups of x must be those of g, and x must match the loss
    @test_throws ArgumentError coordinate_descent!(AtomIterate(p, 10), CDLeastSquaresLoss(y, X), g)
    @test_throws DimensionMismatch coordinate_descent!(AtomIterate(p, groups), CDLeastSquaresLoss(y, X[:, 1:end-1]), g)
  end
end

@testset "coordinate order" begin
  @test CDOptions().order === :random
  @test CDOptions(;randomize=false).order === :cyclic
  @test CDOptions(;randomize=true).order === :random
  @test !CDOptions(;order=:cyclic).randomize
  @test_throws ArgumentError CDOptions(;order=:active)       # removed in 0.7
  @test :randomize in propertynames(CDOptions())
  @test CDOptions(100, 1e-7, false, true, 50, true) == CDOptions(;max_iter=100, order=:cyclic)
  @test_throws ArgumentError CDOptions(;order=:shuffled)
  @test_throws ArgumentError CDOptions(;order=:cyclic, randomize=false)

  # every order gives the same solution
  n, p, s = 200, 500, 10
  X = randn(n, p)
  Y = X[:, 1:s] * ones(s) + randn(n)
  g = NormL1(0.05 * maximum(abs, X'Y) / n)
  f = CDLeastSquaresLoss(Y, X)
  xref = SparseIterate(p)
  coordinate_descent!(xref, f, g, CDOptions(;order=:cyclic, opt_tol=1e-12, gaptol = 0))
  for order in (:random, :cyclic), warm_start in (true, false)
    x = SparseIterate(p)
    info = solve!(x, f, g, CDOptions(;order, opt_tol=1e-12, warm_start, gaptol = 0))
    @test info.converged
    @test Vector(x) ≈ Vector(xref) atol=1e-8
  end
end

# least squares ‖y - Xx‖² / (2n) as a user-defined loss with a value, which
# Anderson acceleration needs: it undoes an extrapolation with initialize!
struct UserLS <: CoordinateDifferentiableFunction
  y::Vector{Float64}
  X::Matrix{Float64}
  r::Vector{Float64}
end
UserLS(y, X) = UserLS(y, X, copy(y))
HighDimensionalModels.initialize!(f::UserLS, x) = (f.r .= f.y .- f.X * Vector(x); nothing)
HighDimensionalModels.num_coordinates(f::UserLS) = size(f.X, 2)
HighDimensionalModels.gradient(f::UserLS, x, k) = -dot(view(f.X, :, k), f.r) / length(f.y)
HighDimensionalModels.coordinate_scale(f::UserLS, k) = norm(view(f.X, :, k)) / sqrt(length(f.y))
HighDimensionalModels.response_scale(f::UserLS) = norm(f.y) / sqrt(length(f.y))
function HighDimensionalModels.descend_coordinate!(f::UserLS, g, x, k)
  a = sum(abs2, view(f.X, :, k)) / length(f.y)
  old = x[k]
  new = prox_entry(g, old - HighDimensionalModels.gradient(f, x, k) / a, k, 1 / a)
  h = new - old
  iszero(h) && return h
  x[k] = new
  f.r .-= view(f.X, :, k) .* h
  h
end
(f::UserLS)(x) = sum(abs2, f.r) / (2 * length(f.y))

@testset "Anderson acceleration" begin
  @test CDOptions(;order=:cyclic, anderson=5).anderson == 5
  @test CDOptions(;randomize=false, anderson=5).anderson == 5
  @test_throws ArgumentError CDOptions(;anderson=5)                 # needs the cyclic order
  @test_throws ArgumentError CDOptions(;order=:cyclic, anderson=-1)

  # strongly correlated columns and more columns than rows, where coordinate
  # descent needs many passes
  n, p, s = 200, 800, 20
  X = randn(n, p)
  for j in 2:p
    @views X[:, j] .= 0.8 .* X[:, j-1] .+ 0.6 .* X[:, j]
  end
  Y = X[:, 1:s] * randn(s) + randn(n)
  w = 0.5 .+ rand(n)
  λ = 0.02 * maximum(abs, X'Y) / n
  λs = 0.2 * maximum(abs, X'Y) / norm(Y)
  plain = CDOptions(;order=:cyclic, opt_tol=1e-10, gaptol = 0)
  accelerated = CDOptions(;order=:cyclic, opt_tol=1e-10, anderson=5, gaptol = 0)

  # the same solution in fewer passes, for every built-in loss
  for (f, g, F) in ((CDLeastSquaresLoss(Y, X), NormL1(λ), x -> sum(abs2, Y - X*x) / 2n + λ * norm(x, 1)),
                    (CDLeastSquaresLoss(Y, X; intercept=true), NormL1(λ), x -> sum(abs2, (r = Y - X*x; r .- mean(r))) / 2n + λ * norm(x, 1)),
                    (CDWeightedLSLoss(Y, X, w), NormL1(λ), x -> sum(w .* (Y - X*x).^2) / 2n + λ * norm(x, 1)),
                    (CDQuadraticLoss(X'X/n, -X'Y/n), NormL1(λ), x -> sum(abs2, Y - X*x) / 2n + λ * norm(x, 1)),
                    (CDSqrtLassoLoss(Y, X), NormL1(λs), x -> norm(Y - X*x) + λs * norm(x, 1)),
                    (CDLeastSquaresLoss(Y, X), NormL1(λ, rand(p)), nothing))
    x0, x1 = SparseIterate(p), SparseIterate(p)
    info0 = solve!(x0, f, g, plain)
    info1 = solve!(x1, f, g, accelerated)
    @test info1.converged
    @test info1.passes < info0.passes / 1.5
    @test Vector(x1) ≈ Vector(x0) atol=1e-6
    F === nothing || @test F(Vector(x1)) ≈ F(Vector(x0)) rtol=1e-10
  end

  # without a penalty, from a path of penalties, and along LassoPath
  Xs, Ys = X[:, 1:100], Y
  x0, x1 = SparseIterate(100), SparseIterate(100)
  coordinate_descent!(x0, CDLeastSquaresLoss(Ys, Xs), ZeroFunction(), plain)
  coordinate_descent!(x1, CDLeastSquaresLoss(Ys, Xs), ZeroFunction(), accelerated)
  @test Vector(x1) ≈ Vector(x0) atol=1e-6
  f = CDLeastSquaresLoss(Y, X)
  x0, x1 = SparseIterate(p), SparseIterate(p)
  coordinate_descent!(x0, f, NormL1(λ), CDOptions(;order=:cyclic, opt_tol=1e-10, warm_start=false, gaptol = 0))
  coordinate_descent!(x1, f, NormL1(λ), CDOptions(;order=:cyclic, opt_tol=1e-10, warm_start=false, anderson=5, gaptol = 0))
  @test Vector(x1) ≈ Vector(x0) atol=1e-6
  λpath = [0.2, 0.1, 0.05, 0.02] .* (maximum(abs, X'Y) / n)
  path0 = LassoPath(X, Y, λpath, plain; standardize_x=false)
  path1 = LassoPath(X, Y, λpath, accelerated; standardize_x=false)
  @test all(path1.converged)
  @test sum(path1.iterations) < sum(path0.iterations)
  for k in eachindex(λpath)
    @test Vector(path1.βpath[k]) ≈ Vector(path0.βpath[k]) atol=1e-6
  end

  # a user-defined loss and penalty that define value
  x0, x1 = SparseIterate(p), SparseIterate(p)
  info0 = solve!(x0, f, ElasticNetPenalty(λ, 0.5), plain)
  info1 = solve!(x1, UserLS(Y, X), ElasticNetPenalty(λ, 0.5), accelerated)
  @test info1.converged
  @test info1.passes < info0.passes
  @test Vector(x1) ≈ Vector(x0) atol=1e-6

  # the groups of an AtomIterate, and a loss without a value, run without it
  groups8 = [((k-1)*8+1):(k*8) for k in 1:100]
  gg = GroupNormL2(λ, groups8)
  x0, x1 = AtomIterate(p, groups8), AtomIterate(p, groups8)
  coordinate_descent!(x0, f, gg, plain)
  coordinate_descent!(x1, f, gg, accelerated)
  @test Vector(x1) ≈ Vector(x0) atol=1e-6
  x = SparseIterate(3)
  coordinate_descent!(x, SeparableLoss([1., -2., 3.]), NormL1(0.5), accelerated)
  @test Vector(x) ≈ [0.5, -1.5, 2.5]
end


@testset "momentum" begin
  @test CDOptions().momentum === nothing           # used where it is known to help
  @test CDOptions(;momentum=true).momentum
  @test CDOptions(;momentum=false).momentum === false
  @test CDOptions(;order=:cyclic).momentum === nothing
  @test_throws ArgumentError CDOptions(;order=:cyclic, momentum=true)   # can stall
  @test_throws ArgumentError CDOptions(;order=:cyclic, anderson=5, momentum=true)
  o = CDOptions(;order=:cyclic, anderson=5)
  @test CDOptions(o.max_iter, o.opt_tol, o.order, o.warm_start, o.num_steps, o.relative_tol, o.anderson) == o

  # a common factor between the columns, where the random order is fast and
  # the momentum makes it faster
  n, p, s = 300, 1000, 20
  X = sqrt(0.5) * randn(n) .+ sqrt(0.5) * randn(n, p)
  Y = X[:, 1:s] * randn(s) + randn(n)
  w = 0.5 .+ rand(n)
  λ = 0.02 * maximum(abs, X'Y) / n
  λs = 0.2 * maximum(abs, X'Y) / norm(Y)
  plain = CDOptions(;opt_tol=1e-10, momentum=false, gaptol = 0)
  mom = CDOptions(;opt_tol=1e-10, momentum=true, gaptol = 0)

  # the same solution, in fewer passes over all losses together; UserLS has a
  # cache that ProximalSolvers does not know, and is initialized at the
  # extrapolated point instead
  passes0, passes1 = 0, 0
  for (f, g) in ((CDLeastSquaresLoss(Y, X), NormL1(λ)),
                 (CDLeastSquaresLoss(Y, X; intercept=true), NormL1(λ)),
                 (CDWeightedLSLoss(Y, X, w), NormL1(λ)),
                 (CDQuadraticLoss(X'X/n, -X'Y/n), NormL1(λ)),
                 (CDSqrtLassoLoss(Y, X), NormL1(λs)),
                 (CDLeastSquaresLoss(Y, X), NormL1(λ, rand(p))),
                 (UserLS(Y, X), NormL1(λ)))
    x0, x1 = SparseIterate(p), SparseIterate(p)
    info0 = solve!(x0, f, g, plain)
    info1 = solve!(x1, f, g, mom)
    @test info1.converged
    @test Vector(x1) ≈ Vector(x0) atol=1e-6
    passes0 += info0.passes
    passes1 += info1.passes
  end
  @test passes1 < passes0 / 1.3

  # without a penalty, from a path of penalties, and along LassoPath
  f = CDLeastSquaresLoss(Y, X)
  xref = SparseIterate(p)
  coordinate_descent!(xref, f, NormL1(λ), plain)
  x = SparseIterate(p)
  coordinate_descent!(x, f, NormL1(λ), CDOptions(;opt_tol=1e-10, warm_start=false, momentum=true, gaptol = 0))
  @test Vector(x) ≈ Vector(xref) atol=1e-6
  x0, x1 = SparseIterate(100), SparseIterate(100)
  coordinate_descent!(x0, CDLeastSquaresLoss(Y, X[:, 1:100]), ZeroFunction(), plain)
  coordinate_descent!(x1, CDLeastSquaresLoss(Y, X[:, 1:100]), ZeroFunction(), mom)
  @test Vector(x1) ≈ Vector(x0) atol=1e-6
  λpath = [0.2, 0.1, 0.05, 0.02] .* (maximum(abs, X'Y) / n)
  path0 = LassoPath(X, Y, λpath, plain; standardize_x=false)
  path1 = LassoPath(X, Y, λpath, mom; standardize_x=false)
  @test all(path1.converged)
  for k in eachindex(λpath)
    @test Vector(path1.βpath[k]) ≈ Vector(path0.βpath[k]) atol=1e-6
  end

  # on the groups of an AtomIterate when asked for, from ProximalSolvers 0.3.1,
  # with the solution of the plain passes; earlier releases refuse it
  groups = [((k-1)*10+1):(k*10) for k in 1:100]
  gg = GroupNormL2(λ, groups)
  if pkgversion(ProximalSolvers) >= v"0.3.1"
    xg0, xg1 = AtomIterate(p, groups), AtomIterate(p, groups)
    coordinate_descent!(xg0, f, gg, plain)
    coordinate_descent!(xg1, f, gg, mom)
    @test Vector(xg1) ≈ Vector(xg0) atol=1e-6
  else
    @test_throws ArgumentError coordinate_descent!(AtomIterate(p, groups), f, gg, mom)
  end

  # by default it is used with a random order, a SparseIterate and a built-in
  # loss, and not otherwise: the same runs as momentum = true or false
  passes(f, g, x, o; seed = 7) = (Random.seed!(seed); solve!(x, f, g, o).passes)
  auto = CDOptions(;opt_tol=1e-10, gaptol = 0)
  @test passes(f, NormL1(λ), SparseIterate(p), auto) == passes(f, NormL1(λ), SparseIterate(p), mom)
  @test passes(f, NormL1(λ), SparseIterate(p), auto) != passes(f, NormL1(λ), SparseIterate(p), plain)
  uf = UserLS(Y, X)
  @test passes(uf, NormL1(λ), SparseIterate(p), auto) == passes(uf, NormL1(λ), SparseIterate(p), plain)
  cyc = CDOptions(;opt_tol=1e-10, order=:cyclic, gaptol = 0)
  @test passes(f, NormL1(λ), SparseIterate(p), cyc) ==
        passes(f, NormL1(λ), SparseIterate(p), CDOptions(;opt_tol=1e-10, order=:cyclic, momentum=false, gaptol = 0))
  @test passes(f, gg, AtomIterate(p, groups), auto) == passes(f, gg, AtomIterate(p, groups), plain)
end

end
