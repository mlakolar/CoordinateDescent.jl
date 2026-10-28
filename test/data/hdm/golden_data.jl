# U1 input fixtures. Run with Julia 1.13.3 and one BLAS thread; see the README.
# All randomness uses the explicitly constructed Xoshiro, never default_rng().
#
# hetero (seed 11801, n=200, p=50), highdim (11802, n=100, p=150):
#   Z_ij, v_i, e_i iid N(0,1); X_1=Z_1,
#   X_j=.5 X_{j-1}+sqrt(.75) Z_j (AR(.5), independent rows).
#   d=.25 + X_[1,3,5] * [1.5,-1.5,1.5] + .5v;
#   y=1 + .5d + X_[1,3,5] * [1.5,-1.5,1.5]
#       + (.3+.3|X_7|)e. All nonzero structural slopes have magnitude >= .5.
# forced reuses hetero bit for bit, with forced indices [40,43]. They must be
# absent from both nuisance supports, for both post choices, in both packages.
#
# nearthreshold (seed 11803, n=120, p=10): form Q from the unpivoted QR of
# [ones(n) randn(rng,n,12)], and W=sqrt(n)*Q[:,2:13]. Then
#   X=W[:,1:10], X_2 *= 1e6;
#   d=1.5X_1 + 6e-7 X_2 + .8W_11;
#   y=.5d + X_1 + 5e-7 X_2 + .8W_12.
# X is centered and orthogonal. The small raw-coordinate slopes (6e-7 and
# 8e-7 in the two reduced forms) deliberately exercise hdm's 1e-6 coefficient
# cutoff, despite appreciable fitted contributions. This is NOT a parity case.
#
# empty (seed 12138, n=100, p=20): independent iid N(0,1) X, y, d, with
# y and d centered. This fixed draw selects nothing even with post=false.
# Draw order: X, then y, then d (unlike the treatment-first designs above).

using DelimitedFiles, LinearAlgebra, Printf, Random, Statistics
BLAS.set_num_threads(1)

function write_data(root, name, X, y, d; forced = Int[])
  dir = joinpath(root, name)
  mkpath(dir)
  for (file, a) in (("X", X), ("y", y), ("d", d))
    open(joinpath(dir, file * ".csv"), "w") do io
      for i in axes(a, 1)
        println(io, join((@sprintf("%.17g", a[i, j]) for j in axes(a, 2)), ','))
      end
    end
  end
  isempty(forced) || writedlm(joinpath(dir, "forced.csv"), forced, ',')
end

function ar_data(n, p, seed)
  rng = Xoshiro(seed)
  X = randn(rng, n, p)
  for j in 2:p
    @views X[:, j] .= 0.5 .* X[:, j-1] .+ sqrt(0.75) .* X[:, j]
  end
  b = X[:, [1, 3, 5]] * [1.5, -1.5, 1.5]
  d = 0.25 .+ b .+ 0.5 .* randn(rng, n)
  y = 1 .+ 0.5 .* d .+ b .+ (0.3 .+ 0.3 .* abs.(X[:, 7])) .* randn(rng, n)
  X, y, d
end

function generate(root)
  X, y, d = ar_data(200, 50, 11801)
  write_data(root, "hetero", X, y, d)
  write_data(root, "forced", X, y, d; forced = [40, 43])
  write_data(root, "highdim", ar_data(100, 150, 11802)...)

  rng = Xoshiro(11803)
  n = 120
  W = sqrt(n) .* Matrix(qr([ones(n) randn(rng, n, 12)]).Q)[:, 2:13]
  X = copy(W[:, 1:10]); X[:, 2] .*= 1e6
  d = 1.5 .* X[:, 1] .+ 6e-7 .* X[:, 2] .+ 0.8 .* W[:, 11]
  y = 0.5 .* d .+ X[:, 1] .+ 5e-7 .* X[:, 2] .+ 0.8 .* W[:, 12]
  write_data(root, "nearthreshold", X, y, d)

  rng = Xoshiro(12138)
  X, y, d = randn(rng, 100, 20), randn(rng, 100), randn(rng, 100)
  y .-= mean(y); d .-= mean(d)
  write_data(root, "empty", X, y, d)
end

generate(isempty(ARGS) ? (@__DIR__) : only(ARGS))
