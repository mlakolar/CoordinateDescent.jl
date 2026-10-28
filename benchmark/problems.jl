# Inputs of the benchmark cases: for a scenario key and a repetition,
# scenario(key, rep) returns a NamedTuple of raw Float64 arrays, drawn with the
# standard library only. Keys are family-[variant-]shape, as ls-sparse-500x200;
# SCENARIOS lists them. Fields by family: X and y; weights and offset (weighted
# variants); groups; Y (multitask and low rank); x, the locations of a signal;
# Sigma and mu (knockoffs); d, the treatment (inference); truth, the
# coefficients that generated the data, and effect, the treatment effect.
# Float32 copies are made by the caller.

using LinearAlgebra, Random, SparseArrays, Statistics, SHA

# Only the random draws are the same on every Julia version; BLAS, LAPACK and
# Statistics may round differently. So the inputs are computed from the draws
# by elementwise operations and by the products and moments below, which add
# in a fixed order, and are then identical on Julia 1.10 and 1.13.
function product(X::AbstractMatrix, b::AbstractVector)
  y = zeros(size(X, 1))
  for j in axes(X, 2), i in axes(X, 1)
    y[i] += X[i, j] * b[j]
  end
  y
end

function product(X::SparseMatrixCSC, b::AbstractVector)
  y = zeros(size(X, 1))
  rows, vals = rowvals(X), nonzeros(X)
  for j in axes(X, 2), k in nzrange(X, j)
    y[rows[k]] += vals[k] * b[j]
  end
  y
end

product(X::AbstractMatrix, B::AbstractMatrix) = reduce(hcat, [product(X, B[:, j]) for j in axes(B, 2)])

function sample_mean(v)
  s = 0.0
  for t in v
    s += t
  end
  s / length(v)
end

function sample_sd(v)
  m, s = sample_mean(v), 0.0
  for t in v
    s += abs2(t - m)
  end
  sqrt(s / (length(v) - 1))
end

# The stream of a key and repetition: Xoshiro seeded with the first 8 bytes of
# sha256("$key|$rep") read as a big-endian integer (not Base.hash, which can
# change between Julia versions). Its scalar uniform draws are the same on
# Julia 1.10 and 1.13, but randn and randexp are not (their tail algorithms
# differ), nor are the bulk fills of long arrays. So every draw is a scalar
# rand: normals by Marsaglia's polar method and exponentials by inversion, with
# log and sqrt, which also agree; randperm and sampling from ranges are not used.
function scenario_rng(key::AbstractString, rep::Integer)
  digest = sha256("$key|$rep")
  Xoshiro(foldl((a, b) -> (a << 8) | UInt64(b), digest[1:8]; init = UInt64(0)))
end

function normal_draw(rng)
  while true
    u, v = 2rand(rng) - 1, 2rand(rng) - 1
    s = u * u + v * v
    0 < s < 1 && return u * sqrt(-2log(s) / s)
  end
end

exponential_draw(rng) = -log1p(-rand(rng))
gaussian(rng, dims::Integer...) = reshape([normal_draw(rng) for _ in 1:prod(dims)], dims...)
uniform(rng, dims::Integer...) = reshape([rand(rng) for _ in 1:prod(dims)], dims...)

# k distinct indices of 1:p, in random order
draw_indices(rng, p, k) = sortperm(uniform(rng, p))[1:k]
draw_signs(rng, k) = [rand(rng) < 0.5 ? -1.0 : 1.0 for _ in 1:k]

# a coefficient vector with k nonzeros of magnitudes in [0.5, 1.5]
function sparse_truth(rng, p, k)
  β = zeros(p)
  β[draw_indices(rng, p, k)] .= draw_signs(rng, k) .* (0.5 .+ uniform(rng, k))
  β
end

# the noise level that gives a signal-to-noise variance ratio snr
noise_sd(signal, snr) = (σ = sample_sd(signal); σ > 0 ? σ / sqrt(snr) : 1.0)

# Gaussian columns with shifted means and scales in [0.5, 2], so that centering
# and loadings matter; ρ > 0 makes the columns equicorrelated
function gaussian_design(rng, n, p; ρ = 0.0)
  Z = gaussian(rng, n, p)
  ρ > 0 && (Z .= sqrt(1 - ρ) .* Z .+ sqrt(ρ) .* gaussian(rng, n))
  m, s = 0.3 .* gaussian(rng, p), 0.5 .+ 1.5 .* uniform(rng, p)
  (Z .+ transpose(m)) .* transpose(s)
end

sigmoid(t) = t >= 0 ? inv(1 + exp(-t)) : exp(t) / (1 + exp(t))

# a Poisson draw by inversion of the distribution function
function poisson_draw(rng, μ)
  u, k = rand(rng), 0
  term = exp(-μ)
  total = term
  while u > total && k < 10_000
    k += 1
    term *= μ / k
    total += term
  end
  Float64(k)
end

# Laplace noise, heavier tailed than Gaussian
laplace(rng, n) = [(rand(rng) < 0.5 ? -1.0 : 1.0) * exponential_draw(rng) for _ in 1:n]

function least_squares(rng, n, p; k = 5, ρ = 0.0)
  X = gaussian_design(rng, n, p; ρ)
  β = sparse_truth(rng, p, k)
  signal = product(X, β)
  (; X, y = 1.0 .+ signal .+ noise_sd(signal, 4.0) .* gaussian(rng, n), truth = β)
end

# about 5% of the entries stored, the columns on scales in [0.5, 2]
function sparse_least_squares(rng, n, p; k = 5)
  stored = uniform(rng, n, p) .< 0.05
  values = gaussian(rng, n, p) .* transpose(0.5 .+ 1.5 .* uniform(rng, p))
  X = sparse(stored .* values)
  β = sparse_truth(rng, p, k)
  signal = product(X, β)
  (; X, y = 1.0 .+ signal .+ noise_sd(signal, 4.0) .* gaussian(rng, n), truth = β)
end

# positive weights and an offset; the noise variance is inverse to the weight
function weighted_least_squares(rng, n, p; k = 5)
  X = gaussian_design(rng, n, p)
  β = sparse_truth(rng, p, k)
  weights, offset = 0.5 .+ uniform(rng, n), 0.3 .* gaussian(rng, n)
  signal = product(X, β)
  y = offset .+ 1.0 .+ signal .+ noise_sd(signal, 4.0) .* gaussian(rng, n) ./ sqrt.(weights)
  (; X, y, weights, offset, truth = β)
end

function logistic(rng, n, p; k = 5, weighted = false)
  X = gaussian_design(rng, n, p)
  β = 0.5 .* sparse_truth(rng, p, k)
  weights = weighted ? 0.5 .+ uniform(rng, n) : nothing
  offset = weighted ? 0.3 .* gaussian(rng, n) : nothing
  η = -0.3 .+ product(X, β) .+ (weighted ? offset : 0.0)
  y = Float64.(uniform(rng, n) .< sigmoid.(η))
  weighted ? (; X, y, weights, offset, truth = β) : (; X, y, truth = β)
end

function poisson(rng, n, p; k = 5)
  X = gaussian_design(rng, n, p)
  β = 0.25 .* sparse_truth(rng, p, k)
  η = 0.5 .+ product(X, β)
  (; X, y = [poisson_draw(rng, exp(t)) for t in η], truth = β)
end

# twelve groups of five consecutive columns, correlated within the group; three
# active groups with three nonzero coefficients each
function grouped(rng, n, ngroups, size)
  p = ngroups * size
  groups = [collect((g - 1) * size + 1:g * size) for g in 1:ngroups]
  X = gaussian_design(rng, n, p)
  for G in groups
    sd = transpose([sample_sd(X[:, j]) for j in G])
    X[:, G] .= sqrt(0.7) .* X[:, G] .+ sqrt(0.3) .* gaussian(rng, n) .* sd
  end
  β = zeros(p)
  for g in draw_indices(rng, ngroups, 3)
    G = groups[g]
    β[G[draw_indices(rng, size, 3)]] .= draw_signs(rng, 3) .* (0.5 .+ uniform(rng, 3))
  end
  signal = product(X, β)
  (; X, y = 1.0 .+ signal .+ noise_sd(signal, 4.0) .* gaussian(rng, n), groups, truth = β)
end

# five nonzero rows shared by the q responses
function multitask(rng, n, p, q; k = 5)
  X = gaussian_design(rng, n, p)
  B = zeros(p, q)
  B[draw_indices(rng, p, k), :] .= 0.7 .* gaussian(rng, k, q)
  signal = product(X, B)
  Y = transpose(gaussian(rng, q)) .+ signal .+ noise_sd(vec(signal), 4.0) .* gaussian(rng, n, q)
  (; X, Y, truth = B)
end

# a coefficient matrix of rank two
function lowrank(rng, n, p, q; rank = 2)
  X = gaussian_design(rng, n, p)
  B = product(gaussian(rng, p, rank), gaussian(rng, rank, q)) ./ sqrt(p)
  signal = product(X, B)
  Y = transpose(gaussian(rng, q)) .+ signal .+ noise_sd(vec(signal), 4.0) .* gaussian(rng, n, q)
  (; X, Y, truth = B)
end

function quantile_regression(rng, n, p; k = 4)
  X = gaussian_design(rng, n, p)
  β = sparse_truth(rng, p, k)
  signal = product(X, β)
  (; X, y = 1.0 .+ signal .+ noise_sd(signal, 4.0) .* laplace(rng, n), truth = β)
end

# a piecewise-constant signal on unit-spaced locations, with segments of random
# lengths and levels
function piecewise_constant(rng, n; segments = 6)
  cuts = sort!(draw_indices(rng, n - 1, segments - 1))
  levels = 2 .* gaussian(rng, segments)
  θ = [levels[searchsortedfirst(cuts, i)] for i in 1:n]
  (; x = collect(1.0:n), y = θ .+ 0.5 .* gaussian(rng, n), truth = θ)
end

# a continuous piecewise-linear signal on irregular, strictly increasing
# locations
function piecewise_linear(rng, n; knots = 4)
  x = cumsum(0.5 .+ uniform(rng, n))
  at = sort!(x[draw_indices(rng, n - 2, knots) .+ 1])
  slopes = gaussian(rng, knots + 1) ./ sample_sd(x)
  θ = slopes[1] .* x
  for k in 1:knots
    θ .+= (slopes[k+1] - slopes[k]) .* max.(x .- at[k], 0.0)
  end
  (; x, y = θ .+ 0.25 * sample_sd(θ) .* gaussian(rng, n), truth = θ)
end

# regression whose coefficients are piecewise constant in the column index;
# n > p, so X has full column rank
function fused_regression(rng, n, p)
  X = gaussian(rng, n, p)
  β = zeros(p)
  b = p ÷ 6
  β[b+1:3b] .= 1.0
  β[4b+1:5b] .= -1.0
  signal = product(X, β)
  (; X, y = signal .+ noise_sd(signal, 4.0) .* gaussian(rng, n), truth = β)
end

# rows drawn from N(0, Σ), Σ = diag(d) R diag(d) with R autoregressive (0.4) and
# scales d in [0.5, 2], so that diag(Σ) is not one; the columns are the
# autoregression X₁ = Z₁, Xⱼ = 0.4Xⱼ₋₁ + √0.84 Zⱼ, scaled by d
function knockoff(rng, n, p; k = 10)
  powers = accumulate(*, fill(0.4, p - 1); init = 1.0)
  R = [i == j ? 1.0 : powers[abs(i - j)] for i in 1:p, j in 1:p]
  d = 0.5 .+ 1.5 .* uniform(rng, p)
  # the same order of products in both triangles, so that Sigma is symmetric
  Sigma = [d[min(i, j)] * R[i, j] * d[max(i, j)] for i in 1:p, j in 1:p]
  X = gaussian(rng, n, p)
  for j in 2:p
    X[:, j] .= 0.4 .* X[:, j-1] .+ sqrt(0.84) .* X[:, j]
  end
  X .*= transpose(d)
  β = sparse_truth(rng, p, k)
  signal = product(X, β)
  (; X, y = signal .+ noise_sd(signal, 4.0) .* gaussian(rng, n), Sigma, mu = zeros(p), truth = β)
end

# the partially linear model y = 0.5 d + Xβ + ε of R's hdm, with the treatment
# d = Xγ + v depending on a few of the controls
function inference(rng, n, p; k = 5)
  X = gaussian_design(rng, n, p)
  β, γ = sparse_truth(rng, p, k), sparse_truth(rng, p, k)
  d = product(X, γ) .+ gaussian(rng, n)
  (; X, y = 0.5 .* d .+ product(X, β) .+ gaussian(rng, n), d, truth = β, effect = 0.5)
end

const SCENARIO_BUILDERS = [
  # quick shapes
  "ls-200x50" => rng -> least_squares(rng, 200, 50),
  "ls-sparse-500x200" => rng -> sparse_least_squares(rng, 500, 200),
  "ls-tall-5000x40" => rng -> least_squares(rng, 5000, 40),
  "ls-wide-100x400" => rng -> least_squares(rng, 100, 400),
  "ls-corr-300x100" => rng -> least_squares(rng, 300, 100; ρ = 0.5),
  "ls-weighted-200x50" => rng -> weighted_least_squares(rng, 200, 50),
  "logistic-300x60" => rng -> logistic(rng, 300, 60),
  "logistic-weighted-300x60" => rng -> logistic(rng, 300, 60; weighted = true),
  "poisson-300x40" => rng -> poisson(rng, 300, 40),
  "groups-200x60" => rng -> grouped(rng, 200, 12, 5),
  "multi-200x40x5" => rng -> multitask(rng, 200, 40, 5),
  "lowrank-150x30x10" => rng -> lowrank(rng, 150, 30, 10),
  "lowrank-wide-60x100x10" => rng -> lowrank(rng, 60, 100, 10),
  "quantile-200x30" => rng -> quantile_regression(rng, 200, 30),
  "signal-200" => rng -> piecewise_constant(rng, 200),
  "fusedreg-150x60" => rng -> fused_regression(rng, 150, 60),
  "trend-200" => rng -> piecewise_linear(rng, 200),
  "slope-150x60" => rng -> least_squares(rng, 150, 60; k = 8),
  "knockoff-300x50" => rng -> knockoff(rng, 300, 50),
  "hdm-200x80" => rng -> inference(rng, 200, 80),
  # extended shapes
  "ls-2000x1000" => rng -> least_squares(rng, 2000, 1000; k = 10),
  "logistic-2000x500" => rng -> logistic(rng, 2000, 500; k = 10),
  "quantile-1000x100" => rng -> quantile_regression(rng, 1000, 100; k = 6),
  "signal-5000" => rng -> piecewise_constant(rng, 5000; segments = 12),
]

const SCENARIOS = String[first(b) for b in SCENARIO_BUILDERS]

function scenario(key::AbstractString, rep::Integer)
  k = findfirst(b -> first(b) == key, SCENARIO_BUILDERS)
  k === nothing && throw(ArgumentError("unknown scenario $key; the scenarios are $(join(SCENARIOS, ", "))"))
  rep >= 1 || throw(ArgumentError("the repetition must be positive, but it is $rep"))
  last(SCENARIO_BUILDERS[k])(scenario_rng(key, rep))
end

# A lowercase hex SHA-256 (64 characters) of the inputs over their sizes,
# element types and bytes, stable across processes and Julia versions: fields
# in the order of their names, arrays in column-major order, sparse matrices by
# their compressed columns, numbers little-endian
function hash_data!(io, a::SparseMatrixCSC)
  write(io, "SparseMatrixCSC{$(eltype(a))}")
  hash_data!(io, size(a))
  hash_data!(io, a.colptr)
  hash_data!(io, rowvals(a))
  hash_data!(io, nonzeros(a))
end

function hash_data!(io, a::AbstractArray)
  write(io, "Array{$(eltype(a))}")
  hash_data!(io, size(a))
  if isbitstype(eltype(a))
    for v in a
      write(io, htol(v))
    end
  else
    foreach(v -> hash_data!(io, v), a)
  end
end

function hash_data!(io, a::Tuple)
  write(io, htol(Int64(length(a))))
  foreach(v -> hash_data!(io, Int64(v)), a)
end

hash_data!(io, a::Union{Integer, AbstractFloat}) = (write(io, string(typeof(a))); write(io, htol(a)))
hash_data!(io, a::Bool) = write(io, a ? "true" : "false")
hash_data!(io, a::Union{AbstractString, Symbol}) = (s = string(a); write(io, htol(Int64(ncodeunits(s))), s))
hash_data!(io, ::Nothing) = write(io, "nothing")

function data_hash(nt::NamedTuple)
  io = IOBuffer()
  for name in sort!(collect(keys(nt)); by = string)
    write(io, string(name), '\0')
    hash_data!(io, getproperty(nt, name))
  end
  bytes2hex(sha256(take!(io)))
end
