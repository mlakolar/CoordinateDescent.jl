facts("kernels") do
  x = 0.3
  y = 0.4
  k = GaussianKernel(1.)
  @fact evaluate(k, x, y) --> roughly(exp(-0.01))

  function evaluate!(g::GaussianKernel{T}, buffer::Array{T}, x::Array{T}, y::Float64) where {T <: AbstractFloat}
    size(buffer) == size(x) || throw(DimensionMismatch())

    @inbounds @simd for i=1:length(x)
       buffer[i] = evaluate(g, x[i], y)
    end
    buffer
  end

  x = rand(100, 1000)
  y = rand()
  k = GaussianKernel(0.5)

  o1 = zeros(x)
  o2 = zeros(x)
  @. o1 = evaluate(k, x, y)
  evaluate!(k, o2, x, y)

  @fact o1 --> roughly(o2)
end

facts("expand_X") do
  X = reshape(collect(1.:6.), 2, 3)
  z = [0.2, 0.4]
  z0 = 0.3

  tX = similar(X)
  @fact CoordinateDescent._expand_X!(tX, X, z, z0, 0) --> X

  tX = zeros(2, 6)
  tX1 = zeros(2, 6)
  Q = [1. -0.1; 1. 0.1]
  for i=1:2
    tX1[i, :] = kron(X[i,:], Q[i, :])
  end
  @fact CoordinateDescent._expand_X!(tX, X, z, z0, 1) --> roughly(tX1)

  tX = zeros(2, 9)
  tX1 = zeros(2, 9)
  Q = [1. -0.1 0.01; 1. 0.1 0.01]
  for i=1:2
    tX1[i, :] = kron(X[i,:], Q[i, :])
  end
  @fact CoordinateDescent._expand_X!(tX, X, z, z0, 2) --> roughly(tX1)
end


facts("locpoly") do




end
