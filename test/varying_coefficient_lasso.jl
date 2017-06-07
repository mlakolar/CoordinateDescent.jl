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

facts("expand_X multiplications") do

  p = 10
  X = randn(100, p)
  Y = randn(100)
  z = rand(100)
  w = zeros(100)
  k = GaussianKernel(0.2)
  @. w = evaluate(k, z, 0.5)

  for degree=0:2
    cp = p*(degree+1)
    eX = zeros(100, cp)
    ewX = zeros(100, cp)

    CoordinateDescent._expand_X!(eX, X, z, 0.5, degree)
    @fact CoordinateDescent._expand_wX!(ewX, w, X, z, 0.5, degree) --> roughly(diagm(w)*eX)

    Xt_w_Y = zeros(cp)
    @fact CoordinateDescent._expand_Xt_w_Y!(Xt_w_Y, w, X, z, Y, 0.5, degree) --> roughly(eX' * diagm(w) * Y)

    Xt_w_X = zeros(cp, cp)
    @fact CoordinateDescent._expand_Xt_w_X!(Xt_w_X, w, X, z, 0.5, degree) --> roughly((eX'*diagm(w))*eX)
  end
end

facts("locpoly") do

  n, p = 500, 2
  Y, X, Z, betaMat = genData(n, p)
  zgrid = collect(0.01:0.2:0.99)

  gk = GaussianKernel(0.4)

  w = zeros(n)

  for degree=0:2
    @fact locpoly(X, Z, Y, zgrid, degree, gk) --> roughly(locpoly_alt(X, Z, Y, zgrid, degree, gk))

    z0 = 0.5
    cp = p*(degree+1)

    eX = zeros(n, cp)
    _expand_X!(eX, X, Z, z0, degree)

    @. w = evaluate(gk, Z, z0)

    @fact locpoly(X, Z, Y, z0, degree, gk) --> roughly((eX' * diagm(w) * eX)\(eX' * diagm(w) * Y))
  end
end



facts("locpolyl1") do

  n, s, p = 500, 10, 50
  gk = GaussianKernel(0.1)
  zgrid = collect(0.01:0.1:0.99)
  opt = CDOptions(;randomize=false)

  for i=1:NUMBER_REPEAT
    Y, X, Z, betaT = genData(n, s)
    X = [X zeros(n, p-s)]

    λ0 = rand() / 10

    for degree=0:2
      o1 = locpolyl1(X,Z,Y,zgrid,degree,gk,λ0, opt)
      o2 = locpolyl1_alt(X,Z,Y,zgrid,degree,gk,λ0, opt)

      @fact maximum( maximum(abs.(o1[i] - o2[i])) for i=1:length(zgrid) ) --> roughly(0.; atol=1e-4)
    end
  end

end
