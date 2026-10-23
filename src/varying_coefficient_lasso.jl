# Varying-coefficient models
#
#   y_i = ∑_j X_ij β_j(z_i) + ε_i
#
# estimated by local polynomial regression around a point z0: each β_j(z) is
# approximated by a polynomial of degree q in (z - z0), so that the data matrix
# is expanded to X_i ⊗ [1, (z_i - z0), ..., (z_i - z0)^q] and the observations
# are weighted by a kernel K_h(z_i, z0). Coefficients of the expanded design
# are ordered by variable: for variable j, entries (j-1)(q+1)+1 : j(q+1) hold
# β_j(z0), β_j'(z0), ..., up to scaling by factorials.

############################################################
#
# kernel functions
#
############################################################

"""
    SmoothingKernel{T}

Supertype of kernels `K_h(x, y)` with bandwidth `h` used to weight observations
in local polynomial regression. Evaluate with [`evaluate`](@ref).
"""
abstract type SmoothingKernel{T} end

"""
    GaussianKernel(h)

Gaussian kernel `K_h(x, y) = exp(-(x - y)² / h) / h`.

Note that `h` enters as a variance (`h = 2σ²`), not as a standard deviation.
"""
struct GaussianKernel{T} <: SmoothingKernel{T}
  h::T
end

"""
    EpanechnikovKernel(h)

Epanechnikov kernel `K_h(x, y) = 0.75 (1 - u²) / h` for `|u| < 1` and zero
otherwise, where `u = (x - y) / h`.
"""
struct EpanechnikovKernel{T} <: SmoothingKernel{T}
  h::T
end

"""
    createKernel(KernelType, h)

Construct a kernel of type `KernelType` (for example `GaussianKernel{Float64}`)
with bandwidth `h`. Used to loop over bandwidths when only the kernel type is
known.
"""
createKernel(::Type{GaussianKernel{T}}, h::T) where {T <: AbstractFloat} = GaussianKernel{T}(h)
createKernel(::Type{EpanechnikovKernel{T}}, h::T) where {T <: AbstractFloat} = EpanechnikovKernel{T}(h)

"""
    evaluate(k::SmoothingKernel, x, y)

Value of the kernel `K_h(x, y)`.
"""
evaluate(k::GaussianKernel{T}, x::T, y::T) where {T <: AbstractFloat} = exp(-abs2(x - y) / k.h) / k.h
function evaluate(k::EpanechnikovKernel{T}, x::T, y::T) where {T <: AbstractFloat}
    u = (x - y) / k.h
    abs(u) >= 1 ? zero(T) : T(0.75) * (1 - abs2(u)) / k.h
end


############################################################
#
# local polynomial regression with lasso
#
############################################################

"""
    locpolyl1(X, z, y, zgrid, degree, kernel, λ0, refit, options=CDOptions())

Local polynomial estimate of a sparse varying-coefficient model at each point
`z0` of `zgrid`. At `z0` it solves the weighted lasso

    minimize  ∑ᵢ wᵢ (yᵢ - X̃ᵢ'β)² / (2n) + λ0 ∑ₖ sₖ |βₖ|

where `wᵢ = K_h(zᵢ, z0)`, `X̃ᵢ = Xᵢ ⊗ [1, (zᵢ - z0), ..., (zᵢ - z0)^degree]` and
`sₖ = sqrt(∑ᵢ wᵢ X̃ᵢₖ² / n)` standardizes the columns of the expanded design.
Each problem is warm started from the solution at the previous grid point.

The loss scales with the kernel and the loadings `sₖ` with its square root, so
`λ0` is on the scale of the kernel: multiplying the kernel by a constant `c`
gives the same estimate when `λ0` is multiplied by `√c`.

Returns `(B, BR)`, sparse matrices of size `p(degree + 1) × length(zgrid)`.
Column `k` of `B` holds the lasso estimate at `zgrid[k]`. If `refit` is true,
column `k` of `BR` holds the weighted least squares refit on the selected
variables (all `degree + 1` coefficients of a variable are refit when any of
them is nonzero, and the refit of minimum norm is taken when it is not
unique); otherwise `BR` is zero.
"""
function locpolyl1(
  X::Matrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int64,
  kernel::SmoothingKernel{T},
  λ0::T,
  refit::Bool,
  options::CDOptions=CDOptions()) where {T <: AbstractFloat}

  # the weighted loss below checks its data only when it is built, and then
  # reads it in place
  _check_finite(X, "X")
  _check_finite(z, "z")
  _check_finite(y, "y")
  _check_finite(zgrid, "zgrid")

  # construct inner options because we do not want warmStart = false
  # we want to start from the previous iteration since the points
  # on the grid should be close to each other
  opt = _withWarmStart(options)

  n, p = size(X)
  ep = p * (degree + 1)
  out = spzeros(T, ep, length(zgrid))
  outR = spzeros(T, ep, length(zgrid))

  # temporary storage, initialized since the loss and the penalty check it
  w = zeros(T, n)
  expandX = zeros(T, n, ep)
  S = BitArray(undef, ep)
  stdX = zeros(T, ep)
  m = CDWeightedLSLoss(y, expandX, w)        # reads expandX and w, which change in the loop
  β = SparseIterate(T, ep)

  ind = 0
  for z0 in zgrid
    ind += 1

    # the following two should update f
    w .= evaluate.(Ref(kernel), z, Ref(z0))
    _expand_X!(expandX, X, z, z0, degree)
    _stdX!(stdX, w, expandX)

    # solve for β
    _solve!(β, m, NormL1(λ0, stdX), opt)
    out[:, ind] = β

    if refit
        get_nonzero_coordinates!(S, β, p, degree, true)
        outR[S, ind] = _wls(view(expandX, :, S), w, y)
    end
  end
  out, outR
end

"""
    lvocv_locpolyl1(X, z, y, degree, hArr, KernelType, λ0, options=CDOptions())

Leave-one-out cross-validation error of the refitted [`locpolyl1`](@ref)
estimator for each bandwidth in `hArr`.

For each observation `i`, the model is fit at `z0 = zᵢ` with observation `i`
given zero weight. The lasso penalty is `λ0 σ̂`, where the noise level `σ̂` is
estimated jointly with the coefficients as in the scaled lasso. The selected
variables are refit by weighted least squares, taking the refit of minimum
norm when it is not unique, and used to predict `yᵢ`. As in `locpolyl1`, `λ0`
is on the scale of the kernel.

Returns the vector of summed squared prediction errors, one per bandwidth.
"""
function lvocv_locpolyl1(
    X::Matrix{T}, z::Vector{T}, y::Vector{T},
    degree::Int64,                                     # degree of the polynomial
    hArr::Vector{T},
    kernelType::Type{<:SmoothingKernel},
    λ0::T,
    options::CDOptions=CDOptions()) where {T <: AbstractFloat}

    # the weighted loss below checks its data only when it is built, and then
    # reads it in place
    _check_finite(X, "X")
    _check_finite(z, "z")
    _check_finite(y, "y")

    n, p = size(X)
    numH = length(hArr)
    MSE = zeros(T, numH)

    opt = _withWarmStart(options)

    ep = p * (degree + 1)
    w = zeros(T, n)
    wX = zeros(T, n, ep)
    stdX = zeros(T, ep)
    r0 = Vector{T}(undef, n)
    S = BitArray(undef, ep)

    m = CDWeightedLSLoss(y, wX, w)             # reads wX and w, which change in the loop
    solver = _solver(opt)
    β = SparseIterate(T, ep)
    σcur = Ref(zero(T))
    function update(g, β, m)
      σnew = _getSigma(w, residuals(m))
      change = abs(σnew - σcur[]) / σcur[]
      σcur[] = σnew
      NormL1(λ0 * σnew, stdX), change
    end

    for indH = 1:numH
        kernel = createKernel(kernelType, hArr[indH])
        for i = 1:n
            # update variables
            z0 = z[i]
            w .= evaluate.(Ref(kernel), z, Ref(z0))
            w[i] = zero(T)
            _expand_X!(wX, X, z, z0, degree)
            _stdX!(stdX, w, wX)
            _findInitResiduals!(w, wX, y, min(10, ep), r0)

            # alternate between the lasso fit and the noise level
            σcur[] = _getSigma(w, r0)
            reweight!(β, m, NormL1(λ0 * σcur[], stdX), update, solver; maxiter = 10, tol = 1e-2)

            # refit and make prediction
            get_nonzero_coordinates!(S, β, p, degree, true)
            Yh = dot(wX[i, S], _wls(view(wX, :, S), w, y))
            MSE[indH] += abs2(Yh - y[i])
        end
    end
    MSE
end

"""
    refit_locpolyl1(X, z, y, z0, degree, kernel, β)

Refit the local polynomial estimate at `z0` using only the variables selected
by `β`, the expanded coefficient vector returned by [`locpolyl1`](@ref) at
`z0`. A variable is selected when any of its `degree + 1` coefficients is
nonzero.

Returns `(βr, S)`, where `S` is a `BitVector` of length `p` marking the
selected variables and `βr` is the [`locpoly`](@ref) estimate on `X[:, S]`.
"""
function refit_locpolyl1(
    X::Matrix{T}, z::Vector{T}, y::Vector{T},
    z0::T,
    degree::Int64,
    kernel::SmoothingKernel{T},
    β::Union{SparseIterate{T}, SparseVector{T}}
    ) where {T <: AbstractFloat}

    p = size(X, 2)
    S = BitArray(undef, p)
    get_nonzero_coordinates!(S, β, p, degree, false)
    βr = locpoly(view(X, :, S), z, y, z0, degree, kernel)
    (βr, S)
end


############################################################
#
# local polynomial regression low dimensions
#
############################################################

# Solves the local polynomial least squares problem at z0 using preallocated
# storage wX (n × p(degree+1)) and w (n). Both buffers are overwritten.
function _locpoly!(
  wX::AbstractMatrix{T}, w::AbstractVector{T},
  X::AbstractMatrix{T},
  z::AbstractVector{T},
  y::AbstractVector{T},
  z0::T,
  degree::Int64,
  kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

  w .= sqrt.(evaluate.(Ref(kernel), z, Ref(z0)))    # square root of kernel weights
  _expand_wX!(wX, w, X, z, z0, degree)              # √w ⋅ x ⊗ [1 (zi - z0) ... (zi-z0)^q]
  @. w *= y                                         # √w ⋅ y
  qr!(wX) \ w
end

"""
    locpoly(X, z, y, z0::Real, degree, kernel=GaussianKernel(1))
    locpoly(X, z, y, zgrid::Vector, degree, kernel=GaussianKernel(1))

Local polynomial estimate of the varying-coefficient model
`yᵢ = ∑ⱼ Xᵢⱼ βⱼ(zᵢ) + εᵢ` at `z0`, or at every point of `zgrid`.

At `z0` it solves the weighted least squares problem

    minimize  ∑ᵢ K_h(zᵢ, z0) (yᵢ - X̃ᵢ'β)²

with `X̃ᵢ = Xᵢ ⊗ [1, (zᵢ - z0), ..., (zᵢ - z0)^degree]`. Returns the coefficient
vector of length `p(degree + 1)`, or a matrix with one such column per grid
point. Coefficient `(j-1)(degree+1) + 1` is the estimate of `βⱼ(z0)`.
"""
locpoly(
  X::AbstractMatrix{T}, z::Vector{T}, y::Vector{T},
  z0::T, degree::Int64, kernel::SmoothingKernel=GaussianKernel(one(T))) where {T <: AbstractFloat} =
    _locpoly!(Array{T}(undef, length(y), size(X, 2) * (degree+1)), similar(y), X, z, y, z0, degree, kernel)

function locpoly(
  X::AbstractMatrix{T}, z::Vector{T}, y::Vector{T},
  zgrid::Vector{T},
  degree::Int64,                                     # degree of the polynomial
  kernel::SmoothingKernel{T}=GaussianKernel(one(T))) where {T <: AbstractFloat}

  n, p = size(X)
  ep = p * (degree + 1)
  out = Array{T}(undef, ep, length(zgrid))
  w = Array{T}(undef, n)
  wX = Array{T}(undef, n, ep)

  ind = 0
  for z0 in zgrid
    ind += 1
    out[:, ind] = _locpoly!(wX, w, X, z, y, z0, degree, kernel)
  end
  out
end

"""
    getResiduals!(ϵhat, X, z, y, zgrid, βhat, degree)

Residuals `ϵhatᵢ = yᵢ - ∑ⱼ Xᵢⱼ β̂ⱼ(zᵢ)` of a varying-coefficient fit, where
`βhat` holds the expanded coefficients at the points of `zgrid` (as returned
by [`locpoly`](@ref) or [`locpolyl1`](@ref)) and `β̂(zᵢ)` is obtained by
linear interpolation with [`get_beta!`](@ref).
"""
function getResiduals!(
     ϵhat::Vector{T},
     X::AbstractMatrix{T}, z::Vector{T}, y::Vector{T},
     zgrid::Vector{T},
     βhat::Union{Matrix{T}, SparseMatrixCSC{T}},
     degree::Int64
    ) where {T <: AbstractFloat}

    n, p = size(X)
    ep = p * (degree + 1)
    βi = zeros(T, ep)
    for i=1:n
        get_beta!(βi, zgrid, βhat, z[i])
        # only the zeroth order coefficients enter the fitted value at zᵢ
        ϵhat[i] = y[i] - dot(view(X, i, :), view(βi, 1:(degree+1):ep))
    end

    ϵhat
end

"""
    getStandardError(X, z, σ2, z0, degree, kernel)

Variances of the local polynomial estimates `β̂ⱼ(z0)`, `j = 1, ..., p`, under
homoscedastic noise with variance `σ2`. With `W = diag(K_h(zᵢ, z0))` and the
expanded design `X̃`, the covariance of the estimator is

    σ2 (X̃'WX̃)⁻¹ X̃'W²X̃ (X̃'WX̃)⁻¹

and the function returns the diagonal entries that correspond to `β̂ⱼ(z0)`.
Take square roots to obtain standard errors.
"""
getStandardError(
    X::AbstractMatrix{T}, z::AbstractVector{T},
    σ2::T,
    z0::T,
    degree::Int64,                                     # degree of the polynomial
    kernel::SmoothingKernel{T}) where {T <: AbstractFloat} =
  getStandardErrorHEW(X, z, fill(σ2, length(z)), z0, degree, kernel)

"""
    getStandardErrorHEW(X, z, ϵ_sqr, z0, degree, kernel)

Heteroscedasticity-robust (Eicker–Huber–White) version of
[`getStandardError`](@ref): the variances of `β̂ⱼ(z0)` computed from

    (X̃'WX̃)⁻¹ X̃'W Ψ W X̃ (X̃'WX̃)⁻¹,   Ψ = diag(ϵ_sqr),

where `ϵ_sqr` holds squared residuals, for example from
[`getResiduals!`](@ref).
"""
function getStandardErrorHEW(
    X::AbstractMatrix{T}, z::AbstractVector{T},
    ϵ_sqr::AbstractVector{T},
    z0::T,
    degree::Int64,                                     # degree of the polynomial
    kernel::SmoothingKernel{T}) where {T <: AbstractFloat}

    n, p = size(X)
    ep = p * (degree + 1)
    out = Array{T}(undef, p)
    w = Array{T}(undef, n)
    XtwX = Array{T}(undef, ep, ep)
    XtwΨwX = Array{T}(undef, ep, ep)

    w .= evaluate.(Ref(kernel), z, Ref(z0))
    _expand_Xt_w_X!(XtwX, w, X, z, z0, degree)

    @. w = w * w * ϵ_sqr
    _expand_Xt_w_X!(XtwΨwX, w, X, z, z0, degree)

    A = inv(XtwX)
    varMat = A * XtwΨwX * A

    for j=1:p
        out[j] = varMat[(j-1)*(degree+1)+1, (j-1)*(degree+1)+1]
    end

    out
end

"""
    lvocv_locpoly(X, z, y, degree, hArr, KernelType)

Leave-one-out cross-validation error of [`locpoly`](@ref) for each bandwidth in
`hArr`: observation `i` is predicted from the fit at `z0 = zᵢ` computed without
it. Returns the vector of summed squared prediction errors, one per bandwidth.
"""
function lvocv_locpoly(
    X::Matrix{T}, z::Vector{T}, y::Vector{T},
    degree::Int64,                                     # degree of the polynomial
    hArr::Vector{T},
    kernelType::Type{<:SmoothingKernel}) where {T <: AbstractFloat}

    n, p = size(X)
    numH = length(hArr)
    MSE = zeros(T, numH)

    ep = p * (degree + 1)
    w = Array{T}(undef, n-1)
    wX = Array{T}(undef, n-1, ep)
    indOut = BitArray(undef, n)

    for indH = 1:numH
        kernel = createKernel(kernelType, hArr[indH])
        for i = 1:n
            fill!(indOut, true)
            indOut[i] = false
            Xview = view(X, indOut, :)
            Yview = view(y, indOut)
            Zview = view(z, indOut)

            hbeta = _locpoly!(wX, w, Xview, Zview, Yview, z[i], degree, kernel)

            # make prediction
            Yh = dot(X[i, :], hbeta[1:(degree+1):ep])
            MSE[indH] += abs2(Yh - y[i])
        end
    end
    MSE
end

"""
    split_locpoly(X, z, y, Xtest, ztest, ytest, zgrid, degree, hArr, KernelType)

Validation error of [`locpoly`](@ref) for each bandwidth in `hArr`. The model
is fit on `(X, z, y)` at the points of `zgrid`, interpolated to `ztest` with
[`get_beta!`](@ref), and used to predict `ytest`. Returns the vector of summed
squared prediction errors, one per bandwidth.
"""
function split_locpoly(
    X::Matrix{T}, z::Vector{T}, y::Vector{T},
    Xtest::Matrix{T}, ztest::Vector{T}, ytest::Vector{T},
    zgrid::Vector{T},
    degree::Int64,                                     # degree of the polynomial
    hArr::Vector{T},
    kernelType::Type{<:SmoothingKernel}) where {T <: AbstractFloat}

    p = size(X, 2)
    numH = length(hArr)
    MSE = zeros(T, numH)
    ep = p * (degree + 1)
    βhati = zeros(T, ep)

    for indH = 1:numH
        kernel = createKernel(kernelType, hArr[indH])
        βhat = locpoly(X, z, y, zgrid, degree, kernel)
        for i = 1:length(ytest)
            get_beta!(βhati, zgrid, βhat, ztest[i])

            # make prediction
            Yh = dot(Xtest[i, :], βhati[1:(degree+1):ep])
            MSE[indH] += abs2(ytest[i] - Yh)
        end
    end
    MSE
end


############################################################
#
# utils
#
############################################################

"""
    get_beta!(out, zgrid, beta_grid, z0)

Coefficients at `z0` obtained by linear interpolation between the columns of
`beta_grid` at the two points of the sorted grid `zgrid` that bracket `z0`.
Outside the grid the nearest column is used.
"""
function get_beta!(
       out::Union{SparseVector{T}, Vector{T}},
       zgrid::Vector{T},
       beta_grid::Union{Matrix{T}, SparseMatrixCSC{T}},
       z0::T) where {T <: AbstractFloat}
    # zgrid[id1] ≤ z0 ≤ zgrid[id2]; id1 == id2 when z0 is a grid point
    id1 = searchsortedlast(zgrid, z0)
    id2 = searchsortedfirst(zgrid, z0)

    if id1 == 0                               # z0 < zgrid[1]
        out .= view(beta_grid, :, 1)
    elseif id2 > length(zgrid)                # z0 > zgrid[end]
        out .= view(beta_grid, :, length(zgrid))
    elseif id1 == id2
        out .= view(beta_grid, :, id1)
    else
        z1 = zgrid[id1]
        z2 = zgrid[id2]
        α = (z0 - z1) / (z2 - z1)
        out .= (1-α) .* view(beta_grid, :, id1) .+ α .* view(beta_grid, :, id2)
    end
    out
end

"""
    get_nonzero_coordinates(β, p, degree, expanded)

Variables with at least one nonzero coefficient in the expanded coefficient
vector `β` of length `p(degree + 1)`. If `expanded` is true, returns a
`BitVector` of length `p(degree + 1)` that marks all `degree + 1` coefficients
of each selected variable; otherwise a `BitVector` of length `p`.
"""
get_nonzero_coordinates(
    β::Union{SparseVector{T}, SparseIterate{T}},
    p::Int,
    degree::Int,
    expanded::Bool
    ) where {T <: AbstractFloat}  =
       expanded ? get_nonzero_coordinates!(BitArray(undef, p*(degree+1)), β, p, degree, expanded) : get_nonzero_coordinates!(BitArray(undef, p), β, p, degree, expanded)


function get_nonzero_coordinates!(
    S::BitArray,
    β::Union{SparseVector{T}, SparseIterate{T}},
    p::Int,
    degree::Int,
    expanded::Bool) where {T <: AbstractFloat}

    fill!(S, false)
    for j = 1:p
        nonzero = false
        for k=((j-1)*(degree+1)+1):(j*(degree+1))
            nonzero = nonzero | !iszero(β[k])
        end
        if nonzero
            if expanded
                for k=((j-1)*(degree+1)+1):(j*(degree+1))
                    S[k] = true
                end
            else
                S[j] = true
            end
        end
    end
    S
end


"""
Computes matrix whose each row is equal to
w[i] ⋅ (X[i, :] ⊗ [1, (z - z0), ..., (z-z0)^q])
where q is the degree of the polynomial.

The output matrix is preallocated.
"""
function _expand_wX!(
  wX::AbstractMatrix{T},
  w::AbstractVector{T},
  X::AbstractMatrix{T},
  z::AbstractVector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  for j=1:p
    @inbounds for i=1:n
      v = X[i, j] * w[i]
      df = z[i] - z0
      col = (j-1)*(degree+1) + 1
      @inbounds wX[i, col] = v
      for l=1:degree
        v *= df
        @inbounds wX[i, col + l] = v
      end
    end
  end
  wX
end


# expands the data matrix X
# each row becomes
# (X[i, :] ⊗ [1, (z - z0), ..., (z-z0)^q])
function _expand_X!(
  tX::Matrix{T},
  X::AbstractMatrix{T},
  z::AbstractVector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  for j=1:p
    for i=1:n
      v = X[i, j]
      df = z[i] - z0
      col = (j-1)*(degree+1) + 1
      @inbounds tX[i, col] = v
      for l=1:degree
        v *= df
        @inbounds tX[i, col + l] = v
      end
    end
  end
  tX
end


# computes X̃' diag(w) X̃ for the expanded design X̃ without forming X̃;
# only the lower triangle is accumulated and then copied to the upper one
function _expand_Xt_w_X!(
  Xt_w_X::Matrix{T},
  w::AbstractVector{T}, X::AbstractMatrix{T},
  z::AbstractVector{T}, z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  fill!(Xt_w_X, zero(T))
  @inbounds for j=1:p
    @inbounds for k=j:p
      @inbounds for i=1:n
        v1 = X[i, j] * w[i]
        df1 = z[i] - z0

        col=(j-1)*(degree+1)+1
        for jj=0:degree
          v2 = X[i, k]
          df2 = z[i] - z0
          if k != j
            krange = 0:degree
          else
            krange = jj:degree
            v2 = v2 * df2^jj
          end

          row = (k-1)*(degree+1)+1
          for kk=krange
            # need X[i, j] * (z[i] - z0)^(jj+kk) * X[i, k] * w[i]
            # v2 = (z[i] - z0)^kk * X[i, k]
            # v1 = (z[i] - z0)^jj * X[i, k] * w[i]
            Xt_w_X[row+kk, col+jj] += v2 * v1
            v2 *= df2
          end
          v1 *= df1
        end
      end
    end
  end

  @inbounds for c=1:size(Xt_w_X, 2)
    for r=c+1:size(Xt_w_X, 1)
      Xt_w_X[c, r] = Xt_w_X[r, c]
    end
  end

  Xt_w_X
end

# computes X̃' diag(w) y for the expanded design X̃ without forming X̃
function _expand_Xt_w_Y!(
  Xt_w_Y::Vector{T},
  w::AbstractVector{T}, X::AbstractMatrix{T}, z::AbstractVector{T}, y::AbstractVector{T},
  z0::T, degree::Int64) where {T <: AbstractFloat}

  n, p = size(X)
  fill!(Xt_w_Y, zero(T))
  @inbounds for j=1:p
    @inbounds for i=1:n
      v = X[i, j] * w[i] * y[i]
      df = z[i] - z0

      col=(j-1)*(degree+1)+1
      @inbounds for jj=0:degree
        # need X[i, j] * (z[i] - z0)^jj * Y[i, k] * w[i]
        Xt_w_Y[col+jj] += v
        v *= df
      end
    end
  end

  Xt_w_Y
end
