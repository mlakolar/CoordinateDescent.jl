
"""
Helper function that finds an initial estimate for σ that is needed
for Lasso and ScaledLasso procedures.

The procedure works as follows:

* s input variables that are most correlated with the response y are found
* y is regressed on those s features
* σ is estimated based on the residuals, which gives an upper bound on the true sigma
"""
function findInitSigma(
  X::AbstractMatrix{T},
  y::AbstractVector{T},
  s::Int) where {T <: AbstractFloat}

  n, p = size(X)
  if s > p
    s = p
  end
  c = (X'*y) / n
  @. c = abs(c)

  # find value of s-th largest element in abs(c)
  h = binary_maxheap(T)
  for i=1:p
      push!(h, c[i])
  end

  val = zero(T)
  for i=1:s
    val = pop!(h)
  end

  # solve OLS with s most correlated values
  S = c .> val
  βh = X[:, S] \ y
  std(y - X[:,S]*βh)
end
