module HighDimensionalModelsProximalAlgorithmsExt

using HighDimensionalModels
using LinearAlgebra
import ProximalBase
import ProximalAlgorithms

const CD = HighDimensionalModels
const PB = ProximalBase
const PA = ProximalAlgorithms

struct RegressionLoss{D,G,T}
  data::D
  gradient::G
  energy::T
end
RegressionLoss(d, g) = RegressionLoss(d, g, sum(abs2, d.Y)/(2size(d.X, 1)))
(f::RegressionLoss)(B) = sum(abs2, f.data.Y - f.data.X * B)/(2size(f.data.X, 1))
function PA.value_and_gradient(f::RegressionLoss, B)
  # Preserve the original low-precision arithmetic. The Float64 shortcuts
  # below are checked against the original-design loss near cancellation.
  eltype(B) === Float64 || return (f(B), CD._lr_grad(f.data, f.gradient, B))
  d, g = f.data, f.gradient
  n = size(d.X, 1)
  if g.A === nothing
    residual = d.X * B - d.Y
    return (sum(abs2, residual)/(2n), transpose(d.X) * residual/n)
  end
  G = CD._lr_grad(d, g, B)
  quadratic, linear = dot(B, G), dot(B, g.C)
  value = f.energy + (quadratic - linear)/2
  scale = f.energy + (abs(quadratic) + abs(linear))/2
  # A small residual can be lost by subtracting the quadratic terms. Fall
  # back to the original residual evaluation instead of clamping the loss.
  if !isfinite(value) || value <= sqrt(eps(Float64)) * scale
    value = f(B)
  end
  value, G
end

# ProximalCore returns the value at the proximal point, whereas ProximalBase's
# prox! returns that point. Keep this bridge on our own type and reuse the
# nuclear workspace and proximal value rather than decomposing twice.
struct NuclearPenalty{P}
  penalty::P
end
(g::NuclearPenalty)(B) = g.penalty(B)
PA.prox!(Y, g::NuclearPenalty, X, γ) = PB.value_and_prox!(Y, g.penalty, X, γ)

function nuclear_panoc(d, g, λ, start; max_iter, tol)
  T = eltype(d.X)
  penalty = PB.NuclearNorm(T(λ))
  if iszero(g.L)
    return CD._lr_solution(d, zero(start), λ, 0, true,
      sum(abs2, d.Y)/(2size(d.X, 1)), :nuclear)
  end
  loss = RegressionLoss(d, g)
  scale = max(one(T), norm(g.C))
  step = similar(start)
  count = Ref(0)
  function certified(B)
    G = CD._lr_grad(d, g, B)
    @. step = B - G/g.L
    PB.prox!(step, penalty, step, inv(g.L))
    g.L * norm(step - B) <= tol * scale
  end
  function stop(iter, state)
    count[] += 1
    (count[] % 10 == 0 || count[] == max_iter) && certified(PA.default_solution(iter, state))
  end
  B, iterations = PA.PANOC(; maxit = Int(max_iter), stop)(;
    x0 = copy(start), f = loss, g = NuclearPenalty(penalty), Lf = g.L, adaptive = false)
  converged = certified(B)
  # The final loss, coefficient rank and nuclear value use these same
  # returned coefficients. Reuse their residual and singular values.
  residual = d.Y - d.X * B
  singular = svdvals(B)
  cutoff = min(size(B)...) * eps(T) * first(singular)
  coefficient_rank = Base.count(>(cutoff), singular)
  objective = sum(abs2, residual)/(2size(d.X, 1)) + T(λ)*sum(singular)
  CD._lr_solution(d, B, λ, iterations, converged, objective, :nuclear;
    coefficient_rank, residual)
end

end
