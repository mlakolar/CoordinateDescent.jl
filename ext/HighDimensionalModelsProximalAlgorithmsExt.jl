module HighDimensionalModelsProximalAlgorithmsExt

using HighDimensionalModels
using LinearAlgebra
import ProximalBase
import ProximalAlgorithms

const CD = HighDimensionalModels
const PB = ProximalBase
const PA = ProximalAlgorithms

struct RegressionLoss{D,G}
  data::D
  gradient::G
end
(f::RegressionLoss)(B) = sum(abs2, f.data.Y - f.data.X * B)/(2size(f.data.X, 1))
PA.value_and_gradient(f::RegressionLoss, B) = (f(B), CD._lr_grad(f.data, f.gradient, B))

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
  objective = loss(B) + penalty(B)
  CD._lr_solution(d, B, λ, iterations, converged, objective, :nuclear)
end

end
