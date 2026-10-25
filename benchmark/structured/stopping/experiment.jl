# Experimental overlay for comparisons only; the package source is unchanged.
@eval CoordinateDescent begin
const _stopping_calibration_mode = Ref(:baseline)

# For min 1/2||beta-b||^2 + alpha TV(K beta), a clipped cumulative
# representation gives a feasible box dual q. Evaluate the gap as a sum of
# nonnegative stationarity and Fenchel terms to avoid subtracting objectives.
function _stopping_signal_gap!(stationarity, w, K, beta, b, kb, u, rho, alpha)
    T = eltype(beta)
    cumulative = zero(T); previous = zero(T)
    slack = zero(T); penalty = zero(T)
    @inbounds for j in 1:length(w)-1
        cumulative += rho*u[j]
        q = clamp(-cumulative, -alpha, alpha)
        w[j] = previous-q
        delta = kb[j+1]-kb[j]
        penalty += alpha*abs(delta)
        slack += alpha*abs(delta)-q*delta
        previous = q
    end
    w[end] = previous
    mul!(stationarity, transpose(K), w)
    residual_square = zero(T); stationarity_square = zero(T)
    @inbounds for j in eachindex(beta)
        residual = beta[j]-b[j]
        residual_square += abs2(residual)
        stationarity_square += abs2(residual+stationarity[j])
    end
    stationarity_square/2 + slack, residual_square/2 + penalty
end

function _split_fit(A, b, K, g, ρ; max_iter, abstol, reltol, start = nothing,
                    gram = _structured_gram(A), xy = _structured_tproduct(A, b), factor = nothing,
                    bandwidth = nothing)
  T, p, m = eltype(b), size(A, 2), size(K, 1)
  F = factor === nothing ? _split_factor(gram, K, ρ; bandwidth) : factor
  β = start === nothing ? zeros(T, p) : copy(start.β)
  z = start === nothing ? K * β : copy(start.z)
  u = start === nothing ? zeros(T, m) : (start.ρ / ρ) .* start.u
  kb, v, zold = zeros(T, m), zeros(T, m), zeros(T, m)
  rhs, dual, ktu = zeros(T, p), zeros(T, p), zeros(T, p)
  ok, it, pr, dr = false, 0, T(Inf), T(Inf)
  for iter in 1:max_iter
    @. v = z - u
    mul!(rhs, transpose(K), v)
    @. rhs = xy + ρ * rhs
    _structured_solve!(β, F, rhs)
    mul!(kb, K, β)
    copyto!(zold, z)
    @. v = kb + u
    prox!(z, g, v, inv(ρ))
    @. u = u + kb - z
    @. v = kb - z
    pr = norm(v)
    @. v = z - zold
    mul!(dual, transpose(K), v)
    dr = ρ * norm(dual)
    mul!(ktu, transpose(K), u)
    it = iter
    ep = sqrt(m) * abstol + reltol * max(norm(kb), norm(z))
    ed = sqrt(p) * abstol + reltol * ρ * norm(ktu)
    mode = A isa _SignalDesign && F isa _StructuredBandFactor{Float64,1} ?
      _stopping_calibration_mode[] : :baseline
    if mode in (:gap1e6, :gap3e7, :gap1e7)
      if pr <= 5ep && dr <= 5ed && (iter == 1 || iter % 10 == 0 || iter == max_iter)
        gap, objective = _stopping_signal_gap!(dual, v, K, β, b, kb, u, ρ, g.λ)
        tolerance = mode == :gap1e6 ? 1e-6 : mode == :gap3e7 ? 3e-7 : 1e-7
        energy = sum(abs2,b)
        nearzero = objective <= 100eps(T)*energy && pr <= ep && dr <= ed
        if isfinite(gap) && isfinite(objective) && isfinite(energy) &&
           (gap <= tolerance * objective || nearzero)
          ok = true
          break
        end
      end
    else
      scale = mode == :residual2 ? 2 : mode == :residual5 ? 5 : 1
      if pr <= scale*ep && dr <= scale*ed
        ok = true
        break
      end
    end
  end
  β, it, ok, pr, dr, (; β, z, u, ρ)
end
end
