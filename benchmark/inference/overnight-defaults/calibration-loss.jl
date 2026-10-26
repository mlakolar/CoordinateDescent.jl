# Experimental solver building block only. No inference/default promotion.
# Native RCAL objective: mean(W .* (y .* exp.(-eta) + (1 .- y) .* eta)) + lambda*norm(beta,1),
# W normalized to mean one; eta includes offset and an unpenalized intercept.
struct CalibrationLoss <: ProximalBase.ScalarLoss end
function ProximalBase.value_and_deriv(::CalibrationLoss, eta::Real, y::Real)
    # Branch on binary y to avoid 0*Inf on large negative eta for controls.
    y==0 && return (float(eta), one(float(eta)))
    y==1 || throw(ArgumentError("CalibrationLoss needs binary responses"))
    e=exp(-float(eta)); (e,-e)
end
(l::CalibrationLoss)(eta::Real,y::Real)=ProximalBase.value_and_deriv(l,eta,y)[1]
ProximalBase.deriv(l::CalibrationLoss,eta::Real,y::Real)=ProximalBase.value_and_deriv(l,eta,y)[2]
ProximalBase.deriv2(::CalibrationLoss,eta::Real,y::Real)=y==0 ? zero(float(eta)) : exp(-float(eta))
ProximalBase.curvature(::CalibrationLoss)=Inf
