module HighDimensionalModelsHiGHSExt

using HighDimensionalModels, HiGHS, SparseArrays
const CD = HighDimensionalModels
const MOI = HiGHS.MOI

function _quantile_model(X, y, λ, w, τ, intercept, initial_basis, simplex, optimizer_factory)
  n, p = size(X)
  # The private factory permits deterministic native termination tests.
  opt = optimizer_factory()
  MOI.set(opt, MOI.Silent(), true)
  for (name, val) in (("threads",1), ("parallel","off"), ("solver","simplex"),
                      ("simplex_strategy",simplex == :dual ? 1 : 4),
                      ("primal_feasibility_tolerance",1e-8), ("dual_feasibility_tolerance",1e-8))
    MOI.set(opt, MOI.RawOptimizerAttribute(name), val)
  end
  u = MOI.add_variables(opt, n)
  bounds = MOI.add_constraints(opt, u, fill(MOI.Interval(τ-1, τ), n))
  eq = if intercept
    MOI.add_constraint(opt, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(n), u), 0.0), MOI.EqualTo(0.0))
  else
    nothing
  end
  constraints = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.Interval{Float64}}[]
  for j in 1:p
    terms = if X isa SparseMatrixCSC
      [MOI.ScalarAffineTerm(nonzeros(X)[i],u[rowvals(X)[i]]) for i in nzrange(X,j)]
    else
      [MOI.ScalarAffineTerm(X[i,j],u[i]) for i in 1:n if X[i,j] != 0]
    end
    push!(constraints, MOI.add_constraint(opt, MOI.ScalarAffineFunction(terms,0.0), MOI.Interval(-n*λ*w[j], n*λ*w[j])))
  end
  MOI.set(opt, MOI.ObjectiveSense(), MOI.MAX_SENSE)
  MOI.set(opt, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
          MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(y./n,u),0.0))
  if initial_basis == :quantile
    columns = fill(HiGHS.kHighsBasisStatusUpper,n)
    rows = fill(HiGHS.kHighsBasisStatusBasic,p+intercept)
    if intercept
      order=sortperm(y);k=clamp(ceil(Int,n*τ),1,n)
      columns[order[1:k-1]] .= HiGHS.kHighsBasisStatusLower
      columns[order[k]] = HiGHS.kHighsBasisStatusBasic
      rows[1] = HiGHS.kHighsBasisStatusLower
    else
      columns[y .< 0] .= HiGHS.kHighsBasisStatusLower
    end
    ret = HiGHS.Highs_setBasis(opt,columns,rows)
    ret == HiGHS.kHighsStatusOk || error("HiGHS rejected the quantile starting basis")
  end
  opt, u, bounds, eq, constraints
end

function CD._quantile_exact_path(X::AbstractMatrix{Float64}, y::Vector{Float64},
                                  ls::Vector{Float64}, w::Vector{Float64}, τ::Float64, intercept::Bool;
                                  tol, time_limit, initial_basis, simplex, start,
                                  max_hat_s=Inf, optimizer_factory=HiGHS.Optimizer)
  n, p = size(X)
  opt, u, _, eq, constraints = _quantile_model(X,y,ls[1],w,τ,intercept,initial_basis,simplex,optimizer_factory)
  solutions = CD.QuantileLassoSolution[]; status = :optimal
  for (k,λ) in enumerate(ls)
    remaining = time_limit-(time_ns()-start)/1e9
    remaining <= 0 && (status=:TIME_LIMIT;break)
    for j in 1:p
      MOI.set(opt,MOI.ConstraintSet(),constraints[j],MOI.Interval(-n*λ*w[j],n*λ*w[j]))
    end
    isfinite(remaining) && MOI.set(opt,MOI.TimeLimitSec(),remaining)
    MOI.optimize!(opt)
    term = MOI.get(opt,MOI.TerminationStatus());status=Symbol(string(term))
    if MOI.get(opt,MOI.ResultCount()) > 0 &&
       MOI.get(opt,MOI.PrimalStatus()) != MOI.NO_SOLUTION && MOI.get(opt,MOI.DualStatus()) != MOI.NO_SOLUTION
      scores = MOI.get(opt,MOI.VariablePrimal(),u)
      β = Float64[-n*MOI.get(opt,MOI.ConstraintDual(),c) for c in constraints]
      a = intercept ? -n*MOI.get(opt,MOI.ConstraintDual(),eq) : 0.0
      push!(solutions, CD._qsolution(X,y,β,a,scores,λ,w,τ,intercept,:exact,0.0,
                                     status,MOI.get(opt,MOI.SimplexIterations()),term==MOI.OPTIMAL,tol))
    else
      break
    end
    term == MOI.OPTIMAL || break
    k < length(ls) && count(!iszero,solutions[end].β) > max_hat_s && (status=:max_hat_s;break)
  end
  solutions, status
end

function CD._quantile_exact_refit(X::AbstractMatrix{Float64}, y::Vector{Float64},
                                   taus::Vector{Float64}, intercept::Bool;
                                   tol, time_limit, initial_basis, simplex, start,
                                   optimizer_factory=HiGHS.Optimizer)
  n, p = size(X); w = ones(p)
  opt, u, bounds, eq, constraints = _quantile_model(X,y,0.0,w,taus[1],intercept,initial_basis,simplex,optimizer_factory)
  solutions = CD.QuantileLassoSolution[]
  for τ in taus
    # Bound updates retain the current simplex basis, including after a limit.
    for c in bounds
      MOI.set(opt,MOI.ConstraintSet(),c,MOI.Interval(τ-1,τ))
    end
    remaining = time_limit-(time_ns()-start)/1e9
    remaining > 0 || error("quantile refit returned no primal–dual pair (TIME_LIMIT)")
    isfinite(remaining) && MOI.set(opt,MOI.TimeLimitSec(),remaining)
    MOI.optimize!(opt)
    term = MOI.get(opt,MOI.TerminationStatus()); status = Symbol(string(term))
    MOI.get(opt,MOI.ResultCount()) > 0 &&
      MOI.get(opt,MOI.PrimalStatus()) != MOI.NO_SOLUTION && MOI.get(opt,MOI.DualStatus()) != MOI.NO_SOLUTION ||
      error("quantile refit returned no primal–dual pair ($status)")
    scores = Float64[MOI.get(opt,MOI.VariablePrimal(),v) for v in u]
    β = Float64[-n*MOI.get(opt,MOI.ConstraintDual(),c) for c in constraints]
    a = intercept ? -n*MOI.get(opt,MOI.ConstraintDual(),eq) : 0.0
    push!(solutions, CD._qsolution(X,y,β,a,scores,0.0,w,τ,intercept,:exact,0.0,
                                   status,MOI.get(opt,MOI.SimplexIterations()),term==MOI.OPTIMAL,tol))
  end
  solutions
end
end
