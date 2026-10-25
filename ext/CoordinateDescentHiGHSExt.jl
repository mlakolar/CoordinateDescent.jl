module CoordinateDescentHiGHSExt

using CoordinateDescent, HiGHS, SparseArrays
const CD = CoordinateDescent
const MOI = HiGHS.MOI

function CD._quantile_exact_path(X::AbstractMatrix{Float64}, y::Vector{Float64},
                                  ls::Vector{Float64}, w::Vector{Float64}, τ::Float64, intercept::Bool;
                                  tol, time_limit, initial_basis, simplex, start)
  n, p = size(X)
  opt = HiGHS.Optimizer()
  MOI.set(opt, MOI.Silent(), true)
  for (name, val) in (("threads",1), ("parallel","off"), ("solver","simplex"),
                      ("simplex_strategy",simplex == :dual ? 1 : 4),
                      ("primal_feasibility_tolerance",1e-8), ("dual_feasibility_tolerance",1e-8))
    MOI.set(opt, MOI.RawOptimizerAttribute(name), val)
  end
  u = MOI.add_variables(opt, n)
  MOI.add_constraints(opt, u, fill(MOI.Interval(τ-1, τ), n))
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
    push!(constraints, MOI.add_constraint(opt, MOI.ScalarAffineFunction(terms,0.0), MOI.Interval(-n*ls[1]*w[j], n*ls[1]*w[j])))
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
  solutions = CD.QuantileLassoSolution[]; status = :optimal
  for λ in ls
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
      β = -n .* MOI.get(opt,MOI.ConstraintDual(),constraints)
      a = intercept ? -n*MOI.get(opt,MOI.ConstraintDual(),eq) : 0.0
      push!(solutions, CD._qsolution(X,y,β,a,scores,λ,w,τ,intercept,:exact,0.0,
                                     status,MOI.get(opt,MOI.SimplexIterations()),term==MOI.OPTIMAL,tol))
    else
      break
    end
    term == MOI.OPTIMAL || break
  end
  solutions, status
end
end
