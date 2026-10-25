ENV["CD81_IMPROVEMENTS_LIBRARY"]="1"
include(joinpath(@__DIR__,"updated_worker.jl"))
for case in ("medium","wide"),rep in 1:2
  d=problem("inputs/$case-$rep")
  for s in ("cd_rank","cd_nuclear","cd_panoc","cd_sparse_api","cd_sparse_fista","cd_rankpath")
    for frac in (updated_family(s)=="rank" ? (0.0,) : (.5,.1,.02))
      B,it,ok,detail=execute(s,d,frac)
      factors=s in ("cd_sparse_api","cd_sparse_fista") ?
        api_factors(d,frac,B;block_solver=s=="cd_sparse_fista" ? :fista : :cd) : nothing
      obj,gap,res=certificate(d,updated_family(s),B,frac;factors)
      @assert ok && res<=1e-6 && (isnan(gap)||gap<=1e-6) (s,case,rep,frac,ok,gap,res)
    end
  end
end
println("Public API adapters and independent certificates passed on two medium/two wide inputs.")
