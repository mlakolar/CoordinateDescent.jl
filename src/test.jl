using Convex
using JuMP
using Mosek
using HD
reload("HD")

solver = MosekSolver(LOG=0)
typeof(solver)
model = JuMP.Model(solver=solver)
typeof(model)
@JuMP.defVar(model, a[1:10])
typeof()

type QR
  m::JuMP.Model
  beta
  t1
  t2
  u
  v

  function QR(solver, X, Y)

  end
end

