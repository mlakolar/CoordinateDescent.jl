using FactCheck

using CoordinateDescent
using ProximalBase

function try_import(name::Symbol)
    try
        @eval import $name
        return true
    catch e
        return false
    end
end

grb = try_import(:Gurobi)
cvx = try_import(:Convex)
scs = try_import(:SCS)

if grb
  Convex.set_default_solver(Gurobi.GurobiSolver(OutputFlag=0))
else
  Convex.set_default_solver(SCS.SCSSolver(eps=1e-6, verbose=0))
end

srand(1)
const NUMBER_REPEAT = 100

tests = [
  "atom_iterator",
	"lasso",
  "coordinate_descent",
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end



FactCheck.exitstatus()
