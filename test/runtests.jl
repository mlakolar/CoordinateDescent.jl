using FactCheck

using CoordinateDescent
using ProximalBase

using CoordinateDescent: _expand_wX!, _expand_X!, _expand_Xt_w_X!,
                         _expand_Xt_w_Y!, _locpoly!, _stdX!

srand(1)
const NUMBER_REPEAT = 20

include(joinpath(@__DIR__, "..", "benchmark", "locpoly_bench.jl"))

tests = [
  "atom_iterator",
	"lasso",
  "coordinate_descent",
  "varying_coefficient_lasso"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end



FactCheck.exitstatus()
