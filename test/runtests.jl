using CoordinateDescent

include(joinpath(@__DIR__, "..", "benchmark", "locpoly_bench.jl"))

tests = [
  "atom_iterator",
  "coordinate_descent",
  "lasso",
  "varying_coefficient_lasso"
]

for t in tests
	f = "$t.jl"
	println("* running $f ...")
	include(f)
end
