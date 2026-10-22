# Runs every numbered example in this folder, each in a module of its own,
# and stops at the first error:
#
#     julia --project=examples examples/run_all.jl

for file in sort(filter(f -> occursin(r"^\d\d_.*\.jl$", f), readdir(@__DIR__)))
  println("\n", "#"^80, "\n# ", file, "\n", "#"^80, "\n")
  t = @elapsed Base.include(Module(), joinpath(@__DIR__, file))
  println("\n# $file took $(round(t; digits = 1)) s")
end
