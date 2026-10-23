# Compares the runs of run.jl on the two sides: for every case and setting, the
# fastest median time over the runs of each side (load on the machine only
# adds time), the passes, and the largest difference of the solutions.
#
#     julia benchmark/compare/compare.jl

using Printf, Serialization

dir = joinpath(@__DIR__, "results")
load(side) = [deserialize(joinpath(dir, f)) for f in readdir(dir) if startswith(f, side * "_")]
olds, news = load("old"), load("new")
fastest(runs, key) = minimum(r[key].time for r in runs)

println("| case | 0.6 (ms) | 0.6, momentum = false (ms) | 0.7 (ms) | 0.7 / 0.6 | 0.7 / 0.6 without momentum | passes, 0.6 / 0.7 | largest difference |")
println("|---|---:|---:|---:|---:|---:|---|---:|")
cases = sort!(unique(first.(collect(keys(first(news))))); by = c -> (c == "locpolyl1", c))
for c in cases
  t06, t06m, t07 = fastest(olds, (c, "0.6")), fastest(olds, (c, "0.6, momentum = false")), fastest(news, (c, "0.7"))
  o, nw = first(olds)[(c, "0.6")], first(news)[(c, "0.7")]
  d = length(o.solution) == length(nw.solution) ? maximum(abs, o.solution - nw.solution; init = 0.0) : NaN
  @printf("| %s | %.1f | %.1f | %.1f | %.2f | %.2f | %d / %d | %.0e |\n", c, 1e3t06, 1e3t06m, 1e3t07, t07 / t06, t07 / t06m,
          o.passes, nw.passes, d)
end
