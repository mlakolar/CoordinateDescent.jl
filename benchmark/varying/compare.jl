# compares the result files of bench.jl: julia benchmark/varying/compare.jl OLD NEW
using Serialization, Printf
old, new = deserialize(ARGS[1]), deserialize(ARGS[2])
for k in sort(collect(keys(new)))
  tn = new[k][1]
  if haskey(old, k)
    to = old[k][1]
    agree = new[k][2] isa AbstractArray ? @sprintf("max |Δ| %.1e", maximum(abs, new[k][2] .- old[k][2])) :
            new[k][2] === nothing ? "" : "selected $(old[k][2]) old, $(new[k][2]) new"
    @printf "%-16s old %.4f s  new %.4f s  ratio %.2f  %s\n" k to tn tn / to agree
  else
    @printf "%-16s             new %.4f s\n" k tn
  end
end
