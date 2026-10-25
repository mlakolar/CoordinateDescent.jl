# All timings and allocation measurements run under Slurm, never locally.
using CoordinateDescent, LinearAlgebra, SparseArrays, Random, Statistics, Printf, DelimitedFiles
BLAS.set_num_threads(1)
seed, out = parse(Int, ARGS[1]), ARGS[2]
mkpath(out)
rng = Xoshiro(seed)
# Compile all three orders on a separate small signal before measuring.
for k in 0:2
  trend_filter(randn(rng, 32), 1/32; order=k)
end
open(joinpath(out,"timings-$seed.tsv"),"w") do io
  println(io,"seed\tn\torder\tround\tseconds\tbytes\titerations\tconverged\tobjective\tprimal\tdual\tdf")
  for n in (1_000, 10_000, 100_000, 1_000_000)
    t = range(0,1;length=n)
    noise = randn(rng,n)
    y = @. sin(4pi*t) + .6*(t > .55) + .2*noise
    for k in 0:2, round in 1:3
      GC.gc()
      m = @timed trend_filter(y, 1/n; order=k, max_iter=5000, abstol=1e-6, reltol=1e-5)
      s = m.value
      println(io,join((seed,n,k,round,m.time,m.bytes,s.iterations,s.converged,s.objective,
                        s.primal_residual,s.dual_residual,dof(s)), '\t'))
      flush(io)
      s.converged || error("nonconvergence: seed=$seed n=$n order=$k")
    end
  end
end
