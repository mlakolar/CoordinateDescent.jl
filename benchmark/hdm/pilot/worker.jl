include(joinpath(@__DIR__, "pilot.jl"))
using .HDMPilot, LinearAlgebra, Printf

function main(args)
  length(args) == 4 || error("Usage: worker.jl CASE FIRST LAST OUTDIR")
  case, firstrep, lastrep = parse.(Int, args[1:3])
  cs = HDMPilot.cases()
  1 <= case <= length(cs) || error("CASE must be in 1:$(length(cs))")
  1 <= firstrep <= lastrep <= HDMPilot.REPS || error("Require 1 <= FIRST <= LAST <= 500")
  outdir = abspath(args[4])
  path = joinpath(outdir, "case-$case-reps-$firstrep-$lastrep.csv")
  if isfile(path)
    println("skip existing $path")
    return
  end
  BLAS.set_num_threads(1)
  start = time_ns()
  HDMPilot.warmup()
  warm = (time_ns() - start) / 1e9
  start = time_ns()
  rows, timings = HDMPilot.Record[], Tuple[]
  for rep in firstrep:lastrep
    a, b = HDMPilot.run_rep(cs[case], rep)
    append!(rows, a); append!(timings, b)
  end
  seconds = (time_ns() - start) / 1e9
  # Main CSV is the completion marker: publish timing first so a restart
  # cannot skip a task whose timing file has not yet been published.
  if firstrep <= 20
    timingpath = replace(path, ".csv" => "-timing.csv")
    HDMPilot.writecsv(timingpath, HDMPilot.TIMING_HEADER, timings)
  end
  HDMPilot.writecsv(path, HDMPilot.RECORD_HEADER, HDMPilot.recordvalues.(rows))
  @printf("case=%d reps=%d:%d records=%d warmup_seconds=%.6f run_seconds=%.6f seconds_per_rep=%.6f reps_per_second=%.6f\n",
    case, firstrep, lastrep, length(rows), warm, seconds, seconds / (lastrep-firstrep+1),
    (lastrep-firstrep+1) / seconds)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
