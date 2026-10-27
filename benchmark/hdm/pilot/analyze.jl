module PilotAnalysis

include(joinpath(@__DIR__, "pilot.jl"))
using .HDMPilot, DelimitedFiles, Printf, Statistics
const P = HDMPilot
const SUMMARY_HEADER = ["case", "case_name", "arm", "vcov", "group", "metric", "value",
  "lower", "upper", "mcse", "planned", "valid", "datasets", "uncertainty"]

avg(x) = isempty(x) ? NaN : mean(x)
sd(x) = length(x) < 2 ? NaN : std(x)

function wilson(hits, n)
  n == 0 && return (NaN, NaN)
  z, p = P.Z95, hits / n
  middle = (p + z^2 / (2n)) / (1 + z^2 / n)
  half = z * sqrt(p * (1-p) / n + z^2 / (4n^2)) / (1 + z^2 / n)
  (max(0.0, middle - half), min(1.0, middle + half))
end

function normalstats(x; rate = false)
  value = avg(x)
  mcse = length(x) < 2 ? NaN : std(x) / sqrt(length(x))
  lower, upper = value - P.Z95 * mcse, value + P.Z95 * mcse
  if rate
    lower, upper = max(0.0, lower), min(1.0, upper)
  end
  (; value, lower, upper, mcse)
end

function clustermeans(rows, f)
  groups = Dict{Int, Vector{Float64}}()
  for r in rows
    push!(get!(groups, r.rep, Float64[]), f(r))
  end
  [mean(groups[rep]) for rep in sort!(collect(keys(groups)))]
end

function rate(rows, f, pooled)
  datasets = length(unique(r.rep for r in rows))
  if pooled
    return (; normalstats(clustermeans(rows, f); rate = true)..., datasets,
              uncertainty = "dataset_normal")
  end
  lower, upper = wilson(count(f, rows), length(rows))
  (; value = avg(f.(rows)), lower, upper, mcse = NaN, datasets, uncertainty = "Wilson")
end

function add!(summary, c, arm, covariance, group, metric; value = NaN, lower = NaN,
              upper = NaN, mcse = NaN, planned = 0, valid = 0, datasets = 0, uncertainty = "none")
  push!(summary, (; case = c.index, case_name = c.name, arm, vcov = string(covariance), group,
    metric, value, lower, upper, mcse, planned, valid, datasets, uncertainty))
end

function loadinputs(outdir, cs)
  records = Dict{Tuple{Int, Int, Int, String}, P.Record}()
  timings = Tuple[]
  timingkeys = Set{Tuple{Int, Int, Int, String}}()
  files = filter(f -> occursin(r"^case-\d+-reps-\d+-\d+\.csv$", f), readdir(outdir))
  for file in sort(files)
    m = match(r"^case-(\d+)-reps-(\d+)-(\d+)\.csv$", file)
    case, firstrep, lastrep = parse.(Int, m.captures)
    1 <= case <= length(cs) && 1 <= firstrep <= lastrep <= P.REPS || error("Invalid shard $file")
    c = cs[case]
    rows = P.readrecords(joinpath(outdir, file))
    length(rows) == (lastrep-firstrep+1) * P.nslots(c) * length(P.ARMS) || error("Incomplete shard $file")
    for r in rows
      key = (r.case, r.rep, r.slot, r.arm)
      haskey(records, key) && error("Duplicate record $key: overlapping shards are not independent replicates")
      r.case == case && r.case_name == c.name && firstrep <= r.rep <= lastrep &&
        1 <= r.slot <= P.nslots(c) && r.arm in P.ARMS && 1 <= r.target <= c.p || error("Invalid record $key")
      role = r.slot in P.active_slots(c) ? "active" : "null"
      truth = role == "null" ? 0.0 : c.design == "belloni" ? c.alpha : c.signal * (-1.0)^(r.slot-1)
      r.role == role && r.truth == truth || error("Role/truth mismatch $key")
      records[key] = r
    end
    if firstrep <= 20
      path = joinpath(outdir, replace(file, ".csv" => "-timing.csv"))
      isfile(path) || error("Missing timing companion $path")
      a, h = readdlm(path, ',', String; header = true)
      vec(h) == P.TIMING_HEADER || error("Invalid timing header $path")
      length(eachrow(a)) == (min(20, lastrep)-firstrep+1) * P.nslots(c) * 2 || error("Incomplete timing $path")
      for r in eachrow(a)
        row = (parse(Int, r[1]), r[2], parse(Int, r[3]), parse(Int, r[4]), parse(Int, r[5]),
               r[6], parse(Int, r[7]), r[8], parse(Float64, r[9]))
        key = (row[1], row[3], row[4], row[6])
        key in timingkeys && error("Duplicate timing $key")
        row[1] == case && row[2] == c.name && firstrep <= row[3] <= min(20, lastrep) &&
          1 <= row[4] <= P.nslots(c) && row[6] in ("partialling", "doubleselection") || error("Invalid timing $key")
        push!(timingkeys, key); push!(timings, row)
      end
    end
  end
  records, timings
end

function summarize_group!(summary, c, arm, covariance, q, name, slots, allrows)
  planned = P.REPS * length(slots)
  rows = [r for r in allrows if r.slot in slots && P.valid(r, q)]
  valid, datasets = length(rows), length(unique(r.rep for r in rows))
  pooled = startswith(name, "pooled_")
  common = (; planned, valid, datasets)
  add!(summary, c, arm, covariance, name, "availability"; common..., value = planned == 0 ? NaN : valid/planned)
  for (metric, f) in (("coverage", r -> P.covered(r, q)), ("rejection", r -> P.rejected(r, q)))
    st = rate(rows, f, pooled)
    add!(summary, c, arm, covariance, name, metric; planned, valid, st...)
    hits = count(f, rows)
    add!(summary, c, arm, covariance, name, metric * "_planned_bounds"; common...,
      lower = planned == 0 ? NaN : hits/planned,
      upper = planned == 0 ? NaN : (hits + planned - valid)/planned, uncertainty = "failure_bounds")
    if pooled
      add!(summary, c, arm, covariance, name, metric * "_fit_weighted"; common..., value = avg(f.(rows)))
    end
  end
  errors = [r.estimate-r.truth for r in rows]
  for (metric, value) in (("bias", avg(errors)), ("empirical_sd", sd([r.estimate for r in rows])),
                         ("error_sd", sd(errors)), ("mean_se", avg([r.se[q] for r in rows])),
                         ("mean_length", avg([2 * P.Z95 * r.se[q] for r in rows])),
                         ("median_length", isempty(rows) ? NaN : median([2 * P.Z95 * r.se[q] for r in rows])))
    add!(summary, c, arm, covariance, name, metric; common..., value)
  end
  if pooled
    cr = filter(P.converged, rows)
    add!(summary, c, arm, covariance, name, "coverage_converged_only";
      planned, valid = length(cr), rate(cr, r -> P.covered(r, q), true)...)
  end
end

function paired!(summary, records, c, covariance, q, other)
  slots = P.active_slots(c)
  coverage, lengths, countpairs = Float64[], Float64[], 0
  for rep in 1:P.REPS
    dc, dl = Float64[], Float64[]
    for slot in slots
      a, b = get(records, (c.index, rep, slot, "A"), nothing), get(records, (c.index, rep, slot, other), nothing)
      if a !== nothing && b !== nothing && P.valid(a, q) && P.valid(b, q)
        push!(dc, Float64(P.covered(a, q)) - Float64(P.covered(b, q)))
        push!(dl, 2 * P.Z95 * (a.se[q] - b.se[q]))
        countpairs += 1
      end
    end
    if !isempty(dc)
      push!(coverage, mean(dc)); push!(lengths, mean(dl))
    end
  end
  for (metric, values) in (("paired_coverage_difference", coverage), ("paired_mean_length_difference", lengths))
    add!(summary, c, "A-$other", covariance, "pooled_active", metric;
      planned = P.REPS * length(slots), valid = countpairs, datasets = length(values),
      normalstats(values)..., uncertainty = "paired_dataset_normal")
  end
end

function distribution!(summary, c, arm, metric, values; planned)
  values = filter(isfinite, values)
  for (suffix, value) in (("mean", avg(values)), ("median", isempty(values) ? NaN : median(values)),
                          ("p95", isempty(values) ? NaN : quantile(values, 0.95)),
                          ("max", isempty(values) ? NaN : maximum(values)))
    add!(summary, c, arm, "all", "all", metric * "_" * suffix; value, planned, valid = length(values))
  end
end

function diagnostics!(summary, failures, c, arm, rows)
  planned = P.REPS * P.nslots(c)
  for field in (:s_y, :s_d, :s_union, :rank, :dof_residual, :target_variation, :max_leverage)
    distribution!(summary, c, arm, string(field), [getfield(r, field) for r in rows]; planned)
  end
  distribution!(summary, c, arm, "final_seconds", [r.final_seconds for r in rows]; planned)
  if arm in ("A", "D1")
    distribution!(summary, c, arm == "A" ? "post_pair" : "native_pair", "nuisance_seconds",
      [r.nuisance_seconds for r in rows]; planned)
  end
  known = filter(r -> all(x -> x !== nothing,
    (r.converged_y, r.solver_converged_y, r.converged_d, r.solver_converged_d)), rows)
  add!(summary, c, arm, "all", "all", "nuisance_convergence_fraction";
    value = avg(P.converged.(known)), planned, valid = length(known))
  missing = planned - length(rows)
  missing > 0 && (failures[c.index, arm, "record", "MissingRecord"] = missing)
  for r in rows
    if r.fit_status != "ok"
      key = (c.index, arm, "arm", r.fit_status)
      failures[key] = get(failures, key, 0) + 1
    end
    for (q, type) in enumerate(P.VCOVS)
      kind = r.errors[q]
      isempty(kind) && !P.valid(r, q) && (kind = "InvalidEstimateOrSE")
      if !isempty(kind)
        key = (c.index, arm, string(type), kind)
        failures[key] = get(failures, key, 0) + 1
      end
    end
  end
end

number(x) = isfinite(x) ? @sprintf("%.4g", x) : "—"
percent(x) = isfinite(x) ? @sprintf("%.2f%%", 100x) : "—"
function interval(r)
  percent(r.value) * " [" * percent(r.lower) * ", " * percent(r.upper) * "]"
end
flag(x) = x === nothing ? "NA" : x ? "PASS" : "FAIL"

function report(path, summary, cs, records, failures)
  lookup = Dict((r.case, r.arm, r.vcov, r.group, r.metric) => r for r in summary)
  P.atomic_write(path) do io
    println(io, "# Post-selection coverage pilot — DESCRIPTIVE\n")
    println(io, "Defaults are unchanged. This pilot is a descriptive screen, not a validation of coverage.")
    println(io, "Observed $(length(records)) / 180000 planned arm records; all denominators retain 500 datasets per case.")
    println(io, "Missing chunks are unavailable, not successful fits. Conditional rates exclude invalid fits.")
    println(io, "Pooled rates average valid-role means within datasets, then average datasets with a valid role.")
    println(io, "Their 95% normal bounds use dataset MCSE; a single dataset gives no MCSE. Single-role Wilson intervals are in summary.csv.\n")
    println(io, "DESCRIPTIVE flags, in order: availability ≥99% for every nonempty pooled group; active-coverage lower 95% bound ≥92.5%; null-rejection upper 95% bound ≤7.5%.")
    println(io, "NA means no applicable roles or insufficient datasets for uncertainty. No multiplicity adjustment is made.\n")
    for c in cs
      println(io, "## $(c.index): $(c.name)\n")
      println(io, "| Arm | vcov | Valid active / null | Active coverage [95%] | Null rejection [95%] | Mean length active / null | Converged-only active coverage | DESCRIPTIVE flags |")
      println(io, "|:--|:--|:--|:--|:--|:--|:--|:--|")
      for arm in P.ARMS, type in P.VCOVS
        getrow(g, m) = lookup[c.index, arm, string(type), g, m]
        a, z = getrow("pooled_active", "coverage"), getrow("pooled_null", "rejection")
        av, zv = getrow("pooled_active", "availability"), getrow("pooled_null", "availability")
        applicable = filter(r -> r.planned > 0, [av, zv])
        af = isempty(applicable) ? nothing : all(r -> r.value >= 0.99, applicable)
        cf = isfinite(a.lower) ? a.lower >= 0.925 : nothing
        nf = isfinite(z.upper) ? z.upper <= 0.075 : nothing
        lengths = number(getrow("pooled_active", "mean_length").value) * " / " * number(getrow("pooled_null", "mean_length").value)
        conv = percent(getrow("pooled_active", "coverage_converged_only").value)
        println(io, "| $arm | $type | $(percent(av.value)) / $(percent(zv.value)) | $(interval(a)) | $(interval(z)) | $lengths | $conv | $(flag(af)) / $(flag(cf)) / $(flag(nf)) |")
      end
      println(io)
    end
    println(io, "## Paired active comparisons\n")
    println(io, "Common valid roles within each dataset; means and MCSE are over paired datasets. Detailed counts and normal bounds are in summary.csv.\n")
    println(io, "| Case | Pair | vcov | Coverage difference ± MCSE | Length difference ± MCSE | Paired datasets |")
    println(io, "|:--|:--|:--|:--|:--|--:|")
    for c in cs, pair in ("A-B", "A-C"), type in P.VCOVS
      a = lookup[c.index, pair, string(type), "pooled_active", "paired_coverage_difference"]
      b = lookup[c.index, pair, string(type), "pooled_active", "paired_mean_length_difference"]
      a.datasets == 0 && continue
      println(io, "| $(c.name) | $pair | $type | $(number(a.value)) ± $(number(a.mcse)) | $(number(b.value)) ± $(number(b.mcse)) | $(a.datasets) |")
    end
    println(io, "\n## Failures and missing records\n")
    println(io, "Arm and individual-vcov error counts describe different stages and can overlap. MissingRecord means a planned record is absent.\n")
    println(io, "| Case | Arm | Stage | Type | Count |\n|:--|:--|:--|:--|--:|")
    for ((case, arm, stage, kind), count) in sort!(collect(failures); by = first)
      println(io, "| $(cs[case].name) | $arm | $stage | $kind | $count |")
    end
    isempty(failures) && println(io, "| All | All | All | None | 0 |")
    println(io, "\n## Timings (seconds)\n")
    println(io, "Warm worker calls only; pair timings are counted once, not once per arm. Final-stage timings sum all six variance calls, including failures.\n")
    println(io, "| Case | Stage | Mean | Median | p95 | Observations |\n|:--|:--|--:|--:|--:|--:|")
    for c in cs, stage in ("post_pair", "native_pair", P.ARMS..., "convenience_partialling", "convenience_doubleselection")
      metric = stage in ("post_pair", "native_pair") ? "nuisance_seconds" : startswith(stage, "convenience_") ? "seconds" : "final_seconds"
      r = lookup[c.index, stage, "all", "all", metric * "_mean"]
      r.valid == 0 && continue
      med = lookup[c.index, stage, "all", "all", metric * "_median"].value
      p95 = lookup[c.index, stage, "all", "all", metric * "_p95"].value
      println(io, "| $(c.name) | $stage | $(number(r.value)) | $(number(med)) | $(number(p95)) | $(r.valid) |")
    end
    println(io, "\nSupport-size, rank, leverage, empirical SD, mean SE, interval medians, planned-denominator bounds, and convergence sensitivity are retained in summary.csv.")
  end
end

function analyze(outdir, resultsdir)
  cs = P.cases()
  records, timings = loadinputs(outdir, cs)
  bycasearm = Dict((c.index, arm) => P.Record[] for c in cs for arm in P.ARMS)
  for key in sort!(collect(keys(records)))
    r = records[key]
    push!(bycasearm[r.case, r.arm], r)
  end
  summary, failures = NamedTuple[], Dict{Tuple{Int, String, String, String}, Int}()
  for c in cs
    active, null = P.active_slots(c), P.null_slots(c)
    groups = [("active_$i", [slot]) for (i, slot) in enumerate(active)]
    append!(groups, [("null_$i", [slot]) for (i, slot) in enumerate(null)])
    append!(groups, [("pooled_active", active), ("pooled_null", null)])
    for arm in P.ARMS
      rows = bycasearm[c.index, arm]
      for (q, type) in enumerate(P.VCOVS), (name, slots) in groups
        summarize_group!(summary, c, arm, type, q, name, slots, rows)
      end
      diagnostics!(summary, failures, c, arm, rows)
    end
    for (q, type) in enumerate(P.VCOVS), other in ("B", "C")
      paired!(summary, records, c, type, q, other)
    end
    for method in ("partialling", "doubleselection")
      ts = [t for t in timings if t[1] == c.index && t[6] == method]
      distribution!(summary, c, "convenience_$method", "seconds", [t[9] for t in ts]; planned = 20 * P.nslots(c))
      for t in ts
        if t[8] != "ok"
          key = (c.index, "convenience_$method", "timing", t[8])
          failures[key] = get(failures, key, 0) + 1
        end
      end
    end
  end
  mkpath(resultsdir)
  P.writecsv(joinpath(resultsdir, "summary.csv"), SUMMARY_HEADER, (Tuple(r) for r in summary))
  report(joinpath(resultsdir, "RESULTS.md"), summary, cs, records, failures)
  println("analysis records=$(length(records)) summary_rows=$(length(summary)) results=$(abspath(resultsdir))")
  summary
end

function main(args)
  length(args) == 2 || error("Usage: analyze.jl OUTDIR RESULTSDIR")
  analyze(abspath(args[1]), abspath(args[2]))
end

end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && PilotAnalysis.main(ARGS)
