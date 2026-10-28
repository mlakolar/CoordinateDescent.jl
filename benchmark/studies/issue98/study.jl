using HighDimensionalModels
import ProximalBase, ProximalSolvers
using LinearAlgebra, SparseArrays, SHA, TOML, Profile

# Reuse the generators, certificates and package fingerprint without running the harness.
module Harness
include(joinpath(@__DIR__, "..", "..", "run.jl"))
end

const GENERATOR = "t = range(0, 1; length = n); " *
  "y = sin.(4π .* t) .+ 0.6 .* (t .> 0.55) .+ 0.2 .* gaussian(rng, n); " *
  "if irregular: x = cumsum(0.8 .+ 0.4 .* uniform(rng, n)), after drawing y"
const BOUND_TARGET = 1e-6

function study_cases(scale)
  rows = (
    ("linear-1k", "trend_filter", "trend-sine-1000", 1, 1000, Float64, 1, 10.0, 30),
    ("linear-irregular-10k", "trend_filter", "trend-sine-10000-irregular", 1, 10000, Float64, 1, 1.0, 10),
    ("linear-100k", "trend_filter", "trend-sine-100000", 1, 100000, Float64, 1, 1.0, 10),
    ("quadratic-100k", "trend_filter", "trend-sine-100000", 2, 100000, Float64, 1, 0.1, 5),
    ("quadratic-path-10k", "trend_filter_path", "trend-sine-10000", 2, 10000, Float64, 50, 10.0, 3),
    ("tv-path-10k", "trend_filter_path", "trend-sine-10000", 0, 10000, Float64, 50, 10.0, 10),
    ("linear-1m", "trend_filter", "trend-sine-1000000", 1, 1000000, Float64, 1, 1.0, 3),
    ("quadratic-1m", "trend_filter", "trend-sine-1000000", 2, 1000000, Float64, 1, 1.0, 3),
    ("linear-100k-f32", "trend_filter", "trend-sine-100000", 1, 100000, Float32, 1, 1.0, 5),
    ("cv-linear-10k", "cv_trend_filter", "trend-sine-10000", 1, 10000, Float64, 10, 10.0, 3))
  [begin
    id, kind, datakey, order, n, T, levels, λR, rounds = row
    if scale == "smoke"
      n, levels, rounds = max(200, n ÷ 100), min(levels, 5), min(rounds, 2)
    end
    (; id, kind, datakey, order, n, T, levels, λR, rounds,
       irregular = endswith(datakey, "-irregular"))
  end for row in rows]
end

repname(rep) = "rep" * lpad(rep, 2, '0')
case_dir(out, c, rep) = joinpath(out, "cases", c.id, repname(rep))
input_dir(out, c, rep) = joinpath(out, "inputs", c.datakey, repname(rep))
done(dir) = isfile(joinpath(dir, "DONE"))

function atomic_write(f, path)
  mkpath(dirname(path))
  temporary, io = mktemp(dirname(path))
  try
    f(io)
    close(io)
    mv(temporary, path; force = true)
  finally
    isopen(io) && close(io)
    isfile(temporary) && rm(temporary)
  end
end

write_toml(path, data) = atomic_write(io -> TOML.print(io, data; sorted = true), path)
mark_done(dir) = atomic_write(io -> println(io, "complete"), joinpath(dir, "DONE"))
write_f64(path, v) = atomic_write(io -> foreach(x -> write(io, htol(Float64(x))), v), path)
file_hash(path) = bytes2hex(open(sha256, path))

function f64_hash(vectors)
  ctx = SHA.SHA2_256_CTX()
  buffer = IOBuffer()
  for v in vectors
    for x in v
      write(buffer, htol(Float64(x)))
      if position(buffer) >= 65536
        SHA.update!(ctx, take!(buffer))
      end
    end
  end
  SHA.update!(ctx, take!(buffer))
  bytes2hex(SHA.digest!(ctx))
end

field(::Nothing) = ""
field(x::Float32) = repr(Float64(x))
field(x::AbstractFloat) = repr(x)
field(x) = replace(string(x), '\t' => ' ', '\n' => ' ', '\r' => ' ')
row(io, fields...) = println(io, join(field.(fields), '\t'))

function levels_for(c; n = c.n)
  λR = c.levels == 1 ? [c.λR] : 10.0 .* 0.01 .^ range(0, 1; length = c.levels)
  λR, c.T.(λR ./ n)
end

function case_spec(c, rep, scale)
  keywords = Dict{String,Any}("order" => c.order)
  c.irregular && (keywords["x"] = "x.f64")
  if c.kind == "cv_trend_filter"
    keywords["folds"] = "mod1.(1:n, 5)"
    keywords["threads"] = false
  end
  Dict{String,Any}("id" => c.id, "call" => c.kind, "order" => c.order,
    "n" => c.n, "element_type" => string(c.T), "data_key" => c.datakey,
    "rounds" => c.rounds, "levels" => c.levels, "rep" => rep, "scale" => scale,
    "keywords" => keywords)
end

draw(c, rep; n = c.n) = Harness.trend_sine(Harness.scenario_rng(c.datakey, rep), n;
                                         irregular = c.irregular)

function ensure_inputs(c, rep, out, scale)
  raw = draw(c, rep)
  dir = input_dir(out, c, rep)
  hashes = Dict("y.f64" => f64_hash((raw.y,)))
  c.irregular && (hashes["x.f64"] = f64_hash((raw.x,)))
  spec = Dict("data_key" => c.datakey, "rep" => rep, "n" => c.n,
    "irregular" => c.irregular, "generator" => GENERATOR, "sha256" => hashes)
  # A complete draw is immutable, including when a different scale is requested.
  if done(dir)
    TOML.parsefile(joinpath(dir, "spec.toml")) == spec || error("input specification or regenerated hashes differ")
    for (name, digest) in hashes
      file_hash(joinpath(dir, name)) == digest || error("input hash mismatch: $name")
    end
  else
    write_f64(joinpath(dir, "y.f64"), raw.y)
    c.irregular && write_f64(joinpath(dir, "x.f64"), raw.x)
    write_toml(joinpath(dir, "spec.toml"), spec)
    mark_done(dir)
  end
  λR, levels = levels_for(c)
  io = IOBuffer()
  row(io, "level", "lambda_R", "lambda")
  for k in eachindex(levels)
    row(io, k, λR[k], levels[k])
  end
  table = String(take!(io))
  dir = case_dir(out, c, rep)
  specification = case_spec(c, rep, scale)
  if isfile(joinpath(dir, "case.toml"))
    TOML.parsefile(joinpath(dir, "case.toml")) == specification || error("case specification differs")
    read(joinpath(dir, "levels.tsv"), String) == table || error("case levels differ")
  else
    atomic_write(io -> write(io, table), joinpath(dir, "levels.tsv"))
    write_toml(joinpath(dir, "case.toml"), specification)
  end
  (; y = c.T.(raw.y), x = c.irregular ? raw.x : nothing, levels)
end

# Fresh copies are prepared outside the measured public call.
function fresh(c, data)
  (; y = copy(data.y), x = data.x === nothing ? nothing : copy(data.x),
     levels = copy(data.levels), folds = c.kind == "cv_trend_filter" ? mod1.(1:length(data.y), 5) : Int[])
end

function call_fit(c, s; tight = false, fullpath = false)
  options = (; order = c.order)
  s.x === nothing || (options = merge(options, (; x = s.x)))
  tight && (options = merge(options, (; abstol = 0.0, reltol = 1e-12, max_iter = 10^6)))
  if c.kind == "trend_filter"
    trend_filter(s.y, only(s.levels); options...)
  elseif c.kind == "trend_filter_path" || fullpath
    trend_filter_path(s.y, s.levels; options...)
  else
    cv_trend_filter(s.y, s.levels; options..., folds = s.folds, threads = false)
  end
end

function solutions(c, result)
  c.kind == "trend_filter" && return [result]
  path = c.kind == "cv_trend_filter" ? result.path : result
  length(path.solutions) == c.levels || error("incomplete returned path")
  path.solutions
end

timed_fit(c, s) = @timed call_fit(c, s)
optional_ns(t, name) = hasproperty(t, name) ? round(Int, getproperty(t, name) * 1e9) : nothing

function compile_fit(c, rep)
  raw = draw(c, rep; n = 40)
  _, levels = levels_for(c; n = 40)
  data = (; y = c.T.(raw.y), x = c.irregular ? raw.x : nothing, levels)
  timed_fit(c, fresh(c, data))
  nothing
end

function metadata(c, rep, side, scale)
  root = dirname(dirname(pathof(HighDimensionalModels)))
  commit, dirty = Harness.git_identity(root)
  Dict{String,Any}("julia_version" => string(VERSION), "case" => case_spec(c, rep, scale),
    "side" => side, "scale" => scale, "rounds" => c.rounds,
    "julia_threads" => Threads.nthreads(), "blas_vendor" => string(BLAS.vendor()),
    "blas_threads" => BLAS.get_num_threads(), "cpu_model" => Sys.cpu_info()[1].model,
    "word_size" => Sys.WORD_SIZE, "package_version" => string(pkgversion(HighDimensionalModels)),
    "package_hash" => Harness.package_fingerprint(root), "package_commit" => commit,
    "package_dirty" => dirty, "ProximalBase_version" => string(pkgversion(ProximalBase)),
    "ProximalSolvers_version" => string(pkgversion(ProximalSolvers)))
end

function write_first(dir, c, data, result)
  fits = solutions(c, result)
  atomic_write(joinpath(dir, "levels.tsv")) do io
    row(io, "level", "lambda", "iterations", "converged", "objective", "df", "primal_residual", "dual_residual")
    for (k, s) in enumerate(fits)
      row(io, k, data.levels[k], s.iterations, s.converged, s.objective, s.df,
          s.primal_residual, s.dual_residual)
    end
  end
  for (k, s) in enumerate(fits)
    write_f64(joinpath(dir, "coef-L$k.f64"), s.x)
  end
  if c.kind == "cv_trend_filter"
    atomic_write(joinpath(dir, "cv.tsv")) do io
      row(io, "level", "lambda", "cvm", "cvse")
      for k in eachindex(data.levels)
        row(io, k, data.levels[k], result.cvm[k], result.cvse[k])
      end
    end
  end
end

# Strip directory components before rendering profile frames.
function public_frame(f)
  Base.StackTraces.StackFrame(f.func, Symbol(basename(string(f.file))), f.line,
                             f.linfo, f.from_c, f.inlined, f.pointer)
end

function profiles(dir, c, data, seconds, bytes)
  delay = max(seconds / 20000, 1e-6)
  Profile.init(n = 10_000_000, delay = delay)
  Profile.clear()
  s = fresh(c, data)
  Profile.@profile call_fit(c, s)
  data_cpu, frames = Profile.retrieve()
  for (ip, stack) in frames
    frames[ip] = public_frame.(stack)
  end
  samples = count(iszero, Profile.fetch(; include_meta = false))
  atomic_write(joinpath(dir, "profile-cpu.txt")) do io
    println(io, "sample_delay_seconds = ", repr(delay))
    println(io, "total_samples = ", samples)
    if samples == 0
      println(io, "No CPU samples: the call finished before a sampling tick.")
    else
      Profile.print(IOContext(io, :displaysize => (typemax(Int), 100000), :limit => false),
        data_cpu, frames; format = :flat, C = false, mincount = 0, noisefloor = 0,
        sortedby = :count)
    end
  end
  Profile.clear()
  # Cap the rate and reduce it further for calls with large Julia heap traffic.
  rate = min(0.01, 50000 * 64 / max(bytes, 1))
  Profile.Allocs.clear()
  s = fresh(c, data)
  Profile.Allocs.@profile sample_rate=rate call_fit(c, s)
  allocations = Profile.Allocs.fetch().allocs
  sites = Dict{Tuple{String,Int,String},Int}()
  for a in allocations
    stack = filter(f -> !f.from_c, a.stacktrace)
    frames = join(("$(f.func) at $(basename(string(f.file))):$(f.line)" for f in Iterators.take(stack, 5)), " <- ")
    key = (string(a.type), Int(a.size), frames)
    sites[key] = get(sites, key, 0) + 1
  end
  ranked = sort!(collect(sites); by = p -> (-p.first[2] * p.second, p.first))
  atomic_write(joinpath(dir, "profile-alloc.txt")) do io
    println(io, "sample_rate = ", repr(rate))
    println(io, "sampled_allocations = ", length(allocations))
    println(io, "sampled_bytes = ", sum(a -> Int(a.size), allocations; init = 0))
    row(io, "type", "size_bytes", "samples", "sampled_bytes", "top_frames")
    for (key, count) in Iterators.take(ranked, 30)
      type, size, stack = key
      row(io, type, size, count, size * count, stack)
    end
  end
  Profile.Allocs.clear()
end

function fit(c, rep, side, out, scale)
  occursin(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", side) && side != "reference" || error("invalid side label")
  dir = joinpath(case_dir(out, c, rep), side)
  done(dir) && error("refusing to overwrite complete fit")
  data = ensure_inputs(c, rep, out, scale)
  meta = metadata(c, rep, side, scale)
  compile_fit(c, rep)
  first_seconds, first_bytes = 0.0, 0
  atomic_write(joinpath(dir, "rounds.tsv")) do io
    row(io, "round", "time_ns", "bytes", "gc_ns", "compile_ns", "recompile_ns",
        "iterations_total", "converged_all", "coef_sha256")
    for round in 1:c.rounds
      s = fresh(c, data)
      t = timed_fit(c, s)
      # Capture the fresh-process peak before output processing and profiling.
      peak = round == 1 ? Int(Sys.maxrss()) : 0
      fits = solutions(c, t.value)
      row(io, round, round_ns(t.time), t.bytes, round_ns(t.gctime),
          optional_ns(t, :compile_time), optional_ns(t, :recompile_time),
          sum(s -> s.iterations, fits), all(s -> s.converged, fits), f64_hash((s.x for s in fits)))
      if round == 1
        first_seconds, first_bytes = t.time, t.bytes
        atomic_write(joinpath(dir, "memory.tsv")) do mem
          row(mem, "maxrss_bytes")
          row(mem, peak)
        end
        write_first(dir, c, data, t.value)
        if c.kind == "cv_trend_filter"
          meta["lambda_min"], meta["lambda_1se"] = t.value.λmin, t.value.λ1se
        end
      end
    end
  end
  profiles(dir, c, data, first_seconds, first_bytes)
  final = metadata(c, rep, side, scale)
  (final["package_hash"], final["package_commit"], final["package_dirty"]) ==
    (meta["package_hash"], meta["package_commit"], meta["package_dirty"]) || error("package changed during fit")
  write_toml(joinpath(dir, "meta.toml"), meta)
  mark_done(dir)
  println("fit complete: ", c.id, " rep=", rep, " side=", side)
end

round_ns(seconds) = round(Int, seconds * 1e9)

# Harness.box_lsq with identical iterates and explicit termination diagnostics.
function box_lsq_record(D::SparseMatrixCSC{Float64}, g::Vector{Float64}, b::Vector{Float64}; maxiter = 5000)
  m = size(D, 1)
  u = zeros(m)
  Dt = sparse(transpose(D))
  L = m == 0 ? 0.0 : maximum(sum(abs, D; dims = 1)) * maximum(sum(abs, D; dims = 2))
  L > 0 || return (; u, iterations = 0, stopped_by = "zero_operator")
  objective(v) = sum(abs2, Dt * v .- g) / 2
  floor = (4eps())^2 * sum(abs2, g) / 2
  φu = objective(u)
  iterations, stopped_by = 0, "maxiter"
  for iteration in 1:maxiter
    if φu <= floor
      stopped_by = "objective_floor"
      break
    end
    grad = D * (Dt * u .- g)
    pg = Harness.cert_maxabs(u .- clamp.(u .- grad ./ L, -b, b))
    if pg == 0
      stopped_by = "projected_gradient"
      break
    end
    iterations = iteration
    ε = min(1e-8 * maximum(b), pg)
    active = [b[k] == 0 || (u[k] <= -b[k] + ε && grad[k] > 0) || (u[k] >= b[k] - ε && grad[k] < 0) for k in 1:m]
    free, fixed = findall(!, active), findall(active)
    step = zeros(m)
    step[fixed] .= -grad[fixed] ./ L
    if !isempty(free)
      DF = D[free, :]
      H = DF * transpose(DF) + (1e-12 * L) * I
      factor = try cholesky(Symmetric(H)) catch; nothing end
      if factor === nothing
        stopped_by = "factorization_failed"
        break
      end
      step[free] .= -(factor \ grad[free])
    end
    α, accepted, candidate, φc = 1.0, false, u, φu
    for _ in 1:60
      candidate = clamp.(u .+ α .* step, -b, b)
      decrease = α * dot(grad[free], -step[free]) + dot(grad[fixed], u[fixed] .- candidate[fixed])
      decrease > 0 || break
      φc = objective(candidate)
      if φc <= φu - 1e-4 * decrease
        accepted = true
        break
      end
      α /= 2
    end
    if !accepted
      stopped_by = "line_search_failed"
      break
    end
    progress = φu - φc
    u, φu = candidate, φc
    if progress <= 1e-15 * φu
      stopped_by = "no_progress"
      break
    end
  end
  (; u, iterations, stopped_by)
end

function reference(c, rep, out, scale, method)
  method in ("dual-active-set", "admm-tight") || error("unknown reference method")
  dir = joinpath(case_dir(out, c, rep), "reference")
  done(dir) && error("refusing to overwrite complete reference")
  data = ensure_inputs(c, rep, out, scale)
  y = Float64.(data.y)
  D = sparse(Float64.(difference_matrix(c.n, c.order + 1; x = data.x)))
  levels = Float64.(data.levels)
  fits = nothing
  fit_seconds = 0.0
  if method == "admm-tight"
    # CV references concern only the full-data path, not fold selection.
    s = fresh(c, (; y, x = data.x, levels))
    t = @timed call_fit(c, s; tight = true, fullpath = true)
    fits = c.kind == "trend_filter" ? [t.value] : t.value.solutions
    length(fits) == c.levels || error("incomplete tight reference path")
    fit_seconds = t.time
  end
  accepted = Bool[]
  converged = Bool[]
  atomic_write(joinpath(dir, "reference.tsv")) do io
    row(io, "level", "method", "iterations", "stopped_by", "seconds", "primal", "dual", "gap_relative", "bound_relative")
    for (k, λ) in enumerate(levels)
      b = fill(λ, size(D, 1))
      g = method == "dual-active-set" ? y ./ c.n : (y .- Float64.(fits[k].x)) ./ c.n
      t = @timed box_lsq_record(D, g, b)
      u = clamp.(t.value.u, -b, b)
      Dtu = transpose(D) * u
      β = method == "dual-active-set" ? y .- c.n .* Dtu : Float64.(fits[k].x)
      primal = sum(abs2, y .- β) / (2c.n) + λ * sum(abs, D * β)
      dual = dot(u, D * y) - (c.n / 2) * sum(abs2, Dtu)
      gap = primal - dual
      # A material weak-duality violation cannot be accepted as a zero bound.
      valid = all(isfinite, (primal, dual)) && gap >= -64eps() * max(abs(primal), abs(dual), 1.0)
      bound = valid && norm(β) > 0 ? sqrt(2c.n * max(gap, 0.0)) / norm(β) : Inf
      iterations = method == "dual-active-set" ? t.value.iterations : fits[k].iterations
      stop = method == "dual-active-set" ? t.value.stopped_by :
        "admm_$(fits[k].converged ? "converged" : "maxiter");dual_$(t.value.stopped_by)"
      row(io, k, method, iterations, stop, t.time + fit_seconds / length(levels), primal, dual,
          valid ? max(gap, 0.0) / max(abs(primal), floatmin(Float64)) : Inf, bound)
      push!(accepted, bound <= BOUND_TARGET)
      push!(converged, t.value.stopped_by in ("objective_floor", "projected_gradient", "no_progress", "zero_operator"))
      write_f64(joinpath(dir, "beta-L$k.f64"), β)
      write_f64(joinpath(dir, "u-L$k.f64"), u)
    end
  end
  write_toml(joinpath(dir, "meta.toml"), Dict("method" => method, "case" => case_spec(c, rep, scale),
    "bound_target" => BOUND_TARGET, "accepted_by_bound" => accepted,
    "dual_stopped_by_test" => converged, "dual_maxiter" => 5000,
    "tight_path_seconds" => fit_seconds))
  mark_done(dir)
  println("reference complete: ", c.id, " rep=", rep, " accepted=", count(accepted), "/", length(accepted))
end

function main(args = ARGS)
  Threads.nthreads() == 1 || error("one Julia thread is required")
  BLAS.set_num_threads(1)
  scale = get(ENV, "ISSUE98_SCALE", "full")
  scale in ("full", "smoke") || error("ISSUE98_SCALE must be full or smoke")
  cases = study_cases(scale)
  if args == ["list"]
    row(stdout, "id", "kind", "data_key", "order", "n", "element_type", "levels", "rounds")
    for c in cases
      row(stdout, c.id, c.kind, c.datakey, c.order, c.n, c.T, c.levels, c.rounds)
    end
    return
  end
  length(args) >= 4 || error("usage: study.jl list | inputs CASE REP OUT | fit CASE REP SIDE OUT | reference CASE REP OUT [METHOD]")
  command, id = args[1:2]
  index = findfirst(c -> c.id == id, cases)
  index === nothing && error("unknown study case")
  c, rep = cases[index], parse(Int, args[3])
  1 <= rep <= 10 || error("REP must be between 1 and 10")
  if command == "inputs" && length(args) == 4
    ensure_inputs(c, rep, args[4], scale)
    println("inputs complete: ", c.id, " rep=", rep)
  elseif command == "fit" && length(args) == 5
    fit(c, rep, args[4], args[5], scale)
  elseif command == "reference" && length(args) in (4, 5)
    reference(c, rep, args[4], scale, length(args) == 5 ? args[5] : "dual-active-set")
  else
    error("invalid command or argument count")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
