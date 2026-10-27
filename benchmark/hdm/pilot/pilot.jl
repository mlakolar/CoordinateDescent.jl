module HDMPilot

using HighDimensionalModels, DelimitedFiles, LinearAlgebra, Printf, Random, Statistics

const VCOVS = (:hdm, :homoskedastic, :HC0, :HC1, :HC2, :HC3)
const ARMS = ("A", "B", "C", "D1", "D2")
const REPS = 500
const Z95 = 1.959963984540054

struct Case
  index::Int
  name::String
  n::Int
  p::Int
  rho::Float64
  s::Int
  signal::Float64
  support::String
  error::String
  design::String
  alpha::Float64
  cy::Float64
end

function cases(path = joinpath(@__DIR__, "cases.csv"))
  a, h = readdlm(path, ',', String; header = true)
  vec(h) == split("case,n,p,rho,s,signal,support,error,design,alpha,cy", ',') || error("Invalid cases header")
  [Case(i, r[1], parse(Int, r[2]), parse(Int, r[3]), parse(Float64, r[4]),
        isempty(r[5]) ? 0 : parse(Int, r[5]), isempty(r[6]) ? NaN : parse(Float64, r[6]),
        r[7], r[8], r[9], isempty(r[10]) ? NaN : parse(Float64, r[10]),
        isempty(r[11]) ? NaN : parse(Float64, r[11])) for (i, r) in enumerate(eachrow(a))]
end

data_seed(case, rep) = 1_400_000_000 + 1_000_000 * case + rep
solver_seed(case, rep, slot, stream) = 1_500_000_000 + 10_000_000 * case + 10_000 * rep + 10 * slot + stream

# Frozen Gaussian generator conventions: 9e9e587:benchmark/inference/
# overnight-defaults/pilot.jl:7-26. See README for the hetero_target addition.
function dataset(c::Case, rep)
  rng = Xoshiro(data_seed(c.index, rep))
  n, p, rho = c.n, c.p, c.rho
  if c.design == "belloni"
    Z = randn(rng, n, p - 1)
    for j in 2:p-1
      Z[:, j] .= rho .* Z[:, j-1] .+ sqrt(1 - rho^2) .* Z[:, j]
    end
    vy, vd = zeros(p), zeros(p)
    vy[1:5] = 1 ./ (1:5); vy[11:15] = 1 ./ (1:5); vd[1:10] = 1 ./ (1:10)
    d = vd[1] .+ Z * vd[2:end] .+ randn(rng, n)
    X = hcat(d, Z)
    b, b0 = vcat(c.alpha, c.cy .* vy[2:end]), c.cy * vy[1]
    targets = [1; collect(p-4:p)]
  else
    X = randn(rng, n, p)
    for j in 2:p
      X[:, j] .= rho .* X[:, j-1] .+ sqrt(1 - rho^2) .* X[:, j]
    end
    X .-= mean(X; dims = 1)
    X ./= sqrt.(mean(abs2, X; dims = 1))
    ids = c.support == "random" ? randperm(rng, p)[1:c.s] : collect(1:c.s)
    b, b0 = zeros(p), 0.2
    b[ids] = c.signal .* (-1.0) .^ (0:c.s-1)
    targets = [ids[1:min(c.s, 5)]; [j for j in 1:p if b[j] == 0][1:5]]
  end
  err = if c.error == "t3"
    # The denominator intentionally has no division by three: variance one.
    randn(rng, n) ./ sqrt.(vec(sum(abs2, randn(rng, n, 3); dims = 2)))
  elseif c.error == "hetero_target"
    sqrt.((1 .+ X[:, 1].^2 .+ X[:, 6].^2) ./ 3) .* randn(rng, n)
  else
    randn(rng, n)
  end
  (; X, y = b0 .+ X * b .+ err, b, b0, targets)
end

nslots(c) = c.design == "belloni" ? 6 : 10
active_slots(c) = c.design == "belloni" ? (c.alpha == 0 ? Int[] : [1]) : collect(1:5)
null_slots(c) = setdiff(1:nslots(c), active_slots(c))

Base.@kwdef mutable struct Record
  case::Int
  case_name::String
  rep::Int
  slot::Int
  target::Int
  truth::Float64
  role::String
  arm::String
  fit_status::String = "ok"
  estimate::Float64 = NaN
  se::Vector{Float64} = fill(NaN, length(VCOVS))
  errors::Vector{String} = fill("", length(VCOVS))
  s_y::Float64 = NaN
  s_d::Float64 = NaN
  s_union::Float64 = NaN
  rank::Float64 = NaN
  dof_residual::Float64 = NaN
  target_variation::Float64 = NaN
  max_leverage::Float64 = NaN
  converged_y::Union{Nothing, Bool} = nothing
  solver_converged_y::Union{Nothing, Bool} = nothing
  converged_d::Union{Nothing, Bool} = nothing
  solver_converged_d::Union{Nothing, Bool} = nothing
  nuisance_seconds::Float64 = NaN
  final_seconds::Float64 = NaN
end

const RECORD_HEADER = ["case", "case_name", "rep", "slot", "target", "truth", "role", "arm", "fit_status", "estimate",
  ["se_$t" for t in VCOVS]..., ["err_$t" for t in VCOVS]...,
  "s_y", "s_d", "s_union", "rank", "dof_residual", "target_variation", "max_leverage",
  "converged_y", "solver_converged_y", "converged_d", "solver_converged_d", "nuisance_seconds", "final_seconds"]
const TIMING_HEADER = ["case", "case_name", "rep", "slot", "target", "method", "seed", "status", "seconds"]

recordvalues(r::Record) = (r.case, r.case_name, r.rep, r.slot, r.target, r.truth, r.role, r.arm,
  r.fit_status, r.estimate, r.se..., r.errors..., r.s_y, r.s_d, r.s_union, r.rank, r.dof_residual,
  r.target_variation, r.max_leverage, r.converged_y, r.solver_converged_y, r.converged_d,
  r.solver_converged_d, r.nuisance_seconds, r.final_seconds)

function csvfield(x)
  s = x === nothing ? "" : x isa AbstractFloat ? @sprintf("%.17g", x) : string(x)
  occursin(r"[,\"\n\r]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end
csvline(io, row) = println(io, join(csvfield.(row), ','))

function atomic_write(f, path)
  mkpath(dirname(path))
  tmp, io = mktemp(dirname(path))
  try
    f(io)
    close(io)
    mv(tmp, path; force = true)
  finally
    isopen(io) && close(io)
    isfile(tmp) && rm(tmp)
  end
end

function writecsv(path, header, rows)
  atomic_write(path) do io
    csvline(io, header)
    for row in rows
      csvline(io, row)
    end
  end
end

exception_type(e) = string(nameof(typeof(e)))
function fitpair(c, rep, slot, C, y, d, post)
  start = time_ns()
  fits, errors = Any[nothing, nothing], String[]
  for (i, response) in enumerate((y, d))
    stream = post ? i : i + 2
    try
      fits[i] = rlasso(C, response; post, rng = Xoshiro(solver_seed(c.index, rep, slot, stream)))
    catch e
      e isa InterruptException && rethrow()
      push!(errors, exception_type(e))
    end
  end
  (; y = fits[1], d = fits[2], status = isempty(errors) ? "ok" : first(errors),
     seconds = (time_ns() - start) / 1e9)
end

function arm_record(c, rep, slot, j, truth, y, d, C, arm, pair)
  r = Record(; case = c.index, case_name = c.name, rep, slot, target = j, truth,
    role = truth == 0 ? "null" : "active", arm, nuisance_seconds = pair.seconds)
  fy, fd = pair.y, pair.d
  if fy !== nothing
    r.s_y, r.converged_y, r.solver_converged_y = length(fy.selected), fy.converged, fy.solver_converged
  end
  if fd !== nothing
    r.s_d, r.converged_d, r.solver_converged_d = length(fd.selected), fd.converged, fd.solver_converged
  end
  if pair.status != "ok"
    r.fit_status = pair.status
    r.errors .= pair.status
    r.final_seconds = 0.0
    return r
  end
  r.s_union = length(union(fy.selected, fd.selected))
  method = arm in ("C", "D2") ? :doubleselection : :partialling
  refit = arm == "A" ? true : arm == "B" ? false : nothing
  start, successes = time_ns(), 0
  for (i, covariance) in enumerate(VCOVS)
    try
      te = double_selection(y, d, C, fy, fd; method, refit, vcov = covariance)
      r.estimate, r.se[i] = te.estimate, te.se
      r.rank, r.dof_residual = te.rank, te.dof_residual
      r.target_variation, r.max_leverage = te.target_variation, te.max_leverage
      successes += 1
    catch e
      e isa InterruptException && rethrow()
      r.errors[i] = exception_type(e)
    end
  end
  r.final_seconds = (time_ns() - start) / 1e9
  successes == 0 && (r.fit_status = first(r.errors))
  r
end

function run_rep(c, rep; time_convenience = rep <= 20)
  data = dataset(c, rep)
  rows, timings = Record[], Tuple[]
  for (slot, j) in enumerate(data.targets)
    d, C = data.X[:, j], data.X[:, setdiff(1:c.p, j)]
    post = fitpair(c, rep, slot, C, data.y, d, true)
    native = fitpair(c, rep, slot, C, data.y, d, false)
    for arm in ARMS
      pair = arm in ("D1", "D2") ? native : post
      push!(rows, arm_record(c, rep, slot, j, data.b[j], data.y, d, C, arm, pair))
    end
    if time_convenience
      for (stream, method) in enumerate((:partialling, :doubleselection))
        seed, status, start = solver_seed(c.index, rep, slot, stream), "ok", time_ns()
        try
          double_selection(data.y, d, C; method, rng = Xoshiro(seed))
        catch e
          e isa InterruptException && rethrow()
          status = exception_type(e)
        end
        push!(timings, (c.index, c.name, rep, slot, j, method, seed, status, (time_ns() - start) / 1e9))
      end
    end
  end
  rows, timings
end

function warmup()
  # Non-study case zero compiles both post modes, all final stages and timings.
  c = Case(0, "warmup", 60, 12, 0.3, 3, 1.0, "fixed", "normal", "standard", NaN, NaN)
  run_rep(c, 1)
  nothing
end

function readrecords(path)
  a, h = readdlm(path, ',', String; header = true)
  vec(h) == RECORD_HEADER || error("Invalid record header: $path")
  f(x) = parse(Float64, x)
  flag(x) = isempty(x) ? nothing : x == "true" ? true : x == "false" ? false : error("Invalid flag $x")
  [Record(case = parse(Int, r[1]), case_name = r[2], rep = parse(Int, r[3]), slot = parse(Int, r[4]),
    target = parse(Int, r[5]), truth = f(r[6]), role = r[7], arm = r[8], fit_status = r[9], estimate = f(r[10]),
    se = f.(r[11:16]), errors = r[17:22], s_y = f(r[23]), s_d = f(r[24]), s_union = f(r[25]),
    rank = f(r[26]), dof_residual = f(r[27]), target_variation = f(r[28]), max_leverage = f(r[29]),
    converged_y = flag(r[30]), solver_converged_y = flag(r[31]), converged_d = flag(r[32]),
    solver_converged_d = flag(r[33]), nuisance_seconds = f(r[34]), final_seconds = f(r[35])) for r in eachrow(a)]
end

valid(r, q) = r.fit_status == "ok" && isfinite(r.estimate) && isfinite(r.se[q]) && r.se[q] > 0
converged(r) = all(x -> x === true, (r.converged_y, r.solver_converged_y, r.converged_d, r.solver_converged_d))
covered(r, q) = abs(r.estimate - r.truth) <= Z95 * r.se[q]
rejected(r, q) = abs(r.estimate) > Z95 * r.se[q]

end
