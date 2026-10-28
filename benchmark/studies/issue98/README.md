# Trend-filtering study for issue #98

This driver measures warmed public calls before and after a package change.
It records time, Julia heap allocation, convergence diagnostics, coefficients,
peak memory and diagnostic CPU and allocation profiles. Independent reference
records bound coefficient error for the normalized objective
`sum(abs2, y - beta)/(2n) + lambda * sum(abs, D * beta)`.
Convergence flags alone do not establish accuracy.

Keep one fixed copy of this driver and its harness includes for both revisions.
Prepare each environment with `benchmark/setup.jl`, using the base environment
as the reference for the head environment to hold other dependencies fixed.
Use the same Julia version and one Julia thread; the driver sets one BLAS
thread. Each `fit` needs a fresh process. Interleave base and head for each
case and dataset, alternating which side runs first across repetitions.
Keep generated records outside Git.

```sh
study=benchmark/studies/issue98/study.jl
JULIA_NUM_THREADS=1 julia --project=ENV "$study" list
JULIA_NUM_THREADS=1 julia --project=ENV "$study" inputs linear-1k 1 OUT
JULIA_NUM_THREADS=1 julia --project=ENV_BASE "$study" fit linear-1k 1 base OUT
JULIA_NUM_THREADS=1 julia --project=ENV_HEAD "$study" fit linear-1k 1 head OUT
JULIA_NUM_THREADS=1 julia --project=ENV "$study" reference linear-1k 1 OUT
# Optional alternative, using a separate output root:
JULIA_NUM_THREADS=1 julia --project=ENV "$study" reference linear-1k 1 OUT_TIGHT admm-tight
```

`REP` is 1–10. `SIDE` is a simple label such as `base` or `head`.
`list` prints `id kind data_key order n element_type levels rounds` as TSV.
The ten cases include linear and quadratic signals, irregular locations,
order-zero TV paths, Float32 inputs and serial five-fold cross-validation.
Single fits use the fixed R-scale penalties in the case table in `study.jl`.
Paths use 50 logarithmically spaced R-scale levels from 10 to 0.1; CV uses
10 levels. Every passed penalty is `lambda_R/n`. Float32 inputs and penalties
are rounded to Float32 before fitting. Regular cases pass no locations and
therefore fit at the default unit spacing. The normalized `t` shapes only
the signal. CV passes `folds = mod1.(1:n, 5)` and `threads = false`.
All other fitting options retain package defaults.

Set `ISSUE98_SCALE=smoke` for short checks: every size becomes
`max(200, n ÷ 100)`, with at most five levels and two rounds. Both profiles
are still written. Use a separate output root for each scale. Shared data
keys use the same draw in every case, including Float32 cases, which convert
the shared Float64 input. Draws use `scenario_rng(data_key, REP)` and draw
response noise before irregular spacings.

## Records

All files are written through temporary files and renamed. `DONE` is written
last. Complete fit and reference directories cannot be overwritten; incomplete
ones may be retried. Complete shared inputs are checked and reused. Inputs
are regenerated before fitting or reference computation, and both regenerated
hashes and binary-file hashes must match `spec.toml`. Existing case specs and
penalty tables must also match. A changed input, scale or case definition is
an error. Completion marks a complete record, including adverse numerical
results; it does not imply convergence or reference acceptance.

Binary vectors are raw little-endian Float64 with no header. Level numbers
start at one. Float32 coefficients are converted exactly to Float64. TSV
files have headers, integer nanoseconds and bytes, round-trip decimal floats,
and empty fields when a Julia version lacks a timing field. Profile frames
use only source basenames.

| Relative location | Contents |
|---|---|
| `inputs/<data_key>/rep<NN>/y.f64`, `x.f64` | Shared response and optional irregular locations. |
| `inputs/<data_key>/rep<NN>/spec.toml` | Data key, repetition, size, irregular flag, generator formula and SHA-256 of each binary. |
| `cases/<case>/rep<NN>/levels.tsv` | `level lambda_R lambda`; lambda is exactly the passed value, including Float32 rounding. |
| `cases/<case>/rep<NN>/case.toml` | Id, call, order, size, element type, data key, rounds, level count, repetition, scale and explicit keywords. |
| `<SIDE>/rounds.tsv` | `round time_ns bytes gc_ns compile_ns recompile_ns iterations_total converged_all coef_sha256`. |
| `<SIDE>/memory.tsv` | `maxrss_bytes`: fresh-process peak after compilation and one warmed fit, before profiling or certification. |
| `<SIDE>/levels.tsv` | `level lambda iterations converged objective df primal_residual dual_residual` from the first timed call. |
| `<SIDE>/coef-L<k>.f64` | First timed call's coefficient vector at each level. |
| `<SIDE>/cv.tsv` | CV only: `level lambda cvm cvse`. |
| `<SIDE>/profile-cpu.txt` | Sampling delay, actual total sample count and complete flat Julia profile (`C = false`), with a wide display. |
| `<SIDE>/profile-alloc.txt` | Sampling rate, sampled allocation count and bytes; top 30 sites by sampled bytes, grouped by type, size and first five Julia frames. |
| `<SIDE>/meta.toml` | Julia version; case spec; side, scale and rounds; Julia and BLAS threads, BLAS vendor, CPU model and word size; package version, harness-compatible SHA-1 `package_hash`, commit and dirty flag; ProximalBase and ProximalSolvers versions. CV adds `lambda_min` and `lambda_1se`. |
| `reference/beta-L<k>.f64`, `reference/u-L<k>.f64` | Reference coefficients and feasible box-constrained dual vector. |
| `reference/reference.tsv` | `level method iterations stopped_by seconds primal dual gap_relative bound_relative`. |
| `reference/meta.toml` | Method, case, coefficient-bound target, per-level acceptance and dual stop-test flags, dual iteration limit and tight-path time. |

`<SIDE>/` and `reference/` are under `cases/<case>/rep<NN>/`.
The coefficient SHA-256 concatenates all levels as little-endian Float64.
For CV, coefficients, iteration totals and convergence flags describe the
returned full-data path; timings include all folds. `lambda_min` and
`lambda_1se` are the documented selected penalties.

A small fit compiles the same call, order, location type, element type and
level count before measurement. Each measured call receives fresh copies;
copying, hashing and writing are outside timing. `@timed` includes any
remaining compilation, also recorded separately when available. Peak memory
includes preparation and compilation and is process-wide. It is not a
measurement of one fit's incremental memory.

CPU profiling performs one additional call, targeting about 20,000 samples
using the first timed call's duration. Short calls and timer resolution can
yield fewer samples; the actual count is recorded, with an explicit note if
the call finishes before any CPU sample is captured. Allocation profiling
performs one additional call with a rate at most 0.01, reduced further for
large measured heap traffic. Sampled bytes are not total allocated bytes.
Profiles are diagnostic and are not timing observations.

## Reference interpretation

`dual-active-set` uses the projected-Newton box least-squares algorithm from
the harness's `box_lsq`, copied with iteration and stopping diagnostics.
It solves `min_u norm(D'u - y/n)^2/2` with `abs.(u) <= lambda`, allowing
5,000 iterations, then sets `beta_ref = y - n*D'u`. The difference matrix
comes from public `difference_matrix(n, order + 1; x)`; order zero thus uses
first differences. All reference arithmetic is Float64. For Float32 cases,
it uses the Float32-rounded response and penalties converted to Float64.
CV references use the full-data levels.

`stopped_by` distinguishes `objective_floor`, `projected_gradient`,
`no_progress`, `zero_operator`, `maxiter`, `factorization_failed` and
`line_search_failed`. `dual_stopped_by_test` identifies the first four;
stagnation is not an accuracy certificate. `iterations` counts attempted
Newton steps, with zero if a pre-step convergence test fires.

`admm-tight` calls public `trend_filter` or `trend_filter_path` with
`abstol = 0`, `reltol = 1e-12` and `max_iter = 10^6`. Order-zero paths forward
these options to fused lasso. CV uses a full-data path without running folds.
The dual certificate solves `box_lsq(D, (y - beta)/n, lambda)` with the same
diagnostic adapter. Its stop text records both ADMM and dual termination;
iterations count ADMM iterations. For paths, `seconds` apportions total
tight-path time equally over levels and adds each dual solve's time; total
path time is also recorded separately.

Both methods clip the dual vector to its box before evaluating
`dual = u'Dy - (n/2)*norm(D'u)^2`. They report
`gap_relative = max(primal - dual, 0)/max(abs(primal), floatmin(Float64))`
and the strong-convexity coefficient bound
`bound_relative = sqrt(2n*max(primal - dual, 0))/norm(beta_ref)`.
Acceptance requires a finite bound at most `1e-6`, irrespective of stopping
flags. Nonfinite values, zero reference norm or a material weak-duality
violation are rejected. These are certificates evaluated in floating-point
arithmetic, without interval rounding. Retain failed bounds and unconverged
fits when comparing revisions.
