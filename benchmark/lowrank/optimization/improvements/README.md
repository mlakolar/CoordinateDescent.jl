# Solver options and fixed-rank diagnostics

This follow-up measures the public PANOC and block FISTA options and compares
the fixed-rank API with its previous implementation. All numerical execution,
precompilation, tests, timing, profiling, and aggregation run in Discovery
Slurm allocations. MATLAB is excluded. The main queue reached its per-user
job limit; the user explicitly selected a fixed debug CPU model for the full
rerun. Every method and both source versions use Intel Xeon Silver 4116 CPUs.
Do not compare absolute times with the earlier EPYC 7513 report as if hardware
were unchanged.

The shared binary inputs are the original six designs and five independent
inputs per design. Each fit has three timing rounds. Ordinary timing uses four concurrent workers; the recovery allocation uses
eight workers for fresh-process memory only. All workers use one CPU
and one Julia/BLAS/OpenMP thread. Compilation, input loading, independent
certificates, factor replay, and profiles are excluded from fit timing.

The previous public API runs in the frozen original environment; the updated
API runs in a new environment pointing to the submitted package snapshot.
Neither environment points to a changing parent checkout. PANOC remains an
optional extension: load ProximalAlgorithms, then use `solver = :panoc`.
`block_solver = :fista` applies to row-sparse fits. Defaults remain APG and
coordinate descent; both choices propagate through paths and cross-validation.

`updated_worker.jl` loads the original comparison adapters from the parent
directory. It adds the production PANOC option and public sparse APIs to the
comparison, retaining the original generic Julia, Python, and R competitors.
Labels `cd_sparse` and `block_fista` are the original shared-initialization
kernel adapters; `cd_sparse_api` and `cd_sparse_fista` time the public APIs,
including their reduced-rank initialization. Before/after comparisons use
public API timings on both sides. Nonconvex objective differences remain in
the paired tables.

The fixed-rank path computes ranks 0–3; the nuclear path has eight descending
penalties. Independent coefficient checks cover saved cold fits and the
rank-path endpoint. Nuclear path rows certify the last point and retain the
conjunction of native convergence flags across all levels, rather than
independently certifying every intermediate point.

Core scalar certificate thresholds are unchanged: relative dual gap and
normalized proximal-gradient mapping <=1e-6 for nuclear fits; fitted loss and
feasible rank for fixed-rank fits; actual-factor KKT, response stationarity,
and orthogonality for sparse fits. The predictor-block FISTA certificate uses
the actual weights. The native PANOC backend uses its standard 0.95/L step
and ProximalBase's workspace-based nuclear prox; the earlier generic
ProximalOperators adapter retains its measured 1/L step. These are comparisons
of accurate implementations, not identical iteration trajectories.

Completed coverage is 7,110 timing rows, 430 text profiles, 230 fresh-process
memory observations, and 1,860 independent saved-fit checks (1,855 pass; the
five known R normal-equation failures remain). Every memory group has five
independent inputs and passed coverage, fit-certificate, and RSS checks.
An initial focused pass retained 24 complete observations before its wall
limit. The user approved 20 additional running minutes; the missing-only
continuation finished in 16 min 59 s with the same 8-CPU/16-GB maximum.
Profiling covers all five inputs on medium, many-response, and wide designs,
plus both adaptive-weight sparse methods on medium. The known normal-equation
R failure is retained; its QR alternative is included. Profile percentages
from Julia sampling, cProfile, and Rprof are not interchangeable.

Recreation in a new run root:

1. Restore the frozen package/environment references and stage these scripts
   in `code/`, together with parent `julia_worker.jl`, `common.py`,
   `python_worker.py`, and `r_worker.R`. Stage the package as `package/`.
   Use the original shared inputs read-only as `inputs/`; restore the pinned
   Python environment as `venv/`.
2. For a clean run on this slower CPU, submit `prep.sbatch` with a 20-minute
   time override. Historical launchers retain their measured time limits.
   Submit `prep.sbatch` for environment setup and full Julia 1.10/1.13 suites.
   The recorded first run incurred new-CPU precompilation and timed out;
   `retry_tests.sbatch` records the full-suite recovery after correcting a
   test's near-zero absolute tolerance. For clean reproduction, tests already
   contain the correction; retain whichever logs actually verify the sources.
3. After tests pass, use `CD81_ORDINARY_ONLY=1 sbatch --time=00:40:00
   --dependency=afterok:ID comparison.sbatch` for the ordinary comparison.
   Then submit `complete.sbatch` with a 30-minute override and
   `--dependency=afterany:COMPARISON_ID` to reuse successful receipts and
   finish isolated R components, adaptive profiles, checking, and memory.
   These are recreation estimates, not additional authorized submissions.
   Both launchers use exclusive single-core `srun` workers. Ordinary timings
   use concurrency four; memory uses eight, with two 8-GB R workers isolated
   from the ordinary steps.
4. The original submission passed `main` as a positional Python argument;
   the corrected launchers use `--phase main` and `--memory SOLVER`.
   `complete.sbatch` resumes successful receipts with `recover_tasks.py`
   and runs the remaining stages through `remaining.sh`.
   `resume_worker.jl` keeps flushed Julia timing rows and completed profiles
   if a worker reached the wall limit, using the same numerical kernels and
   measurement function. Original logs and partial files are retained.
5. `analyze.py` recomputes certificates from saved coefficients/factors,
   checks distinct observation keys and all five inputs, and writes paired
   summaries. `collect.py` retains compact reporting tables and bundles
   hundreds of small profile/diagnostic files for practical transfer.

The original cluster paths in these launchers are deliberate provenance;
adjust them for a new installation while keeping source and package versions
pinned. The run is `20261025-cd81-solvers`; the parent workspace registry is
`.slurm/runs/20261025-cd81-solvers.json`.

For the historical launcher aliases, stage `worker.sh`/`memory.sh` as both
their canonical names and `worker-v3.sh`/`memory-v3.sh`. Stage `analyze.py` as
`analyze-v3.py` and `analyze-v8.py`; `collect.py` as `collect-v3.py`,
`collect-v8.py`, `collect-v10.py`, and `collect-v11.py`; `adaptive.jl` as
`adaptive-v2.jl`; and `memory_summary.py` as `memory-summary-v2.py`. Keep the
canonical files too. `resume_worker.jl` and `remaining.sh` retain their names.
`collect.py` accepts absent historical recovery receipts in a clean run.
The checked-in launchers additionally record historical recovery jobs; source
archives in the run registry retain the exact earlier submitted versions.

`memory.sbatch` records the submitted 19-minute continuation for the full
memory pass. `resume_memory.py` reused the 24 complete observations, preserved
partial files, and selected only 206 missing steps. It neither reran core
timings/profiles nor recomputed their passed independent certificates. The
continuation is job 13425977; all 230 observations passed final checks.

For this continuation's aliases, stage `resume_memory.py` as
`resume_memory-v12.py`, `memory_summary.py` as `memory-summary-v12.py`, and
`collect.py` as `collect-memory-v12.py`. Exact staged sources are retained in
`memory-v12.tar` in the run registry directory and the report's source archive.
