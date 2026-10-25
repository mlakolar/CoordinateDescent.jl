# Trend product investigation: issue98

The candidate retains the sparse difference matrix for public use and
factorization. Inside trend ADMM it uses row stencils read from that matrix
for orders1/2, including irregular divided-difference weights. Other orders,
nonlocal/malformed matrices and unsupported matrix types keep sparse products.
The existing banded solve, exact TV prox, objectives, rho and stopping/DF
tolerances are unchanged. Regular stencils store constant coefficients;
irregular stencils store the actual per-row weights. Stencil width is a type
parameter to avoid boxing tuple values during construction.

Run all numerical work under Discovery Slurm with slurm-workflow. The owning
registry is `.slurm/runs/20261026-hdm98.json`, remote root
`/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue98`. Baseline295fbf5
and the candidate use the same pinned dependency sources and Julia1.13.2
environments as issue99; each source snapshot/environment is separate.
Do not replace sources beneath running jobs or reuse another task's SSH.

Targeted release/LTS checks compare both product directions with explicit
matrices, dual identities, nonfinite arithmetic, boundaries, Float32/64,
regular/irregular grids, sparse fallbacks, full structured fits/paths/CV and
R golden controls. Tests passed on the first source snapshot; a constructor
specialization is undergoing the same checks on the next snapshot.

`generate.jl` uses distinct saved seeds per case/replication and refuses to
overwrite inputs. Rep0 is calibration only. `julia_worker.jl` measures three
warmed complete public fits or50-level paths, including setup, output and DF,
then separate CPU/allocation profiles. Input loading, warming and independent
checking are outside timers. One fresh process handles each implementation
and input; memory includes the runtime, inputs, warming and profiling rather
than isolated solver workspace. `check_results.py` independently constructs
sparse divided differences and checks saved coefficients, objective, DF and
convergence, plus before/after agreement.

The first calibration tested small/irregular linear cases,100k linear and
quadratic signals, a nearly tied TV path and a linear path. It exposed an
unwarmed wrapper call in the first round and avoidable constructor allocations;
neither is an adoption result. The next immutable calibration uses the same
saved rep0 inputs, a width-specialized constructor and explicit actual-wrapper
warmup before all three rounds. Original results and failures remain retained.
Do not use calibration inputs as independent evaluation evidence. Saved profiles
then motivated explicit width2/3 transpose interiors with boundary handling
outside the loop. This source passed the same release/LTS checks and its
six-case calibration; all original results and controls are retained.

The frozen holdout uses9cases×3freshinputs (rep1–3), including million-point
linear/quadratic signals and the quadratic50-level path. `main.sbatch` uses
4CPUs/16GiB/90min with GNU Parallel/exclusive1CPU workers and rotated native
method order. `main-worker.sh` measures baseline/candidate in separate fresh
processes, then invokes the untimed independent tight glmgen reference in the
existing R4.5.3 container. The reference uses obj_tol1e-12, thinning disabled
and max_iter20000. This is an accuracy comparison, not a timing ranking
against a differently stopped R fit. `reference-canary.sbatch` first checks
the adapter on saved calibration data. The apptainer1.3.6 module is required.

Stage the current Julia benchmark script as `julia_worker-holdout.jl`, the
measured source as candidate3/ with env-candidate3-1.13, and the other checked-in
files under the owning run root. Preserve source/harness identities before
`sbatch --parsable main.sbatch`. Source and input hashes, per-level iterations,
CPU/thread metadata and all process outcomes are recorded. Profile intervals
are1ms, or5ms for million-point/quadratic-path fits, outside all timers.
`check_results.py` checks every native result and every reference column,
including convergence and exact shared bytes. `summarize.py` aggregates saved
results under Slurm. No evaluation outcome may be used to retune this source.

Before an adoption recommendation, freeze source/harness, generate at least
three fresh inputs per case, include million-point linear/quadratic signals
and the quadratic50-level path, and check against tighter independent glmgen
references. Require coefficient agreement<=1e-4, relative objective difference
<=1e-6, finite outputs and convergence, preserving failures and actual counts.
Report paired input ratios, three-round medians, profiles, allocations and
fresh-process baseline/peak RSS. Require a useful repeatable public-API gain
without broad regression before enabling the internal default. No tolerance
relaxation or TV-prox replacement is part of this candidate. Do not merge.

The frozen holdout is complete. [Final evidence](results/discovery-20261026/RESULTS.md)
records all27 inputs/81 processes and supports the automatic supported-order
product route. Keep all tolerances and fallback behavior. The unchanged TV
control has a reversed replication; the report preserves it. The conditional
deadline recovery was a no-op. Do not merge before human review.
