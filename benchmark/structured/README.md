# Structured regression validation

All timings and allocation measurements run on Discovery under Slurm.
`setup.sbatch` installs `genlasso` 1.6.1 and `glmgen` 0.0.3 (commit
`9e6061503d6dd103712a0c07f6be7293c1c58aaf`) into a task-specific R library.
`evaluate.sbatch` expects immutable snapshots in `source`, `base`, `solvers`
and `kernels`; adjust those four paths to the snapshots of the run.

The independent R fixture generator `golden.R` uses seeds 8101–8103. It
computes full `genlasso` paths, then evaluates them at four specified levels,
and computes `glmgen::trendfilter` at those same levels. Cases include fused
regression, graph cycles, and trend orders 0–3 on both unit and irregular
locations. `check_golden.jl` checks coefficients and degrees of freedom against
the exact paths and also checks the `glmgen` solutions against `genlasso`.
The loss convention is `lambda_R = n * lambda_Julia`.

The performance script `run.jl` covers signal lengths 1,000, 10,000, 100,000
and 1,000,000 and trend orders 0, 1 and 2. Each of the three independent seeds
has three measured rounds after compilation on a separate short signal.
It records wall time, allocated bytes, convergence, iterations, objective,
both ADMM residuals and degrees of freedom. The number of measured solves is
108. The tolerances are `abstol=1e-6`, `reltol=1e-5`, with at most 5,000
iterations; the regularization level is `1/n`.

`regression.jl` additionally measures fused regression, graph regression and
trend regression at `(n,p)=(300,80)` and `(1000,200)`, as individual solves
and four-level warm paths. It uses the same three seeds and three measured
rounds (108 measurements), with `abstol=1e-7`, `reltol=1e-6`.

Use GNU Parallel with three exclusive one-CPU `srun` steps, as in the launcher.
Set BLAS/OpenMP and Julia threads to one. The task requests 3 CPUs and 12 GB
for evaluation; setup requests 1 CPU and 4 GB. Scheduler completion and
numerical validation are recorded separately in the project run registry.
Retain final TSV files, golden fixtures, logs and the source/environment
references. No intermediate fits or whole-depot archives are needed.

The checked result tables, including the adverse initial regression attempt,
are in `recorded/`. `RESULTS.md` reports the retained measurements and the
default-parameter repair. Run `summarize.py` inside the Discovery allocation
from a run root (or the `recorded` directory) to check coverage and regenerate
`summary.json`; it retains the original fused measurements and uses the
repaired graph/trend measurements.
