# Targeted linear-trend stopping calibration

This follow-up to `20261025-cd80-competitive` evaluates only Float64 signal
trend filtering of order one. General regression, other signal orders and
`rho` are held fixed. PR #83 remains draft. Execute fits and profiles only in
Discovery Slurm allocations, with one thread per worker.

The user approved 45 additional active allocation minutes, at most 3 CPUs and
12 GB. Run identity: `20261025-cd80-stopping`. The enclosing project's
`.slurm/runs/20261025-cd80-stopping.json` records submissions, attempts, source,
results and allocation union time. Queue time is excluded. This allowance is
additional to the earlier campaign and does not reset its accounting.

`diagnose.jl` follows the unchanged ADMM updates and records reference error,
objective, primal/dual residuals and a feasible primal-dual gap. Diagnostic
reference checks are outside benchmark timing. Reference inputs and fitted
coefficients are reused from the previous campaign on seeds 8201–8203.

For objective `P(beta)=||beta-y||²/2 + alpha*||D1*K*beta||₁`, any
`q` in `[-alpha,alpha]` is dual feasible, with `v=KᵀD1ᵀq` and
`dual=yᵀv-||v||²/2`. The diagnostic obtains `q` by clipping cumulative
ADMM multipliers. Clipping ensures feasibility even before convergence.
The gap can be evaluated without subtracting large objective values:

```
P(beta)-dual = ||beta-y+v||²/2
              + sum_j(alpha*abs((D1*K*beta)_j)-q_j*(D1*K*beta)_j).
```

Every summand is nonnegative in exact arithmetic. This certifies objective
suboptimality, not the coefficient-relative-error requirement; both are
checked against saved references. `experiment.jl` is a benchmark-only overlay
of the same update loop. It compares the original residual test, larger
residual thresholds and candidate gap thresholds with a residual guard.
It does not modify the package source. Setting selection uses only the
calibration datasets.

`calibrate.jl` measures end-to-end public fits and 50-level paths, including
DF/model construction, with three timing rounds per dataset. Setting order
rotates by dataset. Compilation is warmed on unrelated small inputs; input
reading and reference checks are outside timings. It preserves every
candidate's achieved errors and convergence status.

`holdout.R` generates new data using seeds 8301–8303 and frozen cases: lengths
1,000–100,000 at R lambda 0.1/1/10; irregular 10,000-point grids; 50-level paths
at 1,000 and 10,000 points; and a million-point fit at R lambda 1. The previous
selected glmgen controls are transported without a fresh sweep. A separate
1e-12 reference is computed outside timings. Failed transported settings
must be reported or repaired explicitly, not counted as speed wins.

A default change requires all selected cases/datasets to meet coefficient
relative error 1e-4, objective relative difference 1e-6, finite outputs and
solver convergence. Inspect hard cases separately. Useful speedup should
repeat on the new datasets; avoid attributing different achieved precision
to a language-level advantage. If no candidate is reliable, retain current
defaults and report that outcome. Source changes require release/LTS tests,
regression tests for stopping failures and normal GitHub CI.

The completed decision and timing tables are in [RESULTS.md](RESULTS.md).
The retained evidence is in [recorded/](recorded/README.md). No additional
default-accuracy change is implied outside the tested linear signal rule.

## Campaign stages and recreation

The Slurm recipes use explicit immutable folders in the run directory:

1. `harness/diagnose.sbatch`: original-update traces against saved references.
2. `harness/holdout-reference.sbatch`: new inputs and timed/tighter glmgen fits.
3. `harness-calibration-v2/calibrate.sbatch`: four fixed calibration rules,
   three timing rounds per dataset, independent dense certificate checks and
   three baseline CPU profiles. Pass the explicit harness directory as the
   third argument; Slurm spools the submission script elsewhere.
4. `harness-validation/validate.sbatch`: release/LTS tests and the two-case
   wrapper-warmup correction. It uses `calibrate-warmfix.jl`.
5. `harness-holdout/holdout.sbatch`: freezes the calibration selection, then
   runs actual old/new source in separate processes, with order alternating
   by dataset. `holdout-source.jl` explicitly warms the wrapper types and
   captures CPU/allocation profiles after timing.
6. `harness-final/final.sbatch`: revalidates the final source, whose general
   residual branch preserves the original short circuit, and repeats only
   candidate timings using `holdout-source-final.jl`.

The registry preserves exact submission arguments, dependencies and source
snapshot hashes. Respect aggregate resources; the stages were serialized.
`source-gap-v1`/`source-gap-v2` and the inherited dependency environments are
frozen run sources. The historical package name in these recipes is
`CoordinateDescent`. The PR subsequently integrates the canonical
`HighDimensionalModels` name, with a byte-identical calibrated solver core.
Use the frozen sources for exact recreation of these measurements. A current
canonical-package rerun must use its matching environment and imports.

To rebuild the final summary from compact records, without fitting a model:

```sh
python3 summarize.py recorded /tmp/cd83-stopping-summary
```

`calibration-warmfix` overrides exactly two calibration cases.
`holdout/julia-final` overrides candidate timings only. Baseline/R data and
the first actual-source comparison remain visible. The same-allocation table
is derived separately from `holdout/julia` before the final-source override.
All reported times are medians; the three timing rounds do not create three
additional independent datasets.
