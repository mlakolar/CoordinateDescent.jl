# Replicated comparisons with glmgen and genlasso

The findings and limits are in [RESULTS.md](RESULTS.md). Raw timings, common
accuracy checks, candidate selection, profiles and source provenance are in
[`recorded/`](recorded/).

Run fitting, benchmarking and profiling only in Slurm allocations on Discovery.
The 2026-10-25 campaign is `20261025-cd80-competitive`; the project registry is
`juliaPkg/.slurm/runs/20261025-cd80-competitive.json`. Its source snapshots,
immutable harness versions, job IDs, environments and recovery records control
recreation. Do not replace source files underneath a running allocation.

`generate.R` generates three independent datasets (8201–8203) and a common case
table. Both languages read the same exported inputs. Each case has three measured
rounds per dataset. There are signal orders 0–2, lengths 1,000–100,000, R penalty
levels 0.1/1/10, irregular grids, and 50-level paths. Regression cases use 200×40
and 1,000×100 designs, with predictor correlations 0.5 and 0.95, sparse fusion,
cyclic graphs and trend penalties. Compilation/loading is warmed up separately;
input reading and accuracy checking are outside fit timings. Fitting includes
model construction and degrees of freedom. R batches fits below 20 ms to reduce
clock quantization. Thread counts are one for Julia, BLAS and OpenMP.
`million.R` creates a separate million-point extension for orders 0–2 at R λ=1,
using seeds 8201–8203 offset by 10,000. It never changes the main inputs.
It confirms the settings selected at length 100,000 and R λ=1 on new million-point
datasets, with three timings per dataset and a fresh tighter R reference. It does
not repeat the full tolerance search at a million observations.

Objectives are `RSS/(2n) + lambda_Julia * norm(D*beta,1)` with
`lambda_R = n*lambda_Julia`. No standardization or ridge changes the problem.
glmgen thinning is disabled. A supplied 50-level grid is returned by every path
comparison. genlasso internally computes path knots down to the smallest requested
penalty, so its timing includes that work; Julia and glmgen solve a supplied grid.
Single-fit genlasso timings also include the knot traversal required to reach λ.

The initial baseline uses Julia `abstol=1e-7, reltol=1e-6`; the paired baseline and
optimized measurements use `1e-8, 1e-7` respectively. Both use 20,000 iterations.
The cyclic-graph regression cases use `1e-9, 1e-8` in both paired Julia versions:
the initial correlated-design comparison needed an extra decade to meet the
coefficient agreement criterion.
The optimized public-default setting is measured separately as well. The primary
comparison selects the faster passing Julia setting per case, with the choice
recorded; fixed-tolerance baseline/optimized comparisons remain separate.
Measured glmgen uses `obj_tol` 1e-5 (default), 1e-8 and 1e-10, `max_iter=20000`,
with a separate untimed `obj_tol=1e-12` reference. The reported reference is the
fastest tested setting passing the common checks on all three datasets in that
case; tolerance calibration cost is excluded. These are different stopping tests, so matching their
numeric tolerance values is insufficient. `check.jl` checks every fitted grid
column against common inputs and objectives: relative coefficient error ≤1e-4
and relative objective difference ≤1e-6. These are cross-solver agreement checks,
not certified primal-dual optimality gaps. Report convergence flags, accuracy
failures and missing coverage separately; a fast failed case is not a speed win.

Signal genlasso comparisons use `trendfilter`. Regression comparisons select
the fastest passing result among generic default QR, generic `svd=TRUE` and the
recommended `fusedlasso`/`trendfilter` specializations. All candidates, including
failed agreement checks, remain recorded. `diagnose.R` recreates a cyclic-graph
path discrepancy; `dual-check.R` checks the final correlated sparse-fusion path
point. Both use independent bounded dual QPs in base R, outside fit timings.

`paired.sbatch` evaluates baseline and optimized Julia at the same tolerances in
one allocation, then profiles the optimized code over all three datasets.
`julia.jl` uses CPU sampling and separate sampled allocation profiles; profile
overhead is excluded from benchmark times. `summarize.py` reports paired ratios
from the within-dataset median runtimes and preserves failures.

The final source change only affects fused-regression paths and order-zero
signal paths. `gram.sbatch` repeats both precision settings for those four main
cases (72 measurements) and profiles the correlated fused path on three datasets
with ten fits per profile. All other v3 computations are unchanged and reused.
`qr.sbatch` adds the generic R default backend. `million.sbatch` confirms three
previously selected signal settings on new data. Main-case setting selection
uses the measured datasets and excludes calibration cost. Allocations are
nonexclusive and stages use fixed ordering; small timing differences are
tentative. Figure bars are dataset ranges, not confidence intervals.

Memory columns distinguish cumulative Julia allocations from process high-water
RSS. `worker_hwm_bytes` is the cumulative process high-water mark, including
runtime, inputs and earlier fits; it is not an isolated per-fit peak. Scheduler
step RSS is a second worker-level measurement. No R-versus-Julia allocation
comparison or per-fit peak-memory claim follows from these quantities.

To recreate using the recorded environments, stage this folder as `harness/` in
the run directory, resolve `worker.sh`'s old-run reference to the recorded frozen
dependencies, and submit `campaign.sbatch` with mode/output arguments. Run the
canary first for new execution paths. `profile.sbatch`, `special.sbatch`,
`validate-opt.sbatch` and `paired.sbatch` record the campaign stages; their source
and environment paths must refer to immutable recorded snapshots. Respect the
task's aggregate CPU, memory and active-time budget, including retries.
Run the optional `million.sbatch` extension only after checking the remaining
aggregate active-time allowance against its requested wall time.

The checked-in Slurm scripts are the campaign recipes and name immutable staged
folders (`harness`, `harness-qr`, `harness-gram`, `harness-verify`). When recreating
them, stage the appropriate checked-in scripts under those names and resolve
the explicit environment/source paths to the snapshots in the registry.
Do not submit all stages at once: dependencies and aggregate resources matter.

To recreate summaries from the checked-in records without executing a solver:

```sh
python3 summarize.py recorded/main /tmp/cd83-summary --accuracy-root recorded/accuracy
python3 summarize.py recorded/million /tmp/cd83-million-summary
```

The checked-in `summary.json` omits duplicated accuracy rows; the regenerated
full summary includes them. For million-point generation the replicate IDs
remain 8201–8203 in tables, while the actual RNG seeds are 18201–18203. Coefficient
files and large synthetic input matrices are retained remotely and intentionally
excluded from Git. `generate.R`/`million.R` recreate them. The source and harness
must be frozen before executing the fit/check stages on Discovery.
