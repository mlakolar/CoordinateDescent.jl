# Structured stopping accuracy: issue #97

This is the follow-up to PR #83's original comparison and linear-signal
calibration. The source change sets solver-family-specific Float64 defaults
and checks the fused proximal-gradient residual at the returned coefficients
against the original design. It does not extend production objective-gap
stopping beyond the previously calibrated linear signals.

The defaults were selected from the retained 8201–8203 comparison, before
new seeds 8501–8503 were evaluated. `recorded/selection.json` records that
choice. Baseline source is `b570922`; candidate solver/test source is
`7bf2e2e`. The final suite passes 882 checks on each of Julia 1.13.2 and 1.10.12
in Discovery allocations with frozen master dependency checkouts.

## Design

There are 15 workloads: quadratic signals of length 1,000/10,000 at R penalties
0.1/1/10, irregular 10,000-point quadratic fits at the same penalties, 50-level
quadratic paths at lengths 1,000/10,000, and correlated p=100 fused/graph
regression single fits and paths. Every path column is checked.

Each source has three warmed end-to-end timings on each of three independent
datasets. Input loading, compilation, independent references and accuracy
checks are outside timing. Both sources and all three datasets share one
allocation; source order alternates by dataset. Julia, R, BLAS and OpenMP use
one thread per worker. Data generation, fitting and validation run only under
Discovery Slurm. GitHub CI validates the canonical package on its usual matrix.

R supplies untimed `glmgen` fits with `obj_tol=1e-12` for signals and specialized
`genlasso::fusedlasso` references for regression. A separate box-constrained
dual QP supplies a feasible lower bound for the least-penalized regression
level. Feasibility holds after clipping regardless of the QP termination code.
The common acceptance targets remain relative coefficient error <=1e-4 and
relative objective difference <=1e-6; convergence and complete coverage are
also required. A certificate for the objective does not uniformly certify
relative coefficient precision for ill-conditioned/nonidentifiable designs.

## Recreation

The source and dependency commit IDs, environment manifests, source selection,
job accounting and compact outputs are retained with the results. Large input
and coefficient matrices remain in the remote run directory:

`/home1/mkolar/juliaPkg/runs/20261025-cd97-defaults`

Use new task-owned source/dependency folders, following the repository's master
checkout convention, and the recorded revisions when recreating these frozen
measurements. Do not reset existing sibling checkouts or mutate the earlier
campaigns. The source snapshots contain `Project.toml`, `src/`, `ext/`, `test/`
and `scripts/`; the master bootstrap also needs Git metadata in each source
folder. The run's registry records the source archive hashes.

1. Stage the baseline and candidate as `HighDimensionalModels-baseline` and
   `HighDimensionalModels.jl`, beside the recorded master dependency checkouts.
2. Submit `prepare.sbatch`: it resolves separate release/LTS test environments,
   runs structured/independent golden/Aqua checks, and creates a baseline
   environment using exactly the same release dependencies.
3. Submit `campaign.sbatch` only after those checks pass. It loads GNU Parallel
   and Apptainer, and uses exclusive one-CPU steps for seeds 8501–8503.
   `generate.R` creates shared inputs; `reference.R` computes untimed references;
   `measure.jl` records each actual source's fits, errors and residuals.
4. Rebuild the summary without fitting anything:

   ```sh
   python3 benchmark/structured/accuracy/summarize.py \
     benchmark/structured/accuracy/recorded /tmp/cd97-summary
   ```

The final six Float32 regression checks are staged with the updated
`test/structured.jl` as `harness-final/structured-final.jl`; `final.sbatch`
runs the complete 882-check suite on both existing environments. It is queued
after the holdout campaign, with a two-minute limit. The numeric solver source
is identical to the candidate measured in the campaign. Published measurement
helpers use snake_case names; provenance records this identifier-only change
from the frozen run harness.

The project registry is `.slurm/runs/20261025-cd97-defaults.json` in the enclosing
juliaPkg workspace. This run retains the existing task cap (3 CPUs/12 GB) and
the remaining allowance; it does not reset compute accounting. The first setup
attempt lacked source Git metadata and the first holdout launcher omitted the
Apptainer module. Both failed before fitting; their logs and allocation time
remain recorded. No cleanup of other runs is part of this work.
