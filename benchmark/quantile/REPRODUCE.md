# Reproducing the Discovery solver study

The [report](results/discovery-2026-10-25/REPORT.md) contains the interpretation. Everything under `results/discovery-2026-10-25/` is saved evidence; it is not an input directory to overwrite during a new run. Numerical work belongs in a fresh Slurm allocation and run directory, with the slurm-workflow registry. The recorded cluster account, partition and runtime paths are Discovery-specific.

## Source and environment

Stage these public Git revisions into the fresh run's `source/` directory:

|Package/source directory|Repository|Commit|
|---|---|---|
|CoordinateDescent.jl|mlakolar/HighDimensionalModels.jl (formerly CoordinateDescent.jl)|c408831e2b011e83fa427248bf074dc0fc91d870|
|ProximalBase.jl|mlakolar/ProximalBase.jl|0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a|
|ProximalSolvers.jl|mlakolar/ProximalOPT.jl|9041e236f3de8ffdc02103e5a505993463e00bb1|

Use `git archive` of each commit with prefix `source/<directory>/`, or equivalent clean checkouts. The benchmark imports the pinned CoordinateDescent module, independently of the current repository name. Copy this directory's top-level scripts and task TSVs into the fresh run. Copy the saved Project/Manifest if exact dependency resolution is required. Local path dependencies resolve under that run directory. Building/precompiling and all Julia/R commands must run under Slurm. Gurobi needs the site module/license; it is optional in the solver dispatch but some comparison launchers explicitly enable it.

`setup.sbatch` develops/instantiates the three sources, builds the optional Gurobi adapter, precompiles, validates and runs the pilot. `rsetup.sbatch` installs the R references in a task-local Rlib using the clean Apptainer environment. Pin the saved R versions for exact recreation; CRAN's current releases may differ. The Julia Manifest and each R session are saved with the results. The canonical scripts consolidate earlier repairs and use `QUANTILE_RUN_ROOT` or the submission directory. Their historical executed versions are separately retained.

## Stage sequence

Read launchers and verify each previous stage before submission. The commands below describe the sequence, not authorization to repeat the campaign or exceed a new task budget. Record each returned job ID and use Slurm dependencies. Sequential stages are sufficient; the original campaign used a few independent overlaps within16 GiB.

1. `setup.sbatch` and `rsetup.sbatch`: inspect validation and the pilot before proceeding. With Gurobi enabled the final validation suite has1650 assertions. `admm_study.jl` adds18 assertions in its stage.
2. `evaluate.sbatch`: main24 input workers and108 density measurements; four exclusive1 CPU/4 GiB srun workers through GNU Parallel across the two-node allocation.
3. `finalize.sbatch`: corrected R golden comparisons, main/pilot aggregation and28repeated profiles. `golden.R` consumes fixtures emitted by `validate.jl`.
4. `admm.sbatch`, `rbench.sbatch`, `rho.sbatch`, `split.sbatch`: fresh-input factorized ADMM, identical R/Julia inputs, separate-rho tuning/confirmation, and the initial4-second split comparison. Followups may be sequential. The R comparison needs the saved R environment and cleanenv container wrapper.
5. `report.sbatch`: reuses saved coefficients for approximation/Fenchel checks and prepares coverage/paired/profile summaries. Missing later-stage counts are expected at this point.
6. `basis.sbatch`, then `basis_recovery.sbatch`: original analytic-dual cohort, followed by a planned-cell recovery and matched split starts. The original widest split attempts OOMed at4 GiB; a fresh recreation may instead give every basis worker8 GiB, reduce concurrency to2 and allow adequate stage time, recording that change. `basis_recovery` fills absent or validation-interrupted cells and preserves completed scientific budget failures. It adds split initializers. The final target is96 timing observations across16groups and12 inputs, regardless of which cells needed recovery.
7. `reconcile_bounds.py .` and `report_inputs.py shared-bounds`: these are saved-output numerical analyses and should also run under Slurm. Check every final `coverage_complete` value, worker exit codes and Slurm accounting. Scheduler success alone is not sufficient. Keep the original raw bounds and the derived views.

Use a separate development run to extend shapes, lambda grids or tolerances. Timing summaries require all eight levels and external accuracy, but intentionally retain native non-convergence flags when the final numeric check passes. Entire6/6 groups with three independent inputs are the basis for comparative rankings.

## Historical executed sources and outcomes

`executed-sources/` expands the small immutable tar snapshots used during the campaign. `source-snapshot-sha256.json` records their original hashes. It contains historical source, including failed attempts, not a second supported API. Later snapshots overlay the earlier harness; they are not standalone environments.

|Slurm job(s)|Effective harness/snapshot layers|Outcome|
|---|---|---|
|13422781,13422782|harness-v1 and original launchers|Julia worker parse error after858 checks; R inherited-environment setup failed|
|13423092|clean R setup|R references installed|
|13423103|harness-v2|Repaired pilot and licensed solver canary completed|
|13423130|original R comparison|Failed oracle normalization; original CSV retained|
|13423142|harness-v3, main-scripts-v1|1650 checks;552 main timings and108 density timings completed|
|13423157|final-scripts-v1, profile-scripts-v1; corrected golden-v2.R|Six corrected R golden cells and28 CPU traces completed|
|13423180|admm-scripts-v1, pre-run-metadata-v2|72 ADMM comparisons completed|
|13423207|rbench-scripts-v1, pre-run-metadata-v2, rbench-tolerance-v3|90matched R/Julia comparisons completed|
|13423468|rho-scripts-v2|Ten tuning paths and12 conditional confirmation paths completed|
|13423518|split-scripts-v1|54initial split observations completed|
|13423432|report-scripts-v1, accuracy-summaries-v3, report-inputs-v4|Saved-fit approximation and summaries completed|
|13423596|basis-scripts-v2 plus preceding summary layers|30 basis assertions passed; widest split workers OOMed; partial outputs retained|
|13423694|basis-recovery-v4|60 split assertions passed; missing observations and matched initializers completed; final96-cell coverage|

13423151 was cancelled before allocation and replaced by13423157. `accounting-final.tsv` is exact sacct output with UTC timestamps; `accounting-summary.json` includes failures in resource totals. The local task registry is `.slurm/runs/20261025-cd78-quantile.json` in the workspace containing this checkout. The remote record is `/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile`.

Original and recovered basis CSVs are separate; `basis-completed` merges them without silently retrying capped observations and adds memory/attempt fields. `shared-bounds` uses the strongest saved feasible dual for each identical study/input/lambda, preserves the original bound fields and records the contributing method/replicate. It does not alter coefficients, primal objectives or timings. Derived R metadata uses NaN for unmeasured allocations/iterations.

The transfer manifest verifies the small outputs against Discovery. Full R design/response binaries are retained remotely and deliberately omitted here; `rbench.R` recreates them from recorded seeds, and `rbench-input-sha256.json` includes their byte counts/hashes. Small fitted coefficient/lambda binaries and golden fixtures are included. Upstream source revisions replace the need to publish multi-megabyte source archives or an entire Julia depot. License-server setup logs are omitted.
