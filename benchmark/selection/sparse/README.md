# Native sparse SLOPE: Discovery validation and benchmark protocol

This follow-up preserves CSC storage for `slope` and `SLOPEPath`, including
implicit centering and standardization. The estimator calls the screened SLOPE
APG engine from ProximalSolvers commit
`3bee2f91604fe08d34a07d45bcc638e3b4face3a` through the temporary companion
exception in the master bootstrap. The completed broad benchmark uses
the numerically identical SLOPE engine at `335ea905da207a6cc1cd3bd945bb7bbbf9f4eba4`;
the [report](REPORT.md) records the source comparison and current-dependency
integration tests. Statistical penalties and defaults
are unchanged. The old comparator is the exact `src/slope.jl` at HDM commit
`95993030f5fa34b01364dcdf9eebf42a4eab15ba`, retained as `baseline_slope.jl`.

All generation, fits, profiling and numerical summaries run under Slurm
on Discovery, along with the recorded tests. GitHub-hosted CI also runs package
tests as explicitly authorized. No Julia computation runs on the workstation. Existing pinned
Julia/R environments are reused; the final manifests and source receipts are
retained with results. Original and failed pilot results remain available.

## Fixed comparisons

Seven independent processes per backend, identical saved inputs and factors,
three timed batches per case. Import/JIT warm-up, certificates, input/output and
profiling are outside timing. A second warm call calibrates batches to roughly
100 ms, capped at twenty fits. Each process contributes its median batch time;
reported runtime ratios are paired by replication. Backend order rotates and
reverses between replications. All timed workers are serialized with one Julia,
BLAS and OpenMP thread. Profiles of native original-sparse and large-centered
paths are collected separately in every replication.

The original four designs are reused exactly from the preceding campaign:
600x200 dense AR(.3), 200x1000 wide AR(.3), 400x400 AR(.8), and 600x1000
sparse density .02. They have twenty levels down to .2 of the null threshold.
Additional cases cover densities .001/.01/.1; nonzero-mean, centered/scaled
1000x3000 inputs; a fifty-level path down to .02; a 1500x6000 centered design
at density .002; and a 10000x20000 storage canary at density .0001 with ten
levels. Cases 8 and 9 use identical X/y. The original comparator is deliberately
omitted from the 1.6 GB dense-equivalent canary to bound resource use. Input
storage, dimensions, flags and exact densities are recorded in `inputs/cases.csv`.

Native and baseline fits use relative full proximal residual tolerance 1e-7,
SLOPE.jl internal tolerance 1e-9, and R SLOPE auto tolerance 1e-9. These different
internal settings are calibrated to the same independently recomputed relative
proximal certificate <=1e-7, including intercept stationarity. R tolerance1e-8
failed the long-path pilot at 1.2153e-7 and is retained as a failed calibration.
SLOPE.jl tolerance1e-8 also failed the .001-density case at2.7851e-7;
its tighter1e-9 setting is calibrated before resuming. Every path level is checked in original input units, with independent column
moments, scales and matrix products; objectives and predictions are compared.
No reduced-fit stopping flag substitutes for the full certificate.

## Memory and acceptance

Fresh processes measure peak resident memory for cases 4, 10 and 11, seven
replications per backend/size (baseline omitted at case11). The final comparison
uses `memory_fit_only.sbatch`: 77 standalone fitting processes, all independently
certified after their measured process exits. SLOPE.jl's worker imports only its
own backend and the input/output dependencies. Two warm calls and one retained
fit match the original R worker. Python's Linux
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` includes the interpreter,
imports/JIT and child backend, and is reported in MiB. It is not the transformed
matrix's size, nor an exact incremental fitting-memory measure. Memory workers
may run concurrently in up to four separate 4 GB Slurm steps; their elapsed times are not
used for performance rankings. Julia heap allocation totals exclude C++/Fortran
allocation and are reported separately. The earlier 77-process harness-RSS
cohort included Julia certificate/operator-summary work and an HDM/CD import
for SLOPE.jl; its records remain as historical evidence but are superseded for
RSS comparisons. The corrected cohort uses debug/Xeon4116/d05-41; the timing
cohort remains EPYC7313/b11-09. Initial precompilation can inflate maxima; no
replications are trimmed. Owned operator and solver-workspace
storage is measured directly on the final implementation.

Targets: independently certified residual<=1e-7; storage O(nnz+n+p), excluding
the requested coefficient path; no reproducible dense slowdown above about5%;
and at least parity with SLOPE.jl on the original sparse case. Ratios and their
replication ranges are retained even when a target fails. One fixed dataset per
case measures process/runtime variability, not population-wide performance.
The large canary's dense-equivalent size is a reference, not a measured baseline
allocation. Changing storage does not justify rerunning the full FDR campaign.

## Discovery queue and hardware

The full-suite and calibration pilots used main/EPYC7513. The final repeated
campaign uses Discovery's debug partition on b11-09/EPYC7313, after normal QoS
hit its account concurrent-job cap. This is a bounded development test, under
the debug partition's one-hour limit. Job QoS remains normal; the partition's
own debug QoS supplies its separate short-job limit. Only this task's pending
jobs and the coordinated PS guard/extraction jobs were moved. All four backends
and seven repetitions run on the same allocated node. Compare within-campaign
paired ratios; do not combine the earlier7513 absolute times with the7313 table.

## Recreation

Use the task-scoped SSH and lightweight registry described by `slurm-workflow`.
The registry is `.slurm/runs/20261025-cd82-sparse.json` in the project parent.
Remote root: `/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse`.
Runtime: Julia1.10.12; full suite also Julia1.13.2; R4.5.3 container with the
previous campaign's SLOPE2.1.1 library and SLOPE.jl1.3.1. Source snapshots,
checksums and final manifests identify exact runtime dependencies.

For the reported broad cohort, restore harness commit
`2af17278c2f30337f89ce4d2cb293b45f0ab67fa` and the recorded snapshots/manifest;
it predates the Julia module rename. Current active launchers import
`HighDimensionalModels`; retained receipts keep the original module names.
Stage the recorded source snapshots and reuse the recorded environments. The
final launch uses `sbatch --partition=debug --constraint=epyc-7313 campaign.sbatch`;
these flags override the original main/7513 launcher defaults. Run
`generate.jl` under Slurm once; `campaign.sbatch` then dispatches seven timing
replications, seventy-seven fresh memory processes and seven independent audits
using GNU Parallel with exclusive `srun` steps. `summarize.py` consumes retained
records without refitting. `fulltests.jl` tests a copy whose [sources] entries
point to those exact runtime dependencies, avoiding mutable-master drift in
Pkg.test's sandbox. `compatibility_validation.sbatch` separately tests the
final PS36/PB0e98 integration, using any debug CPU because this stage does not
produce comparative performance measurements. The registry preserves actual job IDs, failures, source
overlays and aggregate resource accounting.

For the corrected RSS cohort, create `rss-fit-only/env` from the frozen
`env335` Project/Manifest and add SpecialFunctions as a direct dependency in
that Project (it is already present in the same manifest). Keep all existing
resolved source paths and versions. Link `rss-fit-only/inputs` to the saved
inputs and stage the four `memory_fit_*` files in `rss-fit-only/scripts`.
With `RUN` set to the original remote root, submit `memory_fit_only.sbatch`.
It sets `BENCHMARK_MODULE=CoordinateDescent` for the frozen pre-rename runtime,
launches 77 fresh measurement processes in four exclusive one-CPU/2GB steps,
then runs seven independent audits in separate processes. Summaries are made
only after all audits pass. The final source, environment, commands and
compressed worker receipts are in
[`fit-only-memory`](discovery-20261025/fit-only-memory/). No measured process
launches its verifier, so a verifier cannot contaminate child high-water RSS.
