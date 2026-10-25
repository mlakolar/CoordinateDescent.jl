# Native sparse SLOPE: Discovery validation and benchmark protocol

This follow-up preserves CSC storage for `slope` and `SLOPEPath`, including
implicit centering and standardization. The estimator calls the screened SLOPE
APG engine from ProximalSolvers commit
`ffdc26c1a20703da7afe8c0e7755e1768c07f2b5`. Statistical penalties and defaults
are unchanged. The old comparator is the exact `src/slope.jl` at HDM commit
`95993030f5fa34b01364dcdf9eebf42a4eab15ba`, retained as `baseline_slope.jl`.

All tests, generation, fits, profiling and numerical summaries run under Slurm
on Discovery. No Julia computation runs on the workstation. Existing pinned
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
SLOPE.jl internal tolerance 1e-8, and R SLOPE auto tolerance 1e-9. These different
internal settings are calibrated to the same independently recomputed relative
proximal certificate <=1e-7, including intercept stationarity. R tolerance1e-8
failed the long-path pilot at 1.2153e-7 and is retained as a failed calibration.
Every path level is checked in original input units, with independent column
moments, scales and matrix products; objectives and predictions are compared.
No reduced-fit stopping flag substitutes for the full certificate.

## Memory and acceptance

Fresh processes measure peak resident memory for cases 4, 10 and 11, seven
replications per backend/size (baseline omitted at case11). Python's Linux
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` includes the interpreter,
imports/JIT and child backend, and is reported in MiB. It is not the transformed
matrix's size, nor an exact incremental fitting-memory measure. Memory workers
may run concurrently in up to four separate 4 GB Slurm steps; their elapsed times are not
used for performance rankings. Julia heap allocation totals exclude C++/Fortran
allocation and are reported separately. Owned operator and solver-workspace
storage is measured directly on the final implementation.

Targets: independently certified residual<=1e-7; storage O(nnz+n+p), excluding
the requested coefficient path; no reproducible dense slowdown above about5%;
and at least parity with SLOPE.jl on the original sparse case. Ratios and their
replication ranges are retained even when a target fails. One fixed dataset per
case measures process/runtime variability, not population-wide performance.
The large canary's dense-equivalent size is a reference, not a measured baseline
allocation. Changing storage does not justify rerunning the full FDR campaign.

## Recreation

Use the task-scoped SSH and lightweight registry described by `slurm-workflow`.
The registry is `.slurm/runs/20261025-cd82-sparse.json` in the project parent.
Remote root: `/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse`.
Runtime: Julia1.10.12; full suite also Julia1.13.2; R4.5.3 container with the
previous campaign's SLOPE2.1.1 library and SLOPE.jl1.3.1. Source snapshots,
checksums and final manifests identify exact runtime dependencies.

Stage the recorded source snapshots and reuse the recorded environments. Run
`generate.jl` under Slurm once; `campaign.sbatch` then dispatches seven timing
replications, seventy-seven fresh memory processes and seven independent audits
using GNU Parallel with exclusive `srun` steps. `summarize.py` consumes retained
records without refitting. `fulltests.jl` tests a copy whose [sources] entries
point to those exact runtime dependencies, avoiding mutable-master drift in
Pkg.test's sandbox. The registry preserves actual job IDs, failures, source
overlays and aggregate resource accounting.
