# Selection procedures: R checks and Discovery campaign

All benchmarks and profiles run under Slurm on USC Discovery. The campaign
uses account `mkolar_1314`, partition `main`, constraint `epyc-7513`, Julia
1.13.2, two Julia threads and one BLAS/OpenMP thread per worker. Two workers
run with GNU Parallel and exclusive `srun` steps in a 4-CPU, 12-GB allocation.
Discovery serialized the outstanding `srun` benchmark steps under the allocation's
step-memory semantics. The retained measurements therefore have one active
benchmark process at a time; each threaded stability fit still uses two Julia
threads. The regression job requests 2 CPUs and 4 GB, so the combined request is
6 CPUs and 16 GB. Setup and all retries count against the task's standing
8-CPU/16-GB/two-hour allowance.

`setup.jl` instantiates this environment. Correctness and documentation jobs use
a task-local compiled cache; `cache_inputs.sh` reuses shared installed package
sources and artifacts without sharing their mutable compiled-cache directory. The retained Manifest identifies the
exact ProximalBase and ProximalSolvers trees; no dependency checkout is edited.
`setup.R` installs only missing R references in a run-local library, reusing
Discovery's existing glmnet library. R runs in the site's R 4.5.3 Apptainer
image. `golden.R` saves shared inputs, Gaussian normal draws, complementary
pairs, reference outputs, and R session information. `golden.jl` checks them
with Julia. The golden fixtures are retained in `test/data/selection` and
checked by the normal package test suite without requiring R.

The comparisons cover:

- R SLOPE 2.1.1: four levels with and without an intercept and standardization;
  without an intercept, explicit RMS scaling matches Julia's convention
  (R's built-in SD scaling still centers the scale calculation);
- R knockoff 0.3.6: the equicorrelated diagonal, Gaussian draws with shared
  standard normals, fixed-X Gram/cross-Gram identities (the orthogonal basis
  is implementation-dependent), glmnet coefficient differences and knockoff+;
- R stabs 0.7.1: complementary-pairs selection frequencies, selected columns,
  and its `assumption="none"` bound, with a shared fixed-penalty capped lasso
  base selector. stabs accepts one half per pair and makes its complement.

From a staged checkout and a task-owned run directory:

```sh
export RUN=/home1/mkolar/juliaPkg/runs/UNIQUE_RUN_ID
# Stage source under $RUN/source; keep the source revision or relevant patch.
cd "$RUN"
sbatch --parsable --export=ALL,RUN="$RUN" source/benchmark/selection/prepare.sbatch
# After setup, run one representative replication plus the golden/unit checks.
sbatch --parsable --export=ALL,RUN="$RUN",FIRST_REP=1,LAST_REP=1 source/benchmark/selection/campaign.sbatch
# After the canary passes, run only the remaining six replications.
sbatch --parsable --export=ALL,RUN="$RUN",FIRST_REP=2,LAST_REP=7 source/benchmark/selection/campaign.sbatch
sbatch --parsable --export=ALL,RUN="$RUN" source/benchmark/selection/test.sbatch
python3 source/benchmark/selection/summarize.py "$RUN/results"
```

Add a unique run tag to job names/comments and record submission intent and
returned IDs before monitoring or retrying. The scripts do not submit jobs
by themselves. Setup's unit check and R installation have separate outcomes;
a later failed test does not mean R installation failed.

Each of seven independent processes times nine workloads after warm-up and
profiles each workload for at least 0.2 seconds. Data and solver seeds are
fixed across timing replications, with order rotated. Raw seconds, allocation
bytes and GC time are saved, alongside 63 flat profiles. Internal stability
parallelism is compared with the serial fit using the same pairs and seeds.
The harness does not time R or claim superiority over the R packages.

Each process also runs 50 fresh data/noise seeds, for **350 Monte Carlo
repetitions per procedure**. The orthogonal, normalized Gaussian fixed-X and
Gaussian AR(1) design families come from the SLOPE/knockoff papers; the
bounded development dimensions (n=240, p=100, 20 signals) are not full
reproductions of those papers' figures. The script records the FDP, power,
number selected and number of false selections for each trial, including
empty selections. The summarizer checks coverage, duplicate seeds and the
recorded arithmetic, and reports Monte Carlo standard errors. CPSS's FDP is
descriptive; its theoretical bound concerns expected false selections only
under the documented low-selection-probability condition.

The lightweight run record for this implementation is
`.slurm/runs/20261025-cd82-selection.json` relative to the juliaPkg workspace
root; remote outputs are under
`/home1/mkolar/juliaPkg/runs/20261025-cd82-selection`. Final reports and raw
measurements are retained under `discovery-20261025/` here. No routine archive
or cleanup is part of this campaign.
