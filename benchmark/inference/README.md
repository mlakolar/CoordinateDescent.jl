# Issue 77: Discovery validation

All Julia execution (including precompilation, tests, benchmarks, profiling and
coverage calculations) runs inside Discovery Slurm allocations. No local Julia
measurements are used. Run `20261025-cd77-inference` is recorded in the workspace
registry `.slurm/runs/20261025-cd77-inference.json`; its raw outputs, source
snapshots, manifests and job launchers are retained alongside the record and at
`/home1/mkolar/juliaPkg/runs/20261025-cd77-inference` on Discovery.

The standing allowance is 8 CPUs, 16 GB RAM and 120 minutes of running-time
union across all task jobs and retries. Jobs are serialized with dependencies.
The measurement allocation uses GNU Parallel with exclusive `srun --exact`
steps: two workers of four CPUs for benchmarking, then eight workers of one CPU
for coverage. BLAS/OpenMP threads are one. Julia is 1.13.2, on EPYC 7513 nodes.

## Recreate

1. Stage this package at a pinned revision, with a run-local Julia environment
   developing it and compatible ProximalBase/ProximalSolvers snapshots. The
   original run records all dependency paths and saves the resolved manifest.
2. In a Slurm allocation, run `test/runtests.jl` with that environment. Build the
   documentation with `docs/make.jl` (skip deployment for validation).
3. Run `run.jl bench P REP` for P=1000,10000 and REP=1,2,3, using four Julia
   threads. `CD77_OUTPUT` selects the output directory. Each configuration has
   a full-size warm-up, one timed call and one separate sampling-profile call.
   At p=1000 both the shared Gram and residual methods run. At p=10000 the
   shared Gram method runs. The benchmark uses n=200, AR(1) correlation 0.3,
   penalty `0.75sqrt(2log(p)/n)`, and checks convergence, score normalization
   and agreement of timed/profiled estimates. CSVs record cumulative allocated
   bytes; they do not report peak resident memory. Text profiles use 1ms
   sampling and include all Julia threads.
4. Run `run.jl coverage ID` for ID=1,...,16 with one Julia thread each. These
   use the two covariance designs in Section 4 of van de Geer et al. (2014),
   n=100,p=500, support sizes 3/15, first/random supports and fixed uniform
   signals on [0,2]/[0,4]. Each fixed design/signal has 100 independent Gaussian
   error draws. The nodewise penalty is explicitly 0.1 on population-standardized
   columns; the initial fit is scaled lasso. This studies the paper's designs
   without reproducing its pooled-CV tuning rule. The first 20 draws in each
   scenario also test joint multiplier intervals/FWER with B=499. Seeds are
   fixed in the script; failed or nonconverged fits stop the worker.
5. Run `references.R OUTPUT` in the Discovery R 4.5.3 Apptainer container,
   followed by `check_references.jl OUTPUT` in the Julia allocation. Load
   `apptainer/1.3.6` first. R dependencies are installed in a run-local `Rlib`. The active CRAN endpoint did not provide hdi, so download the pinned source archives before running R:
   `https://codeload.github.com/cran/hdi/tar.gz/3eb71fb52b3c0f5213b1283d5f0317b91f6a133a`
   as `references/hdi-3eb71fb.tar.gz`, and
   `https://codeload.github.com/cran/linprog/tar.gz/f05bd8a1bb3c31d63f924ed2b9c6d582d9fbae52`
   as `references/linprog-f05bd8a.tar.gz`.
   The authors' original Javanmard--Montanari routine is downloaded from
   <https://web.stanford.edu/~montanar/sslasso/sslasso_code.tar.gz> to
   `references/sslasso_code/lasso_inference.r`. Its package-loading statements
   are omitted when sourcing the numerical routines; the QP routine is unchanged.

## What the golden comparison checks

`hdi::lasso.proj` receives a shared initial estimate, known sigma, and nodewise
scores from exact-lambda glmnet fits, avoiding unrelated CV/RNG differences.
The Julia nodewise scores are checked independently against those glmnet fits.
The full `hdi::boot.lasso.proj` path receives fixed sampled errors and fixed
initial/lasso fits through its internal helper bindings, so centered/studentized
distributions can be compared deterministically. The helper replacement uses
an exact glmnet lambda rather than interpolation along its default path. This
is component agreement at matched tuning, not agreement between the packages'
default automatic tuning policies. R's hdi exports centered draws multiplied
by the observed SE; the check converts them back to studentized draws.

Javanmard--Montanari rows are compared at mu=0.12 with a tight stopping rule.
CovarianceSelection's fused neighborhood solver at zero fusion penalty checks
three independent nodewise coefficient vectors. R's `glm` supplies additional
low-dimensional logistic/Poisson MLE, standard-error and null-score references.

The committed raw CSV files are in `results/`; `RESULTS.md` records their
interpretation. Run `python3 benchmark/inference/analyze.py RUN_DIRECTORY` inside
a Slurm allocation to regenerate a run's summary from `outputs/` and `outputs-v2/`,
or omit the argument to summarize the committed `results/`. Sampling profiles
are retained with the run records.

`run.jl glmcoverage ID` covers ID=1,...,12: logistic Toeplitz designs at n=100,
p=500, correlation 0.9, support sizes 3/15, first/random support and fixed uniform
signals on [0,1]/[0,2]/[0,4]. Each scenario has 20 response replications. The
initial logistic penalty is 0.05; nodewise Hessian projections use 0.1 after
standardization. A fixed null target also receives a decorrelated score test
with nuisance penalty 0.05 and projection penalty 0.03. Errors and nonconvergence
remain in the CSVs; summary denominators distinguish converged fits. These
coarse fixed-penalty checks supplement the Gaussian study and low-dimensional
R GLM references; they do not establish nominal calibration. Regenerate the
summary with `summarize_glm.py RUN_DIRECTORY` inside a Slurm allocation.
