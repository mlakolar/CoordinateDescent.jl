# Optimization comparison on Discovery

This campaign compares solvers for fixed objectives. It does not compare
prediction, cross-validation choices, or statistical estimators. MATLAB is
excluded.

The subsequent [solver-option and diagnostic follow-up](results/discovery-solvers-20261025/README.md)
adds public PANOC and block FISTA options and reruns all competitors on one
fixed Discovery debug CPU model, including fresh before/after baselines.

[Measured results and interpretation](results/discovery-20261025/README.md)
cover 5,460 timing observations, 290 profile reports, and 170 fresh-process
memory observations. Production solver code was frozen throughout the comparison.

The implementation under test is branch commit
9b7edcf315f173e862b049519a9731d3c8f02925, using ProximalBase
0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a and ProximalSolvers
9041e236f3de8ffdc02103e5a505993463e00bb1. The latter repository's local
parent folder was renamed to ProximalOPT; the measured module is
ProximalSolvers. The environment records capture the actual revisions and
package versions, rather than following another checkout's changing files.

## Methods

- Fixed-rank least squares: the package API, R rrpack's QR/SVD rrr solver
  and its separate normal-equation rrr.fit routine,
  a NumPy/SciPy compact-SVD implementation, and JuliaFirstOrder projected
  gradient on the medium and wide problems.
  A coefficient-only supplement separates Julia result diagnostics. A
  favorable oracle-assisted projected-gradient ablation uses the known
  optimal loss for stopping, retaining the original native-stopping runs.
- Coefficient nuclear-norm regression: the package's restarted accelerated
  proximal gradient; generic ProximalSolvers accelerated gradient and ADMM;
  ProximalAlgorithms FastForwardBackward and PANOC with ProximalOperators'
  nuclear norm; PyProximal FISTA and ADMM. Plain proximal gradient is an
  ablation on medium and ill-conditioned problems.
- Row-sparse, rank-constrained regression: the package and R rrpack, plus an
  adapter replacing the predictor block's coordinate descent with FISTA.
- Sparse singular-vector regression: one exclusive response component at
  fixed penalties, using the package, R rrpack's C++ component routine, and
  FISTA predictor-block updates. This is not a timing of rank selection or
  information-criterion selection.

The nuclear penalty is on B, not XB. The objective is
F(B) = ||Yc-Xc B||_F^2/(2n) + lambda ||B||_*.
Its proximal operator soft-thresholds singular values. The generic
libraries receive the same smooth loss and penalty, through thin adapters;
they are not advertised as turnkey regression packages.

## Shared inputs and measurements

The six designs are (n,p,q) = (300,50,10), (300,50,100), (300,50,1000),
(200,400,100), (500,100,100) with condition number 1e4, and (300,80,100)
with design rank40. Working rank is3. Each case has five independent
NumPy-generated inputs. All languages read the same Float64 column-major
binary values and the same saved initial factors.

Three fixed positive penalties are 0.5, 0.1, and 0.02 times the shared
objective-specific lambda-max. The nuclear path has eight descending
levels from0.5 to0.02. Sparse core timing uses unit weights. A separate
medium-case check compares adaptive inverse-row-norm-squared weights in
Julia/R on five shared inputs, three penalties, and three timing rounds,
with five profiles per language.

Each input is timed three times in rotating solver order. The reporting
unit is the median of these three timings; final tables summarize the
five independent reporting units. Julia warms both tall and wide gradient
types, including cached/path entry points, before timing. Input loading,
compilation, independent final certificates, and profiling are outside the
fit timer. End-to-end fits include centering and required setup. Separate
kernel, setup, first/cached least-squares prox, and path observations
identify the effect of reusable work.

All workers use one CPU and one BLAS/OpenMP thread. Record BLAS vendors:
an implementation comparison with different numerical libraries does not
isolate programming-language effects. Fresh-process memory observations
include the interpreter/packages and warm-up, with the post-warm-up
baseline stated. Ordinary workers' cumulative high-water RSS values must
not be attributed to individual solvers.

## Accuracy checks

For the convex nuclear problem, evaluate a feasible dual point obtained
from the scaled residual. Check both relative primal-dual gap and the
normalized proximal-gradient mapping against1e-6. Native stopping settings
are calibrated toward a mapping near1e-8 to satisfy the gap target.
Generic Python/JuliaFirstOrder solvers check their mapping every10
iterations, including that cost in fit time. A small independent
CVXPY/SCS solve checks the orthonormal closed form.

Fixed-rank fits are compared with the known optimum in fitted-value loss;
wide designs can have different coefficient representatives. Sparse fits
are checked using the actual A,V factors and both block stationarity
conditions. Reconstructing factors from B can create arbitrary unused
directions after a rank drop. Because the original Julia API returns B
only, its frozen outer loop is replayed outside the fit timer, verifying B
before retaining factors. These certificates establish local stationarity,
not global optimality. Paired runtime ratios also retain objective
differences between attained stationary points.

The R sparse-component solver materializes an n*q by q Kronecker matrix.
At n=300,q=1000 it alone needs2.4GB decimal. These runs use isolated
1-CPU/8-GB steps; ordinary workers use1CPU/2GB. Retain nonconvergence,
timeouts, and errors rather than dropping them from success rates.

## Profiling and reproduction

All five inputs of medium, many-response, and wide cases have separate
profile passes at the middle penalty. Julia uses sampling profiles;
Python cProfile exposes native elapsed time at the SVD/BLAS call boundary;
Rprof identifies R/C++ call boundaries. Percentages from these tools are
not directly interchangeable. Instrumented convergence traces are kept
for the Python and JuliaFirstOrder nuclear solvers.

The Discovery run root is
/home1/mkolar/juliaPkg/runs/20261025-cd81-optcomp.
The parent workspace registry is
.slurm/runs/20261025-cd81-optcomp.json.
The source snapshots record each submitted version, including corrected
pilot attempts. Final numerical source hashes identify which files
produced the measurements.

Reproduction uses the retained original package environment and the
run-local Julia manifest and Python freeze. Run in a **new output root**:

1. Prepare the isolated environment with prep.sbatch under Slurm, or
   restore its recorded environment. For a fresh Python environment,
   install the exact versions in the retained provenance/python-freeze.txt.
   prep.sbatch records the original bootstrap; exact recreation restores the
   retained manifest rather than resolving new package versions. Its
   absolute path dependencies refer to the retained frozen package sources.
   Adjust those paths when restoring the same revisions elsewhere.
2. Stage all these scripts in code/ and copy the desired .sbatch launchers
   to the run root. The historical recovery drivers refer to aliases:
   copy r_worker.R to code/r_worker_hires.R, adaptive.R to
   code/adaptive_hires.R, validate.py to code/validate_r.py, and coverage.py
   to code/coverage_r.py. These are the same numerical sources; see measured
   hashes. The final main.sbatch uses the ordinary, updated filenames.
3. Generate shared inputs inside Slurm with python code/common.py inputs.
4. Validate the adapters and small CVXPY reference with the pilot.
5. Submit main.sbatch with GNU Parallel launching exclusive srun workers.
   Its resume checks reuse completed isolated large-R steps. Use a new
   output root for an independent recreation.
6. The main launcher runs validate.py, summarize.py, and coverage.py under
   Slurm. Core coverage requires 5,250 distinct timing rows, 255 profile
   reports, and 170 fresh-process memory observations, with replications 1–5.
7. Run adaptive.sbatch, then finish.sbatch, then rank_matched.sbatch
   sequentially. These add the adaptive-weight check, coefficient-only rank
   comparison, and known-optimum stopping ablation. finish.sbatch preserves
   the original R adaptive measurements and repeats them with the final
   timer; fresh use of adaptive.R already has that higher-resolution timer.
   rank_matched.sbatch aggregates final coverage: 5,460 timings, 290 profile
   reports, and 1,400 independent saved-fit checks.
8. Run collect_results.py for compact reporting copies, profiles.tar.gz,
   and diagnostics.tar.gz. Shared matrices and coefficients stay at the
   run root. The dated README interprets the generated tables; it is written
   separately from the generated full-table.md.

r_precision.sbatch records the historical R timer/QR recovery. It is not
needed after a fresh main run with the final r_worker.R. finalize.sbatch
only repeats reporting aggregation after all supplements exist; it does
not fit models. The launchers use the campaign's frozen Discovery paths;
adapt environment/storage paths deliberately for another installation.

The launcher deliberately does not assign GNU Parallel's reserved PARALLEL
environment variable. Original failed-launch logs are retained separately
from measurement job logs. A successful scheduler exit is checked against
actual measurement coverage.

See the dated results directory for measured outcomes and limitations.
