# Focused primal–dual API benchmark

This is the second stage of issue78, after the initial solver study. It measures
32-level paths down to0.01lambda_max, and a16-level1000x3000 stress cell down
to0.02lambda_max. Shapes are1000x100 Gaussian,1000x100 tail tau0.1 with
heteroscedastic t3 errors,10000x100,1000x1000 correlated, and1000x3000. R generates
centered sample-SD1 columns for all methods. Three fresh seeds1201–1203 and two
shuffled blocks across Julia and R methods give six observations per method/shape. Solver RNGs are fixed separately from input generation. Input-level means, SDs,
t intervals and paired contrasts use three inputs, never six independent inputs.

The public API returns coefficients, raw observation scores and independent
KKT/gap diagnostics. The exact backend now uses direct MOI ranged rows instead
of the exploratory JuMP formulation. Gurobi uses the fastest QR-style primal
simplex setting from the first study and returns native equality duals. Their
timings include complete fit objects and diagnostics. All methods use one
thread; two exclusive srun workers each have8GiB, under a2CPU/16GiB allocation.

The quantreg public lasso wrapper discards native dual workspace. A local copy
of that wrapper captures the bounded LP variable from rqfnb's third workspace
column after its final swap, shifting it by1-tau. This is labelled
`quantreg-native-dual`, not claimed to be an existing public quantreg API. Its
lambda argument is2*n*lambda. Conquer uses the parabolic kernel/h0.2; its
smoothed scores are evaluated from returned residuals inside the R timer.

These two R fit/score timers exclude startup and binary I/O. Julia then times
the same diagnostics used by the public API; `total_s` sums fit/score and
numerical-diagnostic time across the two processes. It excludes cross-language
transport and is an explicit wrapper cost, not one contiguous wall-clock call.
No reconstructed residual-sign dual is passed off as a native score. The
workspace interpretation must pass box/intercept/slope feasibility and KKT/gap
checks before its timing qualifies.

rqPen BR, FN and Huber expose coefficient paths in the tested interface, without
raw dual vectors. Their timings and exact check-loss objective comparisons are
reported separately, not ranked as complete primal–dual outputs. The Huber
method solves a different approximation from both exact QR and Epanechnikov
smoothing. rqPen BR/FN are wrappers around quantreg, rather than independent
underlying LP solvers. Bounds for coefficient-only checks reuse feasible
certificates on identical inputs; they do not create inference dual vectors.

Eight separate post-timing CPU profiles repeat the public exact/smoothed paths
for at least1.5seconds on small/wide inputs at seeds1201/1202, with native
frames enabled. Slow calls can exceed that target; profiles are not timing
observations.

The external gate is2e-5 KKT and2e-5 relative raw gap at every requested level.
API/Gurobi LP feasibility/optimality tolerances are1e-8, quantreg eps1e-8,
conquer coefficient-step tolerance1e-7, rqPen convergence tolerance1e-8.
Native criteria differ and retained achieved residuals govern interpretation. R
wrappers that return without an error do not report a convergence flag; the
CSV marks their native flag as `not_reported`, and checks their returned pair.
Exact paths have60-second budgets; R methods have a60-second R time limit and
75-second external process cap including startup/loading/warmup. Native R
routines may not check R interrupts immediately. Scientific failures, timeouts
and missing outputs remain explicit; they never become speed measurements.

All numerical work runs under Slurm on Discovery. Reuse the existing solver
Manifest and the pinned PB/PS source directories from the first run, but develop
the new API package source in a fresh run directory. Copy these scripts there;
`focused.sbatch` uses GNU Parallel/srun, and `summarize.py` runs in that allocation.
The run registry records actual committed sources and exact jobs. The API tests
must pass before the main benchmark is submitted. Pivotal calibration, CV,
inference and full parametric breakpoint enumeration are outside this stage.
