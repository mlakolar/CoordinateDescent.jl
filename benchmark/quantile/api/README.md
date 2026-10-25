# Focused primal–dual API benchmark

This follows the initial solver study for issue #78. See [RESULTS.md](RESULTS.md)
for the short decision report. All fitting, profiling and numerical aggregation
run under Slurm on Discovery.

## Design

The five cases are Gaussian 1,000 × 100; a heteroscedastic t3 tail case at
τ = 0.1 with 1,000 × 100; Gaussian 10,000 × 100; correlated 1,000 × 1,000;
and 1,000 × 3,000. R generates the shared inputs with centered, sample-SD-one
columns. Seeds 1201–1203 and two shuffled blocks across Julia and R give six
observations per method/case. The first four cases use 32 penalties down to
0.01 λmax; the last uses 16 down to 0.02 λmax. Solver RNGs are fixed separately.
Input-level means, SDs, t intervals and paired contrasts use three inputs,
not six independent samples.

Each worker has one CPU and 8 GiB; two exclusive srun workers share a
2 CPU / 16 GiB allocation. Fit/score and diagnostic dispatches are warmed before
timing. Eight separate CPU profiles repeat exact/smoothed public API paths on
small/wide inputs at seeds 1201/1202, with native frames enabled. Profiles run
after timing and are not additional timing observations.

## Outputs and qualification

The public API returns coefficients, raw observation scores and independent
KKT/gap diagnostics. Its exact backend uses direct MOI ranged rows and a warm
quantile basis. Gurobi uses the fastest QR-style primal simplex setting from the
first study, retains its model along the path and returns native equality duals.
Both timers include fit objects and diagnostics.

The public quantreg lasso wrapper discards its native dual workspace. A local
copy captures rqfnb's bounded LP variable from workspace column three after its
final swap, then subtracts 1−τ. The label `quantreg-native-dual` denotes this
adaptation, not an existing public accessor. Its penalty argument is 2nλ.
Conquer uses the parabolic kernel with h = 0.2; its scores are evaluated from
returned residuals inside the R timer.

These R fit/score timers exclude startup and binary I/O. Julia then times the
same diagnostics used by the public API. `total_s` sums the two components;
it excludes cross-language transport and is not one contiguous wall-clock call.
No residual-sign vector is presented as a native exact dual.

rqPen BR, FN and Huber expose coefficient paths in the tested interface without
raw dual vectors. Their timings and exact check-loss objective comparisons are
reported separately. Huber solves a different approximation from exact QR and
Epanechnikov smoothing. BR/FN wrap quantreg rather than independent LP engines.
Feasible bounds for coefficient checks come from identical-input certificates;
those bounds do not supply inference duals.

The external gate requires KKT ≤ 2e−5 and relative absolute raw gap ≤ 2e−5 at
every requested level. API/Gurobi LP tolerances and quantreg eps are 1e−8;
conquer's coefficient-step tolerance is 1e−7; rqPen's tolerance is 1e−8.
Native stopping rules differ: inspect achieved residuals as well as timings.
R interfaces do not report a native convergence flag (`not_reported` in CSV).
Exact paths have a 60-second budget; R methods have a 60-second R limit and a
75-second external cap including startup/warmup. Native R routines may defer
interrupt checks. Incomplete paths and failed gates never qualify as speeds.
Summary means for partially qualified groups describe survivors only and must
not be ranked against complete groups.

A separate saved-input followup uses small/tall seed 1201, two repetitions,
API default/tighter controls, conquer tolerances 1e−5/1e−7, and a native quantreg
reference whose intercept design is prepared once per path. Its 20 observations
are not pooled with the primary cohort and have no input-level confidence
interval. Two additional native-only wide controls at seeds 1203 and 1202
compare default/tighter settings twice each. The first control used a passing
input; the second uses the actual failed primary input. All 28 control
observations remain separate from the primary cohort. `accuracy.sbatch` also
summarizes the eight saved CPU profiles.

## Accuracy controls and profiles

The saved small/tall input controls give API default/tight means of
0.021/0.041 and 0.128/0.195 seconds, respectively. Neither reaches the 2e−6
gate; conquer at epsilon 1e−7 takes 0.197/0.889 seconds and passes it. These
single-input controls do not establish equal-precision speedups. The exact
quantreg reference with its intercept design prepared once per path takes
0.902/11.792 seconds and passes both gates. Wrapper setup therefore does not
explain the exact API's main advantage on those shapes.

The failed wide primary input is seed 1202: default settings fail the relative
raw-gap check at level 23 even though the native Newton flag is true. The
corrected control repeats that failure; tighter settings pass the main gate
at 4.48 seconds but still fail 2e−6. Earlier seed-1203 controls are retained as
additional sensitivity evidence, not described as a repair of seed 1202.

Eight CPU profiles point to HiGHS sparse pricing and factorization for the exact
backend and arithmetic/matrix-vector work for smoothing. See the saved flat
profiles and `accuracy/profile-summary.json`; profile durations are excluded
from timing observations. Native inner coordinate warnings are retained in
control logs even when the final external pair passes.

## Recreate

The run record accompanying the results identifies exact source commits, jobs,
software and commands. The measured package still used the CoordinateDescent
module name. The later HighDimensionalModels rename changes the namespace and
extension name; the numerical implementation is unchanged. Use the recorded commits
to reproduce the measured run, or the current scripts with the renamed package.

Develop the package and both dependencies together in a fresh run environment:
ProximalBase `0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a` and ProximalSolvers
`73a1c8e927d8917e4f16483271b053fcdb25b488`. The latter supports the current
`gaptol` path keyword; the older first-study snapshot does not. Reuse the saved
solver Manifest. Copy these scripts into the run directory and submit
`focused.sbatch`, which runs GNU Parallel/srun and then `summarize.py` within the
allocation. Run the API checks first. Pivotal calibration, CV, inference and full
parametric breakpoint enumeration remain outside this stage.
