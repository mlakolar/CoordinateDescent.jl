# Gaussian SLOPE evaluation cohort, declared before generation

Production sourcece00fa7/candidate1; baseline master295fbf5. Passed prototype
release/LTS checks and pinned comparator preparation precede this cohort.
No evaluation input has been generated at this checkpoint. A complete harness
and representative rep0 canary must pass before the main generation gate.
Preserve any protocol/harness correction as a separate immutable version.

## Statistical cohort

Generate each shared raw X/y ONCE under Slurm. Seven independent process shards
per scenario,144 datasets/shard,1008 independent datasets/scenario. Main cohort
has13 scenarios; three stress scenarios are separate. This is a precision plan:
for a variable bounded in[0,1], the worst-case Monte Carlo SE at1008 inputs is
about1.58 percentage points; intervals and achieved uncertainty must be reported,
not an empirical FDR<=q software acceptance rule. No outcome-based tuning of
sequence, signals, selection cutoff or solver tolerance is permitted.

| Main design | n,p | Correlation | k | Signal amplitude | Signs/intercept |
|---|---|---|---|---|---|
| iid | 300,100 | independent Gaussian |20|4.5/sqrt(n)| alternating and same, two scenarios; no intercept |
| AR.4 |300,100|Toeplitz .4^abs(i-j)|20|4.5/sqrt(n)| alternating and same, two scenarios; no intercept |
| AR.8 |300,100|Toeplitz .8^abs(i-j)|20|4.5/sqrt(n)| alternating and same, two scenarios; no intercept |
| weak AR.4 |300,100|Toeplitz .4^abs(i-j)|20|3/sqrt(n)| alternating and same, two scenarios; no intercept |
| wide AR.4 |180,300|Toeplitz .4^abs(i-j)|20|4.5/sqrt(n)| alternating and same, two scenarios; no intercept |
| global null AR.4 |300,100|Toeplitz .4^abs(i-j)|0|0| one scenario; no intercept |
| exactly orthogonal |300,100|center Gaussian draws, thinQR, X=sqrt(n)Q|20|4.5/sqrt(n)| alternating and same, two scenarios; intercept fitted |

Nonzero coefficients occupy the first k columns, as in the earlier retained
study; this does not claim averaging over every support placement. Errors are
independent N(0,1). Known sigma1 and q.1 are primary. Random designs outside
the orthogonal control are not forced to have exactly orthogonal or unit-norm
columns. Standardize_x=false in the statistical cohort, holding the historical
estimator/data law fixed. Centering is applied only where intercept=true.

Stress cohorts: iid and AR.4 alternating-sign scenarios above using supplied
sigma=std(y), explicitly a data-dependent stress convention and not a new package
noise estimator; a separate alternating-sign block covariance design uses
independent equal-sized blocks with AR.8 and AR(-.4), known sigma1. This last
case deliberately violates the independent-column calibration assumption.
Estimated-noise and covariance-misspecified results have no claimed FDR bound.

Seeds:107400000+1000000*scenario_index+10000*shard+dataset_index, with scenario
order matching the table (alternating before same) followed by the three stress
cases in stated order. Pilot rep0 uses a disjoint107000000 seed family and is
excluded. Store one bounded binary per scenario/shard, its hash and dimensions,
true coefficients, noise convention and all common weights before fitting.

## Calibration and fair comparator rules

Use BH, the frozen native paper/first-nondecrease Gaussian rule, and pinned
SLOPE.jl1.3.1 Gaussian weights as three COMMON supplied sequences for native,
SLOPE.jl and R SLOPE2.1.1 fits. Scale by the same supplied sigma/sqrt(n).
Effective dimension for Gaussian constructors is m=n-intercept. Independently
check actual sequences against a separate recurrence/quantile calculation.
R's Gaussian constructor at the same effective dimension is an additional
native-convention analysis; record actual weight differences and its selected
set separately. Do not silently substitute it for a common sequence.

Native Gaussian automatic fits must match explicit common native weights;
explicit-weight precedence/metadata remain part of the public contract.
Comparators receive identical centered/transformed designs and original-scale
coefficient/intercept reconstruction. Common-weight agreement comes before
attributing discrepancies to calibration. Use complete returned fits/paths,
not coefficient-only timings ranked against complete public outputs.

Initial fixed comparison settings follow the validated earlier harness:
native tol1e-9/max_iter100000; Julia tol1e-8/max_it100000 and disabled path
truncation; R auto tol1e-8/max_passes1000000, gamma1, disabled truncation,
one thread. A canary may diagnose backend correctness before main generation;
freeze any justified correction before the main gate. Do not relax accuracy
after evaluating the cohort. An independently computed full relative proximal
fixed-point residual must be<=1e-7, including intercept stationarity, for each
measured Float64 result. Record objective, coefficients, selected indices at
abs(beta)>1e-6, true/false counts, FDP and power. Retain failures/nonconvergence;
an inaccurate backend result does not enter valid runtime/statistical rankings.

Report paired calibration differences, FDR=meanFDP, power (undefined at null),
selection counts, Monte Carlo SE/intervals and backend agreement, separately
for every scenario/sign/noise convention. Preserve correlated/wide inflation
and power loss. Distinguish the known-noise orthogonal BH theorem, Gaussian
approximation and empirical departures. No new penalty/noise default is chosen
by optimizing these evaluation outcomes.

## Runtime, defaults and resources

Seven fresh process replications per engine/case, three fresh shared inputs per
replication and three warmed rounds/input. Dense(600,200,rho.3), wide(200,1000,
rho.3), correlated(400,400,rho.8), sparse(600,1000,density.02). Runtime columns
are scaled to mean square one identically before the paired workload; document
whether preprocessing belongs to a public or prepared-data timer. Rotate method
order. Compare base-weight fits and20-level paths from the null threshold down
to.2 times it, without presenting this shorter path as every automatic-grid call.
Weight construction and complete fit/path costs are separate. Profiles and
independent certificates are outside timers. Use anonymous closures and verify
actual public/prepared output equivalence and setup allocation witnesses.

Probe the retained native tol1e-7 default separately on the runtime inputs with
independent certificates and matched tighter fits. Default conclusions must
distinguish q/calibration/noise assumptions, numerical stopping, screening and
arbitrary path-level selection. Retain BH compatibility unless a different
explicit contract is justified; do not imply that a faster optimizer fixes FDR.

Discovery only, slurm-workflow and the run registry; GNU Parallel inside owned
allocations, exclusive1CPU steps, all Julia/BLAS/OpenMP threads1, EPYC7513.
Julia1.10.12 is used with pinned SLOPE1.3.1 for this comparator ecosystem;
release1.13.2 correctness is separately passed. R4.5.3/SLOPE2.1.1 use the recorded
container/Rlib. A bounded representative canary will size the main allocation;
up to32CPU128GiB4h is the initial compute-as-needed plan, not a submission
receipt. Expected raw inputs are several GiB in112 scenario/shard files, with
bounded row-wise result/certificate outputs. Preserve frozen inputs and failed
receipts; no routine refitting, large archive audit, cleanup or merge.
