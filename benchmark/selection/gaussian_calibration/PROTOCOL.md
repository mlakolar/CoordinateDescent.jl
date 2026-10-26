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

## Harness details fixed after disjoint pilot, before the main gate

Statistical pilot13455149 generated32 rep0 inputs and completed all48 backend
processes. The initial checker correctly stopped on R-native weights: R2.1.1
uses the rational normal-quantile approximation in its pinned `qnorm.cpp`,
whereas SciPy uses a more accurate quantile. The observed relative difference
was about1.2e-9. Recheck13455171 reused every fit and passed352 independent
certificates and192 common-weight pairs. The checker now independently ports
that pinned R rational formula at the original2e-12 weight tolerance, and also
compares its quantiles to SciPy with an explicit5e-9 approximation bound. All
three COMMON sequences retain their original2e-12 comparison, and every fit
retains the full1e-7 certificate. No solver setting or cohort was adjusted.

Runtime inputs use disjoint seeds109400000+1000000*case+10000*rep+dataset;
pilot seeds109000000+10000*case+100*rep+dataset. Each has ten coefficients.4
on the first ten columns, independent unit Gaussian noise and no intercept.
The four case dimensions/data laws are unchanged. The same binary floating
data are read by all backends; sparse cases are converted identically to sparse
storage before each public timer. Column normalization happens once in shared
input generation. No random column is regenerated to avoid an adverse result.

Native public/prepared controls return identical complete solution/path fields;
the prepared control caches data preprocessing only and is a benchmark helper,
not an added production API. Actual full-call allocation differences and all
complete fields are checked outside timers. Default probes use both the native
tol1e-7 and max_iter10000; unscreened tight paths are separate. Comparators'
unexposed convergence flags are recorded as unavailable, not inferred from a
successful return. Independent certificates determine accuracy for all backends.
The original pilot CSV comparator true flags meant successful API return;
their independent certificates, rather than those flags, establish accuracy.

Runtime CPU profiles repeat untimed warmed calls for sampling coverage. The
pilot uses native allocation sample probability.02. Wide native paths allocate
hundreds of MiB cumulatively, so the main probability is.001 to bound profile
I/O and observer overhead; exact warmed-call allocated bytes remain in timers.
Stack text is buffered before its file write to avoid one NFS write per frame.
R CPU profiles and allocations>=16384 bytes are separate untimed calls. Whole
worker RSS includes imports/JIT/warmups/checks/profiles and is not fitted-state
memory. Public fit/path timers include complete returned results and exclude
coefficient extraction, profiling and independent checks; R uses a high
resolution wall clock in the main cohort. Weight construction is timed separately.

Main statistical allocation:16CPU64GiB1h; runtime allocation:16CPU64GiB2h,
at most32CPU128GiB together, under the user's compute-as-needed authorization.
The main gate still requires runtime adapter/certificate pilot completion.
The statistical summary uses pointwise Student-t Monte Carlo intervals over
independent datasets, supplemented by conservative Hoeffding bounds for bounded
FDP and exact Clopper-Pearson binomial intervals for global-null FDR. These
avoid interpreting an all-zero sample's zero estimated MCSE as certainty.
Runtime paired log-ratio intervals first average the
three input log-ratios within each process replication, then use Student-t
uncertainty over the seven process replications. Report interval assumptions,
all accuracy failures, adverse paired ratios and uncertainty without outcome
based changes to sequences, signal design, cutoff or solver settings.

Runtime pilot13455206 was stopped after12m17s because unbuffered native
allocation-stack writes dominated the allocation. All Julia/R and native
sparse workers completed; native dense/wide/correlated workers were partial.
Stage recovery keeps the original directory immutable, reuses complete workers
and provably closed profiles, preserves every paired binary/CSV timing row,
and appends only missing timing rounds. The last active profile is regenerated
because it has no completion proof. Buffered profile output and probability.001
are observer changes; no fit tolerance, signal, sequence or timing row is
replaced. This is a disjoint canary, excluded from main independent replication.

Stopping diagnosis13455366 completed in16s on the same12 rep0 runtime inputs:
36 tol1e-8/max100000 paths,720 independently certified points, zero backend
nonconvergence and maximum independent residual9.99e-9. Some original native
1e-9 paths instead reach the100000-iteration cap and warn, with multi-second
costs and GiB cumulative allocation. These adverse records remain in v1/v3.
Before ANY main generation, the native runtime comparison control is therefore
changed to tol1e-8/max100000 for public/prepared/unscreened calls, matching the
comparators' declared1e-8 controls at the unchanged independent1e-7 certificate.
A fresh native adapter canary at that control reuses the12 inputs and all saved
Julia/R comparator workers; it is a different stopping-setting sensitivity,
not an additional independent cohort or a replacement of old timings. Native
statistical single-fit settings remain1e-9/max100000 as already certified.
Public production tol1e-7/max10000 and screening defaults remain unchanged.
Main generation still waits on the corrected runtime adapter canary gate.

Corrected native runtime canary13455395 completed all native workers in62s,
with complete public/prepared fields bitwise equal and all72 allocation witnesses
positive. Independent checks passed all native comparison/default points but
found6 R path-point failures in dense case1 and42 SLOPE.jl path-point failures
in correlated case3 (three timing rounds counted separately). Their maximum
residuals were1.154e-7 and1.120e-6. Other cases passed. Successful comparator
returns are not accuracy certificates. Before main generation, use runtime
SLOPE.jl/R tol1e-10 (original iteration caps) and recertify all outputs on the
same12 inputs; reuse all already-valid native1e-8/default workers. Preserve
old comparator timings/failures and all native1e-9 sensitivity results. Native
runtime control1e-8, public production defaults and statistical single-fit
comparison controls stay as documented; the shared1e-7 certificate never changes.
