# Gaussian SLOPE calibration: evidence and default settings

Implement the explicit Gaussian calibration API and retain BH as the default.
The adjustment improves false-discovery calibration in the intended iid
Gaussian examples, at a measurable power cost. It does not repair arbitrary
correlation, wide designs, endogenous noise estimates or path-level selection.
The new API supplies a documented normalization, intercept adjustment, flat
tail and provenance metadata without requiring an external calibration package.
This is a statistical/usability improvement, not a new optimizer or a claim of
faster fitting than the pre-change code.

## Primary evidence and its limits

The declared cohort contains 13 main and 3 separate stress scenarios, each with
1,008 independent datasets in seven process shards:13,104 main and 3,024 stress
inputs. Known sigma=1/q=0.1, fixed first 20 nonzeros, alternating/same signs,
iid/AR.4/AR.8/weak/wide/global-null and exactly centered orthogonal controls
were fixed before generation. The main random designs use raw Gaussian
columns, without additional sample normalization. Runtime inputs form a
separate 84-input cohort with columns normalized identically. See
[PROTOCOL](../../PROTOCOL.md) for dimensions, signal strengths, seed law,
pre-generation canary corrections and interval assumptions.

In the iid alternating-sign setting (n=300, p=100, signal=4.5/sqrt(n)), BH FDR is
0.1818[0.1767,0.1869] and Gaussian FDR 0.0733[0.0698,0.0768]. The paired change
is -0.1085[-0.1127,-0.1042]. Power decreases from
0.9464[0.9429,0.9499] to0.8570[0.8519,0.8622], a paired change
-0.0893[-0.0932,-0.0854]. The same-sign iid control gives a similar tradeoff.
These are pointwise95% Monte Carlo intervals, not finite-sample FDR guarantees.

Adverse examples remain central. AR.4 alternating-sign FDR falls from 0.2700
to 0.2010[0.1900,0.2120], still above 0.1. Wide AR.4 alternating-sign FDR is
0.4889[0.4746,0.5032] after adjustment, with power only0.1368. Strongly
correlated alternating-sign power remains low. In the exactly orthogonal
alternating control, BH has FDR 0.0810 and power 0.9868; Gaussian adjustment is
more conservative (FDR 0.0136, power 0.9454). This supports retaining the
orthogonal known-noise BH contract and an explicit Gaussian choice.
The global-null paired FDR change is uncertain. Estimated std(y) and block
covariance results are stress evidence only; std(y) is not introduced as a
noise estimator. Their datasets are separate from the corresponding main
scenarios and must not be treated as paired oracle-versus-estimated-noise data.

[TABLES.md](TABLES.md) contains every main/stress estimate, paired change and
runtime interval. Full mean-FDP/power/false/true-selection tables, MCSEs,
conservative Hoeffding FDP bounds and exact-null Clopper-Pearson intervals are
in [monte-carlo.csv](monte-carlo.csv) and
[paired-differences.csv](paired-differences.csv). Null power is undefined.
Intervals are pointwise; no multiplicity-adjusted superiority claim is made.

## Accuracy and comparator conventions

All 64,512 native statistical fit certificates pass the independently computed
original-design relative proximal residual/intercept criterion<=1e-7. This is
four fits per input, including an automatic/explicit duplicate, not 64,512
independent datasets. All 16,128 automatic/explicit native Gaussian coefficient
pairs are bitwise equal. The separate release/LTS prototype checks include 493
new recurrence/domain/Float32/64/intercept/metadata/precedence/input-preservation
checks per version plus existing selection/sparse/R-golden tests.

The 177,408 statistical certificates include 450 failures from external backends
at their frozen1e-8 controls, maximum residual9.336e-6. These fits remain
excluded from valid-backend statistical rankings; their raw values, failures
and the valid-subset counts are retained in
[statistical-accuracy.csv](statistical-accuracy.csv) and the delivery archive.
External valid-subset FDR means use different denominators and must not be
interpreted as full-cohort differences between backend procedures.
There are no API errors. All 96,768 native-versus-comparator common-weight
selected-set comparisons agree, including comparisons whose external
coefficient certificate fails; matching selections do not relabel an inaccurate
coefficient fit as certified. Maximum coefficient/objective relative differences
are 1.235e-5/5.764e-11 across those comparisons.

Five native calls hit the strict study 1e-9 iteration limit; all still pass the
declared independent 1e-7 criterion. The five records concern four distinct
inputs because one automatic/explicit pair is duplicated. Their stopping flags
are retained in [native-strict-stopping.csv](native-strict-stopping.csv).
No tolerance or output is changed after evaluating the cohort.

Pinned comparators are SLOPE.jl 1.3.1 and R 4.5.3/SLOPE 2.1.1. Three COMMON supplied
sequences isolate solver agreement: BH, the native paper/first-nondecrease
sequence and the Julia package's denominator convention. All weights have
independent SciPy/recurrence checks. Actual R-native Gaussian construction is
an additional convention analysis, not a substituted common sequence.
The R quantile approximation and its checker correction are preserved in the
pilot evidence. Its actual paper-relative weight difference is at most1.158e-9
in these designs and there are no R-native/paper selected-set disagreements.

The Julia denominator convention differs from the paper convention by up to
0.000488 relative to the first paper weight. It changes selected sets on up to
16/1008 inputs within a scenario; paired FDR/power differences are small and
not uniformly distinguishable from zero. See
[actual-calibrations.csv](actual-calibrations.csv) and
[native-convention-paired-differences.csv](native-convention-paired-differences.csv).
The API documents its chosen rule rather than claiming identity to every
Gaussian-weight implementation or an unproved global-minimum tail identity.

## Complete-procedure runtime, profiles and memory

Four cases use seven fresh processes/engine/case, three fresh shared inputs per
process and three warmed rounds/input. Methods rotate. All 94,500 timed returned
point certificates pass, maximum residual9.828e-8. Every paired runtime group
contains all 21 inputs/seven processes. Public fit/path results are completed
inside timers; coefficient extraction, profiling and independent checks are
outside. Native public/prepared controls preserve every complete returned field;
all 504 actual full-call allocation differences are positive. The prepared
control is a benchmark helper, not a new production cache API.

At matched independently verified accuracy, native public 20-level paths beat
both packaged comparators in all 12 case/sequence groups: every paired95%
interval for comparator/native exceeds1. Individual fits are mixed. For the
wide Gaussian sequence, SLOPE.jl/native is 0.3604[0.3359,0.3866] and R/native is
0.6273[0.5932,0.6634]; both external individual fits are faster. Sparse fits
favor SLOPE.jl, whereas sparse paths favor native. Dense individual comparisons
can be uncertain. These are current-procedure comparisons and must not be
attributed to adding the Gaussian selector or metadata.

All external/native runtime point pairs have identical selected sets.
Maximum original-objective relative differences are 7.103e-15 for R and
6.267e-15 for SLOPE.jl. Public/prepared coefficients are bitwise equal.
Default-versus-tighter native objectives differ by at most 6.275e-14, with no
selected-set disagreements; unscreened comparisons also agree in selection.
See [runtime-objective-selection.csv](runtime-objective-selection.csv).

All 2,016 CPU profiles are nonempty; 2,016 allocation reports are preserved.
All 280 emitted Julia source/CPU/thread records matches the pinned source,
Julia 1.10.12, EPYC 7513 and one Julia/BLAS thread; all R sessions record 4.5.3 and
SLOPE 2.1.1. All 84 fresh worker resource receipts report success. Exact timed
heap bytes are separate from sampled allocation reports. R allocation profiles
use a size threshold and are not total heap measurements.

Whole-worker RSS includes imports/JIT/warmups/profiling and different auxiliary
controls across engines. Native workers have higher recorded median peaks
than R and Julia comparator workers in these cases; this is not a matched
fitted-state memory ranking. No memory-improvement claim is made. See
[whole-worker-rss.csv](whole-worker-rss.csv) and [timed-heap.csv](timed-heap.csv).

## Default decisions

- Retain BH/q=0.1 as the compatibility default. Use Gaussian calibration as an
  explicit choice when its independent-Gaussian-design approximation and
  caller-supplied noise scale are intended. Neither rule supplies a general
  correlated-design or arbitrary path-selection FDR guarantee.
- Retain the caller-supplied sigma convention, default 1. Do not silently
  estimate noise with std(y). Unit-variance column units must be supplied or
  requested through standardize_x; this study does not justify changing that
  compatibility default or selecting a rule from a data-dependent threshold.
- Retain production tol=1e-7/max_iter=10000. Every runtime default probe passes
  independent 1e-7 accuracy. Default paths are faster than the tighter native
  control in all 12 groups (ratios 0.498–0.839, all paired intervals below1).
  Individual-fit differences can be uncertain or adverse. The strict1e-9
  research limit produces rare capped calls and is not recommended as a new
  public default.
- Retain screening=true. Unscreened/native path ratios are 3.29–6.74 across
  these groups, all paired intervals above1, with matched objectives and
  selections. This is scoped to the declared 20-level null-to0.2 paths.
- Keep explicit weights authoritative and retain the documented flat-tail and
  intercept normalization. No automatic covariance/noise estimator, solver
  switch, path-selection rule or cache is added.

Runtime comparison controls were native 1e-8/max_iter=100000 and external 1e-10 with
their original iteration caps, diagnosed on rep0 before generation to meet
the shared1e-7 criterion. Statistical single-fit controls remained native 1e-9
and external 1e-8. These different controls are explicit, and failed statistical
comparator results are retained rather than corrected by post-cohort tuning.

## Recreation, delivery and retained failures

Production ce00fa70d5913cd327c99f225997af135cd5bc75 is unchanged throughout.
Baseline master 295fbf5420d6384a1ca261023367ee86d0a40b8c was verified before the
worktree. Frozen main harness 2aa79e3/study-v6.tar has SHA256
`3c8a37922813fdcd8c7dd48252d0c12fbcfa1ac0ffadcac2e865584801056290`.
The comparison environment and frozen sibling package revisions are in the
delivery Project/Manifest and registry20261026-hdm107.

On Discovery under issue107, recreate with
`sbatch --parsable study-v6-code/study_main.sbatch` and
`sbatch --parsable study-v6-code/runtime_main.sbatch` in a NEW output namespace.
Do not regenerate an existing frozen input directory. Actual jobs 13455468 and
13455471 completed in 3m30s and 3m13s; saved-output report 13455573 completed in 3s
and convention/delivery 13455640 completed in 7s. All generation, fits, checks,
profiles and numerical aggregation ran under Slurm. No local Julia was run.

[reporting-delivery-v1.tar.gz](reporting-delivery-v1.tar.gz) contains complete
per-point certificates, common-weight comparisons, all timing/record/witness
CSVs, source/thread/session/resource receipts, uncertainty tables, environment,
frozen harness and scheduler/failed-launch receipts.
[worker-profiles-v1.tar.gz](worker-profiles-v1.tar.gz) contains every runtime
CPU/allocation report. Both transfers match [SHA256SUMS](SHA256SUMS).
Raw shared X/y and fitted coefficient/weight binaries remain in
`/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue107/{stat-main-v6,runtime-main-v6}`.

Pilot inputs/probes are excluded from the main cohorts. Initial staging/path
errors, the R-native quantile checker mismatch, unbuffered profile I/O,
strict-stopping and external-certificate failures, and the first reporter's
JSON scalar error remain in Git/run history and retained receipts. Reporter
repairs reuse saved outputs; no primary fit is repeated to make a result pass.

Within each cohort datasets are independent. Some runtime PRNG seed numbers
overlap statistical scenario seed numbers with different dimensions/data laws;
the cohorts are analyzed separately and are never pooled into one independent
sample. All conclusions are confined to these designs, seeds and stopping
controls; no universal FDR, power, time or memory superiority is claimed.
