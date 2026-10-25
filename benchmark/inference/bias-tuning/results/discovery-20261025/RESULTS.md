# Bias-aware tuning: frozen fresh-data validation

EMPIRICAL. This follow-up tested one frozen experimental policy against the unchanged CV default. The full 1,000-replication cross-method comparison was not submitted.

Development used 100 datasets in each of eight designs (800 datasets, 4,000 converged fits). After development, restricted LDPE with four unpenalized correlations and a 50% noise-SD budget was frozen before using the independent holdout namespace. Fresh validation used 200 datasets per design (1,600 datasets, 3,200 fits). Five active and five null targets belong to each dataset. All Monte Carlo standard errors below use dataset-level summaries, retaining dependence between targets. Replications are independent within each case. Corresponding Gaussian and binomial cases share their covariate draw; 1,600 datasets includes both response families rather than 1,600 distinct covariate matrices.

| Case | Policy | Active coverage ± MCSE | Null coverage | Type I | Power | Length active/null | Median warm automatic seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| binomial-benign | cv | 83.3% ± 1.4 pp | 94.1% | 5.9% | 100.0% | 0.405/0.398 | 0.189 |
| binomial-benign | rldpe4_k50 | 92.8% ± 0.9 pp | 94.5% | 5.5% | 100.0% | 0.473/0.464 | 0.190 |
| binomial-correlated | cv | 85.4% ± 1.2 pp | 95.4% | 4.6% | 15.9% | 1.455/1.477 | 0.232 |
| binomial-correlated | rldpe4_k50 | 96.4% ± 0.6 pp | 96.4% | 3.6% | 9.8% | 3.541/3.727 | 1.938 |
| binomial-moderate | cv | 52.9% ± 1.6 pp | 95.6% | 4.4% | 81.2% | 0.587/0.585 | 1.130 |
| binomial-moderate | rldpe4_k50 | 90.5% ± 1.0 pp | 96.3% | 3.7% | 58.0% | 0.949/0.950 | 3.903 |
| binomial-random | cv | 78.7% ± 1.4 pp | 97.6% | 2.4% | 92.5% | 0.619/0.609 | 1.105 |
| binomial-random | rldpe4_k50 | 92.0% ± 0.9 pp | 98.0% | 2.0% | 61.6% | 1.011/0.997 | 4.123 |
| gaussian-benign | cv | 92.7% ± 0.8 pp | 94.9% | 5.1% | 100.0% | 0.184/0.185 | 0.048 |
| gaussian-benign | rldpe4_k50 | 95.2% ± 0.7 pp | 94.9% | 5.1% | 100.0% | 0.213/0.215 | 0.050 |
| gaussian-correlated | cv | 75.0% ± 1.7 pp | 94.5% | 5.5% | 29.8% | 0.772/0.783 | 0.280 |
| gaussian-correlated | rldpe4_k50 | 95.1% ± 0.7 pp | 95.6% | 4.4% | 17.4% | 1.884/1.983 | 1.926 |
| gaussian-moderate | cv | 71.8% ± 1.7 pp | 96.2% | 3.8% | 99.9% | 0.285/0.285 | 1.217 |
| gaussian-moderate | rldpe4_k50 | 94.7% ± 0.7 pp | 95.5% | 4.5% | 98.2% | 0.459/0.461 | 4.098 |
| gaussian-random | cv | 89.5% ± 1.0 pp | 95.8% | 4.2% | 100.0% | 0.286/0.286 | 1.131 |
| gaussian-random | rldpe4_k50 | 94.2% ± 0.7 pp | 96.2% | 3.8% | 98.2% | 0.463/0.461 | 3.975 |

The nominal marginal targets are 95% coverage and 5% type I error. Paired differences and Monte Carlo errors for every reported quantity are in validation-holdout-checks.json. Mean and median timing summaries and Monte Carlo errors for coverage, error, power and interval length are in validation-holdout-summary.json. These are empirical checks, not finite-sample guarantees.

## Calibration decision

Do not promote the candidate as a calibrated general-purpose default. The approximate 95% Monte Carlo upper limit for active coverage is below 95% in: binomial-benign, binomial-moderate, binomial-random. These limits quantify simulation error only; they do not repair inferential bias. The policy is not retuned against these validation datasets.

The refreshed current-default QP rows also fail calibration: active coverage
was 36% (moderate Gaussian) and 8% (correlated Gaussian) on the original paired
20-dataset pilot. Those are separate from the 200-dataset nodewise holdout.
Numerical QP feasibility and dual-gap checks do not remove this statistical
bias. Both the CV policy and universal QP constraint remain experimental;
the implementation is not ready for an out-of-the-box inference release.


## Runtime, memory and comparison scope

Conditional own-policy projection for all eight designs, 1,000 datasets/design and ten targets: cv needs about 1.59 warm CPU-hours, or 0.25 allocated wall-hours on eight single-CPU workers with a 25% margin. This multiplies measured mean complete fit costs; it excludes competitor fits, process startup, scheduling and storage. It is not a full comparison estimate.
Conditional own-policy projection for all eight designs, 1,000 datasets/design and ten targets: rldpe4_k50 needs about 5.68 warm CPU-hours, or 0.89 allocated wall-hours on eight single-CPU workers with a 25% margin. This multiplies measured mean complete fit costs; it excludes competitor fits, process startup, scheduling and storage. It is not a full comparison estimate.

The fresh validation workers used one CPU each. Across 64 worker processes, the maximum measured peak RSS was 1070.7 MiB. Each worker executed both policies on a dataset shard; this is a combined process peak, not a separate per-method memory estimate. The allocation was eight CPUs and 16 GiB. Warm fit timings include each complete automatic policy, including its own initial CV. Process wall time, CPU usage and peak RSS are retained in the worker resource records.

The earlier matched 20-dataset competitor speed benchmark remains the fixed-penalty engine comparison. It has 140 valid fits: Julia/hdi/EconML use all 500 slopes, Julia/sslasso all 500 QP rows, and SIHR is reported separately as a different ten-target pipeline. The new policy changes projection tuning and interval width, so the earlier engine timings are not its automatic end-to-end speedups. Default CV fitting is unchanged apart from rejecting unusable CV tail entries; the package rename and current master dependency bridge reproduced 120 fits on 24 datasets exactly.

The earlier approximately 14 allocated wall-hour estimate concerns only the seven-task Gaussian matched timing workload, on eight CPUs/16 GiB with a 25% margin. It excludes native competitor defaults and binomial cases. The earlier approximately 50-hour complete-protocol estimate covers the four original pilot designs and is provisional; a final cross-method inferential policy and protocol have not been validated or frozen. No full-study launch is justified by these runtime estimates alone.

## Provenance and limitations

- Numerical source: immutable source-v4/env-v4, based on commit 1c6e35b. Frozen runner recipe: f1b73f8. The registry records full hashes, dependency commits, job IDs and resource requests.
- Four initial cases use n=200,p=500,rho=.3,s=5 or n=100,p=500,rho=.9,s=15. Benign cases use n=500,p=100,rho=.3,s=5; random-support cases use the moderate dimensions. Gaussian signals alternate ±.5 and binomial signals ±.7. The intercept is .2 and Gaussian noise SD is one.
- The initializer uses five-fold 40-level prediction CV with opt_tol=1e-8 and inference gaptol=0. The candidate uses a 60-level LDPE path to .001 of the conditional maximum; m=4, kappa=.5, bias bound sqrt(2log(p)), relaxation .1. Model-based marginal normal intervals are evaluated.
- Development, holdout and earlier speed datasets use separate seed namespaces. The published restricted-projection rule motivates the candidate; the GLM Hessian application is evaluated empirically and has no new theorem here.
- The first development probe tested selected-model refits and halved initial penalties. Neither repaired coverage consistently. Its adverse results remain in development-summary.json.
- The CV tail warm-up failure and R library-path failure were recovered without rerunning valid Gaussian results. The first holdout resource wrapper failed before any Julia fitting because /usr/bin/time was unavailable; its five-second failure and empty-data analysis are retained. A tested Python resource wrapper recovered the untouched study with the same statistical source, runner and seeds.
- Independent glmnet checks passed for ordinary and restricted LDPE, and 142 inference/canary checks passed. The current package/master-dependency bridge passed 9,624 checks with zero maximum numerical difference. Hosted CI and docs passed on the integrated implementation before the recipe-only follow-up.

See the preceding automatic-tuning report for native competitor coverage, matched speed, EconML intercept degrees-of-freedom scaling, and QP feasibility/optimality certificates. Full coefficient records, raw worker logs, failures and accounting are retained in the task run folder.


## Bias and grid diagnostics

Every candidate coefficient attained the original working-design bias-factor bound,
and every recorded noise ratio obeyed the frozen 50% SD budget. These numerical
conditions did not establish coverage: logistic intervals still undercovered.
The selected penalty was a finite-grid endpoint for 2,000/2,000 target rows in
both benign cases, 1,641/2,000 in Gaussian moderate, and 1,568/2,000 in logistic
moderate. This leaves a material grid-boundary limitation; it is not evidence
that an unconstrained continuous-path optimum was found. We did not extend the
grid or retune using holdout results.

The signed active error divided by SE averaged −0.283 for Gaussian moderate and
−0.724 for logistic moderate under the candidate. The corresponding empirical
SD/RMS-SE ratios were 0.968 and 0.943. Remaining logistic undercoverage therefore
cannot be repaired merely by claiming that its reported SEs are too small.
The benign logistic case had a ratio 1.090 and signed bias −0.121, a different
finite-sample calibration pattern. See validation-holdout-diagnostics.json.

## Current-default competitor refresh

CURRENT_COMPARISON.md updates the native Julia default rows on exactly the
original 20 shared datasets per case. All 200 refresh fits succeeded, including
the current nodewise defaults, Gaussian QP defaults, and the frozen experimental
candidate. It reuses the validated competitor and explicit-matched fits rather
than rerunning them. These legacy pilot datasets are separate from the 200 fresh
holdout datasets per case; the refresh does not select or retune the candidate.
Full pipelines still compute different internal target counts. The separate
500-slope/row timing benchmark remains the matched engine comparison.

For the refreshed original 34-cell/four-case protocol at 1,000 datasets/case, measured mean warm fitting costs project to 247.25 CPU-hours: an ideal 30.91 hours on 8 workers, or 38.63 allocated hours with a 25% margin. This excludes process startup, queue delay and storage. Eight CPUs/16 GiB remain the planning allocation. The historical fresh-process estimate was about 50 allocated hours; current Julia automatic-pipeline cold startup has not been measured separately per policy, so 50 hours is not a confirmed current full-run budget. Batching replications amortizes startup. No full comparison was launched.
