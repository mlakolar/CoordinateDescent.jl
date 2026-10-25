# Calibration only: no adoption claim

One independent input in each of six designs, three warmed rounds per public
fit configuration. Unit-weight fractions are0.5/0.1/0.02 for CD/FISTA, plus
the medium adaptive-weight fraction0.1 supplement. These inputs may guide
development and must remain excluded from the future20-input evaluation.

All76 actual-factor/stationarity/rank/result certificates pass. Diagnostic
replays match saved production coefficients and outer iterations; every
timed round has the same coefficient digest. All38 before/after objectives
match under the declared relative1e-6 criterion. Independent predictor KKT
and response-subspace stationarity bounds are1e-6. This is local stationarity,
not global optimality. Raw A/V factors, outputs, profiles, versions, hashes and
source archives remain in the owning Discovery run.

At the middle unit penalty, the many-response CD/FISTA point ratios are
0.769/0.848, and wide FISTA0.791. Preserve adverse controls: wide CD at the
middle penalty is1.052, ill-conditioned strong-penalty CD1.048, and the
medium weak-penalty CD1.018. Every ratio, allocation and objective/coefficient
difference is retained in summary.csv/summary.json. One-input timing movements
do not establish a repeatable speedup or default solver recommendation.

The measured candidate is source snapshotcandidate-v2.tar, SHA256 recorded
in `.slurm/runs/20261026-hdm102.json`, with unchanged frozen dependencies.
Pilot13448682 completed in1m24s. Correctness canary13448664 passed574 tests
on each Julia1.10/1.13, including42 near-cutoff rank controls and52 known
weighted-path checks. Additional thin-residual fixtures are recorded separately.

Remaining work includes the prescribed20 fresh inputs per configuration,
natural/public and controlled-initialization R rrpack comparisons, common-init
Julia kernels, complete target profiles and RSS, component timing and path
performance. The current statistical defaults and block_solver=:cd remain.
