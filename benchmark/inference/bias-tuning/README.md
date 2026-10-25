# Bias-aware automatic tuning development

The user authorized compute as needed for this task and asked not to be queried
for routine compute-budget approval. Allocations remain bounded and proportionate.
The full 1,000-replication comparison remains on hold.

Projection candidate: Zhang and Zhang, Table1,
https://arxiv.org/html/1110.2563v2, on a finite decreasing 60-level lasso grid.
Stage1 chooses the largest penalty attaining eta <= sqrt(2log(p)), or records
an adjusted bound if unattainable. Stage2 chooses the smallest grid penalty
whose noise SD factor is at most 1.25 times the stage1 value. This is a 25%
noise-SD budget, not a 25% variance budget. Target-specific penalties, bias/noise
factors, reference noise, feasibility of the original bias target, and grid
boundary selection are returned. Gaussian theory motivates this rule; application
to the fitted GLM Hessian is evaluated empirically and is not a finite-sample
coverage guarantee.

Development seeds start at91500000, independent of earlier studies. Holdout
seeds start at91600000 and remain reserved until a policy is frozen. The first
probe uses20 datasets in each of the four original cases and compares CV,
LDPE with the CV initializer, selected-model refit, half relaxation toward that
refit, and half the initial CV penalty. Refits fall back to the regularized
initializer on an oversized/rank-deficient model or failed/extreme GLM fit;
status is recorded. These are development candidates, not production defaults.
All methods use the same datasets and ten targets. Row timings measure incremental
candidate computation, not complete production pipelines.

All numerical work, including aggregation and independent glmnet reference
checks, runs on Discovery. Small R and Julia scripts independently implement
finite-grid selection on a common golden design. Raw per-replication metrics,
coefficients, errors and completion flags are retained for bounded recovery.

A second development probe compares SD budgets .25/.5 and the published
restricted LDPE choice m=4 (four largest absolute working correlations get
zero penalty), holding the CV initializer fixed. Its output root is `restricted`;
its independent benign and random-support cases use offsets 20000 and 30000.
No holdout is used for choosing among these candidates. Restriction is an expert
option and remains disabled by default while its calibration is evaluated.
The m=4 choice follows the published simulation, not the known simulation
support. References: Zhang and Zhang, Sections 2.3 and 4, linked above.

The measured immutable source-v1/v2/v3 snapshots predate the module rename from
CoordinateDescent to HighDimensionalModels. Current scripts use the new module
name; reproduce a historical measurement with its recorded snapshot/environment.
A current-source parity bridge is required before calling the renamed source
validated by these measurements. Raw data and historical lockfiles remain frozen.

## Frozen fresh-data validation (2026-10-25 18:12 UTC)

Development finished with 800 datasets and 4,000 converged fits and no fit errors.
The independent residualized-glmnet reference passed. No candidate reached 95%
coverage in every case. Freeze one candidate for honest independent validation:
CV initializer (5 folds, 40 levels, opt_tol=1e-8 and inference gaptol=0), ordinary
model-based standard errors, LDPE with m=4 unpenalized working correlations,
60 levels to .001 of the conditional maximum, SD budget kappa=.5, original
bias bound sqrt(2log(p)), and fallback relaxation .1. All ten targets, dataset
construction and signals remain as in development. The choice balances smaller
remaining active bias against longer intervals and lower power; it is not
certified as calibrated and is not promoted to the production default.

Validate this candidate and the unchanged CV default on 200 new datasets in
each of eight cases (1,600 datasets, 3,200 fits). The previously untouched seed
namespace91600000 is now allocated to this frozen study. There will be no
retuning from these results. `validate.jl` fits each full automatic pipeline
separately with the same CV seed and reports full warm fit time. These are paired
statistical comparisons, not a new matched competitor-engine benchmark.
Numerical fitting source is source-v4/env-v4 from the registry; only the standalone
runner and analysis are new. A 24-dataset parity bridge checks master integration.

## Completed follow-up

The frozen holdout produced all 3,200 converged fits and 32,000 unique finite
coefficient records. All 64 worker resource records, normal p-values and noise
budgets passed checks. The report and compact per-dataset inputs are in
[results/discovery-20261025](results/discovery-20261025/RESULTS.md).
The new rule improves Gaussian coverage substantially, but logistic calibration
still fails and grid-endpoint selections are common. It remains experimental.

`refresh-defaults.jl` replaces the historical Julia default rows on the original
20 paired competitor-pilot datasets per case, reusing validated competitor and
explicit-matched fits. All 200 refresh fits succeeded. These legacy inputs are
separate from the fresh holdout. No rule was selected or retuned from either
post-freeze comparison. Numerical defaults remain untouched for the refresh.
`analyze-refresh.py`, `diagnostics.py` and `final-receipts.py` build its reporting
inputs and revised conditional cost projection on Discovery. The final generator
adds the shared-covariate clarification and diagnostic interpretation from saved
results; no numerical refitting is required to recreate the reports.

The full 1,000-replication comparison was not launched. Raw coefficients, failures,
accounting, source snapshots and pinned environments remain in the task run folder
on Discovery and in the local project run registry. The report includes exact
scope limits for the prior matched speed timings and the new automatic pipelines.
