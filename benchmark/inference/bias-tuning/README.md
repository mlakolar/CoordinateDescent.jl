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
