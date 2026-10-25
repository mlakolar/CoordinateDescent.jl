# Automatic tuning and matched speed follow-up

All numerical computation runs under Slurm on Discovery; one CPU per worker,
eight workers, 16 GiB total. The full 1,000-replication comparison is excluded.

The automatic policy is frozen before inspecting any development or holdout
results: minimum-error 5-fold initial CV on 40 levels, then pooled 5-fold nodewise
CV over at most 20 sampled columns on 20 levels. Explicit penalties bypass
selection. The scaled initial fit and universal nodewise penalty remain options.
QP constraints retain their theoretical default and require separate validation.

Development and holdout each use 100 independent datasets per case in the four
original Gaussian/binomial moderate/correlated designs. Seed namespaces are
91200000 (development), 91300000 (holdout), and 91400000 (speed). The policy is
not selected by looking at truth, coverage, or holdout outcomes. Development
also compares the universal projection at the same CV initializer. Gaussian
known-sigma and oracle-initializer calculations diagnose variance/bias and are
not competing feasible procedures. Metrics use the original five active and
five boundary-null targets; dataset-level Monte Carlo uncertainty is reported.
Per-coefficient results are retained for empirical-SE and bias diagnostics.

The speed benchmark has 20 additional datasets: ten Gaussian moderate and ten
Gaussian correlated. All three nodewise engines infer all 500 slopes with
common initial/projection penalties, fixed at the respective medians selected
on development datasets. Objectives, preprocessing, residual-DF noise estimates
and final numerical accuracy are checked. CV selection is outside these fixed-
tuning timings; the automatic study measures the cost of our complete policy.

The QP component benchmark compares all 500 rows at common mu, checks
feasibility and primal-dual gap, and compares inverse rows. Authors' sslasso
reference receives only the documented diagonal fast-path correction. Its
initialization, variance estimator and bias-inflated intervals are not part of
this component benchmark. SIHR and our QP pipeline additionally receive the
same 10-target task; SIHR keeps its default rescale/filter safeguards. They
implement different procedures, so this timing is not a solver-equivalence
claim. Every timing uses one CPU, independent warm-up, and a fresh process.
Peak RSS/process time include imports and warm-up; fit time excludes them.

The follow-up uses the remaining standing allowance from the entire issue-77
task, not a renewed two-hour allowance. Partial results and failures are kept;
no larger resource budget or full comparison is implicitly authorized.
