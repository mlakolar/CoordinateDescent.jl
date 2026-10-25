# Default accuracy after issue #97

The new defaults pass **15/15 targeted held-out workloads**, versus **6/15**
for PR #83's previous defaults. This is a deliberately difficult subset, not
a revised pass rate for the earlier full 60/63-workload campaign. All three
fresh datasets must pass for a workload to count. Both sources have complete
coverage and solver convergence on every workload.

The maximum candidate relative coefficient error is **1.97e-5**, and the
maximum relative objective difference is **2.46e-7**, against independently
implemented R references. Targets remain 1e-4 and 1e-6, respectively. Every
column of a supplied-grid path is checked. No setting is selected on these
fresh inputs: the defaults were frozen from the retained earlier calibration.

## What changes

For Float64 inputs, residual defaults now depend on the solver family:

| Solver family | Default abstol | Default reltol |
|---|---:|---:|
| Generalized/graph lasso, including paths and CV folds | 1e-9 | 1e-8 |
| Fused regression and other trend fits | 1e-8 | 1e-7 |
| Linear signal fits with the calibrated gap rule | 1e-7 | 1e-6 |
| Other floating types | 1e-7 | 1e-6 |

Explicit tolerance arguments override these defaults. Zero-penalty
least-squares fits and exact fully observed signal proximal steps keep their
existing algorithms. The numerical implementation is source commit
`7bf2e2ec11715404f37e57345ec31fbfdef25905`; the baseline is
`b570922b5cad321b817ee04b309c5cea7c4189db`.

Fused regression now verifies the proximal-gradient residual at the returned
coefficients using the original design for both cached and direct products.
Previously the direct route checked the extrapolated predecessor. The reported
`dual_residual` is the value used by the final returned-point check.

## Accuracy has a runtime cost

Every time below is the median of nine warmed measurements: three independent
datasets, each timed three times. Old and new source run in separate processes
inside the same Discovery allocation, with source order alternating by dataset.
Times include the public fit/path, degrees of freedom and result construction;
I/O, compilation, reference solves and accuracy checks are outside timing.

| Workload | Old defaults, seconds | New defaults, seconds | Old accuracy |
|---|---:|---:|---|
| Quadratic 50-level path, n=10,000 | 38.632 | 71.493 | fail |
| Quadratic signal n=10,000, R lambda=1 | 0.830 | 1.259 | pass |
| Irregular quadratic n=10,000, R lambda=10 | 1.926 | 3.711 | fail |
| Correlated fused 50-level path, p=100 | 0.117 | 0.121 | fail |
| Cyclic-graph 50-level path, p=100 | 0.451 | 0.748 | fail |
| Cyclic-graph single fit, p=100, lambda=0.1 | 0.0141 | 0.0183 | fail |

All new-default rows pass. The stricter controls also take longer on some
previously accurate cases, such as the moderate-penalty quadratic signal.
This is a conservative accuracy calibration. It does not supply a new speedup
claim. Optimizing the additional work remains in #98/#99.

The [complete table](recorded/analysis/comparison.tsv) includes every case,
achieved errors and paired timing ratios. Ratios are medians of three
within-dataset median ratios, so they can differ from ratios of the two pooled
medians. These three dataset ratios describe variation, not a confidence
interval. The EPYC 7513 allocation is shared and nonexclusive, with one CPU and
one BLAS/OpenMP thread per worker. These new datasets and controls differ from
the earlier glmgen/genlasso timing tables; do not combine their times into a
single before/after comparison.

## What the stopping tests establish

For the ADMM split `K*beta=z`, the recorded primal residual is `K*beta-z`.
The dual residual is `rho*K'*(z-z_previous)`. The normal-equation update gives

```
X'*(X*beta-y) + rho*K'*u = -rho*K'*(z-z_previous).
```

The proximal step supplies `rho*u` in the subgradient of the penalty at z.
Thus the residual tests check approximate feasibility and stationarity. Small
residuals do not imply a uniform relative objective or coefficient error
without additional problem-dependent bounds.

The linear-signal feasible objective-gap rule remains confined to its
previously calibrated branch. Its stable dual identity is independently
tested with dense algebra for orders one through three and regular/irregular
operators; this mathematical identity does not by itself select a stopping
threshold for the higher orders. Those fits use the stricter residual controls.

For the fresh full-rank regression designs, a separate bounded dual QP supplies
a feasible vector u at the least-penalized requested level. With
`H=X'X`, `h=X'y`, `alpha=n*lambda`, `bd=H\(h-D'u)`, the stable gap is

```
(beta-bd)'*H*(beta-bd)/2 + alpha*sum(abs,D*beta) - dot(u,D*beta).
```

Clipping u to `[-alpha,alpha]` makes its dual lower bound valid irrespective
of QP termination. The largest candidate relative gap at these checked
regression levels is **2.08e-8**. This is an additional independent optimality
check, not just agreement with another package's coefficient vector.
It does not cover every path level or rank-deficient designs. Coefficient
agreement remains separately checked: even an excellent objective gap can
coexist with coefficient sensitivity on correlated or nonidentifiable designs.

## Validation and provenance

The calibration inputs and default controls were retained from the earlier
8201–8203 campaign. Fresh seeds **8501–8503** cover regular quadratic signals
at n=1,000/10,000 and R penalties 0.1/1/10; irregular n=10,000 at the same
penalties; 50-level paths at both sizes; and correlated p=100 fused/graph
regression single fits and paths.

The campaign retains **270 timed fits/paths** and **90 dataset/source accuracy
records**, plus per-level diagnostics. All 45 untimed R reference fits report
success. Signal references use glmgen `obj_tol=1e-12`; regression uses the
specialized genlasso routine, with independent dual-QP checks. Failed original
defaults remain visible. The data do not retune the candidate or R settings.

The final source passes **882/882 checks on each Julia 1.13.2 and 1.10.12**
on Discovery, following the initial 876/876 checks on each version. The final
suite adds Float32 regression coverage to the exact small references, dense
gap identities, independent R
golden paths, zero/constant cases, explicit/absolute-only tolerances,
rank-deficient fallback checks, CV and Aqua.

The [recorded source/dependency provenance](recorded/provenance.json),
[frozen selection](recorded/selection.json),
[raw timing and accuracy tables](recorded/holdout/),
[allocation accounting](recorded/accounting.tsv), and
[recreation recipe](README.md) accompany the report. Source/test snapshots and
large inputs/reference/Julia coefficient matrices remain in the task's remote
run directory. No local Julia process runs during this work.

Two short launcher failures preceded numerical execution: missing Git metadata
in the source snapshot (15 seconds) and an unloaded Apptainer module (2 seconds).
Their logs, repairs and allocation time are retained. This run uses the existing
remaining task allowance and preserves the 3-CPU/12-GB aggregate cap. Total
active allocation time is **14 min 57 s**, including recovery and final checks;
queue waiting is excluded. All jobs finish and the private SSH session is closed.
