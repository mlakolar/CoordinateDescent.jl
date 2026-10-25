# Linear trend-filter stopping: decision and evidence

The targeted stopping change is retained in PR #83. General tolerance defaults
are unchanged. Float64 linear signal fits use a feasible objective-gap check
with a residual guard; other orders, designs, Float32 fits and absolute-only
tolerances retain residual stopping.

On three fresh datasets, the new rule passes all **15 workloads**, versus
**12/15** for the original defaults. In the experiment that measured old and
new source in one allocation, the million-point linear fit takes **27.64 s
before and 21.96 s after**, about 20% less time. Every number is a median of
nine timings: three independently generated datasets, each timed three times.
The paired speedup is 1.25×, with dataset ratios 1.24–1.30.

## Why a targeted rule

The diagnostic retains the original ADMM updates and `rho`. It checks 13
single-fit cases on each of three original datasets, including the previous
million-point data. The result rejects a global relaxation of the residual
tolerances:

| Diagnostic rule | Dataset/case combinations passing accuracy |
|---|---:|
| Original residual thresholds | 35/39 |
| Residual thresholds ×2 | 33/39 |
| Residual thresholds ×5 | 21/39 |
| Relative gap 1e-6 with residual guard | 39/39 |
| Relative gap 3e-7 with residual guard | 39/39 |
| Relative gap 1e-7 with residual guard | 39/39 |

The full replicated calibration adds two 50-level paths, for 15 workloads and
45 dataset/case combinations. All three gap rules pass every case. The
selected 3e-7 rule keeps a margin below the 1e-6 objective target; the fastest
1e-6 rule approaches the boundary. Selection is frozen in
[selection.json](recorded/selection.json) before evaluating Julia on the new
datasets. The new data do not select or tune the rule.

At a million observations, the old rule takes 457–481 iterations on the
calibration data, compared with approximately 350–360 for the selected rule.
The old rule obtains more precision than this comparison requires. On other
inputs, particularly strong penalties, the residual rule can stop too early
for objective accuracy. A gap check addresses both behaviors.

## Held-out performance and competitiveness

Seeds 8301–8303 generate three new datasets. The cases cover lengths
1,000–100,000 at R penalties 0.1, 1 and 10; irregular 10,000-point grids;
50-level paths at 1,000 and 10,000 points; and a million-point fit at R λ=1.
Both languages receive the same exported inputs. Previous glmgen controls
are transported, without a new tolerance sweep, and checked against fresh
untimed `obj_tol=1e-12` references.

The following old/new measurements share one allocation. Source order
alternates by dataset, and each source runs in its own process with warmup.

| Workload | Original defaults, seconds | New rule, seconds | Paired speedup | Original accuracy |
|---|---:|---:|---:|---|
| n=100,000, R λ=1 | 2.372 | 1.851 | 1.27× | pass |
| n=1,000,000, R λ=1 | 27.643 | 21.963 | 1.25× | pass |
| 50-level path, n=10,000 | 7.245 | 6.547 | 1.12× | pass |
| n=1,000, R λ=10 | 0.0122 | 0.0140 | 0.87× | fail |
| Irregular n=10,000, R λ=10 | 0.239 | 0.335 | 0.70× | fail |

All new-rule rows pass accuracy. Strong-penalty cases can take longer because
the new rule obtains the required accuracy. The 1,000-point path also needs
more work to meet the target. Even among cases where the old defaults pass,
the rule is not uniformly faster: regular n=10,000 at R λ=10 takes about
14% longer in the final-source comparison. These outcomes remain in the
[complete paired table](recorded/same-allocation-ratios.tsv) and raw records.

After retaining the original short-circuit residual calculation for all
unaffected models, the final source is validated again and all 135 candidate
timings are repeated. Its signal computations are unchanged. The later
allocation gives the following confirmation and comparison with the
transported glmgen controls:

| Workload | Final Julia, seconds | glmgen, seconds |
|---|---:|---:|
| n=1,000, R λ=1 | 0.0175 | 0.0170 |
| n=10,000, R λ=1 | 0.176 | 0.185 |
| n=100,000, R λ=1 | 1.902 | 2.236 |
| n=1,000,000, R λ=1 | 19.660 | 29.931 |
| Irregular n=10,000, R λ=1 | 0.204 | 0.189 |
| n=1,000, R λ=10 | 0.0151 | 0.0122 |
| 50-level path, n=10,000 | 6.767 | 15.540 |

The linear implementation is competitive on these inputs, with clear R wins
remaining in some small and irregular cases. These new-data timings differ
from the earlier 30.10 s versus 24.18 s table, which used different datasets
and allocations. They should not be combined into a single before/after
speedup. Allocations use shared, nonexclusive EPYC 7513 nodes. The later
confirmation runs separately from the original source and R, so the
shared-allocation experiment above provides the cleaner estimate of the
stopping-rule gain. Dataset ranges are descriptive, not confidence intervals.

## Accuracy and implementation

The acceptance targets remain relative coefficient error ≤1e-4 and relative
objective difference ≤1e-6. They apply against the tighter R fit and the
transported timed R fit. The timed R fit must itself pass against the tighter
reference. All 45 tighter references report success. Coefficients from the
first timing round of each dataset/setting are checked outside timing; all
nine timings retain iterations and convergence flags.

On the held-out data, the final rule has maximum relative coefficient error
**7.35e-6** and maximum relative objective difference **2.99e-7** against the
tighter reference. The original defaults reach maximum objective difference
2.33e-6 and fail the strong-penalty regular/irregular fits and the 1,000-point
path. There is no claim of uniform coefficient-error certification for
unseen data: the mathematical certificate controls objective suboptimality,
and coefficient agreement is checked empirically.

For the unnormalized signal objective
`P(beta)=||beta-y||²/2 + alpha*||D1*K*beta||₁`, any box-feasible
`q in [-alpha,alpha]` gives a dual lower bound, with `v=K' D1' q`.
Clipping cumulative ADMM multipliers supplies a feasible q. The gap is
evaluated as

```
P(beta) - dual = ||beta-y+v||²/2
                + sum_j(alpha*abs((D1*K*beta)_j)-q_j*(D1*K*beta)_j).
```

This avoids subtracting two nearly equal objective values. The implementation
reuses existing work vectors and performs no extra vector allocation for
the check. With positive `reltol`, Float64 linear signal fits require a gap
at most `0.3*reltol*P(beta)` and primal/dual residuals within five times their
usual bounds. Checks occur every ten iterations, plus the first and final
iteration. Near a zero objective, below `100*eps(T)*||y||²`, the original
residual criterion is used. Nonfinite gap/scale calculations cannot report
convergence. Public `abstol` and `reltol` defaults are not changed.

Three CPU and three sampled allocation profiles are retained for each source
on the fresh million-point data, separately from timings. TV denoising and
sparse operator products remain substantial costs. The improvement comes
from fewer iterations, rather than a replacement for those kernels. No
further kernel rewrite is included in this follow-up.

## Validation, recovery and provenance

- The final source passes **510 checks on each of Julia 1.13.2 and 1.10.12**
  on Discovery: 385 structured tests, 114 independent R golden checks and
  11 Aqua checks. New tests independently evaluate the dual formula with
  dense algebra, compare default small fits with exhaustive exact dual
  solutions, and cover constant/zero signals, paths, Float32 and
  absolute-only tolerances.
- Three calibration workers each pass 74 additional dense certificate
  checks. The final 15-case holdout has complete coverage and convergence
  for both sources; the new rule passes 15/15 accuracy gates, versus 12/15.
- The launcher initially resolved its path to Slurm's spool directory.
  That four-second attempt ran no fits. Its log is retained, and the
  corrected launcher uses an explicit staged path.
- Two first-wrapper calibration cases were repeated after making wrapper
  warmup explicit: 72 corrected timings replace only those cells. Original
  records remain available. Holdout timing warms the actual wrapper types.
- The concurrent package rename is integrated at commit `ceb9d36`. The
  calibrated `src/structured.jl` is byte-identical to source commit
  `5004825`; tests and the executable example use `HighDimensionalModels`.
  The current ProximalBase master has the same ordered-operator file as
  the frozen benchmark dependency. Canonical-package CI validates the
  integration with the master dependency workflow.

Run `20261025-cd80-stopping` uses **27 min 7 s** of the approved additional
45 active allocation minutes, including recovery. Concurrent requests never
exceed three CPUs and 12 GB; worker/BLAS/OpenMP thread counts are one. Queue
waiting is excluded. No Julia process, benchmark or profile runs on the
laptop. The [accounting table](recorded/accounting.tsv),
[source/dependency provenance](recorded/provenance.json),
[summary](recorded/summary/summary.json), and
[recreation instructions](README.md) retain the evidence. Large synthetic
inputs and R coefficient matrices remain in the remote run directory.

The [earlier full comparison](../compare/RESULTS.md) retains the other signal
orders and genlasso regression results. This follow-up changes only linear
signal stopping; it does not establish new performance or accuracy claims
for those other estimators. PR #83 remains draft for review.
