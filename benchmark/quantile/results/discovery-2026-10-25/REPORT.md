# Quantile-lasso solver selection on Discovery, 2026-10-25

This study starts [issue78](https://github.com/mlakolar/HighDimensionalModels.jl/issues/78). It supplies experimental solvers, independent golden comparisons, repeated timings, CPU profiles and a measured implementation direction. It does **not** complete the issue: the supported quantile API, pivotal penalty, check-loss CV, post-selection refits and debiased inference remain to be implemented. QR.jl was inspected, not ported.

## Recommended implementation direction

1. **Exact route:** build around the compact dual LP, HiGHS dual simplex, a one-time analytic intercept-only basis, and reuse of the model/basis along a decreasing lambda grid. Keep an exact primal formulation as a validation oracle and alternative backend. The analytic start materially improved the tested LP paths, including completion of all six 100000-row paths under the 30-second cap.
2. **Smoothed route:** expose the existing Epanechnikov convolution loss through the proximal Newton path, with explicit bandwidth and post-fit KKT diagnostics. It was usually the fastest tested route. Keep accelerated proximal gradient as an alternative for very wide designs: its observed mean was lower at n1000,p10000, but the paired three-input interval does not yet establish a stable crossover threshold.
3. **ADMM:** retain these implementations as research baselines; do not select them as the default on this evidence. Factorization, normalization, fixed-rho sensitivity, residual balancing and separate confirmation did not produce a competitive tested exact path. This is not a result about every ADMM implementation.
4. **Parametric simplex:** pursue it after the exact grid API and basis handling are stable. Analytic startup plus grid continuation is implemented here; a breakpoint-enumerating algorithm is not. The dense-grid results support reuse, but do not measure an unimplemented parametric method.

These are conditional engineering choices for the designs, lambda range and tolerances below. Numerical speed does not establish statistical equivalence between the exact and smoothed estimators.

## What QR.jl actually did

At revision `db35d869481847d10c616aa22099bcf2b3e3a203`, the simulation helpers select Gurobi `Method=1`, which is **dual simplex**. The solver constructs the primal residual/absolute-value epigraph LP once and updates its objective along the path. That permits retained optimizer state and basis reuse; there is no explicit coefficient-start or basis-transfer code. The old JuMP/Gurobi stack was not reproduced. The tests' Mosek FREE_SIMPLEX setting does not specify one fixed variant. See the [source audit](../../QR_AUDIT.md) and [Gurobi's Method documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameter:Method).

The current measurements show why formulation and algorithm must be chosen separately. On the QR-style primal epigraph LP, modern Gurobi primal simplex beat its dual simplex on the small, correlated and wide paths. The compact dual LP with HiGHS dual simplex performed better across more of the tested dimensions. A lambda change alters objective coefficients in the primal LP and constraint bounds in the compact dual; these are different reoptimization problems.

## Model, design and timing protocol

All methods target an unpenalized intercept with objective

```
mean(rho_tau(y - a - X*b)) + lambda * sum(abs, b).
```

Columns have RMS1 in the main study. The R/golden fixtures are centered and have sample SD1, matching conquer's internal standardization. Unit penalty weights are held fixed. The compact dual maximizes `y'u/n`, subject to `sum(u)=0`, `tau-1 <= u <= tau`, and `abs(X'u) <= n*lambda`. The primal epigraph and positive/negative coefficient formulations are algebraically equivalent.

Main strata use ten nonzero slopes and an intercept:

|Case|n|p|Design/error|tau|
|---|---:|---:|---|---:|
|small|1000|100|Independent Gaussian / Gaussian|0.5|
|correlated|1000|100|AR(0.9) / heteroscedastic t3|0.2|
|wide|1000|1000|AR(0.5) / heteroscedastic t3|0.5|
|tall|10000|100|AR(0.5) / Gaussian|0.5|
|large|10000|1000|AR(0.5) / heteroscedastic t3|0.5|
|ultrawide|1000|10000|Independent Gaussian / Gaussian|0.5|
|tall100k|100000|100|Independent Gaussian / Gaussian|0.5|
|sparse100k|100000|10000|Sparse Gaussian, density0.001 / Gaussian|0.5|

Each path has eight log-spaced penalties from 1.05 times the empirical intercept-only score maximum down to 0.1 times that maximum. This is not a near-unpenalized path. Three independent input/process seeds per stratum each receive two shuffled timing blocks. Compilation on smaller matching dense/sparse types, input generation, explicit premeasurement GC and external certificates are excluded. A timed path starts fresh: model creation, updates, optimization and solution extraction are included; no basis is shared between repeated paths. Cold LP paths rebuild at every lambda. Gurobi environment startup is excluded. Julia, BLAS and optimizer threads are one, with exclusive core-bound srun workers on AMD EPYC7513 nodes; allocations do not reserve entire nodes.

Means below average the two repetitions within input, then the three input means. SD and 95% Student-t intervals use those three inputs, not six independent samples. Paired contrasts are retained in [paired-contrasts.json](shared-bounds/report-inputs/paired-contrasts.json). These small intervals describe observed input variability and do not exhaust machine-to-machine uncertainty. Incomplete groups are not ranked by the mean of surviving paths.

Exact fits require a finite primal, feasible projected/contracted dual, gap between -1e-7 and `1e-5*(1+abs(primal))`, and normalized intercept-dual residual below1e-10 at all eight penalties. Smoothed fits require proximal KKT residual at most2e-5 at all eight penalties; their check-loss objective gap is reported separately. Native convergence flags remain visible and are not substituted for this external check. In particular, Newton warnings occurred on some difficult paths whose final KKT residuals passed.

Native rules differ: LP feasibility/optimality tolerances1e-7; quantreg timing eps1e-7; conquer timing coefficient-step tolerance1e-6; APG KKT tolerance1e-6; Newton relative objective tolerance1e-10 and inner coordinate tolerance1e-8. R returns coefficients before dual reconstruction; native LP timing includes its available dual extraction, and Julia smooth paths include score extraction. Therefore close timings at different achieved residuals do not establish general package superiority. R allocation/iteration zeros in original CSVs are placeholders; the derived views mark them NaN. Julia allocation counts omit most native LP allocations.

## Exact LP: warm starts and formulation

Selected main-study full-path means, seconds. All numeric entries below have6/6 qualified paths unless a fraction is shown. Analytic bases were introduced in a separate cohort, reported next.

|Case|Gurobi primal LP, dual simplex warm|Gurobi primal LP, primal simplex warm|HiGHS compact dual, dual simplex cold|HiGHS compact dual, dual simplex warm|
|---|---:|---:|---:|---:|
|small|0.7969|0.1996|1.155|0.2079|
|correlated|0.5939|0.1858|1.019|0.1799|
|wide|15.44|3.125|13.21|2.372|
|tall|2/6; not ranked|27.42|13.07|2.247|
|large|0/6|0/6|Not run|21.44|
|tall100k|0/6|0/6|Not run|4/6; not ranked|
|ultrawide|0/6|0/6|Not run|0/6|
|sparse100k|0/6|0/6|Not run|0/6|

The 30-second budget covers an entire LP path, including setup. Budget failures stay in the data and are not timing wins. Warm reuse is not uniformly beneficial: on the tall QR-style primal LP, Gurobi dual simplex cold averaged6.728 seconds with6/6 qualified paths, whereas warm completed2/6 near the cap. These measurements do not isolate whether presolve, basis geometry or another optimizer detail caused that reversal.

For compact-dual cold versus warm paths, paired mean time ratios were5.56[5.18,5.94] on small and5.82[5.34,6.30] on tall. For the QR-style Gurobi primal LP, dual-simplex/primal-simplex paired ratios were4.00[3.54,4.45] on small and4.95[4.10,5.80] on wide. Full method tables, including IPM/barrier and HiGHS primal formulations, are in [the main table](shared-bounds/summary/TABLE.md). Barrier reuse denotes model reuse; it is not claimed to be simplex-basis warm starting.

## Analytic startup and a fair split-LP check

The compact-dual basis sets observations below/above an empirical quantile at their lower/upper bounds, makes one quantile observation basic, and makes the remaining inequality slacks basic. It is applied once through HiGHS's documented C basis setter, then ordinary continuation proceeds. The corresponding split-primal basis uses the intercept and the appropriate signed residuals as basic variables. Both basis constructions give zero startup pivots at a sufficiently large penalty in the canaries;30 compact-dual and60 split-basis assertions passed across n60/61, tau0.2/0.5/0.8 and multiple penalties.

This fresh cohort uses Gaussian AR(0.5) inputs for n1000 and the **identical saved R inputs and lambda grid** for n100000. It does not pool the main study's timings. All completed entries are6/6, with mean, input SD and95% interval in seconds:

|Shape|Default compact dual|Analytic compact dual|Analytic split, primal simplex|Analytic split, dual simplex|
|---|---|---|---|---|
|1000 x100|0.2034; SD0.00594 [0.189,0.218]|**0.1160; SD0.00758 [0.0972,0.135]**|0.1957; SD0.00641 [0.180,0.212]|0.2852; SD0.0220 [0.231,0.340]|
|1000 x1000|2.415; SD0.135 [2.08,2.75]|**1.518; SD0.152 [1.14,1.90]**|2.302; SD0.296 [1.56,3.04]|4.056; SD1.05 [1.46,6.66]|
|100000 x100|0/6 under30s|**22.00; SD0.691 [20.3,23.7]**|Not run|Not run|
|1000 x10000|0/6|0/6|0/6|0/6|

The paired default/analytic compact-dual ratio was1.76[1.38,2.14] on small and1.60[1.38,1.81] on wide. The common-cap ultrawide cohort also tried uninitialized split primal and dual simplex; neither completed a full path. Thus the earlier split study's4-second cap did not alone explain its failure at that shape.

The initial basis job failed when widest split workers exceeded4 GiB. Completed and capped observations were retained. A separate2 CPU/16 GiB allocation ran missing or validation-interrupted cells with8 GiB per worker, and added the matched analytic split starts. The final [96-observation table](shared-bounds/basis-summary/TABLE.md) and [merged raw rows](basis-completed/) record `worker_mem_gb` and `attempt`; original and recovery rows remain separately available. The mixed memory limits and jobs are part of this comparison. The OOM establishes a limitation of this JuMP implementation/allocation, not an intrinsic lower bound on split-LP memory.

## Smoothing and the R comparison

Main-study smoothed path means and95% intervals, seconds, at h0.2. Every entry has6/6 KKT-qualified paths:

|Shape/case|Production proximal Newton|Experimental APG|
|---|---:|---:|
|1000 x100 small|0.003843 [0.00276,0.00492]|0.009321 [0.00702,0.0116]|
|1000 x100 correlated|0.004986 [0.00349,0.00648]|0.04953 [0.0452,0.0539]|
|1000 x1000 wide|0.01565 [0.00913,0.0222]|0.08997 [0.0879,0.0920]|
|10000 x100|0.02551 [0.0220,0.0290]|0.1173 [0.0891,0.146]|
|10000 x1000|0.1070 [0.0733,0.141]|1.587 [0.967,2.21]|
|100000 x100|0.3000 [0.213,0.387]|1.210 [0.889,1.53]|
|1000 x10000|9.262 [6.61,11.9]|5.547 [2.32,8.77]|
|100000 x10000 sparse|0.7822 [0.674,0.890]|1.982 [1.85,2.12]|

Reducing h from0.2 to0.05 increased Newton's wide-path mean from0.01565 to0.4001 seconds while retaining the KKT gate. A fixed h0.2 is not a statistical bandwidth recommendation. At ultrawide the paired APG-minus-Newton time difference was -3.72 seconds, with95% interval[-9.07,1.64]; prefer to retain an alternative solver, not hard-code a crossover from three inputs.

The separate R comparison uses identical binary inputs, three fresh seeds, two blocks and p100. quantreg solves each lambda independently with its specialized Frisch–Newton method. conquer uses the parabolic kernel, h0.2 and its native ascending-lambda order; Julia uses decreasing lambda. Both timings exclude process startup, data generation and binary I/O; conquer's internal normalization and initialization remain included. All values shown below have6/6 qualified paths:

|n|quantreg Frisch–Newton|conquer|Julia proximal Newton, same h|
|---:|---:|---:|---:|
|1000|0.2461 [0.231,0.261]|0.03342 [0.0273,0.0396]|0.003403 [0.00306,0.00374]|
|10000|3.086 [2.85,3.32]|0.2172 [0.195,0.239]|0.02820 [0.0257,0.0307]|
|100000|41.52 [38.0,45.0]|4.246 [3.94,4.55]|0.4167 [0.323,0.510]|

The 22.0-second analytic LP result above used these exact 100000-row inputs in another allocation. Its elapsed times are not paired machine observations with the R run. The native stopping rules and output work also differ. The large observed smooth-path speed difference motivates the production Newton route; it is not a universal claim over conquer's kernels, settings or data distributions.

Golden data cover tau0.2/0.5/0.8 and lambda0.03/0.1. Maximum quantreg objective disagreement was2.20e-13; rqPen agreed to machine precision. The largest smoothed coefficient discrepancy with conquer was9.38e-5, with R KKT below8.00e-9 and Julia KKT below7.85e-6. See [corrected golden comparisons](golden/R-comparison-v2.csv).

A normalization error in the first R check was diagnosed and retained: `rq.fit.lasso` pseudo-observations contribute half their supplied L1 penalty to the sum-loss objective, so the matching argument is `2*n*lambda`, not `n*lambda`. rqPen uses `scalex=FALSE`, `alg="br"`, and the original mean-loss lambda. Both R comparison versions and the inspected quantreg function are saved. No oracle discrepancy was silently discarded.

Quantreg's reconstructed dual was sometimes loose; it initially qualified only2/6 tall paths. The final view uses the strongest independently feasible saved lower bound at each **identical study/input/lambda**, with each candidate's own primal and time unchanged. All6 then qualified, at maximum gap1.26e-10. The transformation also improves other methods' bounds. Original dual/gap fields, source method and replicate are retained, raw files are untouched, and [the audit](shared-bounds/bound-audit.json) reports zero gaps below -1e-7. No bounds are shared across distinct study datasets.

Saved-fit approximation checks separate optimization from estimator discrepancy. Across144 comparisons at h0.2, maximum conquer check-objective excess over the exact fit was2.67e-4/2.75e-5/2.71e-6 at n1000/10000/100000. Maximum training-prediction RMS difference was0.0413/0.0161/0.00612. Feasible smoothed dual gaps were below9.71e-6. The Epanechnikov conjugate calculation passed1206 value/derivative assertions. These finite-design observations do not prove first-order equivalence, coverage or an appropriate bandwidth rate. See [approximation details](shared-bounds/report-inputs/approximation-summary.json).

## ADMM, profiling and path density

The normalized factorized two-block ADMM includes factorization in path time and reuses it along the path, with Woodbury for wide matrices. On fresh AR(0.5) Gaussian inputs it took3.929[3.64,4.22]seconds on1000x100 and27.76[24.0,31.5]seconds on10000x100; the matched compact-dual LP took0.2111 and2.582 seconds. Both methods qualified6/6 there. Wide ADMM paths did not complete under30 seconds. Linearized ADMM was less reliable. See [the ADMM table](shared-bounds/admm-summary/TABLE.md).

To check whether this was merely an unfortunate rho, fixed0.01/0.1/1/10 and residual balancing were tuned on seed901 using certified original-objective gap stopping. Fixedrho0.1 was selected for the small shape and held fixed on seeds902–904: confirmation took1.109[1.07,1.14]seconds versus LP0.2101[0.189,0.231], both6/6. No wide candidate completed its tuning path, so there was no eligible wide confirmation choice. The ten failed/successful tuning paths remain in [tuning.csv](rho-results/tuning.csv). This conditional absence is not missing experimental coverage. The baseline's additional split-residual condition and the later gap-only candidates are distinguished in the code.

There are28 repeated CPU traces: two dimensions, two seeds and seven methods. Fast paths are repeated for at least1.5 seconds after warmup; a slow individual call can extend beyond that target. These traces are separate from timing blocks and some capped LP/ADMM traces represent partial paths. Native frames were enabled. Representative traces at n1000, seed501:

- HiGHS compact dual at p100: 1155/1417 self counts were native HiGHS. Both independent seeds show the native optimizer dominating the LP, with setup/extraction still included in timings.
- Gurobi dual-simplex primal LP:1634/1801 self counts were native Gurobi at p1000.
- Linearized ADMM:1864/1934 self counts were BLAS at p1000; APG1470/1556. Repeated matrix products are a concrete cost in these implementations.
- Newton's p1000 trace has many self samples in inlined scalar arithmetic/indexing under the production coordinate path. The broad automatic category “other Julia/runtime” cannot identify one source-line bottleneck. Its small count of frames literally named ProximalSolvers must not be interpreted as little production-solver work.

Exact counts and full stacks are in [profile-summary.json](shared-bounds/report-inputs/profile-summary.json) and [profiles/](profiles/). The pilot's one-call Newton trace had only three snapshots and is not used for a hotspot claim.

A separate dense-grid study holds the QR-style **primal epigraph formulation** fixed, with3 inputs x3 blocks x3grid sizes x4solver choices:108qualified timings. Mean seconds (input SD):

|Simplex method|8 levels|32 levels|128 levels|
|---|---:|---:|---:|
|Gurobi dual|0.7835 (0.0481)|0.9323 (0.0307)|1.8087 (0.0656)|
|Gurobi primal|0.1829 (0.00856)|0.3497 (0.0155)|0.7771 (0.0225)|
|HiGHS dual|1.2085 (0.0805)|1.3258 (0.0376)|2.0475 (0.178)|
|HiGHS primal|0.5810 (0.0331)|0.6978 (0.0172)|1.1546 (0.0262)|

Grid reuse scales much more slowly than the number of requested lambdas here; for Gurobi dual, mean pivots grow from5172 to6962 between8and128 levels. This does not measure the compact-dual formulation or parametric breakpoint enumeration. For a fixed basis the penalized quantile LP coefficients can be constant over lambda intervals, with nonunique knot solutions. Continuation in the L1 constraint bound is a different parametrization. See [Osborne and Turlach](https://maths-people.anu.edu.au/~mike/PLHty.pdf); least-squares lasso interpolation should not be transplanted unchecked.

## Execution record, limits and next implementation

All Julia/R fits, tests, repeated timings, profiling and numerical aggregation ran on Discovery under Slurm. Actual allocations used97.12 minutes of running-time union,4.476 allocated CPU-hours, at most4 concurrent CPUs and16 GiB allocated memory. The main job was13423142; followups, failures and exact resources appear in [accounting-summary.json](accounting-summary.json). The final recovery13423694 completed, its workers exited0, and all planned study counts passed [coverage checks](shared-bounds/report-inputs/coverage.json):552 main,46 pilot,72 ADMM,90 R,12 conditional-rho,54initial split and96 basis timing paths, plus108 density paths and144 saved-fit approximation comparisons. A “complete observation” may record a scientific time-budget failure; it is not necessarily a successful fit.

The study pins CoordinateDescent0.8.0 at `c408831e2b011e83fa427248bf074dc0fc91d870`, ProximalBase at `0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a`, and ProximalSolvers at `9041e236f3de8ffdc02103e5a505993463e00bb1`. Runtime: Julia1.13.2, HiGHS1.15.1/HiGHS.jl1.26.1, JuMP1.32.1, Gurobi12.0.3/Gurobi.jl1.9.3; R4.5.3, quantreg6.1, conquer1.3.3, rqPen4.2. The repository was subsequently renamed HighDimensionalModels.jl; these measurements use the pinned pre-rename source, not an unmeasured new package version.

The exact `[sources]` directories and resolved Manifest are retained. [REPRODUCE.md](../../REPRODUCE.md) describes the canonical harness and immutable executed versions. Large generated R design binaries remain in `/home1/mkolar/juliaPkg/runs/20261025-cd78-quantile`; their generation seeds and [hashes](rbench-input-sha256.json) are recorded. No claim depends on a local scientific run.

Limits include only three inputs per stratum, predominantly median regression, a lower lambda endpoint0.1lambda_max, two fixed bandwidths, shared-node timing and differing native stopping rules/output work. There is no full dense100000x10000 cell: its design alone requires8GB before solver copies. There is no memory-efficient specialized LP implementation, comprehensive ADMM optimization, low-lambda support stress test, statistical bandwidth study or inference result. The next code stage should add the two public estimators with weighted penalties and shared diagnostics, then pivotal lambda/check-loss CV, with post-selection and debiasing work following separately. Keep issue78 open.
