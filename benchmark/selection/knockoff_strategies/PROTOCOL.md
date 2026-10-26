# Knockoff construction evidence protocol

This protocol is fixed before generating fresh evaluation datasets. The API
and first correctness canary are described in DESIGN.md. Canary and pilot
inputs are developmental and never count toward the independent main cohort.
The user authorized this campaign with compute as needed on Discovery. Nothing
is merged automatically. All numerical computation, including aggregation,
runs under Slurm; local work is source preparation and artifact transfer only.

## Estimands and procedures

The primary contrast changes only the Gaussian knockoff diagonal, keeping the
population covariance, X, y, independent normal innovations, native conditional
sampler, joint-lasso engine, penalty .04 and knockoff+ target .1 fixed. Compare
native equicorrelation, the MVR adapter, and the maximum-entropy adapter.
The native filter performs no intercept fitting or column standardization.
Changing the lasso statistic or selecting its penalty from y is outside this
comparison. Main fitting controls are fixed after a separate pilot certificate
gate and before any main inputs are generated (native cyclic order,tol1e-9,max100000; R glmnet5.1 thresh1e-16,max1000000, after pilot calibration). The independent Float64 relative
proximal fixed-point residual must be <=1e-7; a backend convergence flag alone
is insufficient. Failed constructions/fits remain explicit outcomes.

The adapter uses Knockoffs.jl 3.1.4 at commit
`d23942e9af9103f93be9a20e47ae6f7cad685aa5` only for diagonal optimization.
For MVR/maximum entropy the objectives are on the correlation scale. Validate
original, unperturbed objectives after mapping to input precision/units:
`tr(inv(2R-D))+sum(1/a)` and `-logdet(2R-D)-sum(log(a))` respectively,
where `D=diag(a)`. No objective against a lambda-min perturbed matrix can
replace this check. Retain backend settings and independent residuals, with
actual backend iteration count/status explicitly unavailable.

## Statistical cohort

Each scenario has seven independent worker/process shards with 144 independent
datasets per shard: 1,008 per scenario. The 13 main scenarios are iid, AR(.4),
AR(.8), weak AR(.4), and wide AR(.4), each with alternating and same-sign
signals, plus iid/AR(.4)/AR(.8) global nulls. Ordinary n=300,p=100; wide
n=180,p=300. Non-null support is predictors 1:20. Coefficients have magnitude
4.5/sqrt(n), except weak 3/sqrt(n); alternating signs start positive. Rows
are independent Gaussian with mean zero and the named population covariance;
noise is independent Gaussian with known standard deviation one. Global-null
power is undefined and is not replaced by zero.

Three separate stress scenarios use iid, AR(.4), and AR(.8) alternating-sign
laws but construct knockoffs with a deliberately misspecified covariance:
AR(.4), iid, and AR(.4), respectively. They are departures from exact model-X
assumptions, and cannot support an FDR guarantee. No covariance estimator is
introduced or tuned. The true covariance and the supplied covariance are both
saved. The total planned statistical cohort is 16,128 inputs, analyzed by
scenario rather than pooled into a universal performance estimate.

Main seeds are `106_600_000_000 + 1_000_000*scenario + 10_000*shard + dataset`.
Pilot shard zero is excluded. Within each dataset the draw order is X standard
normals, response noise, and knockoff innovations. No scenario, signal, lambda,
selection cutoff, seed or stopping control changes after examining main outcomes.

Report FDR (mean FDP), power, true/false/total selections, failure rates and
paired method differences. Give dataset-level Monte Carlo SEs and 95% intervals;
use paired intervals for differences and Wilson intervals for global-null
nonempty-selection probabilities and failures. Any conditional-on-success
summary must state its denominator and cannot hide unsuccessful datasets.
Use conservative worst-case FDP bounds when failures preclude an unconditional
point estimate. Include worker-level summaries to reveal shard anomalies.
The worst-case SE of a mean in [0,1] at 1,008 datasets is <=.0158. Smaller
contrasts may remain unresolved; a positive sample difference alone is not a
recommendation. The old 840-input study is context, never pooled as fresh data.

For a bounded independent solver comparison, use the exact saved augmented
matrices for the first three inputs of every shard/scenario with pinned R
glmnet. Check every fit's objective/residual/selected sets; do not assign solver
accuracy rankings to failed certificates. The primary construction contrast
uses one common lasso engine throughout, so comparator solver differences do
not become construction effects.

## Independent construction checks and comparators

Before the main cohort, check Float32/64, iid/AR(.4)/AR(.8), unequal scales,
1e-20/1/1e20 units, singular conditional boundaries and infeasible/user diagonals.
Compare the adapter's diagonal with direct pinned Knockoffs.jl optimization
using identical controls; separately retain its package defaults and robust
updates as sensitivity settings. Independently reconstruct the full 2p joint
matrix, its eigenvalues, objective and finite-difference gradient, the equation
Sigma*A=D, and C'C=2D-D*A. Check both fixed-X Gram identities and its rank/n>=2p
restrictions. Recover the sampler's linear map from supplied innovations.
Input preservation, reproducible RNG, forwarding and explicit failure handling
are required; silently substituting a different diagonal is a defect.

For small independent matrices (p=4,12,30), compare against CVXPY convex
formulations with an independent conic solver; report status, feasibility,
stationarity and objective gaps. The oracle is a numerical comparator, not an
exact proof. Pinned R knockoff provides applicable equicorrelated diagonal and
Gaussian/fixed sampler comparisons; it does not implement the same MVR/maxent
optimizer, so do not relabel its SDP as one. Hash installed relevant source and
retain actual Julia/R/Python package versions. External full samplers must pass
root/joint identities before valid statistical rankings. Historical upstream
factor-orientation failure is checked, retained, and never silently repaired.

## Runtime, defaults and profiling

Use seven independent process replications and three fresh shared inputs per
case, at least three warmed timing rounds per input, rotated method order,
separate profiling calls and exclusive one-CPU srun steps launched by GNU
Parallel. All BLAS/OpenMP/Julia threads are one. Runtime seeds use the disjoint
family `106_900_000_000 + 1_000_000*case + 10_000*rep + input`.

Cases: Gaussian iid (600,200), AR(.4) (600,200), AR(.8) (400,400), wide
AR(.4) (200,600), and fixed-X AR(.4) (600,200). The fixed-X design has full
column rank; wide fixed-X is unavailable and must not be approximated. Test
unequal predictor scales separately from timing. Compare diagonal selection,
sampling with a supplied validated diagonal, full construction and full
filtering. The baseline and candidate receive identical data/innovations;
record identical legacy equicorrelation outputs as a compatibility witness.
Compare direct established optimizer plus native sampler with the adapter,
and the applicable R equicorrelation/fixed constructors. Unavailable matched
external procedures are stated explicitly. Do not claim a new optimizer or a
universal speedup from wrapping the existing one.

Report per-process medians, paired ratios and uncertainty, CPU/allocation
profiles, actual source/CPU/thread records and resource exit outcomes. Numerical
checks run outside timed regions. Julia heap bytes exclude foreign allocations;
whole-process peak RSS also contains imports/observer state. Keep their scopes
explicit and avoid unjustified fitted-state memory rankings. No caching/factor
fast path is introduced here; those belong to issue108.

Equicorrelation and safety1e-8 remain the compatibility defaults. The provisional
optimized controls (niter1000,tol1e-10,lambda_min1e-10,robustfalse,independent
relative stationarity <=max(1e-6,100eps(T))) are investigated on a fixed matrix
suite before the main cohort. Retain package-default, robust-update and tighter
sensitivity results. A failure must be explicit; do not choose controls per
main dataset. Statistical results can justify scoped usage guidance without
changing the compatibility default. A default change needs separate evidence
of benefit, failures and cost across the tested laws; no procedure gains an
FDR guarantee by selecting an optimized diagonal.

## Retention and completion

Keep job IDs, immutable source/env references, fixed settings/seeds, raw reporting
rows, failures, certificates, timing/profiling evidence and a recreation recipe.
Reuse valid outputs during recovery. Preserve adverse outcomes and exclusions.
Transfer compact reporting inputs and profiles; raw generated matrices can
remain on Discovery while needed. No routine whole-run archive or cleanup.
A completed scheduler job is not evidence that scientific acceptance passed.
The PR remains draft until independent correctness, replicated evidence and
all final-head hosted CI/docs checks pass. Never merge or enable auto-merge.
