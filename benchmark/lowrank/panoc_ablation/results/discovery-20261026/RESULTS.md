# Guarded nuclear PANOC: replicated evidence and default recommendation

Adopt the guarded Float64 loss-product and final-diagnostic reuse in PR114.
The change improves the complete public workloads on the six tested designs,
preserves independent accuracy and returned rank/DF, and beats the matched
generic Julia comparison. Retain APG as the dependency-free package default;
explicit PANOC is a useful choice with the optional dependency loaded. Retain
PANOC's .95/L step, certificate cadence10, tolerance and iteration defaults.
The weak many-response fit and the controlled alternatives prevent a universal
solver/default or memory claim.

## Study and correctness

Production candidate14e057b, baseline295fbf5420d6384a1ca261023367ee86d0a40b8c,
frozen study2f03f2c. Discovery13451665 completed0:0 in57m31s:120 fresh shared
inputs (six designs×20 independent seeds),1800 successful fresh processes,
43200 warmed timing rows and39600 independent original-design certificates.
Paired reporter13453673 completed in1m57s; final saved-output analysis13454388
completed in56s. All numerical generation, fitting, validation, profiling and
aggregation ran under Slurm. Julia1.13.2, fixed EPYC7513, one Julia/BLAS/OpenMP
thread per worker. Sources, dependencies, inputs, CPU/thread records and recipe
are frozen in [PROTOCOL.md](../../study/PROTOCOL.md) and registry20261026-hdm103.
Rep0 is excluded throughout. No data-dependent rule/tolerance tuning occurred.

All39600 certificates pass relative original-design primal-dual gap and
normalized proximal-gradient mapping<=1e-6, including every path point in both
public/cached modes and every complete-result comparator. Maximum gap is
1.28722e-7 and maximum mapping9.99956e-9. Returned objectives, intercepts,
rank, selected rows, DF and native residuals pass independent checks.
The2640 actual native before/after saved point pairs have maximum relative
coefficient difference9.70695e-9 and relative objective difference5.01062e-16;
all ranks and DF agree. All19800 public/cached point pairs are bitwise equal.
These are comparisons of existing saved points, not new inputs/certificates.

All1800 source/provenance records pass;3600 CPU profiles are present and
nonempty, with no truncated preferred profile. All720 native cold setup
allocation witnesses have strictly greater public than cached allocations.
Exact untimed source-counter replays match timed coefficients and iterations;
no instrumentation runs inside a timer. Release/LTS correctness and116 new
arithmetic/cancellation/Float32/64/rank/zero/iteration checks, plus66 independent
prototype certificates, are recorded in the separate prototype directory.

## Complete public before/after workloads

Cold fits use fraction.1 of the common lambda maximum; paths contain eight
levels from.5 to.02. Values are medians across independent inputs of the
within-input median of three warmed rounds. Ratios are paired within inputs.
The interval is Student t on paired log ratios, exponentiated: it estimates a
geometric mean ratio, not the median ratio, and has no multiplicity adjustment.
Both complete native public APIs include returned diagnostics. Paths call the
actual public constructor with the same minimal solution-extraction adapter.

| Design | Before/after cold seconds | Median cold ratio | Before/after path seconds | Median path ratio | Path geomean95% interval |
|---|---|---:|---|---:|---|
| small | .0021372 / .0016403 | .76649 | .012250 / .0089081 | .72132 | .71426–.73097 |
| medium | .016438 / .014096 | .85839 | .12841 / .10617 | .82298 | .81935–.83151 |
| many responses | .12221 / .081528 | .67288 | 1.0767 / .74137 | .68944 | .66946–.70918 |
| wide | .22462 / .20987 | .93852 | 1.6202 / 1.4536 | .89893 | .88896–.91410 |
| ill-conditioned | .078843 / .063939 | .78706 | .70308 / .57553 | .81244 | .80075–.81722 |
| rank deficient | .022512 / .019128 | .85642 | .17659 / .14709 | .84010 | .82114–.86013 |

All public before/after group95% upper limits are below1, including the three
cold fractions separately. Public and common cached results remain separate
in comparison-summary.csv. Across all modes there are19 individual paired
timing regressions. In particular the weak, cached many-response case has
median ratio.91397 but geometric-mean interval.85816–1.03706, with8 inputs
slower and worst ratio1.41152. Do not claim uniform per-input/cached improvement.

## Alternatives, attribution and defaults

Matched generic Julia PANOC uses the same safe .95/L step and complete outputs
with ProximalOperators; PB95 holds the surrounding controlled adapter fixed.
Both pass every independent certificate. The candidate improves the six middle
public fits and paths against this generic Julia comparator. Pinned PyProximal
FISTA/ADMM complete-result wrappers also pass all cold/path certificates. Their
middle/path timings are slower here, but library/wrapper/runtime overhead is
part of those complete workloads, so the ratios are not isolated algorithmic
complexity claims. Full fractions, paired intervals and adverse observations
are retained in the tables and worker reports.

The guarded-loss control accounts for substantial gains relative to PB95.
Final-diagnostic reuse is a smaller contribution with uncertain individual
effects. These are controlled contrasts, not an additive decomposition.
For the many-response middle fit, baseline median counters show20 value-gradient
calls and21 separate loss-value calls; the candidate retains20 value-gradient
calls and removes those separate loss evaluations. Wide fits share the residual
for value and gradient; tall fits use Gram products with a direct residual
fallback near cancellation. The current PB bridge already reuses its prox
workspace/value; this change does not repair a supposed second SVD per prox.
The coefficient-only control is attribution evidence and is never ranked as
an implementation with equivalent public results.

Keep certificate cadence10. Cadence1 adds cost; cadence20 has mixed effects:
the wide path ratio is1.0718 (interval1.0586–1.0932) and the ill-conditioned
path ratio1.0434 (1.0212–1.0573), despite gains on some other paths. Safe.99/L
has small, inconsistent gains and uncertainty; it does not support a global
step change. The mathematically quadratic trait gives no uniform improvement
and regresses on the rank-deficient path (ratio1.0466, interval1.0274–1.0559).
Neither control is included in production. The historical boundary1/L pilot
is outside the cited paper's strict step range and remains rejected as a default.

Keep APG as the default while documenting explicit PANOC for the measured
workloads. PANOC improves all six middle fits and paths against APG, but the
weak many-response cold fit is slower: ratio1.11949, interval1.06416–1.23164,
with16/20 inputs slower. Strong many-response/rank-deficient fits have uncertain
relative gains. There is no supported automatic solver-selection threshold.
APG also remains available without installing/loading the optional dependency.

The comparison uses tol1e-8 and max_iter20000 for matched accuracy, rather than
the public API defaults tol1e-7 and max_iter10000. It supports the implementation
and the stated matched-accuracy rankings; it is not a timing measurement of
every default call. Existing default/correctness tests remain in place. No
new statistical penalty-selection, noise-estimation or inference claim follows.

## Memory and retained evidence

Cumulative allocations decrease substantially on the target workloads: the
many-response middle public fit goes from143960952 to43157288 bytes, and its
path from1133439600 to326210656. Whole-process RSS is not uniformly lower.
The small-case timing peak and rank-deficient whole-worker peak regress;
many-response and wide medians improve. See memory-summary.csv and per-input
memory.csv. Timing, post-profile and whole-worker peaks are separate, include
Julia/runtime state, and are not optimizer-only memory. Components are warmed
cached probes, not an additive causal decomposition of end-to-end time.

The adjacent CSV/JSON/gzip records contain paired inputs, all fractions/modes,
39600 certificate rows, source/profile coverage, cached-allocation witnesses,
actual native before/after point checks, memory, components and counter summaries.
worker-reporting-inputs.tar.gz preserves the full worker sidecars/profiles and
reporting inputs; SHA256
7fed86a1aeb3e3afb569b6b3f95648ac479dcc9f6bd27219375ca5ac39398c88.
Raw coefficient/intercept binaries and120 original shared inputs remain in
the registry's frozen Discovery paths. Earlier invalid calibration timings,
targeted harness recoveries and first-prototype failures remain separately
recorded; none is counted as independent holdout evidence. Scientific acceptance
is complete; current reporting-head hosted CI must pass before PR readiness.
