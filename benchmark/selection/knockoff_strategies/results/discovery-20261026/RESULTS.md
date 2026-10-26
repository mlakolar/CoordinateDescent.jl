# MVR and maximum-entropy knockoff construction: Discovery evidence, 2026-10-26

Explicit optimized diagonal choices are useful, but the evidence supports retaining equicorrelation as the default. The optional Knockoffs.jl 3.1.4 adapter supplies only diagonal optimization. Sampling, feasibility checks, joint lasso and knockoff+ remain native. The change adds an owned diagonal/diagnostics API, forwards construction options through the filter, and rejects failed original-objective stationarity without silently changing strategy. It does not implement a new diagonal optimizer or a covariance cache.

## Statistical benefit and adverse results

The main cohort contains 16,128 independent Gaussian datasets: 1,008 per scenario across seven independent process shards. Only the diagonal changes; X, y, independent normal innovations, lambda=.04, cyclic lasso controls and knockoff+q=.1 are shared. All 48,384 native original-design certificates pass (maximum residual 2.77591e-09); there are no internal stopping flags, fit errors or failed public-filter forwarding witnesses. R glmnet 5.1 receives the exact saved augmented matrices on 336 of those inputs: all 1,008 comparator fits pass, all selected sets agree, maximum relative objective difference 7.91589e-14. The R subset is not additional independent data.

For AR(.4) with alternating signals, MVR power rises from 0.0064 to 0.3990: paired gain 0.3926, 95% Monte Carlo interval [0.3725, 0.4127]. MVR estimated FDR is 0.0692 (MC SE 0.0032). This supports exposing an explicit alternative to the old filter policy.

| Design | Equicorrelated power | MVR power | Maximum-entropy power | MVR minus equicorrelation, 95% MC interval |
|---|---:|---:|---:|---|
| iid-alt | 0.8665 | 0.8665 | 0.8665 | [0.0000, 0.0000] |
| ar04-alt | 0.0064 | 0.3990 | 0.3601 | [0.3725, 0.4127] |
| ar04-same | 0.9164 | 0.8653 | 0.8790 | [-0.0589, -0.0432] |
| ar08-alt | 0.0000 | 0.0025 | 0.0018 | [0.0008, 0.0042] |
| ar08-same | 0.6154 | 0.4398 | 0.4960 | [-0.1934, -0.1578] |
| weak-ar04-alt | 0.0014 | 0.0436 | 0.0378 | [0.0337, 0.0506] |
| weak-ar04-same | 0.5200 | 0.3956 | 0.4224 | [-0.1446, -0.1042] |
| wide-ar04-alt | 0.0007 | 0.0151 | 0.0147 | [0.0097, 0.0191] |
| wide-ar04-same | 0.7666 | 0.6865 | 0.7066 | [-0.0949, -0.0653] |

Same-sign power losses are substantial and remain in the recommendation. Severe-correlation and wide alternating-signal power remain poor. Three explicit covariance-misspecification stress cases are reported separately; they establish no misspecified-model FDR guarantee. Global-null power is undefined, and null FDR intervals use Wilson binomial intervals. All-sample empirical zero SEs do not prove an event impossible; zero-event-bounds.csv supplies conservative rare-event bounds. Full FDR/power/selection counts, paired differences and worker means are in STATISTICS.md and the CSVs. No automatic strategy selection or statistical guarantee beyond the stated construction/statistic assumptions is introduced.

## Construction correctness and optimizer controls

Julia 1.10.12 and 1.13.2 each pass 556 core/selection/golden/construction checks plus 587 optional-extension checks. They cover Float32/64, unequal scales, 1e-20/1/1e20 units, supplied/zero/boundary diagonals, conflicts, owned inputs, seeded RNG, supplied noise, fixed Gram identities and explicit optimizer failure. The Float32 legacy promotion and all existing result fields/old positional constructors are preserved.

Sixteen fixed covariance matrices compare raw and adapted optimizers under package-default, proposed, robust and tighter controls. All 64 proposed/robust adapted results are feasible; maximum independent full joint gradient ratio is 4.812e-9. All 20 small CVXPY/CLARABEL problems report optimal and are feasible; relative objective differences are at most 6.24e-10. These are numerical comparators, not exact global error bounds. All raw package-default results are feasible, but 30/32 fail the adapter original-objective stationarity requirement (maximum raw full joint ratio 4.744e-5). Their failed outputs/errors remain in OBJECTIVES.md and the reporting bundle.

The independent conditional-map study passes 16/16 native and 16/16 R samplers (maximum root errors 4.020e-12 and 1.702e-12). The pinned Knockoffs.jl full sampler passes only 1/16; its orientation defect remains, with maximum relative root error.999624 on this unequal-scale suite. It is excluded from valid FDR rankings. Its diagonal optimizer remains usable with the independently validated native sampler. R Gaussian defaults differ slightly from native (maximum measured diagonal difference 1.012e-6); R fixed uses an unshrunk SVD boundary diagonal while native retains safety 1e-8. Actual maps, Gram identities and installed source hashes are retained.

## Runtime and memory

The primary runtime cohort has 105 fresh shared inputs, seven process replications, three inputs per process and three warmed rounds: 70 native and 35 R workers, 6,804 native and 315 R timing rows. All 1,260 runtime filter certificates pass (max 2.38469e-09); all 630 timed raw optimizer full joint certificates pass (max 7.88598e-09). There are 2,268 nonempty CPU and 2,268 nonempty allocation reports, 2,268 positive direct allocation witnesses, 70 actual Julia/CPU/thread metadata records and 105 successful resource receipts.

The primary launch placed baseline workers before candidate workers. All primary observations are preserved. A separately declared alternating-order sensitivity reuses the same 105 inputs in 70 fresh processes, without pooling them as another data cohort. Each process pair runs both variants serially on the same allocated CPU and alternates which runs first. Both studies preserve complete legacy results on 105 shared inputs (315 combined input/matrix/result assertions). Use ORDER-SENSITIVITY.md when assessing small before/after differences; method/payload order within candidate workers was already rotated for the primary strategy comparisons.

| Runtime case | MVR / equicorrelated full filter | Maximum entropy / equicorrelated full filter |
|---|---:|---:|
| iid n=600 p=200 | 1.078 [1.071, 1.094] | 1.301 [1.289, 1.304] |
| AR(.4) n=600 p=200 | 4.747 [4.433, 4.829] | 3.513 [3.181, 3.606] |
| AR(.8) n=400 p=400 | 7.889 [7.863, 7.969] | 6.157 [6.104, 6.203] |
| wide AR(.4) n=200 p=600 | 9.280 [8.896, 9.408] | 6.245 [6.215, 6.482] |
| fixed AR(.4) n=600 p=200 | 6.784 [6.480, 7.030] | 11.788 [11.079, 12.356] |

These optimized choices have a real runtime cost in this study. Preparation, sampling, full construction, full filtering and raw-optimizer costs are separated in RUNTIME.md. The diagonal factory row includes feasibility/objective certification and is distinct from the unchanged legacy vector selector. Applicable public R constructors receive identical X/covariance or binary unit-norm fixed designs; their RNG algorithms and stated diagonal defaults differ. R has no matching MVR/maxent optimizer here; the independent convex objective comparator covers those objectives. Heap allocation excludes foreign memory. Worker maximum RSS includes imports/JIT, checks and profiles and is not fitted-state memory; no cross-language memory advantage is claimed.

## Recommended defaults

| Setting | Recommendation | Evidence/limit |
|---|---|---|
| strategy | retain equicorrelation | Mixed-sign gains accompany same-sign losses and larger optimized-procedure cost; no universal winner |
| equicorrelation safety | retain 1e-8 | Existing seeded arithmetic/precision and scale-relative boundaries remain compatible |
| optimized backend | niter=1000, tol=1e-10, lambda_min=1e-10, robust=false | Original-objective certificates pass; package-default stopping does not supply that assurance |
| independent optimality threshold | max(1e-6, 100eps(input precision)) | Certify actual rounded diagonal/original covariance; Float32 needs its precision margin |
| q / threshold | retain 0.1 / knockoff+offset=1 | Compatibility; this is a target, not an empirical guarantee under arbitrary misspecification |
| lasso penalty / fitting | retain explicitly supplied positive lambda; do not retune CDOptions globally | This study fixes 0.04 and stricter comparison controls; it supplies no data-driven penalty selector |

Backend actual iteration count/convergence status are unavailable and remain labeled unavailable. The independent stationarity result is separate. Robust updates are explicit user options, not a silent fallback. Gaussian model-X still requires the supplied population law; fixed-X still requires full rank, n>=2p and the documented sufficiency/statistic/error assumptions. No covariance estimator or factor cache is added.

## Recreation and retained evidence

Scientific production source is candidate2 at `f14306b8b0350428f2cbb03c8126347fe84f0a3e`, with baseline `295fbf5420d6384a1ca261023367ee86d0a40b8c`. Source snapshots, actual environment manifests, source hashes, all reporting rows, failures and job receipts are in reporting-inputs-v1.tar.gz; all primary worker profiles are in worker-profiles-v1.tar.gz. The same raw X/y/innovations, coefficient and constructor/result-field records remain on Discovery under the recorded run directory. They are not routinely archived or transferred. The source/seed/environment recipe recreates them.

Failed parse/dependency/stdlib/resource-wrapper canaries, old R 1e-14 certificate failures, the R CSV-dimname failure and the incorrect fixed-diagonal helper assumption remain recorded. None caused outcome-based main-cohort signal/sequence/penalty/tolerance tuning. The alternating-order sensitivity is explicitly separate. No primary outputs are replaced and no merge/auto-merge is performed.

