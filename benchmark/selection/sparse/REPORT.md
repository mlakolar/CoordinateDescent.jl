# Native sparse SLOPE: results and recommendation

Keep the native sparse implementation and the shared ProximalSolvers engine.
All three agreed targets pass on the replicated Discovery workloads: accuracy,
sparse storage, and performance. A replacement backend is not warranted by
these measurements. Statistical calibration and knockoff construction choices
remain separate priorities from the [earlier selection study](../compare/RECOMMENDATION.md).

| Target | Evidence | Outcome |
|---|---|---|
| Independent full relative proximal residual ≤1e-7 | Maximum 6.366e-8 over every level of 301 timed paths; maximum 4.119e-8 in 77 fresh memory processes | Pass |
| Storage O(nnz+n+p), excluding the requested coefficient path | Largest transformed design 637,184 bytes; outer solver workspace 1,517,840 bytes; dense equivalent 1,600,000,000 bytes | Pass |
| No reproducible dense slowdown >5%; sparse parity with SLOPE.jl | Dense paired median time ratios 0.927, 0.922 and 0.937 versus the old code; original sparse ratio 0.468 versus SLOPE.jl | Pass |

The workspace measurement includes references to the design and must not be
added to the design measurement. Inner APG scratch adds O(n+p) storage; returning
the full path requires O(pL) coefficients. The 1.6 GB figure is the size of a dense
transformed matrix, not a measured old-backend allocation at that size.

## What changed

Sparse CSC input stays sparse through scaling, screening, and restricted fits.
Centering is a rank-one correction in forward and transpose products, so an
intercept never fills implicit zeros. The caller's matrix is preserved. Dense
input retains the explicit transformation and BLAS products. Input validation,
penalty units, intercept recovery and statistical defaults are unchanged.

HighDimensionalModels owns preprocessing and statistical result objects.
ProximalSolvers owns the accelerated path solver, scratch storage, strong-rule
screening and full optimality checks and repairs. This removes the duplicated
numerical engine rather than maintaining two implementations.

## Runtime competitiveness

These are warm path times in milliseconds: medians of seven independent process
medians, each based on three timed batches. All four backends used identical
saved inputs, path factors, penalty weights and preprocessing conventions on
one EPYC 7313 node, with one Julia/BLAS/OpenMP thread. Imports, JIT compilation,
I/O, profiling and independent certificates are outside the timings.

| Case | Native sparse/shared engine | Old implementation | SLOPE.jl | R SLOPE auto |
|---|---:|---:|---:|---:|
| Dense, 600×200 | 3.225 | 3.450 | 7.778 | 13.750 |
| Wide dense, 200×1000 | 6.132 | 6.751 | 14.480 | 19.400 |
| Correlated dense, 400×400 | 4.087 | 4.367 | 9.684 | 15.286 |
| Original sparse, 600×1000, density .02 | 3.524 | 12.036 | 7.565 | 17.000 |
| Sparse, 1000×3000, density .001 | 9.923 | 55.578 | 35.763 | 44.333 |
| Sparse, 1000×3000, density .01 | 10.874 | 50.072 | 19.507 | 29.500 |
| Sparse, 600×1000, density .1 | 7.519 | 14.081 | 20.782 | 32.500 |
| Centered/scaled, 1000×3000, density .01 | 11.529 | 54.669 | 33.096 | 40.333 |
| Same centered input, 50 weaker-penalty levels | 262.675 | 1580.616 | 8034.789 | 6007.000 |
| Large centered/scaled, 1500×6000, density .002 | 17.920 | 305.661 | 47.382 | 55.000 |
| Storage canary, 10000×20000, density .0001 | 28.636 | omitted | 50.550 | 62.000 |

Cases 1–8 and 10 have twenty levels down to .2 of the null
threshold. The long path has fifty levels down to .02. The canary
has ten levels. Exact dimensions, flags and realized densities are in
[cases.csv](discovery-20261025/inputs/cases.csv).

Paired by replication, the original sparse case is **3.40× faster than the old
code, 2.14× faster than SLOPE.jl and 4.84× faster than R**. Its native/SLOPE.jl
ratio ranges from .446 to .485. The large centered case is about **17.1× faster
than the old code, 2.68× faster than SLOPE.jl and 3.07× faster than R**. The three
dense controls improve by about 6–8%; no replication shows a dense slowdown.
Ratios of paired times can differ slightly from ratios of the table medians.

The large advantage on the weaker-penalty path is workload-specific. It uses
the same X/y as the preceding centered case, but follows a much longer path.
The earlier SLOPE.jl pilot at the looser internal tolerance took about 6.6 s,
versus about 8.0 s with the final setting; the tighter tolerance does not by
itself explain the entire difference. This is one dataset per case, so these
runs establish process variability, not performance across a population of
random problems. No universal speed claim follows.

See [timing summaries](discovery-20261025/results/timing-summary.csv) and
[paired ratios and ranges](discovery-20261025/results/paired-ratios.csv).
The old study's EPYC 7513 absolute times should not be mixed with this table.

## Accuracy and comparison settings

Native and old fits use internal tolerance 1e-7; SLOPE.jl and R auto use 1e-9.
These settings were calibrated against the same independently recomputed full
relative proximal certificate ≤1e-7, including intercept stationarity. Every
path level is checked using original X/y, independently computed moments and
scales, and full matrix products. Reduced-fit stopping flags are insufficient.
The maximum relative objective difference across matched backends is 1.51e-14.

Two failed calibrations are retained: R at 1e-8 reached 1.2153e-7 on the long
path, and SLOPE.jl at 1e-8 reached 2.7851e-7 on the .001-density case. The final
cohort uses the fixed tighter settings throughout. Warm time comparisons at
unequal, unverified accuracy are excluded.

The primary campaign contains 28 timed processes, 903 batch rows, 301 path
accuracy records and 14 separately collected profiles. A corrected memory
cohort contains 77 standalone fit processes with external certificates; the
original 77 harness-memory processes are retained separately.
The [validation receipt](discovery-20261025/results/validation.json),
[accuracy summaries](discovery-20261025/results/accuracy-summary.csv), and
[memory accuracy](discovery-20261025/results/memory-accuracy.csv) retain the checks.

## Process memory is still a tradeoff

Median fresh-process peak resident memory, MiB, over seven replications:

| Case | Native | Old | SLOPE.jl | R auto |
|---|---:|---:|---:|---:|
| Original sparse | 390.9 | 381.6 | 391.3 | 205.9 |
| Large centered/scaled | 410.9 | 503.9 | 404.4 | 210.7 |
| Storage canary | 441.8 | omitted | 422.6 | 212.5 |

Native sparse handling reduces large-case process RSS relative to the old
adapter, but it is slightly higher on the small case. SLOPE.jl is similar on
the original sparse case and uses less RSS on the larger cases. **R uses
substantially less total process memory**, roughly half as much, despite slower
warm fits. The structural storage target does not imply the lowest process RSS.

The final table uses job **13424760** on Xeon 4116 node d05-41. All backends
perform two warm calls and one retained fit; imports/JIT, output serialization
and any package precompilation children are inside the measured process tree.
Each Python measurement process captures its fitting child's
`RUSAGE_CHILDREN.ru_maxrss` and exits before a separate verifier starts.
SLOPE.jl's worker does not import HDM/CD. Thus every backend's independent
certificate is outside its RSS measurement. All 77 paths pass, with maximum
relative residual 4.119e-8 and objective discrepancy 8.88e-16. See
[corrected summaries and ranges](discovery-20261025/fit-only-memory/results/summary.csv)
and the [validation receipt](discovery-20261025/fit-only-memory/results/validation.json).
The old comparator is its standalone SLOPE driver with minimal numerical
imports, while native loads the model package. These are different complete
process footprints; the owned design/workspace measurements isolate the
algorithm's storage requirement more directly.

The original harness-RSS cohort on EPYC 7313 included Julia certificate and
operator-summary work inside the measured worker, and imported HDM/CD even for
SLOPE.jl; R verification was external. Its [original measurements](discovery-20261025/results/memory-summary.csv)
are preserved but superseded for cross-backend RSS conclusions. The two memory
cohorts also used different CPU models, so their difference does not isolate a
pure harness effect. Neither cohort estimates incremental fitting memory.
Initial Julia cache compilation causes some large maxima (up to 930.5 MiB for
SLOPE.jl); seven-replication medians and all ranges are retained without trimming.
Fresh memory workers run concurrently and their times are excluded from speed
rankings. Julia heap counts exclude foreign allocations and cannot establish
cross-language memory superiority.

## Tests and source provenance

The sparse tests cover forward/transpose/adjoint products, five-argument
`mul!`, sparse column restriction, all intercept/standardization combinations,
stored zeros, constant and zero columns, rank deficiency, Float32, Int32
indices, caller input preservation, full certificates and storage canaries.
The existing selection and R golden tests also pass in the validated runtime.

The broad cohort used Julia 1.10.12, SLOPE.jl 1.3.1 and R 4.5.3/SLOPE 2.1.1,
with ProximalSolvers `335ea905da207a6cc1cd3bd945bb7bbbf9f4eba4` and the
installed ProximalBase `86sFe` source identified by the retained manifest and
[exact source archive](discovery-20261025/metadata/primary-pb-source.tar.gz).
The HDM sparse runtime has SHA256
`b32489f93b06453f2301ffe6074b966038fff470af379d0a66fa3e87966bc386`.

The pre-rename integration used ProximalSolvers
`36f623a1747dfec6cac964fba8341819685c5146`, which also accepts the gap-stopped
lasso API. Its SLOPE, APG, image-model, solve and types
blobs equal those of the measured 335 revision. The measured PB source and
current PB `0e98c8eddc2e251be7e0b963c4d32e9a5a69231d` have identical
least-squares constructors, caches, value/gradient methods and sorted-penalty
code; only unrelated partial/nonconvex/entropy sources differ. This is a
comparison with the PB source actually used here, not the different PB source
used by the earlier dense extraction study. After the concurrent package rename,
the renamed validation used companion API `3bee2f91604fe08d34a07d45bcc638e3b4face3a`.
PR #90 subsequently merged as `b9cf11c46a12b0d686f242ee63162692a9091a85`;
the bootstrap now uses the normal sibling master checkout. SLOPE itself remains byte-identical; APG changes
are canonical field names, with the same update operations. Current PB master
`b6c949f3dd26cad09252e1c2c9e49810eb69403e` retains the same relevant loss and
sorted-penalty code. Historical measurements are retained with their actual
335/PB86 source labels rather than relabelled as a new timing campaign. See the
[source identity receipt](discovery-20261025/metadata/runtime-identity.json).

Full HDM tests passed 2732/2732 on Discovery with the earlier integrated API,
and the guarded PS engine passed 53/53 on both Julia 1.10.12 and 1.13.2.
Job 13424680 then confirmed PS36/PB0e98: **2742/2742 full-suite assertions on
Julia 1.13.2 and 245/245 selection/sparse/R-golden plus 53/53 engine assertions
on Julia 1.10.12**. The [logs and manifests](discovery-20261025/compatibility/)
identify the exact pre-module-rename source snapshot. A Julia 1.10 CI failure
in the old test snapshot was an overly strict absolute-only matrix comparison;
the correction adds a 100eps(Float64) relative roundoff bound and leaves all
solver/certificate thresholds unchanged. Job **13424722** then passed the same
**2742 full-suite and 245+53 targeted assertions** with the renamed package at
HDM `8a6589c244f55dcaa1b8798b4fd8dcaf15fc4606`, PS3bee, PBb6 and
KernelSmoothing653784. Its [logs and manifests](discovery-20261025/renamed/)
are retained separately. Benchmark measurements remain labelled with their actual dependency
cohort; unchanged numerical kernels justify reusing those runs, while newer
package imports could change fixed RSS overhead.

## Recommended next work

1. Keep native sparse preprocessing and the shared PS engine. The sparse
   bottleneck identified in the earlier study is resolved for these workloads;
   retain dense BLAS handling and the independent full certificate.
2. Prioritize statistical calibration and MVR/max-entropy knockoff choices from
   the earlier study. Sparse storage changes do not repair the correlated/wide
   FDR limitations, and replacing the SLOPE optimizer would not repair them.
3. Optimize further only against a specific needed workload. All seven profiles
   at each of the two profiled sparse sizes show array indexing/writing and
   sparse products; sorting/proximal work also appears. Generic leaf labels do
   not identify a unique phase or prove that one cache change will help. Use
   those traces to choose a bounded follow-up before changing solver algorithms.
4. Keep standalone R or SLOPE.jl workflows available when process RAM or broader
   features are decisive. Calling an alternative from a process that already
   imports HDM does not inherit the standalone worker's footprint. Startup
   speed, complete GLM and cross-validation workflows were not ranked here.

Run `20261025-cd82-sparse`, primary completed job `13424223`, has its
[protocol and recreation recipe](README.md) and lightweight project registry
at `.slurm/runs/20261025-cd82-sparse.json` in the project parent. Compact reporting
inputs and adverse calibration results are retained here; large per-path
coefficient outputs remain at
`/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse` on Discovery.

The final accounting remains within the approved shared allowance: at most
158.1 running-union minutes of 180 and a conservative 11.4 allocated CPU-hours
of 16, including coordinated solver work and retries. Current-phase allocations
never exceed 8 CPUs or 16 GB. At historical source head `8a6589c`, platform
tests, docs, examples and patch coverage passed, while the project coverage
gate reported 99.75% versus an older 99.88% baseline. Although Codecov's summary
said modified coverable lines were tested, inspection of its raw report found
untested Gaussian/fixed knockoff wrapper routes and the threaded CPSS branch.
The latter had not run in hosted CI because Julia used one thread.

After solver PR #90 merged, the temporary companion exception was removed;
the bootstrap again matches normal sibling-master policy. The integration
follow-up adds direct public-wrapper construction tests and configures two
Julia threads in CI so the existing serial/threaded CPSS agreement test
exercises both branches. No coverage thresholds were reduced. These changes
do not modify the numerical solver or the benchmark inputs. PR #84 remains
draft for its model/statistical review.
