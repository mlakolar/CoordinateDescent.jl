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

The first ten ordinary paths have twenty levels down to .2 of the null
threshold, except the long path, which has fifty levels down to .02. The canary
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

The campaign contains 28 timed processes, 903 batch rows, 301 path accuracy
records, 77 fresh-process memory records and 14 separately collected profiles.
The [validation receipt](discovery-20261025/results/validation.json),
[accuracy summaries](discovery-20261025/results/accuracy-summary.csv), and
[memory accuracy](discovery-20261025/results/memory-accuracy.csv) retain the checks.

## Process memory is still a tradeoff

Median fresh-process peak resident memory, MiB, over seven replications:

| Case | Native | Old | SLOPE.jl | R auto |
|---|---:|---:|---:|---:|
| Original sparse | 412.5 | 451.9 | 473.8 | 206.0 |
| Large centered/scaled | 466.5 | 564.1 | 494.7 | 210.6 |
| Storage canary | 491.8 | omitted | 512.5 | 212.1 |

Native sparse handling reduces total process memory relative to the measured
old code and SLOPE.jl. **R still uses substantially less total process memory**,
roughly half as much on these cases, despite slower warm fits. The structural
storage target does not imply the lowest process RSS.

Linux `getrusage(RUSAGE_CHILDREN).ru_maxrss` includes imports, JIT and backend
startup. It does not isolate incremental fitting memory. Fresh memory workers
may run concurrently; their times are excluded from the speed rankings.
Julia heap allocation counts exclude foreign allocations and cannot establish
cross-language memory superiority. See [RSS ranges](discovery-20261025/results/memory-summary.csv).

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

The final integration pins ProximalSolvers
`36f623a1747dfec6cac964fba8341819685c5146`, which also accepts the current
HDM master's gap-stopped lasso API. Its SLOPE, APG, image-model, solve and types
blobs equal those of the measured 335 revision. The measured PB source and
current PB `0e98c8eddc2e251be7e0b963c4d32e9a5a69231d` have identical
least-squares constructors, caches, value/gradient methods and sorted-penalty
code; only unrelated partial/nonconvex/entropy sources differ. This is a
comparison with the PB source actually used here, not the different PB source
used by the earlier dense extraction study. See the
[source identity receipt](discovery-20261025/metadata/runtime-identity.json).

Full HDM tests passed 2732/2732 on Discovery with the earlier integrated API,
and the guarded PS engine passed 53/53 on both Julia 1.10.12 and 1.13.2.
Current-dependency confirmation is recorded separately with its exact sources
and logs. Benchmark measurements remain labelled with their actual dependency
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
4. Keep R available when total process RAM, startup cost or its broader features
   are decisive. These measurements rank matched warm least-squares paths,
   not complete GLM or cross-validation workflows.

Run `20261025-cd82-sparse`, primary completed job `13424223`, has its
[protocol and recreation recipe](README.md) and lightweight project registry
at `.slurm/runs/20261025-cd82-sparse.json` in the project parent. Compact reporting
inputs and adverse calibration results are retained here; large per-path
coefficient outputs remain at
`/home1/mkolar/juliaPkg/runs/20261025-cd82-sparse` on Discovery.
