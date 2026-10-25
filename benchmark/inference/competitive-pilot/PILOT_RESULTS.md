# Discovery comparison pilot

20 fresh datasets per case; five active and five null targets; one CPU per worker.
Results are exploratory. Coefficients within a dataset are dependent; Monte Carlo SEs use dataset-level means.

| Case | Engine | Tuning | Successful | Coverage active/null | Type I | Power | Length active/null | Fit seconds median | Peak RSS MiB max |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| binomial-correlated | SIHR | default | 20/20 | 0.900/0.990 | 0.010 | 0.080 | 1.575/1.580 | 12.213 | 745.0 |
| binomial-correlated | SIHR | matched | 20/20 | 0.290/0.940 | 0.060 | 0.070 | 0.976/0.999 | 4.546 | 625.3 |
| binomial-correlated | hdi | default | 20/20 | 0.870/0.970 | 0.030 | 0.150 | 1.456/1.474 | 55.064 | 427.1 |
| binomial-correlated | hdi | matched | 0/20 | nan/nan | nan | nan | nan/nan | nan | nan |
| binomial-correlated | ours | default | 20/20 | 0.270/0.940 | 0.060 | 0.050 | 1.006/1.007 | 0.006 | 779.3 |
| binomial-correlated | ours | matched | 20/20 | 0.270/0.940 | 0.060 | 0.050 | 1.006/1.007 | 0.006 | 780.6 |
| binomial-moderate | SIHR | default | 20/20 | 0.610/0.990 | 0.010 | 0.770 | 0.642/0.627 | 21.541 | 678.4 |
| binomial-moderate | SIHR | matched | 20/20 | 0.580/0.970 | 0.030 | 0.700 | 0.604/0.587 | 5.158 | 681.3 |
| binomial-moderate | hdi | default | 20/20 | 0.600/0.990 | 0.010 | 0.780 | 0.600/0.596 | 208.173 | 429.2 |
| binomial-moderate | hdi | matched | 0/20 | nan/nan | nan | nan | nan/nan | nan | nan |
| binomial-moderate | ours | default | 20/20 | 0.670/0.980 | 0.020 | 0.700 | 0.703/0.694 | 0.009 | 779.5 |
| binomial-moderate | ours | matched | 20/20 | 0.670/0.980 | 0.020 | 0.700 | 0.703/0.694 | 0.009 | 778.6 |
| gaussian-correlated | SIHR | default | 20/20 | 0.690/0.920 | 0.080 | 0.190 | 0.793/0.795 | 13.538 | 730.7 |
| gaussian-correlated | SIHR | matched | 20/20 | 0.070/0.890 | 0.110 | 0.220 | 0.395/0.406 | 4.851 | 624.8 |
| gaussian-correlated | econml | default | 20/20 | 0.660/0.920 | 0.080 | 0.240 | 0.753/0.759 | 24.492 | 262.9 |
| gaussian-correlated | econml | matched | 20/20 | 0.120/0.880 | 0.120 | 0.210 | 0.438/0.436 | 0.825 | 252.0 |
| gaussian-correlated | hdi | default | 20/20 | 0.760/0.910 | 0.090 | 0.270 | 0.822/0.833 | 57.666 | 422.1 |
| gaussian-correlated | hdi | matched | 20/20 | 0.120/0.880 | 0.120 | 0.210 | 0.440/0.439 | 0.944 | 259.8 |
| gaussian-correlated | ours | default | 20/20 | 0.300/0.650 | 0.350 | 0.420 | 0.479/0.478 | 0.004 | 784.1 |
| gaussian-correlated | ours | matched | 20/20 | 0.120/0.880 | 0.120 | 0.210 | 0.440/0.439 | 0.004 | 784.9 |
| gaussian-correlated | oursqp | default | 20/20 | 0.140/0.870 | 0.130 | 0.280 | 0.430/0.442 | 0.005 | 770.9 |
| gaussian-correlated | oursqp | matched | 20/20 | 0.070/0.880 | 0.120 | 0.220 | 0.395/0.406 | 0.005 | 784.9 |
| gaussian-correlated | sslasso | default | 20/20 | 0.050/0.980 | 0.160 | 0.210 | 0.461/0.467 | 20.477 | 279.6 |
| gaussian-correlated | sslasso | matched | 20/20 | 0.220/0.940 | 0.120 | 0.280 | 0.553/0.565 | 99.842 | 271.4 |
| gaussian-moderate | SIHR | default | 20/20 | 0.730/0.980 | 0.020 | 1.000 | 0.283/0.282 | 22.228 | 673.6 |
| gaussian-moderate | SIHR | matched | 20/20 | 0.350/0.950 | 0.050 | 1.000 | 0.215/0.215 | 5.213 | 681.1 |
| gaussian-moderate | econml | default | 20/20 | 0.730/0.960 | 0.040 | 1.000 | 0.286/0.288 | 62.615 | 264.5 |
| gaussian-moderate | econml | matched | 20/20 | 0.700/0.950 | 0.050 | 1.000 | 0.285/0.285 | 0.766 | 254.5 |
| gaussian-moderate | hdi | default | 20/20 | 0.820/0.930 | 0.070 | 1.000 | 0.321/0.322 | 207.726 | 428.1 |
| gaussian-moderate | hdi | matched | 20/20 | 0.700/0.950 | 0.050 | 1.000 | 0.286/0.286 | 1.249 | 264.0 |
| gaussian-moderate | ours | default | 20/20 | 0.620/0.910 | 0.090 | 0.960 | 0.355/0.355 | 0.006 | 773.8 |
| gaussian-moderate | ours | matched | 20/20 | 0.700/0.950 | 0.050 | 1.000 | 0.286/0.286 | 0.006 | 781.0 |
| gaussian-moderate | oursqp | default | 20/20 | 0.120/0.890 | 0.110 | 0.930 | 0.265/0.265 | 0.006 | 780.4 |
| gaussian-moderate | oursqp | matched | 20/20 | 0.350/0.950 | 0.050 | 1.000 | 0.214/0.214 | 0.006 | 776.9 |
| gaussian-moderate | sslasso | default | 20/20 | 0.790/0.980 | 0.050 | 1.000 | 0.326/0.327 | 52.502 | 355.2 |
| gaussian-moderate | sslasso | matched | 20/20 | 0.650/0.990 | 0.060 | 1.000 | 0.324/0.326 | 6.168 | 286.8 |

## Numerical diagnostics and API warnings

| Case | Engine | Tuning | Failed | Warning fits | Convergence warnings | QP gap warnings | SIHR enlarged supplied mu | SIHR fallback messages |
|---|---|---|---:|---:|---:|---:|---:|---:|
| binomial-correlated | hdi | default | 0 | 20 | 0 | 0 | 0 | 0 |
| binomial-moderate | hdi | default | 0 | 20 | 0 | 0 | 0 | 0 |
| gaussian-correlated | econml | default | 0 | 7 | 7 | 0 | 0 | 0 |
| gaussian-correlated | econml | matched | 0 | 20 | 0 | 0 | 0 | 0 |
| gaussian-correlated | hdi | matched | 0 | 20 | 0 | 0 | 0 | 0 |
| gaussian-moderate | econml | default | 0 | 20 | 20 | 0 | 0 | 0 |
| gaussian-moderate | econml | matched | 0 | 20 | 0 | 0 | 0 | 0 |
| gaussian-moderate | hdi | matched | 0 | 20 | 0 | 0 | 0 | 0 |

EconML emits an API warning for a manually specified alpha; this is separate from a numerical convergence warning. Monte Carlo SEs, min/max timings, solver mus, QP residuals/gaps and all per-fit records are in pilot-summary.csv and pilot-records.csv.

## Conditional cost estimate

Sum over measured supported configurations, 1000 datasets per case: 246.410 CPU-hours of warmed fitting; 317.332 CPU-hours with a fresh process and warm-up per fit.
Ideal eight-worker elapsed lower bounds: 30.801 to 39.666 hours, before scheduling/imbalance margin.
Largest successful per-process RSS: 784.9 MiB. Aggregate RSS must account for all concurrent workers and overhead.
Conservative fresh-process planning budget with 25% margin: 50 allocated hours at 8 CPUs and 16 GiB. Queue time is excluded. This is for the four pilot cases and ten requested targets, not all-p inference or a broader design grid.
Retaining all CSV inputs at 1000 replications would use about 7.19 GiB; deterministic generation permits a bounded working set with seed/input-hash records instead.
Every requested supported fit successful: True. Missing/failed configurations invalidate a complete-campaign estimate; their costs are lower bounds.

## Comparability notes

- Marginal 95% intervals and unadjusted two-sided 5% tests. No simultaneous/bootstrap comparison in this pilot.
- Native defaults retain package statistical choices. RNG seeds and single-CPU execution are fixed. Requested targets are five active and five null coefficients.
- Julia and SIHR compute requested targets; hdi, EconML and sslasso compute all 500 slopes internally.
- Julia GLM needs an explicit initial lambda: the declared common lambda is supplied; this is a required-input default, not fully automatic tuning.
- Matched nominal initial and projection penalties are common within case; variance estimators and projection objectives remain method-specific. SIHR matched uses rescale=1 and prob.filter=0.
- Native hdi retains default group-test preparation. Matched hdi disables this preparation for the requested individual-coefficient task. Replicated profiling measures this extra preparation separately.
- Native hdi binomial preprocessing does not expose a matched-initial-Fisher hook. Its matched GLM rows are explicitly unsupported, not fabricated.
- sslasso matched repairs the authors' undefined Las helper using exact-lambda glmnet and corrects the diagonal QP fast path from (1-mu0)/G_ii to (1-mu)/G_ii. The iterative solver, noise estimator, interval inflation and testing are retained. Original defaults and superseded matched outputs are preserved.
- Small independent warm-ups precede timing. Peak RSS and process time include runtime/library initialization, warm-up and actual fitting. Cold startup is process time minus warmed fitting only approximately.
- Default author sslasso has a coarse convergence rule; matching nominal penalties does not establish numerical equivalence.
- SIHR can enlarge a supplied mu and can fall back to a diagonal projection after an internal solver error. Instrumentation for replications 2--20 records actual solver mus/statuses without changing outputs; fallback messages are counted.
- sslasso matched post-timing constraint residuals and primal-dual gaps are recorded. Finite coefficient output alone does not establish QP optimality.
- Authors' sslasso intervals include an additional bias allowance that their returned normal p-values do not include; those intervals and tests are not exact duals. Reported coverage and rejection rates use the respective original outputs.
- Coverage, error, power and length summaries condition on successful finite output. Failure and warning counts must accompany comparisons; failed fits are not silently counted as valid inference.

## Replicated profiling

Five independent Gaussian-moderate datasets were profiled at 10 ms intervals under native hdi defaults on Discovery. Every profiled coefficient and SE matched the same-seed timed result to 1e-8.
Median runtime share in nodewise regression: 99.52% (range 99.47--99.53%).
Its custom nodewise cross-validation accounted for 82.76% of total runtime; optional group-test preparation accounted for 0.40%.
Full per-replication stacks and self/total summaries remain in the run's profiles directory; profile-summary.json and profile-totals.csv retain the reporting inputs.
