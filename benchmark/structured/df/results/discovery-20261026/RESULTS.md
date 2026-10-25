# Structured degrees of freedom: Discovery evaluation, 2026-10-26

The component-basis and path-cache implementation reduces complete path time in this fixed evaluation. Every compared coefficient, objective, iteration count and degree-of-freedom value is identical. These results concern diagnostic implementation cost; they do not change or calibrate a statistical estimator.

## Complete public paths

Seven independent datasets per design, three warmed rounds per dataset. Times are medians of within-dataset medians. Ratios and intervals use paired candidate/baseline dataset ratios; the intervals are percentile-bootstrap intervals for their median (10,000 resamples, seven independent units). Smaller ratios mean less candidate time. All imports, compilation, input loading, independent checks and profiling are outside fit timers; model setup and required public diagnostics are inside.

| Case | Baseline ms | Candidate ms | Paired ratio | 95% interval | Paired range | Baseline allocated MiB | Candidate allocated MiB |
|---|---:|---:|---:|---|---|---:|---:|
| fused100 | 69.085 | 7.222 | 0.104 | 0.100–0.107 | 0.097–0.108 | 46.46 | 3.57 |
| fused300 | 505.528 | 42.972 | 0.087 | 0.084–0.089 | 0.083–0.091 | 200.01 | 10.97 |
| fused40 | 9.934 | 1.643 | 0.163 | 0.158–0.172 | 0.158–0.183 | 9.10 | 1.07 |
| graph100 | 384.616 | 373.409 | 0.961 | 0.956–0.969 | 0.947–0.971 | 12.67 | 1.52 |
| rankdef100 | 22.015 | 14.444 | 0.670 | 0.602–0.700 | 0.596–0.765 | 13.17 | 7.74 |
| trend100 | 36.237 | 31.029 | 0.848 | 0.844–0.864 | 0.833–0.878 | 13.71 | 9.06 |
| trend40 | 17.132 | 12.760 | 0.752 | 0.748–0.760 | 0.742–0.778 | 6.79 | 2.60 |
| wide100 | 21.868 | 9.281 | 0.429 | 0.398–0.468 | 0.359–0.518 | 12.08 | 2.07 |

Allocated bytes are cumulative allocation, not peak resident memory. The individual dataset ratios, every timing round and allocated-byte observations are retained in summary.json and timings-*.csv. The seven-input intervals describe this experiment; they do not establish a universal speed ranking.

## Degrees-of-freedom work and setup

These are separate diagnostic measurements on the returned path coefficients, with a fresh one-pattern cache per round. They are not additive decompositions of the public-path timings. Baseline diagnostic calls have no reusable design-rank preparation. Candidate preparation includes the design SVD and row-oriented penalty copy; this cost is included in public-path timings above.

| Case | Baseline DF ms | Candidate DF ms | Candidate setup ms | Distinct patterns |
|---|---:|---:|---:|---:|
| fused100 | 55.422 | 0.176 | 1.646 | 35–42 |
| fused300 | 493.205 | 0.471 | 17.052 | 26–29 |
| fused40 | 13.401 | 0.063 | 0.193 | 27–34 |
| graph100 | 15.060 | 0.073 | 1.191 | 16–20 |
| rankdef100 | 14.976 | 7.087 | 0.805 | 19–20 |
| trend100 | 15.849 | 8.797 | 1.400 | 20–20 |
| trend40 | 5.780 | 1.537 | 0.174 | 30–35 |
| wide100 | 14.473 | 1.857 | 0.254 | 18–20 |

The p=1000 diagnostic, including one workspace setup and one DF evaluation, took 231.166 ms at baseline and 185.789 ms with the candidate; paired median ratio 0.831, interval 0.803–0.868, range 0.786–0.871. This is a single diagnostic workload, not a public p=1000 path measurement or a large-signal CV claim.

The retained cache owns a sparse row representation of the fixed penalty and one inactive-row pattern, with storage proportional to penalty nonzeros and row count. Orthonormal component columns avoid dense null-space storage for graph/pin constraints. A rank-deficient design still requires rank(X*C); counting components alone would be incorrect. General contrasts and ambiguous rank/weight cases retain dense fallback work.

## Correctness, profiles and memory

datasets=7 cases=8 path_points=1820 max_coefficient_difference=0.0 max_objective_difference=0.0 exact_df=true exact_iterations=true all_converged=true

This is 56 shared input/design cases and 1,820 paired path levels (3,640 returned fits). All final fits converged. All 1,050 timing rows are present. Shared input digests agree between implementations. Final Julia 1.10.12 full suite: 5,877/5,877; Julia 1.13.2: 5,873/5,873. Both include independent dense-null-space graph/contrast/rank checks, the rectangular numerical-rank regression and existing R golden paths.

There are 112 CPU profile reports and 112 allocation reports: eight cases × seven independent shared inputs × two implementations, separate from timings. CPU reports contain nonempty sampled stacks; raw stack-word counts are not wall-time percentages. The directly timed DF reduction and unchanged solver iterations support the source-level explanation: fewer repeated dense null-space/rank calculations, with reusable design checks and constraint patterns. Do not sum inclusive profiler frames or infer exact causal percentages from them.

| Fresh worker | Median peak MiB | Range MiB |
|---|---:|---:|
| baseline | 762.4 | 729.2–801.4 |
| candidate | 734.0 | 706.1–862.6 |

Each worker is a fresh process for one implementation and dataset replication, covering all eight designs and the p=1000 diagnostic. RSS includes imports/JIT, independent checks and separate profiles; it is neither optimizer workspace nor a per-design/per-fit memory bound. metadata-*.txt retains post-warm current RSS by case and final peaks. resource-*.json records child-process accounting.

The resident-memory ranges overlap, and the candidate has the larger maximum worker peak. No uniform peak-memory reduction is established.

## Defaults and scope

Use the component calculation and bounded path cache automatically for the supported constraints: complete public-path gains, unchanged results and reduced allocations support that choice on these cases. Keep the dense fallback for general contrasts and numerical ambiguity. Retain df_tol=1e-4 and the existing family/precision-specific stopping settings; this experiment provides no statistical reason to retune them. DF near fusion/rank transitions remains tolerance-sensitive. Generic trend cases, rank-deficient and wide designs remain in the tables. The cache setup has a cost, so do not extrapolate these multi-level path gains to arbitrary one-off problems.

No penalty-selection, coverage, prediction-risk or FDR improvement is claimed. Large signal cross-validation remains issue100. This change does not establish high-dimensional scaling beyond the measured path dimensions.

After the timed study, the zero-penalty sparse fused-fit shortcut was made to use the original dense numerical-rank convention. Every timing grid is strictly positive, so that change does not execute in the measured paths. The final structured/golden suites passed 1,658 checks on each Julia version in job13447394. The full-suite totals above refer to the measured production source; final-head hosted CI checks the additional boundary fix. Post-study logs and the final source hash are included here.

## Source and recreation

Baseline: master 295fbf5420d6384a1ca261023367ee86d0a40b8c. Measured candidate production: 49289d477850cd8eeb052b39cb4af79e1b08d2fe. Candidate structured.jl SHA256:8bd9c38fcb4f183815163b1acccafb858b68a5247c35d4ed3d789983ff7a19fb; baseline:05bd75aa15fe19aa5ef7def00c0a8bc229cdf58779b4320884e8e040d0d8ae9d. Source hashes are emitted by every worker. Dependencies are held fixed: PB b6c949f3dd26cad09252e1c2c9e49810eb69403e, PS 531ff55fcf8a5ce2c6997c144d4f57e85cf79f3e, KS 1feff286d0020764d2c1b03d733ab316cf948ead. Resolved manifests and snapshots remain with the run.

Discovery mkolar_1314/main, EPYC7513, one Julia/BLAS/OpenMP/MKL thread per worker, Julia1.13.2 for timing; one 8CPU/32GiB allocation, seven exclusive workers. Alternate baseline/candidate process order by replication. Tests: job13447103; evaluation/validation/aggregation:job13447106. Seed-zero calibration is separate from final seeds1–7. No candidate policy was tuned on final outcomes.

See the [harness README](../../README.md) and executable scripts. Actual remote run root:/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99; final tree is candidate2/ with env-candidate2-1.13; baseline is baseline/ with env-baseline-1.13. Launch: sbatch --parsable --dependency=afterok:13447103 measure-v2.sbatch. The latter uses worker-v2.sh and the same committed bench/validate/summarize sources. Initial launcher and incomplete-source staging attempts are preserved in the task registry; they do not contribute timings here. All final reporting inputs, failures, profiles and accounting are retained. Human review is required; no merge performed.
