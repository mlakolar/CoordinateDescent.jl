# Discovery validation of issue 82

Seven independent process replications, two Julia threads and one BLAS thread per process. Wall times exclude warm-up and profiling. All nine workloads were profiled in every replication, for at least 0.2 seconds per workload. Tables summarize saved output; no estimators are rerun.

| Workload | Median ms | Min–max ms | Median allocated MiB |
|---|---:|---:|---:|
| fixed-filter | 4.462 | 4.098–5.683 | 5.03 |
| gaussian-construction | 2.906 | 2.596–3.981 | 3.61 |
| slope-path-full | 4.002 | 3.896–5.276 | 6.93 |
| slope-path-screened | 1.495 | 1.394–1.867 | 2.32 |
| slope-single | 0.505 | 0.400–0.663 | 0.60 |
| slope-wide-full | 62.427 | 59.551–64.758 | 11.46 |
| slope-wide-screened | 49.381 | 45.598–52.593 | 10.40 |
| stability-serial | 9.725 | 6.021–10.767 | 12.67 |
| stability-threaded | 4.783 | 4.087–8.927 | 12.67 |

Paired time ratios (full/ screened or serial/threaded), summarized across the same seven replications:

| Comparison | Median ratio | Min–max |
|---|---:|---:|
| slope-path-full / slope-path-screened | 2.76 | 2.50–3.14 |
| slope-wide-full / slope-wide-screened | 1.28 | 1.15–1.40 |
| stability-serial / stability-threaded | 1.26 | 1.09–2.60 |

The small stability workloads have substantial timing spread; the paired median ratio differs from the ratio of marginal medians. These are measurements of the listed workloads, not general speedup guarantees.

Monte Carlo uses 350 independent data/noise repetitions per procedure. The orthogonal known-noise model follows the SLOPE paper; normalized random fixed designs and a Gaussian AR(1) model with correlation 0.4 follow the knockoff design families. Dimensions are n=240, p=100, with 20 signals of size 4.5/sqrt(n) and sigma=1. These are bounded development experiments, not full reproductions of the papers.

| Procedure | Mean FDP (MC SE) | Mean power (MC SE) | Mean false selections |
|---|---:|---:|---:|
| CPSS | 0.0000 (0.0000) | 0.1859 (0.0038) | 0.000 |
| fixed-knockoff+ | 0.0701 (0.0048) | 0.7300 (0.0110) | 1.477 |
| model-X-knockoff+ | 0.0859 (0.0046) | 0.8817 (0.0087) | 1.991 |
| slope | 0.0822 (0.0031) | 0.9856 (0.0015) | 1.854 |

SLOPE and both knockoff+ procedures target q=0.1 in these designs. CPSS uses 20 complementary pairs, a base cap q=10 and cutoff=0.8; its reported distribution-free low-selection-probability bound is 10²/(100×0.6)=1.667. Its FDP is descriptive. Interpreting its bound as PFER requires that all null variables have half-sample selection probability at most q/p. Monte Carlo is not a proof of FDR control.
