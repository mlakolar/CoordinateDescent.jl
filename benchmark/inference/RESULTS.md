# Issue 77 Discovery validation

Run: `20261025-cd77-inference`. All Julia execution and numerical/reference computations ran inside Slurm allocations.

Final timings use three independent design seeds, one untimed full-size warm-up, one timed call, and one separately profiled call per replication. Four Julia threads, one BLAS thread, n=200, AR(1) correlation 0.3.

| p | Gram | replications | median seconds | range seconds | median allocated MiB | median inverse nonzeros |
|---:|:---:|---:|---:|---:|---:|---:|
| 1000 | false | 3 | 0.0908 | 0.0872–0.1374 | 179.0 | 5649 |
| 1000 | true | 3 | 0.0502 | 0.0482–0.0670 | 170.4 | 5649 |
| 10000 | true | 3 | 3.3204 | 2.8449–3.3878 | 15261.3 | 80534 |

Allocated bytes are cumulative allocations per call, not peak resident memory. Slurm accounting records peak memory separately. Initial v1 timing files include predictor-count-dependent sparse concatenation compilation and are retained as diagnostic results; they are excluded from final timing claims.

## Coverage

Fixed n=100,p=500 Gaussian designs: Toeplitz correlation 0.9 and equicorrelation 0.8; support sizes 3/15, first/random support, fixed U[0,2]/U[0,4] signals. Each scenario has 100 independent error draws. Initial fit: universal scaled lasso; nodewise penalty 0.1 on population-standardized columns. These are the paper’s designs, with an explicit fixed nodewise penalty rather than its pooled-CV tuning. Simultaneous coverage and null FWER use 20 error replications per scenario and B=499 multipliers.

| design | s | support | c | reps | active coverage | null coverage | active length | simultaneous coverage | null FWER |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| toeplitz | 3 | first | 2 | 100 | 0.803 | 0.950 | 0.684 | 0.950 | 0.050 |
| equicorrelated | 3 | first | 4 | 100 | 0.847 | 0.913 | 0.638 | 0.400 | 0.550 |
| equicorrelated | 3 | random | 2 | 100 | 0.910 | 0.913 | 0.616 | 0.400 | 0.600 |
| equicorrelated | 3 | random | 4 | 100 | 0.847 | 0.911 | 0.599 | 0.500 | 0.500 |
| equicorrelated | 15 | first | 2 | 100 | 0.395 | 0.889 | 0.419 | 0.000 | 1.000 |
| equicorrelated | 15 | first | 4 | 100 | 0.296 | 0.886 | 0.409 | 0.000 | 1.000 |
| equicorrelated | 15 | random | 2 | 100 | 0.507 | 0.900 | 0.450 | 0.000 | 1.000 |
| equicorrelated | 15 | random | 4 | 100 | 0.500 | 0.901 | 0.463 | 0.000 | 1.000 |
| toeplitz | 3 | first | 4 | 100 | 0.810 | 0.951 | 0.678 | 0.950 | 0.050 |
| toeplitz | 3 | random | 2 | 100 | 0.897 | 0.945 | 0.752 | 0.950 | 0.050 |
| toeplitz | 3 | random | 4 | 100 | 0.923 | 0.946 | 0.771 | 0.950 | 0.050 |
| toeplitz | 15 | first | 2 | 100 | 0.643 | 0.948 | 0.682 | 0.400 | 0.100 |
| toeplitz | 15 | first | 4 | 100 | 0.615 | 0.950 | 0.684 | 0.350 | 0.000 |
| toeplitz | 15 | random | 2 | 100 | 1.000 | 0.941 | 3.433 | 0.850 | 0.150 |
| toeplitz | 15 | random | 4 | 100 | 1.000 | 0.939 | 6.786 | 0.950 | 0.050 |
| equicorrelated | 3 | first | 2 | 100 | 0.847 | 0.913 | 0.627 | 0.600 | 0.400 |

Validated coverage rows: 1600. Coverage fractions within a response draw are correlated across coefficients; the 100 response draws, rather than 100*p independent trials, are the replication unit. Simultaneous estimates have coarse Monte Carlo resolution (1/20). Poor coverage is an adverse finite-sample result, not a numerical convergence failure.

## Interpretation

The matched-tuning numerical comparisons check implementation agreement, not
finite-sample validity. In these fixed-penalty experiments, the sparse Toeplitz
null intervals were close to nominal, while active-variable intervals could
undercover. Equicorrelated designs produced substantial undercoverage and
poor family-wise error control, including at s=3. With s=15, first-support
active coverage was as low as 0.296 and simultaneous coverage was zero in all
four equicorrelated settings. These results do not support a claim of general
95% coverage for the supplied tuning. They are retained as adverse results.
The study uses the published covariance/signal designs but does not reproduce
the paper's pooled-CV nodewise tuning or its exact fixed data realization.

The numerical methods remain asymptotic procedures with design, sparsity and
tuning requirements. Golden-data agreement and convergence do not override
these observed limits. QP inference and alternative tuning are available, but
this coverage study does not establish their empirical calibration.

## Correctness and provenance

The final external reference check passed 24/24 comparisons: exact-lambda
glmnet nodewise scores, `hdi::lasso.proj` estimates/SEs/p-values, the full hdi
bootstrap distributions and equal-tail/max-T p-values at matched nuisance fits
and sampled errors, the authors' Javanmard--Montanari QP rows, three zero-fusion
CovarianceSelection neighborhoods, and R `glm` logistic/Poisson MLE/SE/score
limits. The package's new unit and committed golden tests passed 81+17 checks
on Discovery. The full suite passed in two partitions: 2,184 checks through
multiresponse and 401 remaining checks including Aqua, plus 85 deprecated-API
checks in the subprocess. Documentation built successfully. Initial staging
failures and run-environment corrections remain in the registry.

Final benchmark/profile job: 13423152. Coverage job: 13423153. Golden regression
job: 13423187. All run on Discovery. Source: upstream c408831 plus saved v3
implementation snapshot and unchanged upstream extension/example supplement.
Julia 1.13.2; ProximalBase 0.5.3, ProximalSolvers 0.1.1, CovarianceSelection
0.1.0 from the pinned snapshots in the run manifest. R 4.5.3, hdi 0.1-10,
glmnet 5.1. Full sampling profiles and failed-attempt logs remain in the run
output directory. No Julia code ran on the local Mac for this task.

Final full-suite jobs: 13423202 (2,184 passed; stopped at a missing direct
run-environment dependency) and 13423217 (401 remaining checks, deprecated
subprocess, Aqua, documentation and aggregation passed). No library changes
were required by those environment corrections.

The additional logistic study is in [GLM_RESULTS.md](GLM_RESULTS.md): 240
response replications, all converged. It also shows strong-signal undercoverage
and does not establish general finite-sample calibration. Both studies,
including adverse outcomes, are retained in `results/`.
