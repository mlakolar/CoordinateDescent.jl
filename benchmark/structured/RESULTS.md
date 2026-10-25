# Discovery validation, 2026-10-25

For the optimized implementation's replicated comparisons against glmgen and
genlasso, see the newer [competitiveness report](compare/RESULTS.md). The tables
below retain the original implementation measurements and their original
controls; they are not the final cross-package comparison.

Run: `20261025-cd80-structured`. Benchmarks used Julia 1.13.2, one CPU per worker on Discovery (`main`, `epyc-7513`), and one BLAS/OpenMP thread. Each cell contains three independently generated data sets (seeds 8101–8103), each measured three times after compilation on a separate short problem. Values below are median wall-clock seconds over the nine measurements.

All 108 signal measurements and all 108 retained regression measurements converged. Signal tolerances were `abstol=1e-6`, `reltol=1e-5`, with `lambda=1/n` and at most 5,000 iterations. Regression tolerances were `abstol=1e-7`, `reltol=1e-6`.

## Signal fits

| Length | Order 0 | Order 1 | Order 2 |
|---:|---:|---:|---:|
| 1000 | 0.000246593 | 0.0111667 | 0.0714086 |
| 10000 | 0.00136051 | 0.164926 | 0.760804 |
| 100000 | 0.0167378 | 2.12488 | 8.55866 |
| 1000000 | 0.2091 | 22.404 | 95.9939 |

Order zero is the exact TV proximal step. Orders one and two use specialized ADMM. The largest signal workers used about 1.4 GiB peak RSS, within their 3.8 GB step limit. The raw TSV files also contain cumulative allocated bytes, iteration counts, objectives, residuals and degrees of freedom; cumulative allocation is not peak resident memory.

## Regression fits and warm paths

| n | p | Penalty | Single fit | Four-level path |
|---:|---:|---|---:|---:|
| 300 | 80 | fused | 0.00187512 | 0.00522726 |
| 300 | 80 | graph | 0.00143631 | 0.00986739 |
| 300 | 80 | trend | 0.00153724 | 0.0053567 |
| 1000 | 200 | fused | 0.0106494 | 0.0382441 |
| 1000 | 200 | graph | 0.0110665 | 0.230111 |
| 1000 | 200 | trend | 0.0101933 | 0.0398862 |

The regression paths use levels `[0.1, 0.05, 0.02, 0.01]`; fused paths use `l1_ratio=0.25`. The graph benchmark uses a chain, while the independent golden tests also include graph cycles. These timing tables do not compare speed with R.

## Independent correctness checks

- 137 structured-regression tests cover exact small dual solutions, sparse and dense designs, Float32, intercepts, irregular locations, graph null spaces, zero penalties, CV folds, termination and input errors.
- 114 independent golden checks compare coefficients and degrees of freedom with complete `genlasso` 1.6.1 paths and compare `glmgen` 0.0.3 trend fits on the same inputs. They cover three seeds, trend orders 0–3, and unit/irregular grids. R uses `lambda_R = n*lambda_Julia`.
- The structured tests, fixed golden fixtures and Aqua passed 262 checks on each of Julia 1.13.2 and Julia 1.10.12 under Slurm. Both versions also passed all 85 deprecated-API checks in the prescribed separate process. Documentation, doctests, cross-references and all 11 executable examples passed on Discovery. Deployment and repository edit/source links were disabled for this source-tar validation.
- Earlier broad package regression checks passed in two parts; deprecated API tests are run in their prescribed separate `--depwarn=yes` process.

## Recovery and provenance

The initial regression attempt exposed nonconvergence with a data-independent ADMM `rho` default on the larger design. The default now accounts for the centered design curvature. All 72 affected graph/trend regression measurements were repeated and converged. The 36 completed fused-regression measurements were reused, since their solver was unchanged. The initial failure and its partial results remain in the run record. Signal defaults and computation were unchanged, so the 108 successful signal measurements were reused.

| Job | Purpose | Outcome |
|---|---|---|
| 13423093 | Install independent R references | Completed |
| 13423114 | R golden comparisons and 108 signal timings | Completed, all checks passed |
| 13423141 | Initial regression benchmark | Failed on ADMM nonconvergence; partial results retained |
| 13423143 | Initial dependent validation | Cancelled while pending after the failed dependency |
| 13423164 | Repaired solver checks and 72 affected regression timings | Completed, all checks passed |
| 13423165 | Julia release/LTS checks and documentation | All numerical/Aqua/deprecated checks passed; documentation wrapper used the wrong root |
| 13423204 | Documentation wrapper correction, full docs/examples build | Completed, all document checks passed |

Source provenance is retained as immutable snapshots in the run record: `source-v2.tar` for signals/fused regression and `source-fixed.tar` for the repaired ADMM regressions and final checks. `source-refs-v2.json` and the lightweight registry record hashes and dependency commits. The final implementation is commit `47f92eb`; its only source-snapshot difference relevant to resolution is the minimum ProximalBase compatibility bound, raised to 0.5.3 to require the ordered operators used here.

Recreate the work using `setup.sbatch`, `evaluate.sbatch`, `golden.R`, `run.jl` and `regression.jl` in this folder, with the recorded snapshots/environments and seeds. The local run record is `.slurm/runs/20261025-cd80-structured.json` in the enclosing juliaPkg workspace; remote results are under `/home1/mkolar/juliaPkg/runs/20261025-cd80-structured`.

GitHub CI was initially skipped following a Discovery-only relay. The user clarified that GitHub CI is allowed and that the concern is excessive simultaneous Julia processes on the laptop. Normal GitHub CI is now enabled. The Slurm results above remain independent of the GitHub CI outcome; benchmarking and profiling stay on Discovery.
