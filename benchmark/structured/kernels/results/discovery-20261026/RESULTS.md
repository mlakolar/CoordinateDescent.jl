# Trend product holdout study

Three fresh independent shared inputs per case, three warmed complete public fits/paths per input and implementation. Paired before/after comparisons use within-input medians, with method order rotated. The seed-zero calibration is excluded. All numerical work ran under Discovery Slurm on one fixed EPYC7513 allocation, with one Julia/BLAS/OpenMP thread per exclusive worker.

| Case | Inputs | Baseline s | Candidate s | Paired median | Input ratio range |
|---|---:|---:|---:|---:|---|
| small | 3 | 0.013789 | 0.011501 | 0.869 | 0.834–0.876 |
| irregular | 3 | 0.149757 | 0.130307 | 0.862 | 0.845–0.870 |
| linear100k | 3 | 1.696660 | 1.451598 | 0.855 | 0.842–0.863 |
| quadratic100k | 3 | 6.966461 | 5.723122 | 0.820 | 0.799–0.822 |
| tvpath | 3 | 0.023522 | 0.022528 | 0.958 | 0.876–1.065 |
| linearpath | 3 | 6.067690 | 5.057387 | 0.832 | 0.824–0.833 |
| quadraticpath | 3 | 72.858649 | 58.293360 | 0.794 | 0.793–0.800 |
| linear1m | 3 | 17.714129 | 15.043289 | 0.840 | 0.824–0.854 |
| quadratic1m | 3 | 279.175477 | 217.253187 | 0.816 | 0.778–0.821 |

Every input ratio and descriptive bootstrap interval is in summary.json. Three independent inputs provide limited precision; timing rounds are not additional independent replications. Preserve small/reversed effects and all controls. Public timers include setup, factorization, iterations, results and DF. Imports, input loading, compilation/warming, independent checks and profiles are outside timers.

The independent glmgen fits use obj_tol=1e-12, thinning disabled and the same original input bytes/grid. They provide cross-solver agreement checks, not a certified primal-dual gap or a timed performance ranking. Coefficient agreement must be<=1e-4 and objective difference<=1e-6 for every returned level. The candidate changes product kernels only; rho, TV prox, banded factors and stopping/DF tolerances are fixed.

## Validation

```json
{
  "phase": "main",
  "cases": 27,
  "expected_cases": 27,
  "errors": []
}
```

Full per-input native and reference checks are in inspection.json. Process errors, incomplete coverage and native nonconvergence remain visible and preclude an adoption claim. Allocation totals are cumulative fit/path bytes. Memory records distinguish warmed current/peak, timing peak and final whole-worker peak, which includes separate profile work; none is isolated solver workspace. Profiles cover every dataset separately and record snapshots, intervals and sampled allocations. Inclusive frames overlap.

## Provenance and decision

Baseline295fbf5 and the frozen candidate use the same pinned PB/PS/KS dependencies and Julia1.13.2. Source hashes, CPU/thread metadata, saved seeds/input hashes, process outcomes, R sessions and exact launchers are retained. Raw input/fitted arrays and source/environment snapshots remain in the owning run. See README and run registry. No merge is authorized. A recommendation requires complete validated coverage and repeatable useful public gains; inspect the final ratios and adverse evidence before changing the internal default.

## Final recommendation

Enable the bounded private stencil route automatically for its supported
linear/quadratic trend products. All eight non-TV cases improved on each of
three independent inputs, including the two million-point signals. The paired
median ratios range from0.794 for the quadratic path to0.869 for the small fit.
Keep the sparse fallback for unsupported products and all existing numerical
and statistical defaults. This evidence supports a kernel optimization, not a
new TV algorithm, looser stopping rule or changed estimator.

The TV path executes unchanged production code. Its input ratios0.876–1.065
include a regression; no TV speedup is attributed to this PR. Three independent
inputs provide limited precision, and whole-process memory includes warming,
profiling and runtime effects. The per-input memory/allocation records remain
visible; no uniform peak-memory reduction is claimed.

Main13448193 completed0:0 in85m18s within its90-minute limit:27 shared inputs,
162 timing rows,81 completed processes,54 CPU and54 allocation reports,468
returned levels per Julia variant. Every native level passed finite-output,
convergence, objective/DF and before/after checks; tight glmgen agreement passed
for every level. Final-source structured/golden plus428 new product/boundary/
nonfinite/dual/fallback checks passed on both Julia1.10/1.13. Source0b4b186
passed all hosted release/LTS/macOS/Windows/examples CI and documentation checks.

The attempted wall-time extension had been denied by Slurm. Conditional
recovery13449627 found no missing/failed process stages and performed no fits.
The receipt is retained. Raw input/coefficient arrays and source/environment
snapshots remain on Discovery; the adjacent tables, diagnostics, profiles,
metadata and logs are the reporting inputs. Nothing was merged.
