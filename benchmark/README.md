# Benchmarks

This suite measures the time and Julia heap allocation of public estimator
calls. Every timed output is checked by an accuracy certificate computed from
the inputs and the returned coefficients, independently of the solver's stopping
rule; an output that misses its target is a recorded failure, not a timing. It
does not measure native (non-Julia) allocations or the statistical quality of
the estimates. Peak memory is recorded only as the process-wide `maxrss_bytes`.

## Setup and selections

From the package root, with Julia 1.10 or later and Python 3.11 or later:

```sh
julia benchmark/setup.jl "$PWD" benchmark/results/env
JULIA_NUM_THREADS=1 julia --project=benchmark/results/env benchmark/run.jl
python3 benchmark/summarize.py benchmark/results benchmark/results/summary
```

`setup.jl PACKAGE_ROOT ENV_DIR [REFERENCE_ENV]` creates a fresh environment with
the dependencies of `benchmark/Project.toml` and the checkout at `PACKAGE_ROOT`
developed into it, which also works on Julia 1.10, where `[sources]` is ignored.
It refuses an existing `ENV_DIR/Project.toml`. The runner needs one Julia
thread, and sets one BLAS thread itself. The default is one quick Float64
repetition: an exploratory check, with no between-run interval. Use a fresh
output directory for each measurement series; existing records are never
overwritten.

| Setting | Default | Use |
|---|---|---|
| `HDM_SUITE` | `quick` | `extended` runs the quick cases and adds larger, wide and correlated shapes, sparse, weighted and Float32 inputs, and more paths and variants. |
| `HDM_GROUPS` | All groups but `controls` | Comma-separated groups, selected before any data are drawn. |
| `HDM_CASES` | All cases | Comma-separated globs on case ids (`*`, `?`), within the suite and groups. |
| `HDM_REPETITION` | `1` | Process and input draw, numbered 1–10. |
| `HDM_OUTPUT` | `benchmark/results` | Directory for the process records. |
| `HDM_SAMPLES` | `50` quick; `100` extended | Sample limit, with at least three samples. |
| `HDM_SECONDS` | `1` quick; `3` extended | Limit on the accumulated timed seconds of a case. |
| `HDM_CASE_SECONDS` | `60` | Limit on a single compiled call; see `timeout` below. |
| `HDM_PROFILE_SECONDS` | `0` | Positive values write diagnostic profiles of each `ok` case. |
| `HDM_EXPECTED_PACKAGE_ROOT` | Unset | The checkout that must be loaded, verified before any case is built. |
| `HDM_LIST` | `false` | `true` prints the selected case ids, suites and targets, and exits. |

A case id is `group/estimator/variant`. Run the group of the code you change;
before a pull request that touches shared code (losses, coordinate descent,
paths, standardization), run the quick suite of every group.

| Group | Estimators | Run it for changes to |
|---|---|---|
| `lasso` | `lasso` (dense, sparse CSC, Gram, weights and offset, wide, correlated, large, Float32), `sqrt_lasso`, `scaled_lasso`, `LassoPath` with and without screening, `refit_lasso_path` | `src/lasso.jl`, `src/losses.jl`, `src/coordinate_descent.jl`, the coordinate descent and paths of ProximalSolvers |
| `glm` | `glm_lasso` (logistic and Poisson; logistic with weights and offset, elastic net, MCP, SCAD, a large shape, Float32), `GLMLassoPath` | `src/glm.jl`, `src/penalties.jl`, proximal Newton |
| `group` | `group_lasso` (`:orthonormal`, `:none`), `GroupLassoPath`, `sparse_group_lasso`, `SparseGroupLassoPath` | `src/group.jl`, `src/sparsegroup.jl` |
| `multitask` | `multi_lasso`, `MultiLassoPath` | `src/multiresponse.jl` |
| `lowrank` | `reduced_rank` (also wide and Float32), `nuclear_regression`, `sparse_reduced_rank`, `LowRankPath` | `src/lowrank.jl` |
| `quantile` | `quantile_lasso` (exact with HiGHS, with an offset, a large shape; smoothed), `QuantileLassoPath` | `src/quantile.jl`, `ext/HighDimensionalModelsHiGHSExt.jl` |
| `structured` | `fused_lasso` (signals, regression), `trend_filter` (signal, regression), `generalized_lasso`, `fused_lasso_path`, `trend_filter_path` | `src/structured.jl` |
| `selection` | `slope`, `SLOPEPath`, `equicorrelated_s` with `gaussian_knockoffs`, `knockoff_filter`, `stability_selection` | `src/slope.jl`, `src/knockoffs.jl`, `src/stability.jl` |
| `inference` | `rlasso`, `post_lasso`, `double_selection` | `src/hdm.jl` |
| `controls` | A lasso stopped after one pass, which must end `target_missed` | The harness itself; never selected by default |

For example, list the extended structured cases without measuring them:

```sh
HDM_SUITE=extended HDM_GROUPS=structured HDM_LIST=true \
  julia --project=benchmark/results/env benchmark/run.jl
```

## Certificates and targets

`problems.jl` draws the inputs of each scenario from Xoshiro streams seeded by
its key and the repetition, the same on every supported Julia version.
`certificates.jl` states each problem from the estimator's documented objective,
with its loadings, intercept, weights and offset computed from the inputs, and
reads only the returned coefficients and intercepts on the original scale. For
the adaptive procedure `rlasso` it also reads the penalty level and loadings of
the final solve, so that certification is conditional on them. It never reads
the package's gaps, KKT fields or `converged`.

Most targets are thresholds on a relative duality gap, a stationarity residual
or a coefficient error, at settings fixed per case in `cases.jl`; where the
defaults stop short of the threshold (the path gap tolerance, the group lasso
without orthonormalization, the smoothed quantile tolerance), the case tightens
them and says why. The nonconvex `sparse_reduced_rank` has the target `record`:
the certificate must be valid, and its value is recorded without a threshold.
MCP and SCAD are certified as stationary points only. The selections of
`knockoff_filter` and `stability_selection` have no certificate, but their
post-processing (the knockoff+ threshold of the returned statistics, the
frequencies of the returned selections) is checked exactly, at a threshold of
1e-12. `post_lasso` must refit the requested columns, and `double_selection` is
certified on its effect regression, both nuisance lassos and the union of their
supports. A path certifies every returned level and reports the
worst. `check_certificates.jl` tests each certificate with analytic problems,
designed nonoptimal points, weak duality and dual feasibility; agreement with the
package's own diagnostics is secondary.

| Status | Meaning |
|---|---|
| `ok` | The certificate is valid and its value meets a threshold target (any value for `record`). |
| `target_missed` | The certificate is valid and its value exceeds the threshold. |
| `incomplete` | A path returned fewer levels than requested. |
| `uncertifiable` | The certificate is outside its domain or nonfinite, for example a square-root lasso with an exact fit. |
| `error` | Building the inputs, the call or the certification throws. |
| `timeout` | A compiled call took longer than `HDM_CASE_SECONDS`; the case stops after that call, which is not interrupted. |

A case takes the most severe status of its observations, in the order `error`,
`timeout`, `incomplete`, `uncertifiable`, `target_missed`. Only `ok` samples are
timing-eligible; failed samples are recorded and never filtered.
`HDM_GROUPS=controls` checks this end to end: its case ends `target_missed`,
with no timing-eligible sample, and the runner exits with 2.

## Compare revisions

Keep one copy of this harness fixed and measure two checkouts with it, in two
fresh environments. The second environment copies the manifest of the first, so
every dependency other than HighDimensionalModels is identical; `setup.jl`
fails if any other manifest entry changes. Interleave the repetitions in fresh
processes, alternating which revision runs first:

```sh
before=/absolute/path/to/before
after=/absolute/path/to/after
julia benchmark/setup.jl "$before" benchmark/results/env-before
julia benchmark/setup.jl "$after" benchmark/results/env-after benchmark/results/env-before
runs=3
for rep in $(seq 1 "$runs"); do
  if [ "$((rep % 2))" -eq 1 ]; then set -- before after; else set -- after before; fi
  for side in "$@"; do
    code=0
    case "$side" in before) checkout="$before" ;; after) checkout="$after" ;; esac
    HDM_EXPECTED_PACKAGE_ROOT="$checkout" HDM_REPETITION="$rep" \
      HDM_OUTPUT="benchmark/results/$side" JULIA_NUM_THREADS=1 \
      julia --project="benchmark/results/env-$side" benchmark/run.jl || code=$?
    [ "$code" -eq 0 ] || [ "$code" -eq 2 ] || exit "$code"
  done
done
python3 benchmark/compare.py benchmark/results/before benchmark/results/after \
  benchmark/results/comparison --expected-runs "$runs"
```

Choose `runs` from 1 to 10. Use the same suite, groups, sample and time limits,
machine, and Julia and BLAS versions on both sides; different Julia versions need
separate environments (`setup.jl` refuses a reference made by another version).
The runner checks `HDM_EXPECTED_PACKAGE_ROOT` before building any case. For an
A/A comparison, set `before` and `after` to the same revision; it shows the
noise level of the machine.

`compare.py` writes `REPORT.md`, `comparison.csv` and `coverage.json` to a fresh
directory. It reports after/before ratios of the repetition medians with paired
95% intervals on log ratios, the worst certificate values, and the mean and
maximum bytes with their changes. Single-run ratios are exploratory, and an
interval containing one is inconclusive. A case with a non-`ok` repetition is
excluded from the ratios, with its reasons in the report. Invalid records,
missing repetitions, unmatched cases, or differences in settings, catalog,
data hashes, or dependency versions and tree hashes prevent a comparison. A
dependency tracked by path has an empty tree hash and is compared by version
only, so the compared environments should use registered dependencies.

## Measurement and evidence

The phase is always `call`: the warmed public call, including its own setup
and result construction. Drawing the inputs, preparing their element type and
storage, compilation, resets and certification are outside timing. The first
call of a case compiles; it is certified but contributes no sample. Each sample
then starts from fresh copies of the prepared inputs and a reset random number
generator (the default one is reseeded too), times one call with `@timed`, and
certifies that output. Sampling stops at `HDM_SAMPLES` or `HDM_SECONDS`, with
at least three samples unless an error or timeout stops the case. The order of
the cases is shuffled per repetition with a seeded generator.

Intervals describe variation across processes and input draws, including
machine noise; they do not correct for correlated noise or multiple
comparisons. Profiles are diagnostic, not latency estimates. The default suite is
not a substitute for the package tests or for broad coverage of parameter
regimes. Do not select the fastest run or discard adverse runs; put the concise
finding and its limitations in the pull request, and preserve the manifests and
any uncommitted sources behind a claim.

## Records

Each process writes files with the prefix `julia-<VERSION>-rep<NN>` under
`HDM_OUTPUT`. The TSV files have headers, empty missing fields, integer
nanoseconds and bytes, and round-trip floats:

| Suffix | Columns or contents |
|---|---|
| `-catalog.tsv` | `id group scenario estimator variant phase suite inputs measure target` |
| `-metrics.tsv` | `id group phase rep status samples ok_samples median_ns mean_ns sd_ns min_ns max_ns p95_ns gc_share median_bytes mean_bytes min_bytes max_bytes measure target worst_value median_value data_hash` |
| `-samples.tsv` | `id rep sample time_ns gc_ns bytes value status` |
| `-errors.tsv` | `id rep status exception stack`; one row per non-`ok` case |
| `-profiles.txt` | Optional profiles, with `HDM_PROFILE_SECONDS > 0` |
| `-metadata-start.toml`, `-metadata.toml` | Provenance at the start; the same with the final keys at the end |

The metadata records the settings; the Julia, BLAS and CPU identity; the
package path, commit and dirty status; SHA-1 hashes of the package's
`Project.toml`, `src/` and `ext/` (`package_hash`) and of the Julia files of this
folder (`harness_hash`); and a `[dependencies]` table with the version and tree
hash of every package of the manifest that is not a standard library, other than
HighDimensionalModels (a package tracked by path has an empty tree hash). The
SHA-256 `data_hash` of a case covers its inputs and penalty levels. It is
specific to the Julia version: the scenario inputs are identical on every
version, but penalty levels computed from them can differ by one ulp. The final
metadata adds `cases`, `ok_cases`, `nonok_cases`, `elapsed_s`, `maxrss_bytes` and
`finished = true`; start metadata without it marks an interrupted process, which
is kept and restarted in a fresh series.

The runner exits with 0 when every case is `ok`, 2 for complete records with
non-`ok` cases, and 3 when it cannot start or write its records. Validation
checks the integrity of records separately from numerical success, so records
with failed cases can be valid:

```sh
python3 benchmark/validate.py benchmark/results benchmark/results/validation.json
python3 benchmark/check_analysis.py
JULIA_NUM_THREADS=1 julia --project=benchmark/results/env benchmark/check_certificates.jl
```

Pass `--expected-runs N` to `validate.py`, `summarize.py` and `compare.py` for a
series of `N` repetitions; `validate.py` and `summarize.py` also take
`--versions VERSION ...`. `summarize.py` writes `SUMMARY.md` and `summary.csv`.
Generated results, environments and profiles are ignored by Git.

## Historical studies

The studies that preceded this harness, with their scripts, comparisons with
other packages and run records, remain in the
[snapshot of the retired benchmarks](https://github.com/mlakolar/HighDimensionalModels.jl/tree/abddaf56a790b86178de376b175a1185020c8381/benchmark).
These measurements have no tool here:

- the setup cost of cross-validation and GLM paths (#73),
  [`path_setup_bench.jl`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/abddaf56a790b86178de376b175a1185020c8381/benchmark/path_setup_bench.jl);
- coordinate orders, momentum and Anderson acceleration,
  [`order_anderson_bench.jl`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/abddaf56a790b86178de376b175a1185020c8381/benchmark/order_anderson_bench.jl);
- the Gram matrix against the residuals, which calibrated `gram = :auto`,
  [`gram_bench.jl`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/abddaf56a790b86178de376b175a1185020c8381/benchmark/gram_bench.jl);
- the varying-coefficient models,
  [`varying/`](https://github.com/mlakolar/HighDimensionalModels.jl/tree/abddaf56a790b86178de376b175a1185020c8381/benchmark/varying).

The generators of the golden test data now live next to their fixtures, under
`test/data/` and `test/golden/`.
