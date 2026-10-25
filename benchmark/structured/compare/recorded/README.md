# Recorded comparison evidence

See [the report](../RESULTS.md) for interpretation and limitations.

- `main/inputs/cases.tsv`: all 60 workloads and penalty grids.
- `main/baseline-full/`: initial Julia defaults and R 1e-10 references; the
  tighter untimed reference success flags and R environment provenance.
- `main/baseline-matched/`: pre-optimization Julia at the calibrated controls.
- `main/optimized-matched/`, `main/optimized-default/`: optimized v3 measurements.
- `main/optimized-gram-matched/`, `main/optimized-gram-default/`: final v4
  measurements overriding exactly four changed path cases. All other v3
  computations are unchanged.
- `main/reference-special/`, `main/reference-qr/`, `main/reference-loose/`:
  specialized genlasso, generic default QR and the two additional glmgen
  objective thresholds. Generic SVD is in `baseline-full/`.
- `accuracy/`: every common coefficient/objective check, including failures,
  after applying the final four-case override.
- `summary.json` and `*-ratios.tsv`: all stages and the fastest passing setting
  selection. `reference_candidates` preserves rejected fast settings.
- `optimization-ratios.tsv`: same-control before/after measurements and
  cumulative allocations, paired within dataset.
- `million/`: independent larger-input confirmation, with its own case table,
  54 timings, reference flags, accuracy rows and summaries. Replicate IDs
  8201–8203 correspond to RNG seeds 18201–18203.
- `profiles/`: CPU sampling and largest sampled allocation stacks for all
  three datasets, including the intermediate compact-solve bottleneck.
  Profile overhead is excluded from timing data.
- `diagnostics/`: independent dual checks and the adverse generic R cases.
- `logs/`, `accounting.tsv`, `provenance.json`: validation, failures/repairs,
  exact scheduler records, source snapshots and dependency versions.

Each timing TSV reports its actual solver, replicate, round and controls via
its stage. R `glmgen` means measured objective threshold 1e-10;
`glmgen-default` means 1e-5, and `glmgen-1e8` means 1e-8. All use the raised
20,000-iteration cap. A timing `converged=true` flag does not imply that the
cross-solver accuracy target passes; use the separate accuracy gate.

Julia `bytes` is cumulative allocated memory. `worker_hwm_bytes` is cumulative
process high-water RSS, not an isolated per-fit peak. CPU profile counts are
sampling counts, not independent repetitions or elapsed seconds. The final
cached-path profiles contain ten fits each; the earlier CPU profiles contain
one fit each after warmup. Allocation profiles sample at rate 0.01.

Large coefficient binaries and synthetic input matrices remain in the remote
run directory recorded in `provenance.json`. Seeds and generators recreate
inputs. The adjacent harness README gives commands to rebuild summaries from
these compact records without fitting any model.
