# Retained stopping-calibration records

See [the results](../RESULTS.md) for the default decision and limitations.

- `diagnostics/`: all original-dataset iteration traces and stopping-rule hits.
- `calibration-v2/`: four fixed rules, 15 workloads, three datasets and three
  rounds; three ordinary baseline profiles. The experimental overlay is
  separate from the package source.
- `calibration-warmfix/`: 72 replacement timings for two initial wrapper
  cases, with explicit wrapper warmup. The original records are retained.
- `holdout/inputs/cases.tsv`: the 15 new-data workloads and transported R
  controls. Generators and seeds recreate the large input matrices.
- `holdout/reference/`: 135 timed glmgen measurements and R provenance.
  `holdout/reference-status.tsv` aggregates all 45 tighter-reference flags.
- `holdout/julia/`: 270 actual-source measurements, with source order
  alternating by seed, plus replicated CPU and sampled allocation profiles.
  This allocation contains both the original source and first candidate.
- `holdout/julia-final/`: 135 final-source candidate measurements, overriding
  only the candidate records in the final summary.
- `same-allocation-ratios.tsv`: the old/new timing comparison from the first
  actual-source allocation, before the final-source override.
- `summary/`: final medians, coverage/accuracy/convergence gates, paired
  dataset ratios and comparisons with the transported R settings.
- `selection.json`: candidate selection frozen before new-data Julia fits.
- `validation/`, `validation-final/`: release/LTS test logs and worker outcomes.
- Root job logs, `accounting.tsv` and `provenance.json`: failed launcher,
  recovery, exact intervals, source snapshots and dependency references.

Each setting/case has nine timing rows, with a deterministic fit repeated
three times per dataset. First-round coefficient and objective errors are
checked against both R results outside timing; other rounds keep iteration
and convergence records. R-versus-tight errors are also retained. A solver's
convergence flag alone does not qualify a speed comparison.

`bytes` records cumulative Julia allocation, not peak resident memory.
CPU sampling and allocation profiling run separately from benchmark timing.
Profiles from the first candidate remain applicable to its unchanged signal
branch; final-source timing repetitions confirm the production code. The
nonfatal DWARF address-range warnings are retained in worker logs.

Large coefficient/input binaries and frozen environments remain in the remote
run directory in `provenance.json`. No hidden refit is performed when rebuilding
the summary from these records.
