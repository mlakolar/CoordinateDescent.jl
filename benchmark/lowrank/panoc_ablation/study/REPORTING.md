# Paired reporting, independent of active worker code

`finalize.py` reads the immutable primary summary and saved worker results under
Slurm. It never fits, retimes, or changes primary receipts. Stage as
`report-v1-code/finalize.py` and `report-v1-code/finalize.sbatch`, separately from
the running `study-v1-code/` directory.

Canary (30 existing processes, two rep0 inputs; excluded from holdout):

```sh
sbatch --parsable report-v1-code/finalize.sbatch --root results/study-canary-v1 --out results/report-canary-v1 --canary
```

After the main study:

```sh
sbatch --parsable --dependency=afterany:13451665 report-v1-code/finalize.sbatch
```

Outputs include paired per-input ratios and geometric-mean 95% intervals, all
cold fractions and paths, public and cached modes separately, complete-result
comparators, controlled ablations, saved-coefficient cross-mode hashes and
independent certificate comparisons, cached setup allocation witnesses, source
and worker hashes, profiles, stage-specific memory, exact replay counters and
cached component measurements. Failed primary records remain in the report.
Intervals use inputs, not repeated timing rounds, and have no multiplicity
adjustment. A speed comparison requires both original-design certificates to
pass; incomplete paths do not become complete-path speed observations.

The scripts emit measurements, not an automatic adoption/default decision.
Review regressions, uncertainty, correctness and practical limits before making
that recommendation. No new default is selected from the reporting canary.
