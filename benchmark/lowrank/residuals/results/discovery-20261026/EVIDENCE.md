# Retained evidence

The completed main study is Discovery job13447768 (5m27s, completed0:0).
Poststudy13447870 reads retained records only. Baseline295fbf5 and production
candidate9dd42f8 share frozen dependencies: ProximalBaseb6c949f,
ProximalSolvers531ff55 and KernelSmoothing1feff28. Frozen harnesscc02719
predates generation of evaluation inputs1–20. Per-worker source hashes were
verified against the actual measured lowrank.jl files.

- RESULTS.md: decision, public and coefficient-only comparisons, uncertainty,
  limitations and defaults.
- all-timings.csv, aggregate.csv, comparisons.csv and summary.json: every
  timing round, within-input medians, all paired ratios/ranges and bootstrap
  intervals. There are20 independent inputs per design, not60.
- certificates.csv and paired-certificates.csv:4200 independent output
  certificates and1200 native before/after comparisons.
- components.csv: cached-input component measurements, kept separate from
  public timings. These are not an additive cost breakdown.
- profile-counts.csv and poststudy.json: calls/sample counts, inclusive
  hotspot indicators, source/CPU/thread verification, numerical error maxima
  and target allocation measurements. Inclusive frames overlap.
- memory.csv: warmed current/peak, timing peak and whole-worker peak RSS.
  No uniform memory reduction is supported.
- worker-reports.tar.gz: all120 input folders' per-implementation timing and
  component records,440 CPU reports,80 allocation reports, metadata, saved
  result fields, process outcomes, R sessions and logs. Raw numerical arrays
  are excluded from this compact reporting archive.
- inputs/: seed, dimensions and exact shared-input SHA256/MD5 records.
- provenance/: frozen Julia manifests and exact main-job accounting.
- pilot/: six seed-zero calibration cases, independent certificates,
  retained failed R extraction logs and installed rrpack source body.

Raw input/reference/coefficient/residual arrays, source archives and original
launch receipts remain in
`/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue101/study-v1`;
source/environment roots are recorded in environment.sh. The owning project
registry is `.slurm/runs/20261026-hdm101.json`. No additional fits are needed
to recreate the tables or interpretation.

Pilot13447678 failed before fitting because PARALLEL is GNU Parallel's
options environment variable; use HDM_PARALLEL for its executable path.
Pilot13447703 retained successful small/medium Julia/SciPy outputs, but R
failed because coef(z) returns only the IC-selected coefficient. Recovery
13447742 reused successful results and reconstructed the native path from
coef.ls/A/Spath; all210 certificates and60 bitwise coefficient pairs passed.
These failures are excluded from evaluation and retained in the run records.

Targeted correctness job13447150 passed720 checks on each Julia1.10.12 and
1.13.2, including the existing golden/boundary controls and240 new residual,
objective, prediction, rank and support checks. Hosted checks are reported
for their exact final PR head separately. No merge is authorized.
