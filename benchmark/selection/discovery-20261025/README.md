# Discovery validation, 25 October 2026

[Results: timings, paired ratios, and Monte Carlo](results/summary.md).
The raw inputs to that report are seven timing CSVs and seven Monte Carlo
CSVs in `results/`; all 63 replicated flat profiles are retained there too.
The generator and recreation instructions are in [the parent guide](../README.md).

Successful measurement jobs:

| Job | Node | Work |
|---|---|---|
| 13423174 | a01-16 | 17 R goldens, 80 unit checks, timing/profile replication 1, 50 Monte Carlo seeds |
| 13423195 | a01-09 | timing/profile replications 2–7, 300 additional Monte Carlo seeds |
| 13423193 | a01-04 | full Julia 1.13.2 suite: 2,584 checks, including Aqua |
| 13423196 | a01-03 | Julia 1.10.12 compatibility: 80 unit checks plus 17 R goldens |
| 13423250 | a01-04 | doctests, cross-references and HTML documentation rendering |

The measurement code is the runtime source at `6aa6c11`, staged as
`source-v7.tar.gz` plus the corrected reference/launcher overlays recorded in
the run registry. Later changes clarify the fixed-X intercept assumption in docstrings and
stop a screening repair when fewer than two inner iterations remain. Every
measured SLOPE fit converged; its iteration limit was not binding. The
separate guard validation covers the new budget edge case. The final R fixtures are committed under
`test/data/selection`; R is not needed for normal package tests.

[The Julia 1.13 environment](environment-1.13.toml) is the Manifest from
`benchmark/selection`, with ProximalBase 0.5.3 tree
`9b416f6169aafedda92450c3cbd99e277e48d21a` and ProximalSolvers 0.1.1 tree
`9fdc989118e73fab57119db634bbca306dc248d0`.
[Julia 1.10](environment-1.10.toml) develops those same installed sources.
These are environment references; relative/path entries retain their original
run context. To recreate the benchmark environment, place the 1.13 Manifest
beside the parent `Project.toml` in `benchmark/selection` of a source checkout.

The fixtures use R 4.5.3, SLOPE 2.1.1, knockoff 0.3.6 and stabs 0.7.1;
`test/data/selection/R-session.txt` records the remaining R versions.
[Golden-check receipt](golden-validation.log), [full suite](full-tests-1.13.log),
and [Julia 1.10 checks](compatibility-1.10.log) retain the actual outcomes.

Wide SLOPE profiles have BLAS matrix-vector multiplication as their largest
non-wait leaf frame in all seven replications. Gaussian construction profiles
are dominated by the symmetric eigendecompositions. Stability profiles show
lasso passes, half-sample copies and thread waiting. Flat profiles include idle
thread frames; those frames should not be treated as numerical work.

The timings cover fixed data with independent Julia processes, rotated case
order, and per-case warm-up. GNU Parallel kept two `srun` requests outstanding;
Discovery serialized their steps under the allocation's memory semantics.
Each benchmark process used two Julia threads and one BLAS thread. This is
why the report distinguishes a paired speed ratio from a ratio of marginal
medians, particularly for the small stability workload.

The Monte Carlo designs are bounded instances of the papers' orthogonal,
Gaussian fixed-X and Gaussian AR(1) design families, with 350 repetitions per
procedure. The report preserves CPSS's low power for its chosen cap/tuning;
it does not claim a universal advantage or an unconditional FDR guarantee.

Run registry: `/Users/mladen.kolar/projects/juliaPkg/.slurm/runs/20261025-cd82-selection.json`.
Remote results: `/home1/mkolar/juliaPkg/runs/20261025-cd82-selection`.
The registry retains the early failed preparation/reference checks and their
repairs. Completed numerical experiments were not repeated for transfer or
report generation.
