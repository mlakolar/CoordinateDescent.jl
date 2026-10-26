# Explicit knockoff construction choices

The [final evidence report](results/discovery-20261026/RESULTS.md) separates
statistical benefit, adverse results, correctness, runtime and defaults.
Equicorrelation remains the default. MVR/maxent use the optional Knockoffs.jl
3.1.4 diagonal optimizer with the native sampler and original-objective checks.
The implementation does not copy the optimizer or add covariance caching.

`DESIGN.md` records the API before implementation. `PROTOCOL.md` fixes the
main laws, seeds, controls, comparators and timing analysis. The separate
`ORDER_SENSITIVITY.md` records the before/after variant-order concern and its
alternating-order sensitivity before execution. No primary rows are replaced.
Historical canary/recovery scripts retain their attempt-specific paths; they
are not fresh-environment entry points. For a portable optional-extension
check with workspace dependencies use `scripts/test_knockoff_extension.jl`.

## Recreate the numerical evidence

Use slurm-workflow on Discovery; no numerical work runs on the Mac. Original
root: `/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue106`. Source
baseline is `295fbf5420d6384a1ca261023367ee86d0a40b8c`; scientific candidate
source is `f14306b8b0350428f2cbb03c8126347fe84f0a3e` (candidate2). It is
unchanged throughout the main studies. Freeze sibling packages at:

- ProximalBase `b6c949f3dd26cad09252e1c2c9e49810eb69403e`;
- ProximalSolvers `531ff55fcf8a5ce2c6997c144d4f57e85cf79f3e`;
- KernelSmoothing `1feff286d0020764d2c1b03d733ab316cf948ead`.

Restore the source archives and actual Project/Manifest files from the
reporting bundle into a new immutable run directory, remapping frozen path
dependencies deliberately. Scientific fitting uses Julia 1.10.12 on EPYC7513,
one Julia/BLAS/OpenMP thread. Release/LTS correctness additionally uses
Julia 1.13.2. The comparison environment pins Knockoffs3.1.4; its source is
`d23942e9af9103f93be9a20e47ae6f7cad685aa5`, MIT licensed. R 4.5.3 uses
knockoff 0.3.6 and glmnet 5.1, with actual session/source receipts. Python
NumPy/SciPy/CVXPY/CLARABEL versions are retained in the objective summary.

The main statistical source archive is `study-main-v1.tar`, SHA256
`09f3ef6d127b465455bde39780fc8278ab1c67b4a21cf824823fdddd9068ebf4`.
Run `study_main.sbatch` after a separate shard-zero pilot and comparator
accuracy gate. This generates16,128 inputs, 48,384 native fits/certificates,
and 1,008 R fits on 336 of those same inputs. It writes its own summary/tables.
Generation seeds and draw order are in the protocol; never count the R subset
as additional independent data. Raw X/y/innovations and coefficients remain
available on Discovery and can be regenerated from the pinned recipe.

The primary runtime archive is `runtime-main-v1.tar`, SHA256
`7a19cefb5fde6445b0e015371e0f7ccfef0d9835243075aba5288a99642908a9`.
Run `runtime_main.sbatch`, then `runtime_report.sbatch`. It uses105 fresh
shared inputs, 70 native and 35 R workers, three warmed rounds per input and
separate profiles. The primary launch had baseline/candidate phase grouping;
its before/after results remain explicitly qualified. The order sensitivity
archive is `runtime-order-check-v1.tar`, SHA256
`df614fd53aea735eabb028a89aa9b6c201e87fbb3a680fa337e357e8432601f1`.
It reuses those105 inputs in 70 fresh processes, alternates variant order per
case/rep pair on the same CPU, and does not pool its timings with the primary.

Use the exact GNU Parallel Spack `x86_64_v3` path in launchers; it is a platform
path, not a run-version label. Workers use exclusive one-CPU srun steps.
Project run registry: `.slurm/runs/20261026-hdm106.json`, with submission IDs,
source/environment hashes, failed attempts and stage outcomes. The compact
reporting/profile artifacts retain all reporting rows, failures, source
snapshots, provenance and the recreation recipe. They omit generated matrices
and serialized full fitted objects that remain on Discovery. No routine
cleanup, refitting for transfer, or whole-run archive is needed.
