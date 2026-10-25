# Row-sparse setup and result diagnostics: issue102

This investigation keeps the row-group objective, weights, requested rank,
intercept convention, stopping tolerances and solver choices fixed. The current
candidate reuses design/response sufficient statistics across a path and obtains
FISTA's spectral scale from the already computed design SVD, padded by
1+sqrt(eps(T)). Final rank uses a small factor core with the original
dimension-dependent cutoff and conservative dense fallback; residuals reuse
the actual thin factors when their multiplication count is smaller. Full
coefficients and the original support/DF definitions remain available. The
automatic setup/diagnostic reuse is separate from choosing CD versus FISTA.

All numerical work runs under Discovery Slurm with slurm-workflow and the
owning registry `.slurm/runs/20261026-hdm102.json`. Remote run root is
`/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue102`. Baseline295fbf5
and candidate snapshots use the same frozen PB/PS/KS dependencies as issue99,
with separate source directories, Julia environments and layered depots.
Current setup/path and factor-rank checks passed on Julia1.10/1.13, including
weighted orthogonal known solutions, zero/rank-deficient cases and Float32/64.

Calibration uses six seed-zero input cases, three warmed public-fit rounds per
CD/FISTA configuration, fractions0.5/0.1/0.02 of one shared objective-specific
lambda-max, plus the medium adaptive-weight comparison at fraction0.1.
`prepare.jl` uses the frozen baseline to save shared RRR coefficients/factors,
adaptive weights and penalty scales. `julia_worker.jl` keeps all production
timing and profiling methods unmodified. Afterward `probe.jl` loads the exact
source equations with state/count capture for deterministic untimed replays.
Replay coefficients and outer counts must match the saved production result;
every timed round's coefficient digest is checked. Actual A/V factors and
weights, inner sweeps/steps and KKT-call counts are then retained. This replay
instrumentation does not run inside any reported production timer.

`check_results.py` independently checks predictor KKT and response-subspace
stationarity<=1e-6, orthogonality, reconstruction, rank feasibility, objective,
residuals, intercept, support and DF. It records before/after objective and
coefficient differences; different stationary points must remain separate
from matched-objective timing comparisons. Stationarity is local, not a
certificate of a global optimum. Profiles and fresh-process RSS are separate
from fit timers; whole-worker memory including replay is distinguished.

The first setup-only pilot passed76 certificates. The factor-diagnostic pilot
also passed76, with target many-response/wide-FISTA gains but small or reversed
controls preserved. These are one-input calibration observations, not evidence
for adoption or a default solver change. No evaluation inputs have been generated.

Before readiness, freeze source/protocol and evaluate20 fresh shared inputs per
six-design/three-penalty group plus20 medium adaptive inputs, with three rotated
warmed rounds, paired ratios/ranges/uncertainty and independent certificates.
Compare complete natural public fits, common-initialization CD/FISTA kernels
and R rrpack srrr separately. Preserve R's native initialization and report
its differences from a controlled shared A/V experiment. Recheck all actual
factors, source/CPU/BLAS/thread metadata and every target dataset's profiles
and baseline/peak RSS. Measure setup/initialization/updates/diagnostics as
separate components without presenting cached kernels as additive public cost.
Add path performance and retain path/CV, weighted FISTA and near-cutoff tests.
The full20-input/R/common-initialization campaign is not yet implemented.
Retain current block_solver=:cd and all statistical defaults until the evidence
supports a justified recommendation. Never merge or enable auto-merge.
