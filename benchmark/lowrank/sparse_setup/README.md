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

## Frozen independent-input evaluation

`full_main.sbatch` uses 20 new shared inputs per design (replications 1–20;
replication 0 is calibration only), three warmed rounds, rotated process and
configuration order, EPYC 7513, one Julia/BLAS/OpenMP thread per worker. The
production baseline/candidate and dependencies remain fixed. Generation,
preparation, fitting, profiling, certificates and aggregation all run in Slurm.

`full_worker.jl` measures natural public CD/FISTA separately from
`common_kernel.jl`, which starts from identical frozen A/V factors and weights.
The latter excludes centering and RRR/coefficient-SVD initialization but includes
sufficient statistics, production block methods/stopping and dense result
construction. Actual factors are recovered after all timers/profiles by exact
source replay; every timed coefficient digest agrees across rounds and replay.
`r_worker.R` measures rrpack's unmodified public natural initialization separately
from its compiled solver with the same shared A/V factors. Both use explicit
shared weights and lambda_R=n*lambda_Julia. R's relative coefficient-change
stopping rules differ from Julia's KKT rules: R uses outer 1e-10 and inner 1e-12,
with limits 1000/10000, and must pass the same independent 1e-6 stationarity
certificate. Those tighter values were calibrated only on replication 0.
R's timed actual factors are saved directly. Public R outer counts/flags come
from an untimed traced replay; inner counts are unavailable from that interface.

`path_components.jl` measures complete eight-level public paths from 0.5 to 0.02
lambda-max, including both block solvers and the medium adaptive supplement.
Every path level has an actual-factor certificate. Its separately timed cached
components expose centering, SVD/projection/reconstruction, Gram/cross-product,
eigensolver, predictor/Procrustes/coefficient updates, KKT and result work. These
are cost probes, **not additive decompositions** of public timings.

Expected full coverage: 120 shared inputs, 960 fresh worker processes, 13,080
timing rows, 21,600 component rows, 3,800 individual-fit and 4,480 path-level
certificates. All 20 target many-response/wide/ill-conditioned datasets have
separate profiles and warmed/current/peak RSS records. RSS includes the runtime,
imports, JIT, prior fits and profiling, rather than only optimizer workspace.
Estimated retained raw output is several GB and tens of thousands of files;
reporting inputs/profiles are delivered separately from raw matrices/factors.

`full_check.py` and `path_check.py` check actual weighted objectives, predictor
KKT and response tangent residuals <=1e-6, V orthogonality, B reconstruction and
rank. Julia residuals/intercepts/support/rank/DF and native convergence are checked
as well. Stationarity is local, not a certificate of global optimality.
Objective equivalence is relative 1e-6, declared before evaluation. Unmatched
stationary points, timeouts, flags, regressions and incomplete coverage remain
visible. `full_summary.py` resamples independent inputs for paired median-ratio
95% bootstrap intervals; timing rounds are summarized within an input first.
A matched-objective R timing also has a different native result/IC contract and
must not be presented as a universal language ranking.

Adapter canary 13448961 retained 152 passed Julia certificates but failed before
R startup (`source /etc/profile` returned nonzero under errexit). Recovery
13448988 reused those Julia outputs, changed startup to modules.sh, and passed
all 190 individual-fit plus 224 path-level certificates on the same six
calibration inputs. No fresh evaluation inputs were used to tune adapters.
