# Sparse reduced-rank setup: final evidence and defaults

The frozen candidate improves complete public fits and paths by reusing
sufficient statistics and the existing design SVD, and by computing result
rank/residual diagnostics from thin factors with conservative dense fallbacks.
Coefficients, objective, weights, support/DF conventions, intercept and stopping
rules are preserved. Recommend the bounded optimization with the existing CD
solver default; this is an optimization study, not a statistical risk comparison.

Main13449014 completed in23m32s:20 fresh shared independent inputs per design,
three warmed rounds,120 inputs,960 processes,13080 timing rows and21600 cached
component rows. All3800 individual-fit and4480 path-level actual-factor
certificates passed. Python checked predictor KKT/response stationarity<=1e-6,
orthogonality, rank, reconstruction, weighted objective and Julia returned
residual/intercept/support/DF. All1520 before/after individual-fit pairs match
objectives within1e-6; maximum coefficient difference1.55e-14. Stationarity is
local, not a global-optimum certificate. Max main KKT9.96e-10 and response4.19e-15.

## Public-fit improvements and tradeoffs

Ratios are paired candidate/baseline medians over20 inputs at0.1 lambda-max;
intervals are95% bootstrap intervals resampling inputs (10000 frozen-seed draws).
Three rounds first produce one within-input median. Intervals are descriptive
per configuration and do not adjust for simultaneous comparisons.

| Design/solver | Baseline ms | Candidate ms | Paired ratio [interval] |
|---|---:|---:|---|
| Many responses,CD |13.481|11.497|0.854 [0.840,0.861]|
| Many responses,FISTA |12.992|11.510|0.860 [0.836,0.912]|
| Wide,CD |34.923|33.513|0.962 [0.938,1.003]|
| Wide,FISTA |41.077|34.826|0.854 [0.844,0.864]|
| Ill-conditioned,FISTA |17.989|16.786|0.946 [0.932,0.954]|
| Rank-deficient,FISTA |4.631|3.943|0.842 [0.828,0.864]|

The wide default-CD single-fit gain is uncertain. Small CD at the middle penalty
regresses: ratio1.016 [1.001,1.047],1.719→1.739ms. Preserve all configurations
and their ranges/regressions in comparison.csv/poststudy.json; no universal
single-fit improvement is claimed.

Complete eight-level public paths improve across every retained design/solver
and the adaptive supplement. Many-response paired ratios are0.705 for both CD
and FISTA; wide0.857/0.744; medium adaptive0.815/0.750; ill0.956/0.876;
rank-deficient0.851/0.657. All path intervals are below1, although the ill-CD
upper endpoint0.9995 is close to1. Path and cached component timings are separate
experiments. Components are not additive decompositions of public timings.

## Fair R comparator and retained adverse stationary points

The primary frozen campaign used R outer1e-10/inner1e-12, which could over-solve.
Follow-up13449649 reran only240 R processes on the identical20 inputs with
historical outer1e-8/inner1e-9. All760 new actual-factor certificates passed the
unchanged1e-6 accuracy requirements (max KKT5.97e-9); Julia fits were reused.
Use this **globally chosen faster tested R setting** for the alternative ranking.
The stricter setting remains sensitivity evidence, not the primary speed claim.
See r-tolerance/ for its tables, metadata, profiles and reporting inputs.

At the middle wide penalty, R public natural initialization takes60.235ms versus
33.513ms for candidate CD and34.826ms for candidate FISTA. Candidate/R paired
ratios are0.550 [0.537,0.579] and0.580 [0.560,0.611]. The fresh pre-change Julia
baseline was already faster than this R route: do not attribute the historical
R-gap reversal entirely to this PR. CPU/stack/stopping settings differ from the
historical Xeon study, and the public output/IC contracts differ by package.

At the strong wide penalty, R reaches a different stationary objective on1/20
inputs: R objective468.636 versus Julia420.430 (symmetric relative difference0.1029). That outcome remains visible and
is excluded from matched-objective timing ranks. Public natural initialization
and controlled common-A/V fits are reported separately. Shared weights and
lambda_R=n*lambda_Julia preserve the objective; natural initializers may differ.
This study supports no universal package or programming-language ranking.

## Defaults, profiles and memory

Keep block_solver=:cd. On the candidate, FISTA/CD paired ratios for wide strong
and middle penalties are1.166 [1.064,1.264] and1.065 [1.001,1.104]. In contrast,
FISTA is useful on ill-conditioned middle/weak inputs (0.461/0.219) and
rank-deficient middle/weak inputs (0.734/0.401). Medium adaptive0.986
[0.972,1.001] gives no clear solver-wide improvement. Document explicit FISTA
use for those difficult designs rather than adding an unvalidated automatic
condition-number threshold. Retain gamma=2, all tolerances, iteration limits
and other statistical defaults; this experiment does not optimize statistical
regularization or rank choice.

Poststudy13449864 verified960 source/CPU/thread records,1680 native CPU profiles
and360 R CPU/memory profiles (plus360 new R profiles in the tolerance study).
All20 target wide/many-response/ill-conditioned inputs have separate profiles
and baseline/peak records. Native allocation reports and all timing-byte counts
are retained. Outer/inner/KKT counts are in iteration-counts.csv; R inner counts
are unavailable from its native outer interface. Counts/factors come from
untimed deterministic replay, after all uninstrumented timers/profiles, with
returned coefficients and outer iterations checked against production fits.

No uniform peak-memory reduction is claimed. Wide native timing-peak medians
are475066368/475779072 bytes before/after; whole-worker medians680155136/680638464.
The candidate wide maximum whole-worker peak747405312 exceeds baseline717496320.
Whole-process RSS includes imports, JIT, retained data and profiles, not isolated
optimizer workspace. Full per-input memory and allocations remain available.

The final production source passed574 targeted checks and20 additional known
rank-two/Float32/Float64 fixtures on each Julia1.10/1.13, including weighted
paths, zero/rank-deficient and near-cutoff cases. Hosted platform/LTS/release/
examples/docs CI passed on unchanged production source. Raw arrays/factors,
source and environment snapshots remain on Discovery. Tables/JSON and the
compressed reporting-input/profile delivery are adjacent. No merge is authorized.
