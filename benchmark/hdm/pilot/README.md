# Post-selection coverage pilot

This is a small **descriptive pilot**, not validation of coverage or a basis for
changing defaults. It compares partialling and double selection at fixed
nuisance fits, and native post-false tuning separately. The study runs
on a cluster under Slurm. Local work is limited to short smoke checks.

## Frozen designs

`cases.csv` is the design authority; its row number is the case index. There
are 500 independent datasets per case. The Gaussian generator follows
`9e9e587:benchmark/inference/overnight-defaults/pilot.jl:7–26` from PR #86,
with the specified target-dependent heteroskedastic error added here.

| Case | n | p | rho | s | signal | support | error | design | alpha | cy |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|--:|--:|
| moderate | 200 | 500 | .3 | 5 | .5 | fixed | normal | standard | — | — |
| correlated | 100 | 500 | .9 | 15 | .5 | fixed | normal | standard | — | — |
| benign | 500 | 100 | .3 | 5 | .5 | fixed | normal | standard | — | — |
| hetero-target | 300 | 500 | .3 | 5 | .5 | fixed | hetero_target | standard | — | — |
| heavy | 300 | 500 | .3 | 5 | .5 | random | t3 | standard | — | — |
| weak-signal | 200 | 500 | .3 | 5 | .2 | random | normal | standard | — | — |
| belloni-a0 | 200 | 250 | .5 | — | — | — | normal | belloni | 0 | .75 |
| belloni-a05 | 200 | 250 | .5 | — | — | — | normal | belloni | .5 | .75 |

Standard designs draw `X=randn(rng,n,p)`, apply the AR recursion
`X[:,j]=rho*X[:,j-1]+sqrt(1-rho^2)*X[:,j]`, center columns and divide each
by `sqrt(mean(abs2,column))`. Support is `1:s` or `randperm(rng,p)[1:s]`.
Coefficients on that support are `signal .* (-1).^(0:s-1)`; the intercept
is .2. Targets are the first five active indices in support order, followed
by the first five zero-coefficient indices. Thus the correlated case still
has five active target roles despite having 15 active coefficients.

Errors are drawn after X and any support permutation. Normal errors are
`randn(rng,n)`. The t3 convention is
`randn(rng,n) ./ sqrt.(vec(sum(abs2,randn(rng,n,3);dims=2)))`, with variance
one: do **not** divide the denominator sum by three. The hetero_target error
is `sqrt.((1 .+ X[:,1].^2 .+ X[:,6].^2)./3) .* randn(rng,n)`.
Then `y=b0 .+ X*b .+ err`.

Belloni designs do not standardize. Draw `Z=randn(rng,n,p-1)` and apply the
same AR recursion. Set `vy[1:5]=vy[11:15]=1 ./ (1:5)` and
`vd[1:10]=1 ./ (1:10)`, with other entries zero. Draw
`d=vd[1] .+ Z*vd[2:end] .+ randn(rng,n)`, then set `X=hcat(d,Z)`,
`b=vcat(alpha,cy.*vy[2:end])`, `b0=cy*vy[1]` and draw fresh normal outcome
errors. Targets are `[1; p-4:p]`, six roles. All six are null for belloni-a0;
belloni-a05 has one active and five null roles.

For target position `slot`, set `j=targets[slot]`, `d=X[:,j]` and
`C=X[:,setdiff(1:p,j)]`; truth is the original-unit coefficient `b[j]`.
Individual active/null role summaries follow **target order**, not a fixed
column number across randomized supports.

Data RNG: `Xoshiro(1_400_000_000 + 1_000_000*case + rep)`.
Solver RNG: `Xoshiro(1_500_000_000 + 10_000_000*case + 10_000*rep + 10*slot + stream)`.
Here `rep=1:500`, `slot=1:10` (1:6 for Belloni), and streams 1–4 are y-post,
d-post, y-native, d-native. These blocks are disjoint from PR #86. No global
RNG is used. Timing-only convenience calls use streams 1 and 2, respectively,
for partialling and double selection; they do not alter the study fits.

## Arms and records

Each role computes four `rlasso` fits once, using unchanged defaults and
private RNGs, then reuses them through `double_selection`'s precomputed API:

| Arm | Method | Nuisance post | refit |
|:--|:--|:--|:--|
| A | partialling | true | true |
| B | partialling | same fits as A | false |
| C | doubleselection | same fits as A | nothing |
| D1 | partialling | false, separate native fits | nothing |
| D2 | doubleselection | same fits as D1 | nothing |

A–B isolates refitting at the stored penalty and loadings. D1/D2 also change
native tuning (including default c from 1.1 to .5). Every arm attempts all six
`vcov` choices independently: hdm, homoskedastic, HC0, HC1, HC2, HC3. An HC2/3
exception cannot discard another variance. No forced controls are used.

`case-I-reps-FIRST-LAST.csv` contains one row per case/rep/slot/arm:

- Identity: `case` (1-based row index), `case_name`, `rep`, `slot`, `target`,
  `truth`, `role` (active/null), `arm`.
- `fit_status`: ok if at least one final call returned; otherwise the first
  exception type from the nuisance pair or final calls. Nuisance exceptions
  are caught separately; an unavailable pair produces all its arm rows.
- `estimate`; six `se_<vcov>` and six `err_<vcov>` fields. Errors store the
  exception type, not a message; failed variances are NaN. Non-convergence is
  not an exception and is retained. Interrupts propagate.
- `s_y`, `s_d`, `s_union`: nuisance support sizes and union size; `rank`,
  `dof_residual`, `target_variation`, `max_leverage` from a successful final call.
- `converged_y`, `solver_converged_y`, `converged_d`, `solver_converged_d`.
  Unknown flags are blank, unknown numerical diagnostics are NaN.
- `nuisance_seconds`: time for both fits in that pair, repeated identically
  for A/B/C or D1/D2. `final_seconds`: total elapsed time for the six final
  calls, including errors, zero when the pair was unavailable.

Numerical CSV fields use 17 significant digits. Each worker warms the same
code paths on a discarded case-zero dataset (n=60, p=12). Warm-up is excluded
from all reported study timings. Throughput after warm-up includes generating
datasets and the extra timing calls; writing files is excluded. First-use
specialization outside the warm-up can still affect short timings.

For reps 1–20, `case-I-reps-FIRST-LAST-timing.csv` records case/name/rep/slot/
target/method/seed/status/seconds for warm, post-true convenience calls to both
methods. These calls include nuisance fitting and one default `vcov=:hdm`
final stage. Their cost is not directly comparable to six final stages.

There are 50 rows per standard dataset and 30 per Belloni dataset: **180,000
arm rows** and 2,880 convenience timing rows in the full plan. Pair timing
summaries deduplicate the repeated A/B/C and D1/D2 pair times. Only small CSV
records are retained, not designs or fit objects (roughly 70 MB total, depending
on numeric formatting and error frequency).

## Setup and execution on a Slurm cluster

Use a git worktree at the reviewed harness commit, with `ProximalBase.jl` and
`ProximalSolvers.jl` as siblings of this repository. Julia 1.11+ is needed for
`[sources]`; the recorded run used Julia 1.13.3. Record the worktree commit, sibling commits,
Julia version, run tag, submission command, job IDs and output locations in the
existing run registry. Do not commit a Manifest or change package defaults.

From `benchmark/hdm/pilot`, resolve the path sources using cached transitive
dependencies. This command must not fetch packages; if dependencies are absent,
the orchestrator must arrange the cache separately. Precompilation and fitting
belong inside Slurm jobs.

```sh
cd WORKTREE/benchmark/hdm/pilot
export JULIA=julia   # Julia 1.11 or later
export JULIA_PKG_SERVER="" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0
"$JULIA" --startup-file=no --project=. -e 'using Pkg; Pkg.offline(true); Pkg.instantiate(;allow_autoprecomp=false)'
"$JULIA" --startup-file=no --project=. make-tasks.jl
mkdir -p logs
```

The canary is case 1, reps 1–2. Run it in a separate output directory because
its rep range overlaps the first main task. RUNTAG is passed through the
environment and Slurm-visible metadata; it cannot expand in an SBATCH directive.

```sh
export RUNTAG=hdm-pilot-canary-YYYYMMDD
export TASKS="$PWD/canary.txt" OUTDIR="$PWD/outputs/$RUNTAG"
sbatch --parsable --job-name="$RUNTAG" --comment="$RUNTAG" \
  --account=ACCOUNT --cpus-per-task=2 --mem=MEM --time=LIMIT run.sbatch
```

Inspect the outcome, 100 arm rows, 40 timing rows, failure types and warm
throughput before the main run. Pass the account, CPUs, memory and time limit at
submission; the script runs one single-threaded worker per two allocated Slurm
CPUs and gives each worker step an explicit share of the job memory (`--mem`).
Canary and main execution together must remain inside the approved budget;
the script does not authorize another allocation or an extension of that budget.

```sh
export RUNTAG=hdm-pilot-main-YYYYMMDD
export TASKS="$PWD/tasks.txt" OUTDIR="$PWD/outputs/$RUNTAG"
sbatch --parsable --job-name="$RUNTAG" --comment="$RUNTAG" \
  --account=ACCOUNT --cpus-per-task=CPUS --mem=MEM --time=LIMIT run.sbatch
# Run analysis in an authorized CPU allocation after workers finish:
"$JULIA" --startup-file=no --project=. analyze.jl "$OUTDIR" "$PWD/results/$RUNTAG"
```

`tasks.txt` has 80 tasks, chunks of 50 reps. Restart the same task list/output
directory to skip existing main CSVs. Workers publish the timing companion
first and main CSV last, each by same-directory temporary file and atomic move.
Incomplete temporary files are never read. A killed chunk is repeated from its
start; completed chunks survive. Nonzero worker exits propagate through xargs.
Do not merge canary and main directories; analysis rejects duplicate keys or
overlapping completed shards, malformed shards and missing timing companions.

For the prescribed local smoke test, the already-built test environment can
be used without changing it if offline pilot-environment resolution is
unavailable. From the repository root:

```sh
export JULIA_DEPOT_PATH="$PWD/.codex-orchestrator/julia-depot:$HOME/.julia:"
export JULIA_PKG_SERVER="" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
SMOKE_DIR=$(mktemp -d "$PWD/.codex-orchestrator/hdm-pilot-smoke-XXXXXX")
julia --startup-file=no --project=.master-test-env/v1.13 benchmark/hdm/pilot/worker.jl 3 1 1 "$SMOKE_DIR/records"
julia --startup-file=no --project=.master-test-env/v1.13 benchmark/hdm/pilot/analyze.jl "$SMOKE_DIR/records" "$SMOKE_DIR/results"
```

Expect 50 arm rows, 20 timing rows and RESULTS.md. Keep each local command under
30 seconds; do not infer coverage from one dataset.

## Analysis conventions

`summary.csv` is long format: case/name/arm/vcov/group/metric/value/lower/upper/
mcse/planned/valid/datasets/uncertainty. Groups are active_1… or null_1… by
target order, pooled_active, pooled_null, or all for diagnostics and timing.
All **planned denominators retain 500 datasets**, even for a smoke test or an
incomplete run. Missing chunks count as MissingRecord, not as valid fits.
Belloni-a0 has no planned active roles, so active summaries are undefined.

A fit is valid for a variance iff arm status is ok, its estimate is finite and
its se is finite and positive. Coverage is `abs(estimate-truth)<=Z95*se`;
rejection is `abs(estimate)>Z95*se`, with Z95=1.959963984540054. Single roles use
Wilson 95% intervals. Pooled rates first average over valid roles within each
dataset and then over datasets with at least one valid role. MCSE is the SD
of these dataset means divided by sqrt(number of such datasets); normal 95%
bounds are clipped to [0,1]. With fewer than two datasets MCSE/bounds are NaN.
Extra fit_weighted rows retain the flat valid-fit rates when availability
varies by dataset. There are no naive binomial intervals for pooled roles.

Planned_bounds rows give deterministic bounds: all unavailable roles counted
as misses versus as hits, with the fixed planned denominator. They apply to
coverage and rejection, not to the conditional rate population. Bias, empirical
SD of estimates, SD of estimation errors, mean se and mean/median interval
length are retained. Pooled estimate SD can mix different truths; error_sd
removes that mixture. Pooled converged-only coverage requires all four nuisance
convergence flags true and uses the same dataset uncertainty calculation.

Paired A−B and A−C comparisons use only roles valid for **both** arms at that
vcov within a dataset. They average within-dataset coverage and mean-length
differences, then report the mean, SD/sqrt(R) MCSE and normal bounds across
paired datasets. Paired valid-role and dataset counts remain visible.

Support sizes, rank, dof, target variation and leverage summaries include all
finite observed diagnostics, not only converged fits. Timing summaries include
failed-call elapsed time; final_seconds sums all six variances, convenience
calls use only the default variance. Arm and per-vcov error stages can overlap.

`RESULTS.md` reports pooled rates, lengths, convergence sensitivity, paired
differences, failures and timings. Its **DESCRIPTIVE** flags are ≥99% valid for
each nonempty pooled group; active-coverage lower 95% bound ≥92.5%; and
null-rejection upper 95% bound ≤7.5%. NA means no applicable roles or insufficient
datasets. These are unadjusted screens, not validation, and adverse or missing
results must remain visible.
