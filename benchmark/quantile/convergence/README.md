# Convolution stopping follow-up

Read [RESULTS.md](RESULTS.md) for the short decision. This follows the
[original API comparison](../api/RESULTS.md), preserving that cohort and its
failed accuracy checks. All fitting, tests, profiling and numerical aggregation
ran under Slurm on Discovery. Local work only prepared sources, transferred
evidence and formatted reports.

## Design

The primary cohort uses three fresh R seeds (1301–1303), four shapes and two
shuffled blocks per input. Shapes are small (1,000 × 100), heteroscedastic t3 tail
(same size, τ=0.1), tall (10,000 × 100), and correlated wide (1,000 × 1,000).
The saved wide input 1202 is an additional diagnostic anchor, never pooled with
fresh inputs. All methods receive identical centered, sample-standardized
designs, responses, bandwidth 0.2 and 32 decreasing penalties, from 1.05 times
the exact null score maximum to 0.01 times that maximum. The convolution kernel
is Epanechnikov, called `parabolic` by conquer. There is an unpenalized intercept.

Five methods give 130 primary observations: explicit legacy `gaptol=0`, new API
targets 2e−6 and 2e−7, and conquer coefficient-step settings 1e−6 and 1e−7.
Legacy mode restores the earlier objective-change/relative-inner stopping and
curvature floor, using the same model code and Julia version as the new modes.

The primary conquer 1e−7 wide setting missed the common gate on two of three
fresh inputs. A separate control reruns **both** API 2e−7 and conquer 1e−8 on
the three fresh wide inputs and old anchor: 16 observations. Its timings are
reported separately. Both methods pass the common gate in all 16 runs. Some
conquer runs still fail the secondary 2e−7 raw-gap gate; the primary common gate
is explicitly 2e−6, and achieved diagnostics are saved.

Qualification requires every penalty to satisfy the continuous proximal KKT
residual ≤2e−6 and |raw primal−dual objective gap| ≤2e−6(1+|primal objective|).
The secondary gate substitutes 2e−7 in both checks. Raw scores are never
replaced by projected/scaled certificate scores. Native backend flags are
saved separately. Every attempted path is retained, including incomplete or
inaccurate returns. Partial-group survivor timings are not used for rankings.

## Measurement and uncertainty

All workers use one CPU and one Julia/BLAS/OpenMP thread. GNU Parallel runs two
exclusive `srun` workers within 2-CPU/16-GiB allocations. Relevant dispatch is
warmed on a small input; garbage collection occurs outside timed fitting.
Julia times the complete public API path, including diagnostics. R times native
path fitting plus raw-score recovery, followed by separately timed identical
Julia diagnostics; those times are added. R startup, file transport and data
loading are excluded. This is an explicit R wrapper cost, not one contiguous
cross-language API call. Compilation is excluded from comparisons.

Means first average the two repetitions within each input, then average the
three input means. Saved SDs describe input-to-input variation. Approximate
95% intervals use Student t with two degrees of freedom. Paired contrasts use
the same three input means. There are only three inputs per shape; repetition
does not create six independent datasets. Anchors have no across-input interval.
Two CPU sample profiles were taken **after** timed measurements; full flat
profiles and top-self summaries are retained, without attributing all inclusive
samples to the displayed line.

## Sources and environment

The measured model core is HighDimensionalModels
`7799fff10c467c258e33968e7d81b082ff3ab99b`; the harness is
`0e54d167eaa08fbe96ce5e8157a49270ba0aa3d1`. The measured ProximalSolvers core
is `216ba6f3dc3b3cbd9dc975614847f665644fea01`. Later solver commit
`6268fda7491f9e87644fa238f76ab9b3e2408efb` adds tests only.
ProximalBase is `0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a`; KernelFunctions is
`5d93dff52873d85c60849dd56dd6cadb0391dd59`. Old-backend compatibility uses
ProximalSolvers `73a1c8e927d8917e4f16483271b053fcdb25b488`.

Julia is 1.10.12 throughout this follow-up, R is 4.5.3, and conquer is 1.3.3.
The R session and Julia Project/Manifest files are retained. Earlier API report
timings used Julia 1.13.2 and are not pooled with these measurements. The input
generator is retained verbatim; its generation branch is the one used here.
Its unused older competitor-fitting branch is historical source only.

## Recreate on Discovery

The run root is `/home1/mkolar/juliaPkg/runs/20261025-cd78-api/gap-main`.
The [run record](results/discovery-2026-10-25/run-record.json) identifies jobs,
source versions and retained files. Stage the recorded sources and this harness
in a new run root. Recreate the relative source paths in the saved Manifest, or
develop those exact source checkouts together and resolve on Julia 1.10.12.
`generator.R` in the evidence folder recreates fresh inputs from the seeds.
The original wide-1202 files are retained remotely; they can also be regenerated
from that seed. The primary worker generates fresh files under Slurm as needed.

Submit `sbatch run.sbatch` for the primary cohort and regressions. Copy the
retained `tight-reference` scripts to a sibling control directory, with `inputs`
pointing to the primary inputs and `../ltsenv` to the measured environment;
submit `sbatch --dependency=afterok:PRIMARY_JOB tight.sbatch` there. The control
also runs API checks against `compat-env`, which points to the old solver.
The standalone 40 edge-case assertions are in the recorded later solver test
commit and run with its `test/convolution_gap.jl` under Slurm. Use the supplied
`summarize.py` and `tight-summary.py` inside the allocations, not locally.

Evidence includes per-path and per-penalty CSVs, execution order, worker exit
records, logs, profiles, summaries, failed canaries and accounting. Large
regenerable designs and native R coefficient arrays remain on Discovery.
One research-dual anchor is retained locally: little-endian Float64, Julia
column-major, raw scores and certificate each 1,000 × 32; coefficients 1,001 × 32
with intercept first. They are the primary certified wide-1202 first repetition.
