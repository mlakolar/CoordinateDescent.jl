# Fixed-rank residual construction: issue101

Compare current master with the thin-factor residual candidate on 20 fresh
independent inputs in each of six designs: small (300,50,10), medium
(300,50,100), many responses (300,50,1000), wide (200,400,100), condition
number1e4 (500,100,100), and rank40 (300,80,100). Working rank is3; complete
rank paths contain0:3. Input generation uses distinct saved seeds and hashes,
with the seed-zero pilot excluded from evaluation. The multiplication policy
is frozen before evaluation; no outcomes are used to retune it.

The baseline/candidate complete APIs include centering, decompositions and
returned coefficients, intercept, residuals, objective, rank, support and DF.
Julia coefficient-only controls and SciPy coefficient-only/full-output
adapters separate solver work from required metadata. R uses rrpack's QR/SVD
interface, with a reusable native1:3 rank path and the exact rank-zero result;
its native output contract differs. The adapter reconstructs each path coefficient
from rrpack's documented `coef.ls`, `A`, and `Spath`; `coef(z)` returns only the
IC-selected coefficient. Do not compare an invalid faster normal-
equation shortcut or demand identical nonunique R coefficients on wide inputs.

Independent validation checks all output fields, rank feasibility, closed-form
optimal loss and minimum-norm representatives where promised. Every native
before/after coefficient pair is compared directly. Keep failures and adverse
or insignificant effects. Kernel timings alone cannot justify adoption.

Use slurm-workflow and the project run registry. The original remote directory
is `/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue101/study-v1`.
Stage these files there. `environment.sh` pins already resolved Julia1.13.2
baseline/candidate environments and existing Python/R installations; source
archives and manifests live in the owning run directories. Update absolute
paths deliberately for another installation. Baseline source295fbf5 and
candidate9dd42f8 use the same frozen dependency snapshots from issue99.

Submit `sbatch --parsable pilot.sbatch`, validate and fix the harness on only
the six seed-zero cases, then freeze it and submit `main.sbatch`. Main uses
GNU Parallel with exclusive1CPU/3500MiB srun workers on one EPYC7513 allocation
(8CPUs/32GiB). Three warmed rounds rotate method order. Python/R/Julia/BLAS
thread counts are fixed at one. All fitting, component profiling, validation
and numerical aggregation happen within Slurm. Do not overwrite an input or
use an ambiguous submission timeout as permission to submit a duplicate.

Public timings and coefficient-only controls are separate. Cached-input
component measurements are not additive public-cost decompositions. Every
many-response/wide input has separate CPU profiles; Julia allocation profiles
are outside timers. A fresh process handles each input/implementation, with
post-warm current/peak RSS, peak after timing and final peak recorded. These
include imports/JIT/data/persistence, and final peaks also include separate
component/profile work; they are not solver workspace. Cumulative Julia
allocation is a separate quantity.

Expected main coverage:120 shared inputs,5040 timing rows,4200 independent
output-certificate rows,1200 native before/after coefficient pairs,440 CPU
reports and80 allocation reports. `summary.json` retains paired ratios/ranges
and 10,000-resample intervals over the20 independent input medians; three
timing rounds are not three independent inputs. Retain final reporting inputs,
relevant profiles, failures, source/environment references and a recreation
recipe. Raw coefficient/residual arrays stay with the run for independent
validation; a routine whole-repository archive is unnecessary. Do not merge.

Completed evidence and the conservative default recommendation are in
[the Discovery report](results/discovery-20261026/RESULTS.md). Run
`poststudy.py` under Slurm after `main.sbatch` to reproduce the saved-profile
counts, source/CPU/thread checks and final interpretation without refitting.
The benchmark policy was frozen at `cc02719` before evaluation; the reporting
script reads the saved results and changes no fitting settings.

The R path adapter follows rrpack's documented stored path fields in its
[primary source](https://github.com/cran/rrpack/blob/master/R/rrr.R).
The installed `rrr` body is preserved with the pilot, alongside the original
failed extraction and corrected recovery. See the evidence index for retained
records, failures, exact manifests and raw-array locations.
