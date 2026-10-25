# Low-rank regression: Discovery validation and performance

The [solver-option follow-up](optimization/results/discovery-solvers-20261025/README.md)
measures public PANOC and block FISTA and cheaper fixed-rank diagnostics,
with five independent inputs and fresh before/after baselines.

The subsequent [Python/Julia/R optimization comparison](optimization/results/discovery-20261025/README.md)
uses five independent inputs on six designs, including wide, ill-conditioned,
and rank-deficient cases. It compares the same fixed objectives, including
generic proximal algorithms, and retains accuracy failures and profiler data.
It is separate from the package-only baseline below.

The current `bench.jl` and `oracle.jl` use the renamed HighDimensionalModels
module. Frozen optimization harnesses and recorded environments retain their
historical CoordinateDescent name; use their pinned recreation recipes.

Run timing and profiling on Discovery. The Slurm launcher requests four CPUs,
12 GB, and one hour; it uses GNU Parallel and exclusive single-CPU `srun`
steps, with one Julia/BLAS thread per independent replication.

`bench.jl` covers q = 10, 100, 1000 at n = 300, p = 50, true rank 3 and ten
active predictors. Five independent seeds each warm up the five methods,
then time three rounds in rotating order: fixed-rank, coefficient nuclear
penalty, a five-level nuclear path, row-sparse RRR, and sparse-SVD regression
(eight candidate penalties per component). Each replication also saves a
separate sampling profile per method/response size, using ten calls after
warm-up. Compilation and profiles are excluded from timing. Raw CSVs report
seconds, allocated bytes, rank and convergence; they retain every round.

```sh
srun --account=mkolar_1314 --partition=main --cpus-per-task=1 --mem=4G --time=00:15:00 \
  julia scripts/instantiate.jl benchmark/lowrank
export JULIA=/path/to/julia
sbatch --parsable --comment=YOUR_UNIQUE_RUN_ID benchmark/lowrank/discovery.sbatch
```

The setup script resolves this checkout and sibling master dependencies on
all supported Julia versions, including Julia 1.10.
Pin dependency revisions and save the resolved manifest for a measured run.
Run all Julia preparation, tests, oracles and compilation inside Slurm
allocations as well; local editing and orchestration do not require Julia.

`oracle.jl generate DIR` creates shared deterministic inputs.
`Rscript oracle.R DIR` requires rrpack and writes independent R fits.
`oracle.jl compare DIR` checks them. RRR includes exact Stein degrees of freedom and a wide design (comparing
fitted values there: rrpack returns basic QR coefficients, this solver the
minimum-norm coefficients); row-sparse RRR checks three penalties with unit and
adaptive weights on two identifiable designs. Its R initialization explicitly
uses `U %*% diag(d)`, aligning both methods at the same rank-constrained fit.
The rrpack penalty is `n * λ`. RSSVD compares the whole selected coefficient
matrix at ten candidate penalties, including zero. rrpack 0.1-14 swaps BIC
and BICP labels inside `rssvd.eea`; the harness requests its BICP label and
asserts that the returned coefficients equal selection by the actual BIC
column. The package uses the documented criterion names and formulas.
Singular-vector signs
are immaterial to the comparisons.

The coefficient nuclear penalty has independent orthonormal closed-form and
subgradient-certificate tests in `test/lowrank.jl`; rrpack's fitted-value
adaptive nuclear penalty solves a different objective.

## Validated Discovery baseline

Run `20261025T0910Z-cd81` on 25 October 2026 used Julia 1.13.2, an AMD EPYC
7513 core per worker, and one Julia/BLAS thread. Median milliseconds over
five independent data replications (each the median of three timing rounds):

| q | fixed rank | nuclear | nuclear path, 5 levels | row-sparse | sparse SVD, 8 levels |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.58 | 1.97 | 5.62 | 1.70 | 5.97 |
| 100 | 2.46 | 21.0 | 82.3 | 2.93 | 26.6 |
| 1000 | 8.43 | 113 | 447 | 12.3 | 192 |

All 225 timed fits reported convergence. Each method/size has five separate
profiles, 75 in total. At q = 1000 the nuclear proximal SVD accounts for
about 63% of self samples; sparse SVD spends more time in allocation (28%)
and repeated SVDs (16%). No performance measurements ran on the laptop.

[Full results](results/discovery-20261025/README.md) include replication ranges,
allocated bytes, raw timing CSVs, all profiles, validation counts, pinned
source/dependency references, and scheduler outcomes. Both Julia versions
passed package validation, and the final API/golden checks passed 153 tests
per version.
