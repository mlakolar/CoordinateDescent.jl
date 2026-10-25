# Low-rank regression: Discovery validation and performance

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
julia --project=benchmark/lowrank -e 'using Pkg; Pkg.instantiate()'
export JULIA=/path/to/julia
sbatch --parsable --comment=YOUR_UNIQUE_RUN_ID benchmark/lowrank/discovery.sbatch
```

Julia 1.10 ignores `[sources]`; develop this checkout and add the unregistered
ProximalSolvers and ProximalBase dependencies as in the repository CI setup.
Pin dependency revisions and save the resolved manifest for a measured run.

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

Results of the implementation run are recorded below after validation.
