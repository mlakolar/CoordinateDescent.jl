# Structured degrees of freedom: paired Discovery study

This harness compares the public path APIs and their degrees-of-freedom work
against master `295fbf5420d6384a1ca261023367ee86d0a40b8c`. It does not change
solver tolerances, the lambda grid, `df_tol`, initialization or preprocessing.
The independent diagnostic reference is the original dense formula
`rank(X * null(D[inactive,:]))`. Existing R golden paths are checked separately.

Use the slurm-workflow skill and the project run registry. All fitting,
profiling and numerical aggregation run inside Slurm, with one Julia, BLAS,
OpenMP and MKL thread per worker. The launcher requests one EPYC 7513 node;
seven workers alternate baseline/candidate order by replication. Each method
gets seven independent datasets and three warmed timing rounds per dataset.
The seed-zero pilot is separate from those evaluation datasets.

The eight complete-path cases include fused regression at p=40/100/300,
graph regression, trend regression at p=40/100, duplicate-column regression
and a wide design. Path lengths are 20 to 50. A p=1000 graph diagnostic
measures one DF evaluation including optional path-workspace setup; it is not
a public p=1000 path benchmark. Report that setup cost even when it loses.
CPU/allocation profiles are outside timers. `@timed.bytes` is cumulative
allocation; fresh-worker RSS includes imports, compilation, independent checks
and profiles, and is not optimizer workspace or a per-fit memory bound.

## Recreate

Create a new immutable run directory, with the scripts in this directory at
its root and source trees named `baseline/`, `candidate/`, `ProximalBase.jl/`,
`ProximalOPT.jl/` and `KernelSmoothing.jl/`. Retain Project.toml, src, ext when
present, and the package tests. Pin the baseline above and the candidate
commit reported with the results. Pin dependencies to:

- ProximalBase b6c949f3dd26cad09252e1c2c9e49810eb69403e
- ProximalSolvers 531ff55fcf8a5ce2c6997c144d4f57e85cf79f3e
- KernelSmoothing 1feff286d0020764d2c1b03d733ab316cf948ead

The original run used `/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue99`,
Julia 1.10.12 and 1.13.2 under `/home1/mkolar/juliaPkg`, and a task-owned depot
layered over `/home1/mkolar/juliaPkg/depot`. Update machine-specific paths in
the shell scripts deliberately. Run `setup.jl` under Slurm once per Julia
version; it creates isolated baseline/candidate environments and retains
resolved manifests. Then run the structured and golden suites on both
versions, followed by the seed-zero pilot, before submitting
`sbatch --parsable measure.sbatch`. Run the full package suites separately.

`validate.jl` requires identical coefficients, objectives, iterations and DF
for every returned baseline/candidate path point. `summarize.py` requires the
complete timing grid and converged fits, then reports within-dataset medians,
paired candidate/baseline ratios, ranges and percentile-bootstrap intervals
over seven independent datasets. Repeated rounds are not independent inputs.
Retain failed attempts and incomplete CSVs; do not summarize them as completed
coverage or hide different numerical answers inside a speed ratio.
