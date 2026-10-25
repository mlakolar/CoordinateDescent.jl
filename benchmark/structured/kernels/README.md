# Trend product investigation: issue98

The candidate retains the sparse difference matrix for public use and
factorization. Inside trend ADMM it uses row stencils read from that matrix
for orders1/2, including irregular divided-difference weights. Other orders,
nonlocal/malformed matrices and unsupported matrix types keep sparse products.
The existing banded solve, exact TV prox, objectives, rho and stopping/DF
tolerances are unchanged. Regular stencils store constant coefficients;
irregular stencils store the actual per-row weights. Stencil width is a type
parameter to avoid boxing tuple values during construction.

Run all numerical work under Discovery Slurm with slurm-workflow. The owning
registry is `.slurm/runs/20261026-hdm98.json`, remote root
`/home1/mkolar/juliaPkg/runs/20261026-hdm-overnight/issue98`. Baseline295fbf5
and the candidate use the same pinned dependency sources and Julia1.13.2
environments as issue99; each source snapshot/environment is separate.
Do not replace sources beneath running jobs or reuse another task's SSH.

Targeted release/LTS checks compare both product directions with explicit
matrices, dual identities, nonfinite arithmetic, boundaries, Float32/64,
regular/irregular grids, sparse fallbacks, full structured fits/paths/CV and
R golden controls. Tests passed on the first source snapshot; a constructor
specialization is undergoing the same checks on the next snapshot.

`generate.jl` uses distinct saved seeds per case/replication and refuses to
overwrite inputs. Rep0 is calibration only. `julia_worker.jl` measures three
warmed complete public fits or50-level paths, including setup, output and DF,
then separate CPU/allocation profiles. Input loading, warming and independent
checking are outside timers. One fresh process handles each implementation
and input; memory includes the runtime, inputs, warming and profiling rather
than isolated solver workspace. `check_results.py` independently constructs
sparse divided differences and checks saved coefficients, objective, DF and
convergence, plus before/after agreement.

The first calibration tested small/irregular linear cases,100k linear and
quadratic signals, a nearly tied TV path and a linear path. It exposed an
unwarmed wrapper call in the first round and avoidable constructor allocations;
neither is an adoption result. The next immutable calibration uses the same
saved rep0 inputs, a width-specialized constructor and explicit actual-wrapper
warmup before all three rounds. Original results and failures remain retained.
Do not use calibration inputs as independent evaluation evidence.

Before an adoption recommendation, freeze source/harness, generate at least
three fresh inputs per case, include million-point linear/quadratic signals
and the quadratic50-level path, and check against tighter independent glmgen
references. Require coefficient agreement<=1e-4, relative objective difference
<=1e-6, finite outputs and convergence, preserving failures and actual counts.
Report paired input ratios, three-round medians, profiles, allocations and
fresh-process baseline/peak RSS. Require a useful repeatable public-API gain
without broad regression before enabling the internal default. No tolerance
relaxation or TV-prox replacement is part of this candidate. Do not merge.
