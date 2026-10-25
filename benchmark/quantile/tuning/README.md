# Quantile penalty calibration and check-loss CV

The implementation adds `quantile_lambda(X; ...)` and `CVQuantileLassoPath`.
Inference is tracked separately in [issue #104](https://github.com/mlakolar/HighDimensionalModels.jl/issues/104).
The [user documentation](../../../docs/src/quantile.md) explains the API.

## Validation design

The model core is commit `2fe145ee3fd850fd175cbc78aa94ed80a132faf2`; harness
versions and jobs are recorded with the results. Julia 1.10.12 uses the same
pinned environment and ProximalSolvers core as the preceding convolution
stopping study. R 4.5.3 and quantreg 6.1 run in the existing Discovery container.
All fitting, tests, scientific diagnostics, timing and numerical aggregation
run under Slurm. No local scientific execution is part of this recipe.

The pivotal rule simulates fixed-quantile null scores on the centered design
when an intercept is fitted, divides by the fit's effective penalty loadings
and sample size, and takes a type-7 empirical quantile. The reference implements
that formula independently in R on identical saved uniforms. Public-API tests
also check seeded simulation, dense/sparse agreement, loading changes, shifts,
column rescaling and integer inputs. The fixed-quantile normalization is
documented from [Belloni and Chernozhukov](https://arxiv.org/abs/0904.2931).
The inspected CRAN hdm R sources have no quantile `rq` calibration entry point;
no hdm numerical agreement is claimed. Quantreg's `LassoLambdaHat` has different
loading/probability conventions, so its defaults are not silently equated with
this rule. CV is independently compared with `quantreg::rq.fit.lasso`, passing
`2*n*lambda*effective_loadings` in its augmented-LP normalization.

CV references use 83 observations, 5 shifted/scaled columns, offsets,
unbalanced three-fold labels, three penalties and both standardization settings.
The first fixture uses quantiles .2/.5/.8. Two of its six fold-error comparisons
fail a literal backend-equality check. The complete failed run is retained.
Diagnosis compares native R duals, API duals and training objectives for all
54 fold/penalty pairs. At integer empirical ranks, valid nonunique exact fits
can yield different held-out errors. The separate regular fixture uses
.23/.51/.79, avoiding integer training ranks; all 24 reference assertions pass.
It does not replace the first fixture or erase its adverse evidence.

The API tests include explicit/generated grids, offsets, weighted penalties,
zero effective loadings in calibration, all-fold convergence eligibility,
one-standard-error selection, preserved selected-fit duals, invalid labels,
limits, and dense/sparse and threaded CV. A separate two-thread run uses the
older supported solver to exercise backward compatibility.

## Timing scope

Repeated timings use three Julia-generated inputs (seeds 1501–1503), each
1,000 × 40, and two shuffled repetitions of exact and smoothed CV. Each CV call
fits a 12-penalty full-data path plus five training paths, evaluating held-out
check loss. All levels must qualify. Calibration uses 1,000 simulation draws
on each input, repeated twice. Warm-up and garbage collection precede timing.
Timing includes the public operation and its diagnostics, excluding input
generation and startup. These are operation costs, not new competitor rankings;
see the preceding [solver comparison](../convergence/RESULTS.md) for that.
Means first average repetitions within input, then average the three inputs.

## Recreate on Discovery

Stage the model source and harness at their recorded commits beside `ltsenv`
and the pinned solver checkouts; reuse the preceding stage's Project/Manifest.
Set `QUANTILE_RUN_ROOT` or submit from that directory. `sbatch run.sbatch` runs
the model assertions and regular R reference in two exclusive GNU Parallel
workers (2 CPUs/16 GiB total). `reference.sbatch` reruns only the reference.
`diagnostic.sbatch` recreates the original tied fixture, its native-dual
diagnostics, then the separate regular reference and timing measurements.
It requests 1 CPU/8 GiB for at most five minutes. The saved compatibility
recipe uses the old-solver environment and two Julia threads on two CPUs.

The first launcher failed before R computation because its clean container
cleared `R_LIBS_USER`. The final worker passes it explicitly with `--env`.
All attempts, references, execution outcomes and recreation recipes are retained.
