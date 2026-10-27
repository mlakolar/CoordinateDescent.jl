# hdm U1 fixtures

These freeze the unchanged Julia implementation at
`d6174f5c148ed7e08a4732c915aa691ecb83cb8a`, before issues #118/#119 change
the estimators. Do not refresh the Julia baseline to accommodate those changes.

## Regeneration

From the repository root, with the master test environment built by
`julia scripts/test_master.jl --setup-only` and an R library `$RLIB` holding
hdm 0.3.2 (`R CMD INSTALL -l $RLIB hdm_0.3.2.tar.gz`, from the CRAN archive
with the SHA256 below) next to sandwich 3.1.1, run in order:

```sh
julia --startup-file=no --project=.master-test-env/v1.13 benchmark/hdm/golden_data.jl
R_LIBS=$RLIB Rscript benchmark/hdm/golden.R
julia --startup-file=no --project=.master-test-env/v1.13 benchmark/hdm/baseline.jl
```

Each takes a few seconds with one BLAS thread. The focused tests:

```sh
julia --startup-file=no --project=.master-test-env/v1.13 -e 'using Test; include("test/hdm.jl"); include("test/hdm_golden.jl")'
```

Versions: Julia 1.13.3, R 4.5.1 (2025-06-13), hdm 0.3.2, sandwich 3.1.1.
The CRAN hdm archive SHA256 is
`3354b22752f0aa0c0d9c8c12b38810202b6380f2587e539be11d9a358412fe6a`.
All numeric CSV values use `%.17g`. Two complete regenerations produced
identical SHA256 hashes for all 31 CSV files.

## Inputs and schema

Each directory has headerless `X.csv` (observations by controls), `y.csv`, and
`d.csv` (one column each). `forced/forced.csv` contains 1-based indices `[40,43]`;
its other inputs are byte-identical to `hetero`. These controls are absent from
both nuisance supports, for both post choices, in both packages.

Equations, draw order, and Xoshiro seeds are in `benchmark/hdm/golden_data.jl`:
hetero/forced 11801, highdim 11802, nearthreshold 11803, empty 12138. Empty has
centered noise responses and selects nothing, even with post=false. Centering
avoids hdm's empty-fit quirk: its coefficient vector has zero intercept while
its separate intercept field stores the sample mean.

`reference-r.csv` and `baseline-julia.csv` use `fit,field,index,value` long CSV
rows. Indices number entries within each field, starting at 1. An empty
selection has exactly one sentinel row with index=0,value=0; it is not control 0.

- `reference-r.csv`: `rlasso_{y,d}_{true,false}` fits store hdm's `index`, full
  `coefficients` (intercept then p slopes), separate `intercept`, `residuals`,
  `loadings`, `lambda0`, and `iter`. Lambda0 is native R scale: divide by 2n
  to compare to Julia. Raw hdm `iter` is one more than Julia's number of fits
  for these nonempty cases, because R increments after its last fit. Both
  return 1 when the first fit is empty.
- `effect_{doubleselection,partialling}` in that file store `alpha`, `se`, `t`,
  `pval`, selected `index`, `epsilon`, and `v`, with post=TRUE. Forced I3 is a
  logical vector and applies only to double selection. Double-selection
  epsilon already includes hdm's sqrt(n/(n-s-1)) scaling. Partialling epsilon
  is the unscaled final regression residual and v is dr.
- `final_{doubleselection,partialling}` store `coefficient`, classical `vcov`,
  sandwich variances `HC0` through `HC3`, and `hatvalues`. Partialling also
  stores hdm's nuisance `yr` and `dr`. The final regressions include an
  intercept: y on d and selected controls, or yr on dr. These are reference
  inputs for later units; U1 only checks that they load with the right sizes.
- `baseline-julia.csv`: nuisance fields `selected`, `intercept`, p slope
  `coefficients`, `residuals`, `loadings`, `lambda0`, `iterations`, and `sigma`
  (sample residual standard deviation). Effects store `estimate`, `se`, `t`,
  `pvalue`, two-entry `ci`, and `selected`. The header of `baseline.jl` gives
  the seed formula and call order. Each call immediately follows Random.seed!.
- `metadata.csv`: R/package versions, the supplied archive hash, and generating
  script names. No timestamps or machine-specific paths are stored.

## Frozen comparison tolerances

Julia baselines use the elementwise bound
`abs(actual - expected) <= 1e-12 * max(1, abs(expected))`, allowing one-ulp
cross-platform differences at the nearthreshold design's 1e6 loading scale.
Other comparisons use absolute error with no relative allowance. Selections
and iteration counts are exact. Parity selections are
exact for all four nuisances and both effects on hetero/highdim/forced/empty.

Post=true coefficients/residuals and effect alpha/se/t/pval use 1e-11 for OLS
and normal-CDF roundoff. Post=false coefficients (including intercept) use
1e-5: hdm shooting uses an absolute L1 step below 1e-5; Julia CD uses a scaled
relative coordinate step of 1e-7, warm-starts, and randomizes coordinates. Both
outer loops use 1e-5 on residual standard deviation. This is a solver precision
budget, not a proved optimizer-error bound. The observed maximum is 1.79e-6.
Residual tolerances propagate that coefficient budget:
`1e-5 * (1 + maximum(sum(abs, X[:, selected]; dims=2)))`.
Normalized lambda0 uses 1e-15. Loadings use the same scale-aware Julia baseline bound.
The test prints per-dataset maximum differences for all compared quantities.

Nearthreshold intentionally violates parity. Its centered orthogonal second
column is scaled by 1e6. Julia selects `[1,2]` and hdm selects `[1]` in every
nuisance fit, for both post settings. Julia's raw second coefficients are
8e-7/6e-7 (post=true y/d) and 6.565e-7/4.867e-7 (post=false). hdm zeros them
at its 1e-6 cutoff. Both effect estimates are consequently 0.5 in Julia and
0.8 in hdm. The tests assert these coordinates, coefficient ranges, and effect
values explicitly; the large residual/loading differences are expected.

The 31 CSV fixtures total 1,564,472 bytes (about 1.49 MiB), excluding this README.
