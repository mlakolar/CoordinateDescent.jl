# Benchmarks

Run the scripts from the root of the repository, in the environment of this
folder, which loads CoordinateDescent from this checkout:

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/locpoly_bench.jl    # about a minute
julia --project=benchmark benchmark/cd_bench.jl         # about 20 minutes
```

The checkout is found through `[sources]` in `Project.toml`, which Julia 1.11
and later read. Julia 1.10 ignores it and looks for this version of
CoordinateDescent in the registry, so `Pkg.instantiate()` fails there. Develop
the checkout instead, which does not change `Project.toml`:

```
julia +lts --project=benchmark -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
```

## locpoly_bench.jl

Compares the local polynomial code with the alternatives it was chosen over,
checks that they give the same result, and reports median times
(BenchmarkTools, one BLAS thread):

1. `locpoly`, least squares by QR on the √w-weighted expanded design X̃, against
   the normal equations, formed from X̃ or accumulated without forming it and
   solved by Cholesky;
2. `_expand_Xt_w_X!`, which accumulates X̃' diag(w) X̃ without forming X̃ (for
   `getStandardError`), against a naive loop and against BLAS on the formed X̃;
3. the weighted lasso at one point: `CDWeightedLSLoss` against
   `CDLeastSquaresLoss` on the √w-scaled data;
4. `locpolyl1`, which updates one loss in place and warm starts along the grid,
   against a `CDLeastSquaresLoss` built at every grid point and solved from
   zero.

On an Apple silicon laptop with Julia 1.13.2 and one BLAS thread (indicative
only; Julia 1.10.12 gives similar times):

| part | method | median | max abs difference |
|---|---|---:|---:|
| 1. `locpoly` at 50 points, n = 1000, 30 coefficients | QR (`locpoly`) | 11.6 ms | reference |
| | normal equations from X̃ | 4.8 ms | 3e-13 |
| | normal equations without X̃, Cholesky | 18.9 ms | 8e-13 |
| 2. X̃' diag(w) X̃, 30 × 30, n = 1000 | `_expand_Xt_w_X!` | 0.35 ms | reference |
| | naive loop | 1.35 ms | 5e-13 |
| | BLAS, forming X̃ included | 0.07 ms | 4e-12 |
| 3. weighted lasso, n = 500, 150 coefficients | `CDWeightedLSLoss` | 1.5 ms | reference |
| | `CDLeastSquaresLoss` on √w-scaled data | 1.4 ms | 2e-8 |
| 4. `locpolyl1` at 20 points, n = 500, 150 coefficients | `locpolyl1` | 7.5 ms | reference |
| | a loss per point, from zero | 9.0 ms | 1e-6 |

The normal equations are faster than QR here and agree with it to rounding
error, but they square the condition number of the local design, which matters
when few points fall in the kernel window or the degree is high. The
differences in parts 3 and 4 come from the stopping rule (`optTol = 1e-8`).

## cd_bench.jl

Times, with `@time`, the lasso solvers on one problem with n = 3000, p = 5000
and 100 nonzero coefficients: `scaledLasso!` from its default start and from a
given noise level, and `coordinateDescent!` at λ = 0.001 with and without warm
start, with unit and with standardized penalty loadings. The first call of each
method includes compilation; the last call repeats the one before it.

On the same laptop with Julia 1.13.2 and the default 6 BLAS threads (indicative
only), the script takes 19 minutes:

| call | time |
|---|---:|
| `scaledLasso!`, noise level started by screening (with compilation) | 1.4 s |
| `scaledLasso!`, noise level started at a given value | 0.06 s |
| `coordinateDescent!` at λ = 0.001 from zero | 175 s |
| `coordinateDescent!` at λ = 0.001 along 50 penalty levels from λmax | 208 s |
| the same two with standardized loadings, 100 levels on the path | 191 s, 274 s |
| the last call again | 269 s |

λ = 0.001 is about 3e-4 λmax. With p > n, the solutions there have about 2,980
nonzero coefficients, close to interpolating the 3,000 observations, and the
first three solves at λ = 0.001 take 44,000 to 53,000 passes to meet the
default stopping rule.
