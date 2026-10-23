# Benchmarks

Run the scripts from the root of the repository, in the environment of this
folder, which loads CoordinateDescent from this checkout:

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/locpoly_bench.jl    # about a minute
julia --project=benchmark benchmark/cd_bench.jl         # a few minutes
julia --project=benchmark benchmark/order_anderson_bench.jl    # about 4 minutes
julia --project=benchmark benchmark/scaledlasso_bench.jl       # about 10 minutes
julia --project=benchmark benchmark/gram_bench.jl              # gram = true against false
```

The checkout is found through `[sources]` in `Project.toml`, which Julia 1.11
and later read, as is ProximalSolvers until its version 0.1.1 is registered. Julia 1.10
ignores `[sources]`: develop the checkout and add ProximalSolvers by URL in that
environment first.

`compare/` compares this version with version 0.6.0 (see below).

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

Times the lasso solvers on one problem with n = 3000, p = 5000 and 100 nonzero
coefficients:

- `scaledLasso!` from its default start and from a given noise level (with
  `@time`);
- the lasso at 0.05 λmax and 0.01 λmax, from zero and along a path of 50
  penalty levels from λmax, with unit and with standardized loadings. λmax is
  the smallest penalty with a zero solution, for each kind of loadings. Each
  solve reports its time, passes and number of nonzeros.

`CD_BENCH_N` and `CD_BENCH_P` set a smaller problem, as a quick check that the
script runs.

On Discovery (an EPYC 7513 core, Julia 1.13.2, one BLAS thread; run
20261023T2353Z-cd-gram-bench), the script takes about 15 seconds:

| solve | time (s) | passes | nonzeros |
|---|---:|---:|---:|
| `scaledLasso!`, noise level started by screening (with compilation) | 1.95 | | |
| `scaledLasso!`, noise level started at a given value | 0.03 | | |
| 0.05 λmax, unit loadings, from zero / along 50 levels | 0.05 / 0.25 | 41 / 506 | 454 |
| 0.05 λmax, standardized loadings, from zero / along 50 levels | 0.04 / 0.25 | 38 / 490 | 455 |
| 0.01 λmax, unit loadings, from zero / along 50 levels | 0.80 / 2.21 | 285 / 1232 | 2058 |
| 0.01 λmax, standardized loadings, from zero / along 50 levels | 0.74 / 2.20 | 264 / 1231 | 2058 |

The path costs more than the direct start here because every intermediate
level is solved to the full tolerance (ProximalSolvers issue #19).

Until version 0.7, the script solved at λ = 0.001, about 3e-4 λmax. There the
solutions have about 2,980 nonzero coefficients, close to interpolating the
3,000 observations. Each of the first three solves took 44,000 to 53,000
passes, and the script took 19 minutes measuring a regime that users rarely
fit.

## gram_bench.jl

Compares `lasso` and `LassoPath` on the Gram matrix (`gram = true`) and on the
residuals (`gram = false`), to calibrate `gram = :auto`. The cases are:

- n ∈ {1000, 5000, 20000} and p ∈ {50, 200, 1000};
- independent and AR(1) columns;
- the lasso with an intercept at 0.1 and 0.02 λmax, and a 100-level path.

Each line gives the shortest time of three interleaved runs of each form, the
passes, the largest difference of the coefficients, and what `:auto` picks.
`GRAM_BENCH_QUICK=1` runs small sizes, as a check that the script runs.

## order_anderson_bench.jl

The tables of this section are from version 0.6, whose own coordinate descent
also had `order = :active`. Version 0.7 runs ProximalSolvers's coordinate descent,
which takes the same passes with and without momentum (see compare/ below) but
has no `order = :active`; the script no longer times that order.

Compares the orders of the coordinates, Anderson acceleration (issue #9) and
momentum (issue #20) on the lasso `‖y - Xβ‖² / (2n) + λ‖β‖₁`, solved from β = 0
with the default stopping rule (`optTol = 1e-7`):

- `order = :cyclic`, and `:random` and `:active` without momentum (the default
  before version 0.6 was `:random` without momentum);
- `order = :cyclic` with `anderson = 5`;
- `order = :random` with `momentum = true`, which is the default since 0.6 for
  these built-in losses.

The designs, defined in `common.jl`, are those of the lasso benchmark of
ProximalSolvers.jl, with three more: two more correlated ones and one with three
common factors. They have Gaussian columns scaled to ‖Xⱼ‖² = n, s coefficients
±1 and a signal-to-noise ratio of 4.

| design | n | p | s | correlation between columns |
|---|---:|---:|---:|---|
| A | 500 | 2,500 | 50 | none |
| B | 500 | 2,500 | 50 | AR(1), 0.8 between neighbors |
| B' | 500 | 2,500 | 50 | AR(1), 0.95 between neighbors |
| C | 5,000 | 500 | 50 | AR(1), 0.5 between neighbors |
| D | 200 | 10,000 | 20 | none |
| E | 500 | 2,500 | 50 | 0.5 between every pair (a common factor) |
| E' | 500 | 2,500 | 50 | 0.9 between every pair |
| F | 500 | 2,500 | 50 | three common factors with Gaussian loadings |

Other processes load the machine unevenly, so every case runs the solvers in
turn. Each solver runs until it has half a second of runs, and at least 3 runs
(5 for the random orders). For the random orders the tables give the median
passes and time; for the deterministic ones, the shortest time. The results
below are from an Apple M3 Pro with Julia 1.13.2 and one BLAS thread. Every
solution is as accurate as the cyclic one, except that the cyclic order stops
at `maxIter = 100 000` for E' at 0.01 λmax.

### One λ: milliseconds (passes)

| design | λ/λmax | cyclic | random | active | Anderson | momentum |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.1 | 8.1 (179) | 12.1 (256) | 11.4 (249) | **5.0** (97) | 8.0 (150) |
| A | 0.02 | 203 (3,120) | 301 (4,390) | 320 (4,667) | 89.5 (1,231) | **82.2** (1,152) |
| A | 0.01 | 767 (10,965) | 1,114 (14,748) | 1,018 (13,931) | 263 (3,391) | **202** (2,634) |
| B | 0.1 | 11.4 (333) | 16.6 (472) | 16.2 (471) | **7.1** (172) | 8.5 (211) |
| B | 0.02 | 201 (3,169) | 267 (3,965) | 286 (4,263) | 85.7 (1,240) | **77.3** (1,113) |
| B | 0.01 | 553 (8,164) | 718 (10,267) | 764 (11,033) | 194 (2,579) | **167** (2,272) |
| B' | 0.1 | 10.4 (568) | 20.1 (1,067) | 18.1 (941) | **5.3** (232) | 8.2 (360) |
| B' | 0.02 | 109 (2,364) | 176 (3,624) | 177 (3,689) | **37.1** (717) | 47.8 (875) |
| B' | 0.01 | 443 (7,541) | 640 (9,913) | 674 (10,750) | 130 (2,002) | **120** (1,862) |
| C | 0.1 | 2.5 (16) | 4.2 (35) | 4.1 (34) | **2.4** (14) | 4.0 (29) |
| C | 0.02 | 15.3 (54) | 19.0 (67) | 18.5 (65) | **14.5** (49) | 17.7 (58) |
| C | 0.01 | 23.9 (58) | 32.4 (77) | 31.9 (77) | **22.0** (51) | 29.3 (66) |
| D | 0.1 | 11.2 (835) | 17.7 (998) | 13.2 (937) | **7.1** (363) | 8.8 (368) |
| D | 0.02 | 117 (8,855) | 168 (10,946) | 149 (10,308) | 40.3 (2,511) | **36.9** (2,046) |
| D | 0.01 | 495 (35,655) | 501 (28,447) | 477 (31,709) | **73.5** (4,537) | 80.2 (3,956) |
| E | 0.1 | 6.4 (949) | 1.7 (81) | **1.5** (80) | 2.9 (248) | 1.7 (68) |
| E | 0.02 | 1,415 (36,940) | 33.3 (731) | 35.0 (767) | 450 (10,510) | **17.5** (300) |
| E | 0.01 | 3,348 (66,494) | 131 (2,240) | 147 (2,512) | 1,436 (24,962) | **46.2** (725) |
| E' | 0.1 | 35.8 (9,581) | 3.5 (586) | 3.3 (626) | 9.7 (1,278) | **2.5** (271) |
| E' | 0.02 | 212 (25,164) | 5.6 (525) | 5.4 (522) | 35.3 (3,676) | **3.4** (224) |
| E' | 0.01 | 2,233 (100,000) | 43.6 (1,442) | 43.4 (1,457) | 1,013 (36,694) | **15.0** (428) |
| F | 0.1 | 7.3 (765) | 3.3 (211) | 3.6 (258) | 4.2 (287) | **2.8** (127) |
| F | 0.02 | 404 (9,920) | 50.3 (1,095) | 50.1 (1,098) | 138 (2,991) | **22.7** (399) |
| F | 0.01 | 1,316 (24,067) | 209 (3,505) | 222 (3,590) | 479 (7,915) | **62.1** (961) |

### Path of 50 λ values, λmax to λmax/100 (`LassoPath`): milliseconds

| design | cyclic | random | active | Anderson | momentum |
|---|---:|---:|---:|---:|---:|
| A | 1,223 | 1,623 | 1,566 | **327** | 342 |
| B | 1,165 | 1,700 | 1,662 | **343** | 374 |
| D | 527 | 659 | 693 | 112 | **102** |
| E | 7,559 | 287 | 308 | 2,634 | **107** |
| F | 3,409 | 498 | 528 | 1,354 | **144** |

### Memory K at 0.02 λmax: milliseconds (passes)

| design | K = 3 | K = 5 | K = 10 |
|---|---:|---:|---:|
| A | 90.4 (1,217) | 89.7 (1,231) | 90.1 (1,305) |
| B | 79.2 (1,126) | 84.2 (1,240) | 87.0 (1,334) |
| D | 42.7 (2,552) | 40.5 (2,511) | 45.3 (2,972) |

### What the numbers say

- **Momentum** (`order = :random, momentum = true`, the default since 0.6)
  - It is never slower than the random order without momentum; the two tie on E at 0.1 λmax. At 0.02 λmax and below it is 1.6–6.2 times faster, except when n ≫ p (C), where it is 7–11% faster.
  - On paths it is 2.7–6.5 times faster than without momentum.
  - Under a common factor (E, E' and F) it is the fastest option at 0.02 λmax and below. There it is 1.6–3.4 times faster than without momentum and 6–68 times faster than Anderson acceleration; it also computes the paths of E and F 9–25 times faster than Anderson acceleration.
  - Without a common factor, at 0.02 λmax and below, it is up to 30% faster than Anderson acceleration on A, B and D (D at 0.01 λmax is the exception, 9% slower), and on paths within 9% of it.
  - It is slower than Anderson acceleration when few passes are needed: by 20–67% at 0.1 λmax, by 29% on B' at 0.02 λmax, and by 22–33% when n ≫ p (C). In those cases the cyclic order needs fewer passes than any random one.
  - With `order = :active`, which visits the coordinates of the full passes in the order 1, 2, …, momentum was 2–23% faster than with `order = :random` at 0.1 λmax. Elsewhere it was within 11%, in a separate comparison on A–F and on paths.
  - It is not offered with the cyclic order: on the equicorrelated design that combination stalled 3–18% above the minimum after 100,000 passes. The cyclic order's slowest modes there are complex, which momentum does not handle.
- **Anderson acceleration** (`order = :cyclic, anderson = 5`)
  - Without a common factor (A–D), it is the fastest or close to it in every case. At 0.02 λmax and below on A, B, B' and D, it is 2.3–6.7 times faster than the cyclic order it accelerates.
  - It keeps the weakness of the cyclic order under a common factor, where it is 1.7–6 times faster than the cyclic order but far slower than the random orders.
- **Memory K:** K = 3, 5 and 10 are within 10% of each other.
- **Extrapolations are often rejected while the active set settles, and accepted in runs once it has.** Stopping after a few rejections in a row, as planned in issue #9, prevented any acceleration: in a solve on A, the first three attempts were all rejected.
- **`order = :active`** without momentum is within -27% to +17% of `order = :random`. It never shows the cyclic order's blow-up.
- **Timing under load.** A rerun of this script with other processes loading the machine gave times up to 50% longer. The ratios between the deterministic solvers (minimum of the runs) and the random ones (median) also moved in favor of the deterministic ones. The tables above are from a run on a quiet machine.

## compare/

Compares this version, which runs ProximalSolvers's coordinate descent, with
version 0.6.0, on the cases of the momentum benchmark of 0.6:
- the lasso at 0.05 λmax with an intercept and standardized columns;
- the square-root, scaled and feasible lasso at theory-driven penalties;
- `LassoPath` with 100 λ, an intercept and standardized columns;
- `CDQuadraticLoss(X'X/n, -X'y/n)` at 0.02 λmax;
- each on designs A, E and F, and `locpolyl1` on the model of
  `examples/07_varying_coefficient.jl`.

The old side, in `old/`, is CoordinateDescent 0.6.0 from the registry on
ProximalBase 0.4.1. It runs with its default options, which use momentum, and
with `momentum = false`. The new side, in `new/`, is this checkout on
ProximalBase 0.5 and ProximalSolvers 0.1.0.

```
julia --project=benchmark/compare/old -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/compare/new -e 'using Pkg; Pkg.instantiate()'
benchmark/compare/ab.sh 3
```

`ab.sh` runs the two sides in turn, each in a fresh process, and every run
starts from the same seed. `compare.jl` takes the fastest median time of each
case over the rounds, since load on the machine only adds time. The table is
from three rounds on USC CARC Discovery on 23 October 2026: one core of an AMD
EPYC 7513, Julia 1.13.2, one BLAS thread (Slurm job 13326700), with this
version at `0623e11` on ProximalSolvers `c247846` (then named ProximalSolvers).

| case | 0.6 (ms) | 0.6, momentum = false (ms) | 0.7 (ms) | 0.7 / 0.6 | 0.7 / 0.6 without momentum | passes, 0.6 / 0.7 | largest difference |
|---|---:|---:|---:|---:|---:|---|---:|
| A CDQuadraticLoss | 245.9 | 1050.4 | 234.4 | 0.95 | 0.22 | 1224 / 1224 | 0e+00 |
| A LassoPath | 497.8 | 2218.3 | 498.6 | 1.00 | 0.22 | 7387 / 7387 | 4e-14 |
| A feasibleLasso! | 1.5 | 1.5 | 1.5 | 1.01 | 1.01 | 2 / 2 | 0e+00 |
| A lasso | 25.9 | 54.6 | 25.1 | 0.97 | 0.46 | 397 / 397 | 5e-15 |
| A scaledLasso | 2.0 | 2.2 | 1.9 | 0.96 | 0.88 | 27 / 27 | 7e-16 |
| A sqrtLasso | 5.8 | 5.5 | 5.8 | 1.01 | 1.06 | 11 / 11 | 0e+00 |
| E CDQuadraticLoss | 49.0 | 123.1 | 48.6 | 0.99 | 0.39 | 328 / 328 | 0e+00 |
| E LassoPath | 174.6 | 467.9 | 168.8 | 0.97 | 0.36 | 4548 / 4548 | 2e-14 |
| E feasibleLasso! | 4.8 | 4.5 | 4.5 | 0.94 | 1.01 | 154 / 154 | 2e-15 |
| E lasso | 4.1 | 5.1 | 4.1 | 1.00 | 0.79 | 108 / 108 | 3e-15 |
| E scaledLasso | 1.9 | 1.6 | 1.8 | 0.95 | 1.13 | 84 / 84 | 0e+00 |
| E sqrtLasso | 8.5 | 9.7 | 8.4 | 0.99 | 0.87 | 19 / 19 | 0e+00 |
| F CDQuadraticLoss | 61.9 | 169.2 | 61.7 | 1.00 | 0.36 | 427 / 427 | 0e+00 |
| F LassoPath | 227.9 | 715.7 | 216.3 | 0.95 | 0.30 | 6428 / 6428 | 4e-14 |
| F feasibleLasso! | 4.7 | 4.8 | 4.5 | 0.95 | 0.93 | 282 / 282 | 3e-15 |
| F lasso | 5.4 | 9.1 | 5.3 | 0.96 | 0.58 | 161 / 161 | 4e-15 |
| F scaledLasso | 2.2 | 2.9 | 2.1 | 0.96 | 0.73 | 146 / 146 | 1e-15 |
| F sqrtLasso | 12.1 | 18.5 | 12.0 | 0.99 | 0.65 | 32 / 32 | 0e+00 |
| locpolyl1 | 0.8 | 0.7 | 0.8 | 1.03 | 1.11 | 0 / 0 | 9e-16 |

- **The coordinate descent moved without a change, momentum included.**
  - Against the default of 0.6.0, every case takes the same passes, and the
    solutions agree to 4e-14.
  - The times are 0.94–1.03 of 0.6.0's.
- **What momentum gives**, from the column without it: `LassoPath` takes
  0.22–0.36 of the time, `CDQuadraticLoss` 0.22–0.39 and `lasso` 0.46–0.79. The
  scaled, feasible and square-root lasso at theory-driven penalties take
  0.65–1.13, as in 0.6.
- A first run, before ProximalSolvers took ProximalBase from the registry, gave
  0.95–1.05 with the same passes.

## scaledlasso_bench.jl

Compares the scaled lasso of version 0.6.0, which alternated lasso fits with
updates of σ̂ until the relative change of σ̂ was below `optTol` (1e-2 by
default, and 1e-6), with the square-root lasso solved in one run of coordinate
descent: from the data (`scaledLasso`) and from sufficient statistics
(`CDSqrtQuadraticLoss`). Part 1 fits regressions with independent or AR(1)
columns, ten nonzero coefficients, unit noise, `λ = √(2 log p / n)` and the
column standard deviations as loadings. Part 2 is the precision matrix
estimator of Sun and Zhang (2013): the scaled lasso of every variable on the
others, for AR(1), band and Erdős–Rényi precision matrices, from copies of the
other columns of the data, or from their covariance matrix `S`, shared by all
the columns (built with `check = false`, with an infinite weight on the
response). Seconds on one CPU of USC CARC Discovery (AMD EPYC 7513, Julia
1.13.2; run `20261023T1216Z-cd-scaledlasso`, commit `e6a0bd2`), and in
parentheses the largest difference of the coefficients from a solution at
`optTol = 1e-12`, relative to its largest coefficient. "A, b, c" is the time to
form the sufficient statistics.

| n | p | ρ | 0.6.0, optTol = 1e-2 | 0.6.0, optTol = 1e-6 | scaledLasso | from A, b, c | A, b, c | passes 0.6.0 / new |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 200 | 0.0 | 0.000544 (3e-04) | 0.000904 (5e-08) | 0.000186 (6e-10) | 4.93e-05 (2e-09) | 0.0017 | 35 / 10 |
| 1000 | 200 | 0.5 | 0.000462 (3e-04) | 0.000715 (1e-07) | 0.000207 (4e-07) | 5.76e-05 (1e-07) | 0.00161 | 85 / 20 |
| 1000 | 200 | 0.9 | 0.000801 (3e-05) | 0.001 (2e-06) | 0.000657 (5e-06) | 0.000141 (3e-06) | 0.00165 | 195 / 169 |
| 5000 | 500 | 0.0 | 0.00488 (7e-05) | 0.00701 (4e-08) | 0.00239 (5e-10) | 0.000263 (5e-11) | 0.0294 | 23 / 9 |
| 5000 | 500 | 0.5 | 0.0061 (2e-05) | 0.00774 (3e-07) | 0.00358 (8e-08) | 0.000286 (1e-07) | 0.0285 | 53 / 32 |
| 5000 | 500 | 0.9 | 0.0075 (3e-04) | 0.00908 (4e-06) | 0.00772 (3e-06) | 0.000393 (3e-06) | 0.0287 | 184 / 152 |
| 1000 | 1000 | 0.0 | 0.00204 (8e-04) | 0.00373 (7e-08) | 0.000748 (6e-10) | 0.000998 (3e-10) | 0.0221 | 34 / 11 |
| 1000 | 1000 | 0.5 | 0.00173 (7e-04) | 0.0031 (1e-07) | 0.000786 (8e-08) | 0.00101 (3e-07) | 0.022 | 77 / 40 |
| 1000 | 1000 | 0.9 | 0.00225 (1e-04) | 0.00295 (4e-06) | 0.00126 (6e-06) | 0.0012 (4e-06) | 0.022 | 264 / 244 |
| 200 | 1000 | 0.0 | 0.00114 (2e-02) | 0.00405 (2e-06) | 0.000213 (4e-08) | 0.000995 (1e-07) | 0.00543 | 75 / 17 |
| 200 | 1000 | 0.5 | 0.000916 (2e-03) | 0.00197 (3e-07) | 0.00029 (5e-08) | 0.00103 (1e-07) | 0.00536 | 148 / 31 |
| 200 | 1000 | 0.9 | 0.000923 (1e-04) | 0.0014 (3e-06) | 0.000369 (3e-06) | 0.00122 (3e-06) | 0.00541 | 255 / 173 |
| 200 | 5000 | 0.0 | 0.00527 (3e-02) | 0.0164 (5e-06) | 0.00132 (7e-08) | 0.0329 (3e-08) | 0.172 | 72 / 26 |
| 200 | 5000 | 0.5 | 0.00428 (2e-03) | 0.00978 (3e-07) | 0.00116 (1e-07) | 0.0329 (1e-07) | 0.173 | 99 / 22 |
| 200 | 5000 | 0.9 | 0.00414 (4e-04) | 0.00618 (5e-06) | 0.00159 (4e-06) | 0.0337 (4e-06) | 0.171 | 266 / 181 |

| problem | 0.6.0, optTol = 1e-2 | 0.6.0, optTol = 1e-6 | scaledLasso | from S |
|---|---:|---:|---:|---:|
| AR(1), p = 200, n = 100 | 0.026 (6e-03) | 0.066 (6e-07) | 0.0125 (1e-07) | 0.002 (1e-07) |
| band, p = 200, n = 100 | 0.0261 (3e-02) | 0.0508 (3e-06) | 0.00864 (5e-07) | 0.00255 (6e-07) |
| Erdős–Rényi, p = 200, n = 100 | 0.0171 (8e-03) | 0.0338 (1e-06) | 0.00834 (2e-07) | 0.00231 (2e-07) |
| AR(1), p = 200, n = 400 | 0.0355 (4e-04) | 0.0842 (2e-07) | 0.0422 (2e-07) | 0.00225 (1e-07) |
| band, p = 200, n = 400 | 0.0409 (4e-03) | 0.0787 (5e-07) | 0.0297 (3e-07) | 0.00458 (3e-07) |
| Erdős–Rényi, p = 200, n = 400 | 0.0379 (6e-03) | 0.0633 (6e-07) | 0.0268 (2e-07) | 0.00306 (3e-07) |
| AR(1), p = 500, n = 250 | 0.2 (3e-03) | 0.295 (4e-07) | 0.138 (1e-07) | 0.0134 (1e-07) |
| band, p = 500, n = 250 | 0.247 (9e-03) | 0.463 (1e-06) | 0.122 (3e-07) | 0.024 (4e-07) |
| Erdős–Rényi, p = 500, n = 250 | 0.182 (7e-03) | 0.318 (8e-07) | 0.139 (2e-07) | 0.015 (2e-07) |
| AR(1), p = 500, n = 1000 | 0.504 (1e-03) | 0.68 (1e-07) | 0.366 (1e-07) | 0.0175 (1e-07) |
| band, p = 500, n = 1000 | 0.57 (2e-03) | 0.902 (4e-07) | 0.435 (3e-07) | 0.0242 (3e-07) |
| Erdős–Rényi, p = 500, n = 1000 | 0.532 (4e-03) | 0.81 (7e-07) | 0.366 (9e-07) | 0.0158 (7e-07) |
| AR(1), p = 1000, n = 500 | 1.14 (2e-03) | 1.71 (3e-07) | 0.796 (1e-07) | 0.051 (1e-07) |
| band, p = 1000, n = 500 | 1.17 (4e-03) | 2.23 (7e-07) | 0.894 (3e-07) | 0.0884 (3e-07) |
| Erdős–Rényi, p = 1000, n = 500 | 1.07 (7e-03) | 1.82 (7e-07) | 0.751 (4e-07) | 0.0528 (5e-07) |
| AR(1), p = 1000, n = 2000 | 3.38 (9e-04) | 4.92 (1e-07) | 3.08 (1e-07) | 0.0494 (1e-07) |
| band, p = 1000, n = 2000 | 4.26 (6e-04) | 6.13 (4e-07) | 3.55 (3e-07) | 0.08 (3e-07) |
| Erdős–Rényi, p = 1000, n = 2000 | 3.8 (3e-03) | 5.36 (4e-07) | 3.08 (3e-07) | 0.0524 (6e-07) |

- **Faster, and at the solution.** In all but one case the new `scaledLasso`
  was 1.2–5.4 times faster than the default of 0.6.0, whose coefficients were
  off by 2e-5 to 3e-2; the exception (n = 5000, p = 500, ρ = 0.9) was 3% slower,
  with an error of 3e-6 against 3e-4. At the accuracy of the new version (0.6.0
  with `optTol = 1e-6`) it was 1.2–19 times faster.
- **Sufficient statistics pay off when n > p.** From `A`, `b` and `c` the fit
  was 3.6–20 times faster than from the data when n > p, not counting forming
  them, which costs `O(np²)` (29 ms for n = 5000, p = 500, as much as 4–12 fits
  from the data). When p ≥ n it was slower, up to 25 times for p = 5000,
  n = 200: a coordinate update costs `O(p)` instead of `O(n)`.
- **The precision matrix estimator.** From the shared `S`, all the columns took
  2–90 ms, 3.4–62 times faster than the new `scaledLasso` on copies of the data
  and 7–73 times faster than the default of 0.6.0, with coefficients within
  7e-7 of the solution. Without `check = false`, checking `S` for every column
  costs `O(p²)` each, which dominated for p ≥ 1000.
