# Benchmarks

Run the scripts from the root of the repository, in the environment of this
folder, which loads HighDimensionalModels from this checkout:

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/cd_bench.jl         # a few minutes
julia --project=benchmark benchmark/order_anderson_bench.jl    # about 4 minutes
julia --project=benchmark benchmark/scaledlasso_bench.jl       # about 10 minutes
julia --project=benchmark benchmark/gram_bench.jl              # gram = true against false
julia --project=benchmark benchmark/path_setup_bench.jl        # CV and GLM setup (#73)
```

The checkout is found through `[sources]` in `Project.toml`, which Julia 1.11
and later read, as is ProximalSolvers until its version 0.1.1 is registered. Julia 1.10
ignores `[sources]`: develop the checkout and add ProximalSolvers by URL in that
environment first.

`compare/` compares this version with version 0.6.0 (see below).

## varying/

`varying/bench.jl` times the varying-coefficient code before and after it moved
onto the kernels of KernelSmoothing.jl (#38), on the same data and in separate
environments (`varying/old` takes CoordinateDescent at master 1832be7, the last
with `locpoly`; `varying/new` this checkout):

```sh
julia --project=benchmark/varying/old benchmark/varying/bench.jl old old.jls
julia --project=benchmark/varying/new benchmark/varying/bench.jl new new.jls
julia benchmark/varying/compare.jl old.jls new.jls
julia --project=benchmark/varying/old benchmark/varying/lassocv.jl old
julia --project=benchmark/varying/new benchmark/varying/lassocv.jl new
```

The fits use the Epanechnikov kernel with h = 0.15, degree 1 and 50 grid
points. The unpenalized estimates and the leave-one-out errors of the two
versions agree to 1e-15, and the lasso (λ = 0.1, h = 0.2, 20 grid points)
selects the same 40 variables, summed over the grid points. The table gives the
fastest of three rounds on USC CARC Discovery on 24 October 2026: one core of an
AMD EPYC 7513, Julia 1.13.2, one BLAS thread (Slurm job 13379175, run
20261024T1531Z-cd-varying-ab4).

| case | old | new | new/old |
|---|---:|---:|---:|
| fit, n = 500, p = 3 | 1.73 ms | 0.72 ms | 0.42 |
| fit, n = 2000, p = 3 | 6.78 ms | 2.48 ms | 0.37 |
| fit, n = 2000, p = 10 | 25.5 ms | 7.42 ms | 0.29 |
| fit with `se = true`, n = 500, p = 3 | | 8.81 ms | |
| fit with `se = true`, n = 2000, p = 3 | | 108 ms | |
| fit with `se = true`, n = 2000, p = 10 | | 294 ms | |
| leave-one-out CV, three bandwidths, n = 300 | 22.8 ms | 12.7 ms | 0.56 |
| leave-one-out CV, three bandwidths, n = 1000 | 221 ms | 106 ms | 0.48 |
| lasso, n = 500, p = 50 | 2.37 ms | 1.87 ms | 0.79 |
| group lasso, n = 500, p = 50 | | 8.50 ms | |

The standard errors need the fitted values at every observation, a local fit
at each zᵢ, so `se = true` costs 10 to 40 fits. The old `locpoly` computed none.

`lassocv.jl` times the leave-one-out cross-validation of the lasso with n = 100,
p = 60, degree 1 (120 columns in each local fit) and h = 0.2, 0.3, 0.4, where
#38 found the old code slow at a small penalty level. The old
`lvocv_locpolyl1` solves a scaled lasso, so its errors differ; the noise
variance is 0.09. One run each (Slurm job 13379208, run
20261024T1533Z-cd-varying-lassocv):

| λ | old (scaled lasso) | new, `:lasso` | new, `:group` |
|---|---:|---:|---:|
| 0.1 | 0.77 s, errors 0.70, 0.80, 6.93 | 0.02 s, errors 0.80, 0.17, 0.20 | 0.15 s, errors 0.21, 0.19, 0.21 |
| 0.003 | 4.98 s, errors 1.30, 0.90, 1.08 | 0.64 s, errors 0.36, 0.50, 0.63 | 2.47 s, errors 0.33, 0.42, 0.63 |

## cd_bench.jl

Times the lasso solvers on one problem with n = 3000, p = 5000 and 100 nonzero
coefficients:

- `scaled_lasso!` from its default start and from a given noise level (with
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
| `scaled_lasso!`, noise level started by screening (with compilation) | 1.95 | | |
| `scaled_lasso!`, noise level started at a given value | 0.03 | | |
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

- n ∈ {1000, 5000, 20000} and p ∈ {50, 200, 1000}, or the sizes in
  `GRAM_BENCH_NS` and `GRAM_BENCH_PS`;
- independent and AR(1) columns;
- the lasso with an intercept at 0.1 and 0.02 λmax, and a 100-level path.

Each line gives the shortest time of three interleaved runs of each form, the
passes, the largest difference of the coefficients, and what `:auto` picks.
`GRAM_BENCH_QUICK=1` runs small sizes, as a check that the script runs.

On one core of Discovery (AMD EPYC 7513, one BLAS thread, Julia 1.13.2), the
default sizes and then n ∈ {500, 1000, 2000, 4000, 10000} with
p ∈ {1000, 2000} gave these ratios of the time on the Gram matrix to the time
on the residuals for the 100-level paths, over the two designs:

| n \\ p | 50 | 200 | 1,000 | 2,000 |
|---:|---:|---:|---:|---:|
| 500 | | | 1.03–1.06 | 2.1–2.2 |
| 1,000 | 0.12–0.14 | 0.17 | 0.56–0.60 | 1.3–1.4 |
| 2,000 | | | 0.34–0.37 | 0.66–0.93 |
| 4,000 | | | 0.11–0.14 | 0.37–0.38 |
| 5,000 | 0.03–0.04 | 0.05–0.08 | 0.12–0.15 | |
| 10,000 | | | 0.08–0.11 | 0.17–0.25 |
| 20,000 | 0.02–0.03 | 0.04–0.06 | 0.08–0.13 | |

At one λ the Gram matrix was slower in 93 of the 94 cases, by up to 42 times:
forming it costs O(np²), more than the 7 to 210 passes that it makes cheaper.
So `gram = :auto` uses it for paths on a dense X with n ≥ p and p ≤ 2,000
(then A takes at most 32 MB), and never at one λ. The two forms agreed to 7e-5
in the coefficients, except on the paths at n = p = 1000 of the first run
(6e-4), whose 28,000 to 31,000 passes go down to 1e-4 λmax, close to
interpolation.

## order_anderson_bench.jl

The tables of this section are from version 0.6, whose own coordinate descent
also had `order = :active`. Version 0.7 runs ProximalSolvers's coordinate descent,
which takes the same passes with and without momentum (see compare/ below) but
has no `order = :active`; the script no longer times that order.

Compares the orders of the coordinates, Anderson acceleration (issue #9) and
momentum (issue #20) on the lasso `‖y - Xβ‖² / (2n) + λ‖β‖₁`, solved from β = 0
with the default stopping rule (`opt_tol = 1e-7`):

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
at `max_iter = 100 000` for E' at 0.01 λmax.

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
  `examples/07_varying_coefficient.jl`. Since #38 removed `locpolyl1`, the new
  side skips that case; `varying/` above compares its replacement.

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
| A feasible_lasso! | 1.5 | 1.5 | 1.5 | 1.01 | 1.01 | 2 / 2 | 0e+00 |
| A lasso | 25.9 | 54.6 | 25.1 | 0.97 | 0.46 | 397 / 397 | 5e-15 |
| A scaled_lasso | 2.0 | 2.2 | 1.9 | 0.96 | 0.88 | 27 / 27 | 7e-16 |
| A sqrt_lasso | 5.8 | 5.5 | 5.8 | 1.01 | 1.06 | 11 / 11 | 0e+00 |
| E CDQuadraticLoss | 49.0 | 123.1 | 48.6 | 0.99 | 0.39 | 328 / 328 | 0e+00 |
| E LassoPath | 174.6 | 467.9 | 168.8 | 0.97 | 0.36 | 4548 / 4548 | 2e-14 |
| E feasible_lasso! | 4.8 | 4.5 | 4.5 | 0.94 | 1.01 | 154 / 154 | 2e-15 |
| E lasso | 4.1 | 5.1 | 4.1 | 1.00 | 0.79 | 108 / 108 | 3e-15 |
| E scaled_lasso | 1.9 | 1.6 | 1.8 | 0.95 | 1.13 | 84 / 84 | 0e+00 |
| E sqrt_lasso | 8.5 | 9.7 | 8.4 | 0.99 | 0.87 | 19 / 19 | 0e+00 |
| F CDQuadraticLoss | 61.9 | 169.2 | 61.7 | 1.00 | 0.36 | 427 / 427 | 0e+00 |
| F LassoPath | 227.9 | 715.7 | 216.3 | 0.95 | 0.30 | 6428 / 6428 | 4e-14 |
| F feasible_lasso! | 4.7 | 4.8 | 4.5 | 0.95 | 0.93 | 282 / 282 | 3e-15 |
| F lasso | 5.4 | 9.1 | 5.3 | 0.96 | 0.58 | 161 / 161 | 4e-15 |
| F scaled_lasso | 2.2 | 2.9 | 2.1 | 0.96 | 0.73 | 146 / 146 | 1e-15 |
| F sqrt_lasso | 12.1 | 18.5 | 12.0 | 0.99 | 0.65 | 32 / 32 | 0e+00 |
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
updates of σ̂ until the relative change of σ̂ was below `opt_tol` (1e-2 by
default, and 1e-6), with the square-root lasso solved in one run of coordinate
descent: from the data (`scaled_lasso`) and from sufficient statistics
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
`opt_tol = 1e-12`, relative to its largest coefficient. "A, b, c" is the time to
form the sufficient statistics.

| n | p | ρ | 0.6.0, opt_tol = 1e-2 | 0.6.0, opt_tol = 1e-6 | scaled_lasso | from A, b, c | A, b, c | passes 0.6.0 / new |
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

| problem | 0.6.0, opt_tol = 1e-2 | 0.6.0, opt_tol = 1e-6 | scaled_lasso | from S |
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

- **Faster, and at the solution.** In all but one case the new `scaled_lasso`
  was 1.2–5.4 times faster than the default of 0.6.0, whose coefficients were
  off by 2e-5 to 3e-2; the exception (n = 5000, p = 500, ρ = 0.9) was 3% slower,
  with an error of 3e-6 against 3e-4. At the accuracy of the new version (0.6.0
  with `opt_tol = 1e-6`) it was 1.2–19 times faster.
- **Sufficient statistics pay off when n > p.** From `A`, `b` and `c` the fit
  was 3.6–20 times faster than from the data when n > p, not counting forming
  them, which costs `O(np²)` (29 ms for n = 5000, p = 500, as much as 4–12 fits
  from the data). When p ≥ n it was slower, up to 25 times for p = 5000,
  n = 200: a coordinate update costs `O(p)` instead of `O(n)`.
- **The precision matrix estimator.** From the shared `S`, all the columns took
  2–90 ms, 3.4–62 times faster than the new `scaled_lasso` on copies of the data
  and 7–73 times faster than the default of 0.6.0, with coefficients within
  7e-7 of the solution. Without `check = false`, checking `S` for every column
  costs `O(p²)` each, which dominated for p ≥ 1000.

## glm/

`glm/glm_bench.jl` compares `GLMLassoPath` with GLMNet.jl (glmnet's Fortran)
and Lasso.jl on logistic and Poisson lasso paths with an intercept and
standardized columns, the same objective in all three. Each path has the same
100 levels, from λmax down to 0.01 λmax. The columns are Gaussian with AR(1)
correlation 0.5, and there are 10 nonzero coefficients. Accuracy is the
largest coefficient difference from a glmnet path at tolerance 1e-13, relative
to the largest coefficient. Times are the minimum of 3 interleaved runs after
a warm-up, with one BLAS thread. The script has its own environment:

```
julia --project=benchmark/glm -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/glm benchmark/glm/glm_bench.jl
```

On Discovery (Julia 1.13.2, AMD EPYC 7513, 2026-10-24, run
`20261024T0053Z-cd-glm-bench`), seconds, with the relative error in
parentheses:

| | n | p | glm_lasso | glm_lasso, loose | glmnet | glmnet, tol 1e-10 | Lasso.jl |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 1000 | 100 | 0.064 (9e-7) | 0.031 (5e-4) | 0.070 (1e-3) | 0.137 (5e-5) | 0.018 (1e-3) |
| logistic | 1000 | 2000 | 0.886 (2e-5) | 0.248 (4e-3) | 0.626 (8e-3) | 1.845 (7e-4) | 0.231 (1e-2) |
| logistic | 10000 | 100 | 0.516 (3e-7) | 0.310 (4e-4) | 0.413 (2e-3) | 0.649 (2e-5) | 0.177 (2e-3) |
| logistic | 10000 | 2000 | 11.08 (5e-7) | 4.26 (4e-4) | 7.25 (9e-4) | 17.20 (2e-5) | 4.25 (1e-3) |
| Poisson | 1000 | 100 | 0.024 (2e-7) | 0.013 (9e-5) | 0.040 (4e-4) | 0.074 (2e-5) | 0.015 (9e-5) |
| Poisson | 1000 | 2000 | 0.226 (2e-6) | 0.111 (6e-4) | 0.285 (1e-3) | 0.832 (4e-5) | 0.188 (5e-4) |
| Poisson | 10000 | 100 | 0.171 (2e-7) | 0.109 (5e-5) | 0.279 (2e-4) | 0.412 (2e-5) | 0.188 (1e-5) |
| Poisson | 10000 | 2000 | 2.136 (1e-7) | 1.552 (5e-5) | 2.105 (2e-4) | 2.428 (1e-5) | 3.726 (1e-5) |

"glm_lasso" is the default tolerances (`opt_tol = 1e-7`, `newton_tol = 1e-8`),
and "loose" is `CDOptions(; opt_tol = 1e-4)` with `newton_tol = 1e-5`.

- **Against glmnet.**
  - At their defaults, glmnet stops 400 to 7000 times further from the solution than `glm_lasso`, which takes 0.6–1.5 times its time.
  - At glmnet's accuracy or better, `glm_lasso` with loose tolerances is 1.3–3 times faster.
  - At glmnet's tolerance 1e-10, which still leaves glmnet's paths less accurate than `glm_lasso`'s defaults, `glm_lasso` is 1.1–3.7 times faster.
- **Against Lasso.jl.**
  - On logistic paths, Lasso.jl at its defaults is 2.6–3.8 times faster than `glm_lasso` at its defaults, and 500 to 7000 times less accurate. With loose tolerances `glm_lasso` takes 1.0–1.8 times Lasso.jl's time, and is still 2–5 times more accurate.
  - On Poisson paths, `glm_lasso` is faster at the largest size (0.57) and within 1.6 times elsewhere.

## glm_penalties/

`glm_penalties/compare_glm_penalties.jl` compares `GLMLassoPath` with `alpha`
against R's glmnet 5.1 and with MCP and SCAD against R's ncvreg 3.16.0, for
logistic regression on three designs (n = 400, p = 40, AR(0.5); n = 600,
p = 100; n = 150, p = 300) and Poisson regression on one (n = 400, p = 40), at
the same 25 levels from λmax down to 5% of it:

```sh
julia --project=benchmark/glm_penalties benchmark/glm_penalties/compare_glm_penalties.jl gen DIR
Rscript benchmark/glm_penalties/compare_glm_penalties.R DIR      # needs glmnet and ncvreg
julia --project=benchmark/glm_penalties benchmark/glm_penalties/compare_glm_penalties.jl compare DIR
```

Results on Discovery (runs `20261024T1234Z-cd-glm-penalties` and
`20261024T1259Z-cd-glm-penalties2`, the second with ProximalSolvers#65):

- **The elastic net equals glmnet's** (α ∈ {0.2, 0.5, 0.8}): identical supports
  at all levels and coefficients within 3e-6 on all four designs.
- **MCP and SCAD are stationary:** the largest violation of the stationarity
  conditions over a path is below 5e-13 on the three designs with n > p,
  and below 3e-5 on the design with p > n, whose paths approach separation.
- **ncvreg's GLM solutions are not stationary for these penalties** (violations
  3e-3 to 9e-2 at a = 3, 3.7, 6 and 7). Its `binomial` and `poisson` coordinate
  steps scale the concavity by the weighted curvature of each column at the
  current fit (`glm.c`: the MCP step divides by `v (1 - 1/γ)` with
  `v = xⱼ'Wxⱼ/n`, where the exact step divides by `v - 1/γ`), a penalty that
  changes with the fit; so the paths differ after the first level. At the
  first level where they differ, ours have the lower value of the objective
  with the penalty as documented (ncvreg's may be better for its own). In
  the gaussian family `v = 1` and the two agree (`penalties/`). ncvreg also
  stops paths early when the model is saturated (p > n); the levels it
  returned are compared.

## lla/

`lla/compare_lla.jl` compares the paths of local linear approximation (`lla`,
#69) with R's glmtlp 2.0.3 and ncpen 1.0.1, gaussian with an intercept, on two
designs (n = 200, p = 50; n = 100, p = 300; columns on different scales), at
30 levels from λmax down to 5% of it (10% for p > n), on the columns as given
and standardized:

```sh
julia --project=benchmark/lla benchmark/lla/compare_lla.jl gen DIR
Rscript benchmark/lla/compare_lla.R DIR      # needs glmtlp and ncpen
julia --project=benchmark/lla benchmark/lla/compare_lla.jl compare DIR
```

Results on Discovery (run `20261024T1618Z-cd-lla`):

- **glmtlp's `"tlp"`** (τ = 0.25, its difference-of-convex steps to
  convergence) equals `LassoPath(...; penalty = CappedL1(1, 0.25), intercept =
  true)` with `lla` at its default: coefficients within 5e-14 and intercepts
  within 1e-14 on both designs, for glmtlp's `standardize` FALSE and TRUE as
  `standardize_x`, with the same supports. `lasso` at single levels gives the
  same solutions.
- **ncpen with `x.standardize = FALSE`** equals `lla = LLA(; stages = 1000,
  start = :previous)` with `standardize_x = false`: MCP within 5e-10, SCAD within
  8e-10 and the TLP within 2e-14, with the same supports.
- **ncpen's own standardization** (`x.standardize = TRUE`) gives paths that
  differ by about 1e-2. Dividing the columns by their standard deviations, with
  n or with n - 1, does not reproduce it either, so the documentation claims
  agreement with ncpen only on the columns as given.

## sparsegroup/

`sparsegroup/compare_sparsegroup.jl` compares `SparseGroupLassoPath` with R's
sparsegl 1.1.1 (and SGL 1.3 with a second argument `sgl` to the R script), for
α ∈ {0.05, 0.5, 0.95} on three designs (n = 200, p = 60, groups of 5, AR(0.5);
n = 400, p = 120, groups of 10; n = 100, p = 200, groups of 8), at 25 levels
from λmax down to 5% of it:

```sh
julia --project=benchmark/sparsegroup benchmark/sparsegroup/compare_sparsegroup.jl gen DIR
Rscript benchmark/sparsegroup/compare_sparsegroup.R DIR      # needs sparsegl; add `sgl` for SGL
julia --project=benchmark/sparsegroup benchmark/sparsegroup/compare_sparsegroup.jl compare DIR
```

Results on Discovery (runs `20261024T1354Z-cd-sparsegroup` and
`20261024T1405Z-cd-sparsegroup2`):

- **The convention.** sparsegl and SGL scale the columns to unit Euclidean norm
  (√n times their standard deviation), so their level is ours divided by √n;
  SGL returns the coefficients of its standardized columns and no intercept.
  With that, the solutions of both satisfy the optimality conditions of
  `sparse_group_lasso`'s objective up to their tolerances (sparsegl 3e-3 to 2e-2,
  SGL 7e-7 to 3e-3), against 1e-12 for ours.
- **Against sparsegl on the full paths**, the coefficients agree to 2e-3 through
  1.2e-2, and our objective is lower than or equal to sparsegl's at every level of
  all nine paths, sparsegl's being higher by at most 6e-5 relative: the same
  problem, solved more accurately here.
- **SGL is slow:** its path on the first design at `thresh = 1e-10` did not finish
  in 20 minutes, so the R script runs it only on request.
