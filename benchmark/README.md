# Benchmarks

Run the scripts from the root of the repository, in the environment of this
folder, which loads CoordinateDescent from this checkout:

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/locpoly_bench.jl    # about a minute
julia --project=benchmark benchmark/cd_bench.jl         # about 20 minutes
julia --project=benchmark benchmark/order_anderson_bench.jl    # about 4 minutes
```

The checkout is found through `[sources]` in `Project.toml`, which Julia 1.11
and later read, as is ProximalOPT 0.2 until it is registered. Julia 1.10
ignores `[sources]`: develop the checkout and add ProximalOPT by URL in that
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

## order_anderson_bench.jl

The tables of this section are from version 0.6, whose own coordinate descent
also had `order = :active`. Version 0.7 runs ProximalOPT's coordinate descent,
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
ProximalOPT.jl, with three more: two more correlated ones and one with three
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

Compares this version, which runs ProximalOPT's coordinate descent, with
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
ProximalBase 0.5 and ProximalOPT 0.2.

```
julia --project=benchmark/compare/old -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/compare/new -e 'using Pkg; Pkg.instantiate()'
benchmark/compare/ab.sh 3
```

`ab.sh` runs the two sides in turn, each in a fresh process, and every run
starts from the same seed. `compare.jl` takes the fastest median time of each
case over the rounds, since load on the machine only adds time. The table is
from three rounds on the M3 Pro on 23 October 2026, with Julia 1.13.2 and one
BLAS thread.

| case | 0.6 (ms) | 0.6, momentum = false (ms) | 0.7 (ms) | 0.7 / 0.6 | 0.7 / 0.6 without momentum | passes, 0.6 without momentum / 0.7 | largest difference |
|---|---:|---:|---:|---:|---:|---|---:|
| A CDQuadraticLoss | 220.1 | 929.2 | 922.3 | 4.19 | 0.99 | 5524 / 5524 | 0e+00 |
| A LassoPath | 486.4 | 2199.4 | 2186.3 | 4.49 | 0.99 | 31986 / 31986 | 1e-14 |
| A feasibleLasso! | 1.5 | 1.5 | 1.5 | 1.00 | 0.99 | 2 / 2 | 0e+00 |
| A lasso | 25.1 | 54.8 | 54.6 | 2.17 | 1.00 | 934 / 934 | 1e-14 |
| A scaledLasso | 1.9 | 2.1 | 2.1 | 1.08 | 0.97 | 29 / 29 | 0e+00 |
| A sqrtLasso | 6.0 | 5.8 | 5.8 | 0.96 | 1.00 | 11 / 11 | 0e+00 |
| E CDQuadraticLoss | 43.8 | 109.3 | 110.5 | 2.53 | 1.01 | 946 / 946 | 0e+00 |
| E LassoPath | 173.6 | 464.9 | 459.5 | 2.65 | 0.99 | 10594 / 10594 | 1e-14 |
| E feasibleLasso! | 4.7 | 4.5 | 4.4 | 0.92 | 0.98 | 151 / 151 | 2e-15 |
| E lasso | 3.9 | 5.0 | 5.0 | 1.28 | 1.01 | 173 / 173 | 5e-15 |
| E scaledLasso | 1.9 | 1.6 | 1.6 | 0.83 | 0.98 | 76 / 76 | 0e+00 |
| E sqrtLasso | 8.9 | 10.4 | 10.3 | 1.16 | 0.99 | 23 / 23 | 0e+00 |
| F CDQuadraticLoss | 54.5 | 146.1 | 147.1 | 2.70 | 1.01 | 1282 / 1282 | 0e+00 |
| F LassoPath | 225.6 | 720.3 | 705.5 | 3.13 | 0.98 | 18205 / 18205 | 2e-14 |
| F feasibleLasso! | 4.7 | 4.9 | 4.8 | 1.01 | 0.97 | 501 / 501 | 7e-15 |
| F lasso | 5.2 | 8.8 | 8.8 | 1.71 | 1.01 | 344 / 344 | 8e-15 |
| F scaledLasso | 2.2 | 2.9 | 2.8 | 1.29 | 0.98 | 277 / 277 | 1e-15 |
| F sqrtLasso | 13.0 | 19.8 | 19.4 | 1.49 | 0.98 | 51 / 51 | 0e+00 |
| locpolyl1 | 0.6 | 0.5 | 0.6 | 0.94 | 1.03 | 0 / 0 | 9e-16 |

- **The coordinate descent moved without a change.** Against 0.6.0 without
  momentum, every case takes the same passes and the solutions agree to 2e-14.
  The times are within 3% (0.97–1.03).
- **Momentum has not moved.** Against the default of 0.6.0, which uses
  momentum, this version is slower where momentum helps:
  - `LassoPath`: 2.7–4.5 times as long;
  - `CDQuadraticLoss`: 2.5–4.2 times as long;
  - `lasso`: 1.3–2.2 times as long.
  The scaled, feasible and square-root lasso at theory-driven penalties take
  0.83–1.49 times as long.
