# Benchmarks

Run the scripts from the root of the repository, in the environment of this
folder, which loads CoordinateDescent from this checkout:

```
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark benchmark/locpoly_bench.jl    # about a minute
julia --project=benchmark benchmark/cd_bench.jl         # about 20 minutes
julia --project=benchmark benchmark/order_anderson_bench.jl   # about 5 minutes
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

## order_anderson_bench.jl

Compares the orders of the coordinates and Anderson acceleration (issue #9) on
the lasso `‖y - Xβ‖² / (2n) + λ‖β‖₁`, solved from β = 0 with the default
stopping rule (`optTol = 1e-7`):

- `order = :cyclic`, `:random` (the default) and `:active`;
- `order = :cyclic` with `anderson = 5`.

The designs are those of the lasso benchmark of ProximalOPT.jl, with two more
correlated ones: Gaussian columns scaled to ‖Xⱼ‖² = n, s coefficients ±1 and
a signal-to-noise ratio of 4.

| design | n | p | s | correlation between columns |
|---|---:|---:|---:|---|
| A | 500 | 2,500 | 50 | none |
| B | 500 | 2,500 | 50 | AR(1), 0.8 between neighbors |
| B' | 500 | 2,500 | 50 | AR(1), 0.95 between neighbors |
| C | 5,000 | 500 | 50 | AR(1), 0.5 between neighbors |
| D | 200 | 10,000 | 20 | none |
| E | 500 | 2,500 | 50 | 0.5 between every pair (a common factor) |
| E' | 500 | 2,500 | 50 | 0.9 between every pair |

Each case runs until it has half a second of runs, and at least 3 runs (5 for
the random orders). For the random orders the table gives the median passes
and time; for the deterministic ones, the shortest time. The results below are
from an Apple M3 Pro with Julia 1.13.2 and one BLAS thread. Every solution is
as accurate as the cyclic one, except that the cyclic order stops at
`maxIter = 100 000` for E' at 0.01 λmax.

### One λ: milliseconds (passes)

| design | λ/λmax | cyclic | random | active | Anderson |
|---|---:|---:|---:|---:|---:|
| A | 0.1 | 8.5 (179) | 12.8 (255) | 12.0 (248) | **5.4** (97) |
| A | 0.02 | 212 (3,120) | 318 (4,435) | 326 (4,477) | **93** (1,231) |
| A | 0.01 | 800 (10,965) | 944 (12,442) | 1,013 (13,511) | **275** (3,391) |
| B | 0.1 | 11.7 (333) | 16.7 (465) | 16.1 (452) | **7.2** (172) |
| B | 0.02 | 203 (3,169) | 283 (4,116) | 307 (4,663) | **91** (1,240) |
| B | 0.01 | 586 (8,164) | 783 (10,760) | 834 (11,277) | **202** (2,579) |
| B' | 0.1 | 10.6 (568) | 19.6 (1,019) | 19.1 (980) | **5.4** (232) |
| B' | 0.02 | 114 (2,364) | 175 (3,499) | 180 (3,505) | **38** (717) |
| B' | 0.01 | 458 (7,541) | 668 (10,405) | 702 (9,895) | **134** (2,002) |
| C | 0.1 | **2.5** (16) | 4.5 (35) | 4.2 (33) | **2.5** (14) |
| C | 0.02 | 15.5 (54) | 19.5 (66) | 19.5 (66) | **14.8** (49) |
| C | 0.01 | 24.2 (58) | 33.8 (77) | 30.8 (75) | **22.3** (51) |
| D | 0.1 | 11.9 (835) | 19.5 (1,017) | 14.2 (923) | **7.4** (363) |
| D | 0.02 | 131 (8,855) | 162 (9,067) | 189 (12,050) | **43** (2,511) |
| D | 0.01 | 525 (35,655) | 712 (30,892) | 605 (37,808) | **78** (4,537) |
| E | 0.1 | 6.5 (949) | 1.9 (81) | **1.6** (80) | 2.9 (248) |
| E | 0.02 | 1,464 (36,940) | 37.6 (767) | **35.5** (750) | 465 (10,510) |
| E | 0.01 | 3,499 (66,494) | **140** (2,260) | 159 (2,596) | 1,491 (24,962) |
| E' | 0.1 | 36.1 (9,581) | 3.6 (593) | **3.3** (625) | 9.7 (1,278) |
| E' | 0.02 | 191 (25,164) | 5.8 (539) | **5.5** (530) | 36.3 (3,676) |
| E' | 0.01 | 2,316 (100,000) | **42.6** (1,409) | 45.6 (1,523) | 1,051 (36,694) |

### Path of 50 λ values, λmax to λmax/100 (`LassoPath`): milliseconds

| design | cyclic | random | active | Anderson |
|---|---:|---:|---:|---:|
| A | 1,253 | 1,781 | 1,599 | **346** |
| B | 1,230 | 1,759 | 1,825 | **372** |
| D | 575 | 733 | 734 | **118** |
| E | 8,015 | 368 | **330** | 2,805 |

### Memory K at 0.02 λmax: milliseconds (passes)

| design | K = 3 | K = 5 | K = 10 |
|---|---:|---:|---:|
| A | 94.9 (1,217) | 94.7 (1,231) | 95.3 (1,305) |
| B | 83.8 (1,126) | 90.9 (1,240) | 91.6 (1,334) |
| D | 45.2 (2,552) | 43.0 (2,511) | 49.0 (2,972) |

### What the numbers say

- **Anderson acceleration**
  - Without a common factor (A–D), it is the fastest or tied in every case.
  - On A, B, B' and D, at 0.02 λmax and below, it is 2.2–6.7 times faster than the cyclic order it accelerates, and 3.1–9.1 times faster than the default random order.
  - On paths it is 3.3–4.9 times faster than the cyclic order and 4.7–6.2 times faster than the default.
  - When n ≫ p (C), coordinate descent needs a few dozen passes and acceleration changes little, but it is never slower.
  - It needs the cyclic order, and keeps that order's weakness under a common factor. On E and E' it is 2.2–5.3 times faster than the cyclic order, but the random orders are faster still: 1.5–3 times at 0.1 λmax, 6–25 times at smaller penalties, and 7.6 times on the path.
- **Memory:** K = 3, 5 and 10 are within 10% of each other.
- **Extrapolations are often rejected while the active set settles, and accepted in runs once it has.** Stopping after a few rejections in a row, as planned in issue #9, prevented any acceleration: in a solve on A, the first three attempts were all rejected.
- **`order = :active`** never shows the cyclic order's blow-up, but the default stays `:random`.
  - At 0.1 λmax it is 3–27% faster than `order = :random`.
  - At smaller penalties it ranges from 15% faster to 17% slower.
