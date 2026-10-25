# Variable selection with error control

SLOPE, knockoff+, and complementary-pairs stability selection select columns of
`X`. Their guarantees apply under different assumptions. SLOPE with BH weights
controls FDR for orthogonal Gaussian regression with known noise level.
Knockoff+ controls FDR under the exchangeability and statistic assumptions below.
Stability selection bounds the expected number of selected variables with low
base selection probabilities; interpreting that as a false-selection bound
requires a condition on the noise variables.

## SLOPE

For the loss normalized by the number of observations,

```math
\min_{\beta_0,\beta}\frac{\|y-\beta_0-X\beta\|_2^2}{2n}
+\sum_{j=1}^p\lambda_j|\beta|_{(j)},\qquad
\lambda_1\geq\cdots\geq\lambda_p\geq0.
```

```julia
using HighDimensionalModels, Random, LinearAlgebra
rng = Xoshiro(82)
n, p = 240, 100
X = sqrt(n) * Matrix(qr(randn(rng, n, p)).Q)
y = X[:, 1:20] * fill(0.3, 20) + randn(rng, n)
sol = slope(X, y; q=0.1, sigma=1)
selected = findall(!iszero, sol.x)
path = SLOPEPath(X, y; nλ=40)
coef(path), predict(sol, X)
```

`slope_bh(p; q, sigma, n)` computes
``\lambda_j=\sigma\Phi^{-1}(1-qj/(2p))/\sqrt n``. The stated FDR guarantee
uses ``X^TX/n=I`` and independent Gaussian noise of known standard deviation
``\sigma``. A correlated design, an estimated noise level, or arbitrary path
selection does not inherit this guarantee. An intercept centers both X and y;
with an intercept, orthogonality must hold for the centered design. With
`standardize_x=true`, the penalty is on the coefficients of columns with mean
square one; the returned coefficients are on the original scale.
`SparseMatrixCSC` inputs stay sparse: scaling touches only stored entries.
Fully stored columns are centered before scaling to avoid cancellation for large
offsets; other columns are centered implicitly in matrix products. The transformed design and
solver workspaces use storage proportional to `nnz(X) + n + p`, in addition to
the coefficient path. Dense inputs retain their explicit transformation.

The solver uses accelerated proximal gradient from ProximalSolvers and the
sorted ℓ1 proximal operator from ProximalBase. Paths warm start and use the
strong screening rule of Larsson, Bogdan and Wallin (2020). Every reduced fit
is checked by the *full* proximal fixed-point equation, including its zero
cluster; excluded variables are added and refitted when it fails. This check
is necessary because sorted penalties do not separate across coordinates.
`screening=false` fits the full problem at every level. The path's `λpath`
contains scalar multipliers of `path.lambda`. Convergence, working sizes,
repairs, iterations and full residuals are retained per level. Fits that miss
the requested tolerance return `converged=false` with a warning.

## Knockoff+

```julia
X = randn(rng, n, p)
y = X[:, 1:20] * fill(0.3, 20) + randn(rng, n)
mx = knockoff_filter(X, y, 0.04; mu=zeros(p), Sigma=Matrix{Float64}(I,p,p), q=0.1, rng)
fx = knockoff_filter(X, y, 0.04; method=:fixed, q=0.1, rng)
mx.selected, mx.threshold, mx.W
```

`gaussian_knockoffs` uses the conditional Gaussian construction. For exact
model-X control, the rows of X must follow the Gaussian distribution specified
by the supplied **population** mean and covariance. Substituting estimated
moments provides second-order matching without establishing finite-sample
exchangeability. The filter deliberately requires explicit parameters.

`fixed_knockoffs` requires full column rank and ``n\geq2p``; it preserves the
Gram identities ``\widetilde X^T\widetilde X=X^TX`` and
``X^T\widetilde X=X^TX-\operatorname{diag}(s)``. Columns may have arbitrary
positive norms. Both constructions use an equicorrelated solution computed on
the correlation scale; no SDP dependency is needed. A user-supplied feasible
`s` is also accepted by the constructors. Semidefinite conditional covariances
use an eigenvalue square root, clipping only roundoff-sized negative values.

The statistic is ``W_j=|\widehat\beta_j|-|\widehat\beta_{p+j}|`` from the
joint lasso on ``[X\ \widetilde X]``. The filter fits without an intercept or
standardization **after** constructing knockoffs. Scaling must precede fixed-X
construction. This fixed-X interface assumes a model without an intercept.
For an intercept, project both X and y onto an orthonormal basis orthogonal to
the constant vector, then construct knockoffs in those n−1 coordinates
(requiring n−1 ≥ 2p). Subtracting column means alone does not ensure that the
knockoffs are orthogonal to the removed intercept.
A prespecified positive penalty, independent of y,
preserves fixed-X sufficiency. Cross-validating this penalty does not have the
same fixed-X guarantee. Model-X also requires a swap-antisymmetric statistic;
nonunique lasso solutions and exact ties need care. A failure of the joint
lasso to converge raises an error.

`knockoff_threshold` returns the smallest observed positive magnitude t with

```math
\frac{1+\#\{j:W_j\leq-t\}}{\max(1,\#\{j:W_j\geq t\})}\leq q,
```

or Inf (no selection). Ties are included. `knockoff_filter` always uses this
knockoff+ rule. The standalone threshold also supports `offset=0`, the original
knockoff rule with a modified-FDR guarantee. Supplied knockoffs can be used with
`method=:provided, Xk=...`, leaving their validity to the caller.

## Complementary-pairs stability selection

```julia
st = stability_selection(X, y, 0.12; q=10, cutoff=0.8, npairs=50, rng)
st.selected, st.frequencies, st.pfer_bound
```

Each pair consists of two disjoint half samples. For odd n, one randomly chosen
row per pair is omitted. On each half the base selector fits the lasso at the
supplied penalty and keeps at most q nonzero coefficients, ranked by magnitude
on the fitted scale. This is the fixed-penalty, capped-coefficient selector,
not a lasso-path stopping rule. `standardize_x=true` and `intercept=true` are
the defaults; these transformations are fitted separately on each half.

The frequency of a variable is its average selection indicator over all 2B
halves; variables at or above `cutoff` are retained. `pair_frequencies` records
simultaneous selection in both halves. The result also retains each selection
and the pairs, so the calculation can be inspected. Explicit `pairs` can be
shared with another implementation. Draws and per-half RNG seeds are made
before fitting. `julia --threads=N` enables concurrent fits; `parallel=false`
uses the same draws serially. Failed fits raise an error.

For ``\tau=\mathtt{cutoff}>1/2``, Shah and Samworth's distribution-free bound is

```math
E|\widehat S_{\rm CPSS}\cap L_\theta|
\leq\frac{\theta q}{2\tau-1},\qquad
L_\theta=\{j:P(j\in\widehat S_{\rm half})\leq\theta\}.
```

The deterministic cap makes q an upper bound on the expected base selection
size. The reported `pfer_bound = q^2/(p*(2cutoff-1))` uses ``\theta=q/p``, as
R's stabs does with `assumption="none"`. This bounds expected false selections
only when all noise variables belong to ``L_{q/p}``; exchangeability and a
no-worse-than-random base selector are sufficient conditions. It is **not** an
unconditional FDR guarantee. `stability_bound(...; theta=...)` also exposes the
general bound. The observed average selection size is descriptive and does not
replace q in the guarantee. No unimodality or r-concavity improvement is assumed.

## References and validation

- [Bogdan et al. (2015), SLOPE](https://doi.org/10.1214/15-AOAS842).
- [Su and Candès (2016), minimax adaptation](https://doi.org/10.1214/15-AOS1397).
- [Larsson, Bogdan and Wallin (2020), strong screening](https://proceedings.neurips.cc/paper/2020/hash/a7d8ae4569120b5bec12e7b6e9648b86-Abstract.html).
- [Barber and Candès (2015), fixed-X knockoffs](https://doi.org/10.1214/15-AOS1337).
- [Candès et al. (2018), model-X knockoffs](https://doi.org/10.1111/rssb.12265).
- [Meinshausen and Bühlmann (2010), stability selection](https://doi.org/10.1111/j.1467-9868.2010.00740.x).
- [Shah and Samworth (2013), complementary pairs](https://doi.org/10.1111/j.1467-9868.2011.01034.x).

The R golden-data generator, replicated Discovery benchmark/profile harness,
and Monte Carlo recipe are in `benchmark/selection`. Monte Carlo evidence
checks the listed finite simulation designs; it does not prove a guarantee.

```@autodocs
Modules = [HighDimensionalModels]
Pages = ["src/slope.jl", "src/knockoffs.jl", "src/stability.jl"]
Private = false
```
