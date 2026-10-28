# Low-rank multivariate regression

A coefficient matrix of low rank shares a few linear combinations of the
predictors across responses. For an `n × p` design `X` and `n × q` response
matrix `Y`, these estimators return a `p × q` coefficient matrix `B`.
The intercepts are fitted by default, without penalty, and columns keep their
original scales. Inputs may be integer, Float32 or Float64; sparse inputs are
converted to dense matrices. The methods return `LowRankSolution`s with
`coef`, `predict`, `residuals`, `nobs`, `deviance`, and `dof`.

```julia
using HighDimensionalModels
s = reduced_rank(X, Y, 3)
r = select_rank(X, Y)          # estimates the noise variance by least squares
predict(r.solution, Xnew)
```

`reduced_rank` minimizes the residual sum of squares subject to a rank bound,
using the truncated SVD of the least-squares fitted values. The compact SVD
of the centered design handles wide and rank-deficient designs and chooses
the minimum-norm coefficients. Rank zero gives the intercept-only model.
`select_rank` minimizes `RSS + μ * rank`, the rank selection criterion of
[Bunea, She and Wegkamp (2011)](https://arxiv.org/abs/1004.2995). Its default
is `μ = 2σ²(√rank(Xc) + √q)²`. A supplied `sigma2` avoids estimating the noise
variance; a supplied `μ` sets the threshold directly. A saturated design
requires one of them, since it has no residual degrees of freedom.

## Nuclear-norm regression

```math
\min_{B,\beta_0}\ \frac{\|Y-1\beta_0^T-XB\|_F^2}{2n}+\lambda\|B\|_*.
```

`nuclear_regression` solves this convex problem by accelerated proximal
gradient. ProximalBase's `NuclearNorm` soft-thresholds the coefficient singular
values. The stopping rule checks the proximal gradient mapping at the returned
coefficients. This is a penalty on `B`; thresholding the singular values of
`XB` solves a different problem except for an orthonormal design.

```julia
s = nuclear_regression(X, Y, 0.1)
using ProximalAlgorithms       # optional dependency; activates the PANOC backend
s = nuclear_regression(X, Y, 0.1; solver = :panoc)
p = LowRankPath(X, Y; method = :nuclear, nλ = 30, λminratio = 0.01)
cv = CVLowRankPath(X, Y; method = :nuclear, solver = :panoc, nλ = 30, nfolds = 5)
coef(cv)                     # the minimum CV error solution
predict(cv, Xnew)
```

The path reuses its gradient setup and warm starts each penalty from the
previous fit. For a wide design it computes gradients from `X` rather than
forming a `p × p` Gram matrix. `λmax = ‖Xc'Yc/n‖₂` is the zero-solution
threshold. The default automatic grid starts just above it.
`solver = :apg` remains the default. With ProximalAlgorithms installed and
loaded, `solver = :panoc` uses its limited-memory quasi-Newton acceleration
and line search. Both backends use ProximalBase's nuclear proximal operator
and certify the same normalized gradient mapping at the returned coefficients.

Float64 PANOC fits reuse the loss/gradient products and final residual/singular
values. Tall designs use a quadratic value with a direct residual fallback near
cancellation; wide designs share one residual between value and gradient.
Float32 keeps its original loss/gradient arithmetic. The internal step remains
`.95/L` and the certificate is checked every ten iterations, plus termination.
The replicated study (archive
[`pr114-nuclear-panoc-20261026`](https://github.com/mlakolar/HighDimensionalModels.jl/releases/download/benchmark-evidence/pr114-nuclear-panoc-20261026.tar.gz),
report `results/cluster-20261026/RESULTS.md`) compares complete fits and paths
with independent accuracy checks. PANOC helps
the tested middle-penalty fits and paths, but APG is faster for the weak
many-response cold fit. APG remains the default; no automatic solver-selection
rule or general memory improvement is implied by those measurements.

## Row-sparse reduced-rank regression

```math
\min_{\operatorname{rank}(B)\leq r,\beta_0}\
\frac{\|Y-1\beta_0^T-XB\|_F^2}{2n}
+\lambda\sum_{j=1}^p w_j\|B_{j,:}\|_2.
```

`sparse_reduced_rank` uses the factorization `B = AV'`, `V'V = I`, alternating
exact row coordinate descent for the group-lasso block and an orthogonal
Procrustes update for the response subspace, as in
[Chen and Huang (2012)](https://doi.org/10.1080/01621459.2012.734178).
The initialization is reduced-rank least squares. With `adaptive = true`,
`w` is the inverse row norm of that fit to power `gamma` (default 2), with
a `sqrt(eps(T))` floor. Explicit positive weights override this choice.
The problem is nonconvex: `converged` means the blocks have stabilized, not
that a global optimum was found.
`block_solver = :fista` uses restarted block FISTA for the weighted group-lasso
block, retaining the same response update and KKT stopping conditions.
`block_solver = :cd` remains the default. Either choice applies to paths
and cross-validation.

```julia
s = sparse_reduced_rank(X, Y, 3, 0.1; adaptive = true)
p = LowRankPath(X, Y; method = :sparse, rank = 3, block_solver = :fista, nλ = 30)
best = p.solutions[argmin(lowrank_ic(p; criterion = :bic))]
cv = CVLowRankPath(X, Y; method = :sparse, rank = 3, adaptive = true)
```

A zero fit is restarted from the RRR initialization at the next level, since
zero factors otherwise form an absorbing point of alternating updates. The
rank bound is fixed along a sparse penalty path; compare candidate bounds
using the same explicit folds in separate `CVLowRankPath` calls.

## Sparse singular vectors

`sparse_svd_regression(X, Y, r)` also allows sparse response loadings. It uses
one round of exclusive extraction from the SVD of the RRR coefficients,
with a rank-one weighted lasso in each exclusive response layer. The selected
components need not remain orthogonal. Each layer chooses its penalty by BIC,
BICP or AIC and may select a zero component. This is the RSSVD structure of
[Chen, Chan and Stenseth (2012)](https://doi.org/10.1111/j.1467-9868.2011.01002.x),
implemented without Kronecker design matrices. The return value contains the
`solution`, individual `components`, component penalty grids `λ`, and `scores`.

```julia
ss = sparse_svd_regression(X, Y, 3; nλ = 30, criterion = :bic)
predict(ss.solution, Xnew)
cv = CVLowRankPath(X, Y; method = :sparse_svd, ranks = 0:3, nλ = 20)
```

For this CV path, each training fold selects component penalties by BIC, then
held-out prediction error selects the rank bound. Every fold recomputes its
centering, initial fit, and adaptive weights. Convergence is local.

## Tuning and information criteria

`LowRankPath(...; method = :rank, ranks = 0:5)` shares the design and fitted
SVDs across rank constraints. Fixed-rank results reuse those decompositions
and use a small QR/SVD core for coefficient rank; residuals are computed once.
Selected-design rank, numerical coefficient rank, and exact degrees of freedom
retain their definitions, including custom fitting cutoffs and zero fits.
`lowrank_ic` returns scores on the normalized
scale `log(RSS) + penalty * df/(n*q)`. AIC uses 2 and BIC uses `log(n*q)`;
BICP uses `2log(n*q)` for row-sparse regression and `2log(p*q)` for RSSVD.
These are the scales in rrpack. Additive constants do not affect selection.
The StatsAPI methods `aic(solution)` and `bic(solution)` use
`n*q * log(RSS/(n*q)) + penalty * df`, matching the lasso path convention.

By default `df` counts the parameters of the selected rank/support model:
`r * (rank(X_selected) + q - r)`, plus `q` intercepts. This is an approximation
for penalized estimators. For rank-constrained fits, `df = :exact` uses the
Stein formula of [Mukherjee et al. (2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4684945/);
a cutoff through tied fitted singular values receives an infinite score.
RSSVD component scores count `nnz(u) + nnz(v) - 1`.

`CVLowRankPath` uses a common rank or penalty grid across folds and computes
mean squared error per response, weighted by fold sizes. Supplied `folds`
allow grouped or blocked validation. A nonconverged fold fit raises an error.
For a rank path, request ranks that are estimable in every training fold.

rrpack 0.1-14 interchanges its BIC/BICP selection labels inside `rssvd.eea`.
The R oracle, `test/data/lowrank/oracle.R`, aligns the actual BIC column
explicitly; the methods here use the criterion names according to the formulas
above.

The [low-rank study (snapshot)](https://github.com/mlakolar/HighDimensionalModels.jl/tree/abddaf56a790b86178de376b175a1185020c8381/benchmark/lowrank) records replicated
timing and profiles; the `lowrank` group of the [benchmark harness](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/benchmark/README.md)
measures the current fits. The test suite checks the fits against rrpack fits
saved by the R oracle. rrpack uses unscaled
squared losses: its penalty is `n * λ` for the objectives here. Its adaptive
nuclear-norm estimator shrinks the fitted-value singular values, so it is
not an oracle for the coefficient nuclear penalty; the latter has an exact
orthonormal-design test and a proximal-gradient optimality check.

The sparse-fit setup optimization reuses `X'X/n` and `X'Y/n` within each
fit and across penalty levels in a path. The FISTA step scale comes from the
existing design SVD. Coefficient rank, degrees of freedom and residuals are
computed from thin factors with conservative dense fallbacks. The objective,
weights, support/DF semantics and stopping rules are unchanged. CD results are
unchanged and FISTA results agree within the stopping tolerance. CD remains
the default block solver. The replicated optimization study (archive
[`pr112-sparse-setup-20261026`](https://github.com/mlakolar/HighDimensionalModels.jl/releases/download/benchmark-evidence/pr112-sparse-setup-20261026.tar.gz),
report `results/cluster-20261026/FINAL.md`) used 20 shared independent inputs
per design on an earlier dependency stack and was not rerun at this revision.
At the middle penalty, paired public-fit candidate/baseline
ratios were 0.854 (CD) and 0.860 (FISTA) for many responses, and 0.854 for wide
FISTA. Wide CD was inconclusive at 0.962, with an interval reaching 1.003;
small CD regressed at 1.016. Complete paths improved in every retained
design/solver. FISTA was useful for ill-conditioned and rank-deficient inputs,
while CD was faster on strong/middle wide inputs. The study establishes no
universal solver or default.
