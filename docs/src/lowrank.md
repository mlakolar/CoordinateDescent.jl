# Low-rank multivariate regression

A coefficient matrix of low rank shares a few linear combinations of the
predictors across responses. For an `n × p` design `X` and `n × q` response
matrix `Y`, these estimators return a `p × q` coefficient matrix `B`.
The intercepts are fitted by default, without penalty, and columns keep their
original scales. Inputs may be integer, Float32 or Float64; sparse inputs are
converted to dense matrices. The single-penalty and rank-constrained methods
return `LowRankSolution`s with
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
The [replicated study (snapshot)](https://github.com/mlakolar/HighDimensionalModels.jl/tree/4c506aa8be5f2c485ca0f146255c0b1f4f01179e/benchmark/lowrank/panoc_ablation)
compares complete fits and paths with independent accuracy checks. PANOC helps
the tested middle-penalty fits and paths, but APG is faster for the weak
many-response cold fit. APG remains the default; no automatic solver-selection
rule or general memory improvement is implied by those measurements.

## Simultaneously sparse and low rank

[`sparse_nuclear_regression`](@ref) applies both penalties to the same
coefficient matrix:

```math
\min_{B,\beta_0}\ \frac{\|Y-1\beta_0^T-XB\|_F^2}{2n}
+\lambda_{\mathrm{rank}}\|B\|_*+\lambda_{\mathrm{sparse}}P(B).
```

With `sparsity = :entry`, the default,
`P(B) = ∑ⱼ wⱼ ∑ₜ |B[j,t]|`, the sum of absolute entries with row weights.
With `sparsity = :row`, `P(B) = ∑ⱼ wⱼ ‖B[j,:]‖₂`, selecting predictors
across responses. `weights` must contain one positive finite value per
predictor and defaults to ones. Both levels may be zero. The intercepts
remain unpenalized and columns keep their original scales.

The solver is ProximalSolvers' Condat–Vũ splitting, with smooth least squares,
the weighted sparsity proximal operator, and a dual nuclear-norm step.
Its vector iterates are reshaped for ProximalBase's `NuclearNorm`.
This splitting solves the sum of the penalties; applying their two proximal
operators in succession would generally solve a different problem.

```julia
s = sparse_nuclear_regression(X, Y, 0.1, 0.05; sparsity = :row)
s.gap, s.rank, s.rank_tol, s.activerows, s.nnz
predict(s, Xnew)
```

The returned [`SparseNuclearSolution`](@ref) records the coefficients,
intercepts, residuals, objective, exact entry/row support, and convergence
diagnostics. All describe the returned `B`. Its numerical rank counts
singular values above
`rank_tol = max(√eps(T) σ₁(B), ‖B - B_h‖₂)`, where `B_h` is the last nuclear
proximal point. The solver uses residual balancing without averaging to keep
that point paired with the returned iterate. Balancing is disabled when the
nuclear level is zero. `rank_tol` exposes the scale of
the diagnostic; small positive singular values need not represent recovered
components. `coef`, `predict`, `residuals`, `nobs` and `deviance` apply.
No `dof`, `aic`, `bic` or `lowrank_ic` is supplied for this estimator.

### Certificate and zero region

For centered data, put `R = Yc - Xc*B` and `Θ = R/n`. A dual feasible point
satisfies `Xc'Θ = U + V`, with `‖U‖₂ ≤ λ_rank` and either
`|V[j,t]| ≤ λ_sparse*w[j]` or `‖V[j,:]‖₂ ≤ λ_sparse*w[j]`.
The solver's nuclear dual supplies `U`. Set `V = Xc'R/n - U` and choose the
largest `t ∈ [0,1]` that puts `tV` in its sparsity ball. The nuclear dual is
also checked for roundoff at its boundary. The dual objective is

```math
D(t\Theta)=\frac{t\langle R,Y_c\rangle}{n}
-\frac{t^2\|R\|_F^2}{2n}.
```

`gap = (objective - D)/max(objective, floatmin(T))` bounds relative objective
error up to floating point accuracy. Nonfinite certificates or material
violations of weak duality give `Inf`; negative roundoff is clipped to zero.
Condat–Vũ stops on this gap, and `converged` requires `gap ≤ tol`.
`r_norm`, `s_norm`, `eps_pri` and `eps_dual` also record the solver's KKT
residuals and tolerances. A gap tolerance does not bound coefficient error
on an ill-conditioned or unidentified design.

When `λ_sparse = 0`, the certificate scales `U = Xc'R/n` into the spectral
ball. When `λ_rank = 0`, it uses `U = 0`. With both levels zero, the solver
uses its own KKT stopping rule and reports `gap = NaN`. A design that is
zero after centering returns the intercept-only fit directly.

Writing `C = Xc'Yc/n`, zero coefficients are optimal exactly when
`C = U + V` for elements of the two dual balls. The marginal thresholds are
`‖C‖₂` and the weighted maximum absolute entry or row norm of `C`.
Either marginal threshold alone suffices. The zero region also includes
pairs with both levels below those thresholds.

### Tuning two penalties

```julia
grid = PenaltyGrid(sparse_nuclear_regression, X, Y;
  nλ = (5, 5), λminratio = (0.01, 0.01), sparsity = :row)
grid.names                     # (:rank, :sparse)
grid.solutions[2, 3]            # λ1[2] is rank; λ2[3] is sparsity
cv = CVPenaltyGrid(sparse_nuclear_regression, X, Y;
  λ1 = grid.λ1, λ2 = grid.λ2, sparsity = :row, nfolds = 5)
cv.λmin, cv.cvm, cv.cvse
coef(cv)                       # the minimum-error full-data fit
predict(cv, Xnew)
```

[`PenaltyGrid`](@ref) stores a concrete matrix of solutions. Automatic axes
start at each marginal threshold times `1 + √eps` and decrease on a log
scale. Explicit axes must be nonempty, decreasing, finite and nonnegative;
zeros and repeated levels are allowed. Fits proceed down columns, starting
each column from the head of the previous one. Primal/dual vectors and both
step sizes are copied at each branch; solutions do not share mutable state.

[`CVPenaltyGrid`](@ref) uses the full-data numerical axes in every fold and
recomputes preprocessing from training rows alone. Each fold has its own
seeded generator and workspaces, including mutable SVD caches. `threads`
controls fold parallelism. Explicit `folds` supply row labels, as for
`CVLowRankPath`. `cvm` is mean squared error per response, weighted by fold
sizes; `cvse` is the weighted standard error described in its docstring.
A pair with a nonconverged full-data or training fit gets `NaN` and is
excluded. An empty eligible set raises an error. Ties go to the first pair
in column-major order, preferring larger levels. There is no two-dimensional
one-standard-error rule. `fit(PenaltyGrid, sparse_nuclear_regression, X, Y; ...)`
and the corresponding `fit(CVPenaltyGrid, ...)` are also available.

This combination is useful when a shared response subspace and sparse
coefficients are plausible, with prediction used to assess the tradeoff.
It has no fixed rank bound. In contrast, `sparse_reduced_rank` imposes a rank
constraint and solves a nonconvex problem. An additive model `B = L + S`
allows separate low-rank and sparse components; it imposes different structure
and is outside this estimator. Task-specific designs are not supported by
this estimator. An entry invisible to a task's loss still enters the nuclear
norm, so removing zero or constant columns within tasks, as the multitask
data layer does, would change the problem.

Combining convex penalties need not improve sample complexity over the
better individual penalty. For noiseless matrix recovery from independent
Gaussian linear measurements of the vectorized matrix,
[Oymak, Jalali, Fazel, Eldar and Hassibi (2015)](https://arxiv.org/abs/1212.3753)
give lower bounds for combinations of structure-inducing norms, including
sparse and low-rank models. Those measurement assumptions differ from the
shared-design regression operator here. The result cautions against a
universal recovery claim; this implementation makes none. See
[`examples/14_sparse_nuclear.jl`](https://github.com/mlakolar/HighDimensionalModels.jl/blob/master/examples/14_sparse_nuclear.jl)
for a small prediction comparison with nuclear regression and row lasso.

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
the default block solver. The
[replicated optimization study (snapshot)](https://github.com/mlakolar/HighDimensionalModels.jl/tree/09da5020ffe8aa1c3bd812629794c08fa4a93464/benchmark/lowrank/sparse_setup)
used 20 shared independent inputs per design on an earlier dependency stack
and was not rerun at this revision. Its final report is in the snapshot's
`results` directory. At the middle penalty, paired public-fit candidate/baseline
ratios were 0.854 (CD) and 0.860 (FISTA) for many responses, and 0.854 for wide
FISTA. Wide CD was inconclusive at 0.962, with an interval reaching 1.003;
small CD regressed at 1.016. Complete paths improved in every retained
design/solver. FISTA was useful for ill-conditioned and rank-deficient inputs,
while CD was faster on strong/middle wide inputs. The study establishes no
universal solver or default.
