# The lasso and its tuning-free variants

## The lasso

[`lasso`](@ref)`(X, y, λ)` solves

```math
\underset{\beta_0,\,\beta}{\text{minimize}} \quad \frac{1}{2n}\,\|y - \beta_0 - X\beta\|^2 + \lambda \sum_{j=1}^p \omega_j |\beta_j|.
```

- **The scaling.** The loss is divided by ``2n``, as in glmnet, so that
  ``\lambda`` is on the scale of the gradient ``X_j^\top r / n`` and does not grow
  with the sample size.
- **The intercept.** With `intercept = true`, ``\beta_0`` is not penalized, and
  at the solution it is ``\bar y - \bar x^\top \hat\beta`` for the column means
  ``\bar x`` and the mean ``\bar y``. The columns and ``y`` are centered on the
  fly, so ``X`` is not copied. Without it, ``\beta_0 = 0``.
- **The loadings.** By default ``\omega_j = 1``. With `standardize_x = true`,
  ``\omega_j`` is the standard deviation of column ``j``,
  ``\sqrt{\sum_i (X_{ij} - \bar x_j)^2 / n}`` (centered only with an
  intercept), which penalizes the coefficients as if the columns were
  standardized while keeping them on the scale of the data.
  `lasso(X, y, λ, ω)` takes loadings of your own, as the adaptive lasso does; a
  loading of zero leaves a coefficient unpenalized.
- **The largest useful λ.** All coefficients are zero for
  ``\lambda \ge \lambda_{\max} = \max_j |X_j^\top (y - \bar y)| / (n \omega_j)``,
  with ``\bar y = 0`` without an intercept.

The theory of the lasso takes ``\lambda`` of the order of
``\sigma \sqrt{2 \log(p) / n}`` for noise with standard deviation ``\sigma`` and
standardized columns. That needs ``\sigma``. The variants below do not, and a
[`LassoPath`](@ref) with cross-validation chooses ``\lambda`` from the data (see
[Paths, screening and solver options](paths.md)).

[`lasso`](@ref) returns a [`HighDimensionalModels.LassoSolution`](@ref): the
coefficients `x` (a `SparseIterate`), the `intercept`, the `residuals`,
``\hat\sigma`` (the standard deviation of the residuals), the passes of
coordinate descent and whether it converged. The functions of StatsAPI apply to
it: [`coef`](@ref) (intercept first), [`predict`](@ref), [`residuals`](@ref),
[`nobs`](@ref), [`dof`](@ref) (the number of nonzero coefficients, plus one for
the intercept) and [`deviance`](@ref).

## Formulas and tables

Loading StatsModels.jl adds methods that take a formula and a table (a
DataFrame, a NamedTuple of vectors or any other Tables.jl table) in place of
`X` and `y`: `fit` with [`LassoPath`](@ref), [`CVLassoPath`](@ref),
[`GLMLassoPath`](@ref) and [`GroupLassoPath`](@ref), and [`lasso`](@ref),
[`sqrt_lasso`](@ref), [`adaptive_lasso`](@ref), [`glm_lasso`](@ref),
[`group_lasso`](@ref) and [`rlasso`](@ref).

```julia
using HighDimensionalModels, StatsModels
m = fit(LassoPath, @formula(y ~ x1 + log(x2) + region), df; nλ = 50)
coef(m, 0.1), coefnames(m), predict(m, newdf)
g = group_lasso(@formula(y ~ x1 + region + x2 & region), df, 0.05)   # a group per term
```

The formula sets the intercept (`y ~ 0 + x1` has none). Categorical variables
become dummy columns, coded by `contrasts`. The group lasso takes the columns
of each term as a group. `weights` and `offset` may name columns of the table,
and rows with missing values in the variables of the model are left out. The
result is a [`FormulaFit`](@ref), whose functions of StatsAPI forward to the
fitted model and whose `predict` builds the columns of new data by the formula.

## The elastic net, MCP and SCAD

[`lasso`](@ref) and [`LassoPath`](@ref) take other penalties of ProximalBase
in place of the ℓ1 norm, at the level ``\lambda``. `alpha < 1` gives the
elastic net of glmnet,

```math
\lambda\Big(\alpha\sum_j |\beta_j| + \frac{1 - \alpha}{2}\|\beta\|^2\Big), \qquad 0 < \alpha \le 1,
```

and `penalty` another penalty whose level ``\lambda`` replaces: `NormL1(1, w)`
and `NormL1Nonneg(1, w)`, the lasso and the nonnegative lasso with weights;
`ElasticNet(1, λ2, w)`, with the ridge part ``\lambda_2`` fixed;
`Scaled(ElasticNet(α, 1 - α, w), 1)`, the elastic net with weights on its ℓ1
part; and the nonconvex `MCP(1, a)` and `SCAD(1, a)` of ncvreg, whose
concavity ``a`` is kept. With `standardize_x`, the penalty applies to the
coefficients of the standardized columns, as in glmnet and ncvreg, and the
coefficients returned are those of the columns of ``X``. A nonconvex penalty
has local solutions, and the one found depends on the start: as in ncvreg,
`lasso` follows a path of levels from ``\lambda_{\max}`` down to ``\lambda``,
each warm started from the previous one. The path of the elastic net at a
fixed ``\alpha`` starts at ``\lambda_{\max}`` of the lasso divided by
``\alpha``.

The exact coordinate steps of MCP and SCAD need each coordinate problem to be
convex, ``a v_j > 1`` for MCP and ``(a - 1) v_j > 1`` for SCAD, where ``v_j``
is the mean square of column ``j`` (centered with an intercept, weighted with
weights). Standardized columns have ``v_j = 1``, so that ncvreg's conditions
``a > 1`` and ``a > 2`` suffice; without `standardize_x`, a column with
variance at most ``1/a`` fails, and the path throws an `ArgumentError`.

### Local linear approximation

`lla = LLA(...)` computes a nonconvex penalty by local linear approximation
(reweighted ℓ1): a sequence of weighted lassos, each convex, with the weights
``p_j'(|\beta_j|)/\lambda`` at the previous solution. Besides MCP and SCAD it
takes the capped ℓ1 penalty `CappedL1(1, a)`, ``\lambda\min(|\beta_j|, a)``
(T. Zhang 2010; the truncated ℓ1 penalty of Shen, Pan and Zhu 2012), and the
log penalty `LogPenalty(1, ε)`, ``\lambda\log(1 + |\beta_j|/\varepsilon)``,
which have no exact path:

- `LLA()`, a lasso and one weighted lasso, is the two-step estimator of Fan,
  Xue and Zou (2014), and needs no condition on ``a`` for MCP and SCAD;
- `LLA(; stages = 100)` iterates until the weights stop changing, as the
  multi-stage convex relaxation of Zhang (2010), the difference-of-convex
  algorithm of Shen, Pan and Zhu (2012, glmtlp's `"tlp"`) and the reweighted ℓ1
  of Candès, Wakin and Boyd (2008); `CappedL1` and `LogPenalty` use it when
  `lla` is not given;
- `start = :previous` takes the first weights of each level from the solution
  at the previous level, as ncpen does, instead of starting each level from the
  lasso.

## Bounds on the coefficients

`lower` and `upper` restrict the coefficients to a box,
``l_j \le \beta_j \le u_j``, in [`lasso`](@ref), [`LassoPath`](@ref),
[`CVLassoPath`](@ref), [`adaptive_lasso`](@ref) and [`glm_lasso`](@ref), with
any of the penalties above. Each is a number or a vector with a bound per
coefficient. As for glmnet's `lower.limits` and `upper.limits`, zero must be
feasible, ``l_j \le 0 \le u_j``: `lower = 0` gives the nonnegative lasso, and
``l_j = u_j = 0`` keeps variable ``j`` out. The intercept is not bounded. The
penalty becomes ProximalBase's `Constrained(g, IndBox(l, u))`, which keeps
coordinate descent separable: each step clamps the scalar proximal step to
``[l_j, u_j]``. A coefficient at zero is then optimal while the part of the
gradient that points into the box is at most ``\lambda \omega_j``, so
``\lambda_{\max}`` and the strong rule use only the directions the bounds allow;
for the nonnegative lasso,
``\lambda_{\max} = \max_j (X_j^\top y / n)_+ / \omega_j``.

The adaptive lasso of Zou (2006), [`adaptive_lasso`](@ref)`(X, y, λ)`,
minimizes

```math
\frac{1}{2n}\|y - \beta_0 - X\beta\|^2 + \lambda\sum_j \frac{|\beta_j|}{|\tilde\beta_j|^\gamma}
```

with loadings from a first estimate ``\tilde\beta``: the lasso (`init = :lasso`,
at `initλ`, by default ``\lambda``), ridge regression (`init = :ridge`) or a
vector of coefficients. Large coefficients are penalized less, so that they are
less biased, and a coefficient with ``\tilde\beta_j = 0`` stays at zero.

## Penalized generalized linear models

[`glm_lasso`](@ref)`(X, y, λ, loss)` solves

```math
\min_{\beta_0, \beta}\; \frac{\sum_i w_i\,\ell(\beta_0 + x_i^\top\beta + o_i, y_i)}{\sum_i w_i} + \lambda\sum_j \omega_j|\beta_j|
```

for a scalar loss ``\ell`` of ProximalBase: `ProximalBase.Logistic()`, the
negative log-likelihood of logistic regression (``y_i \in [0, 1]``, usually
``\{0, 1\}``), the default; `ProximalBase.Poisson()`, Poisson regression with
the log link (``y_i \ge 0``); `ProximalBase.Huber(δ)`; or a loss of your own.
The conventions and defaults are those of glmnet: an unpenalized intercept, the
weighted standard deviations of the columns as the loadings
``\omega_j`` (`standardize_x`), and the observation weights ``w_i`` and offset
``o_i`` of `weights` and `offset`. The problem is solved by proximal Newton,
ProximalSolvers' `ProxNewton`, the method of glmnet: each step minimizes the
weighted least-squares model of the loss at the current fit by coordinate
descent, then backtracks. [`GLMLassoPath`](@ref) computes the solutions along a
decreasing sequence of levels with the strong rule, and stops once the fit
explains a fraction `devmax = 0.999` of the null deviance, as glmnet does:
along a logistic path on separable data the coefficients grow without bound.
The solution reports the `deviance` and the `nulldeviance`. `alpha` and
`penalty` give the elastic net, MCP and SCAD, and `adaptive_lasso(X, y, λ, loss)`
the adaptive lasso, as for least squares.

[`CVGLMLassoPath`](@ref)`(X, y, loss)` chooses the level by ``K``-fold
cross-validation, as [`CVLassoPath`](@ref) does for least squares, with the
held-out measures of glmnet's `cv.glmnet`: the deviance (the default), the mean
squared or absolute error of the predicted means, or the misclassification
rate for logistic regression. Weights and offsets follow the observations into
the folds.

## The square-root lasso

[`sqrt_lasso`](@ref)`(X, y, λ)` solves the square-root lasso of Belloni,
Chernozhukov and Wang (2011),

```math
\underset{\beta_0,\,\beta}{\text{minimize}} \quad \|y - \beta_0 - X\beta\| + \lambda \sum_{j=1}^p \omega_j |\beta_j|,
```

with the loadings of standardized columns by default (`standardize_x = true`).
The gradient of the loss at the true coefficients is
``-X^\top \varepsilon / \|\varepsilon\|``, which does not depend on the noise
level, and neither does a good ``\lambda``: with Gaussian noise,
``\lambda = 1.1\, \Phi^{-1}(1 - \alpha / (2p))`` keeps every coefficient at zero
with probability about ``1 - \alpha`` when ``\beta = 0``, whatever ``\sigma``. The
loss is not smooth where the residuals vanish, so each coordinate step
minimizes the objective over the coordinate exactly.

## The scaled lasso

[`scaled_lasso`](@ref)`(X, y, λ, ω)` estimates the coefficients and the noise
level together, by the scaled lasso of Sun and Zhang (2012):

```math
\underset{\beta_0,\,\beta,\,\sigma > 0}{\text{minimize}} \quad \frac{1}{2n\sigma}\,\|y - \beta_0 - X\beta\|^2 + \frac{\sigma}{2} + \lambda \sum_{j=1}^p \omega_j |\beta_j|.
```

The minimum over ``\sigma`` is at ``\hat\sigma = \|y - \beta_0 - X\beta\| / \sqrt n``,
so ``\hat\beta`` solves a square-root lasso, with the loss
``\|y - \beta_0 - X\beta\| / \sqrt n``, which one run of coordinate descent
solves. ``\lambda`` is the penalty for unit noise, such as
``\sqrt{2 \log(p) / n}``: the lasso that ``\hat\beta`` solves has the penalty
level ``\lambda \hat\sigma``, which the solution reports in `penalty`.
[`scaled_lasso!`](@ref) also solves it from sufficient statistics,
``X^\top X / n``, ``X^\top y / n`` and ``y^\top y / n``, and from a covariance
matrix, one variable on the others, as in the precision matrix estimator of
Sun and Zhang (2013).

## The feasible lasso

Under heteroscedastic noise, the penalty loadings of Belloni, Chen,
Chernozhukov and Hansen (2012) depend on the noise of each observation:

```math
\Gamma_j = \sqrt{\frac{1}{n} \sum_{i=1}^n (X_{ij} - \bar x_j)^2 \varepsilon_i^2}.
```

[`feasible_lasso!`](@ref)`(x, X, y, λ0)` alternates the lasso
``\|y - \beta_0 - X\beta\|^2 / (2n) + \lambda_0 \sum_j \Gamma_j |\beta_j|`` with
the update of ``\Gamma`` from its residuals, until the relative change of
``\Gamma`` is below `opt_tol` of [`IterLassoOptions`](@ref); `init_procedure`
chooses the initial residuals. The examples use
``\lambda_0 = 1.1\, \Phi^{-1}(1 - \gamma / (2p)) / \sqrt n`` with
``\gamma = 0.1 / \log n``.

## Inference after selection

[`rlasso`](@ref)`(X, y)` is the lasso with data-driven penalty loadings as R's
hdm computes it (Belloni, Chen, Chernozhukov and Hansen, 2012): the level
[`theory_lambda`](@ref)`(n, p)` ``= c\,\Phi^{-1}(1 - \gamma/(2p))/\sqrt{n}``
with loadings initialized from least squares on the five most correlated
controls. With `post=true`, each lasso is followed by least squares on its
selected controls; the loadings are updated until the residual standard
deviation settles or `max_iter` is exhausted. The first post-lasso solve uses
half the nominal penalty.
The defaults are ``c=1.1`` with `post=true` and ``c=0.5`` with `post=false`.
The solution keeps the final penalized coefficients, intercept, penalty and
loadings in `lasso`, `lasso_intercept`, `lasso_λ0` and `lasso_loadings`, separately
from the returned coefficients and updated loadings.
[`post_lasso`](@ref)`(X, y, sol)` refits least squares on selected controls and
gives heteroskedasticity-robust (HC1) standard errors for that regression.

### Model and orthogonal score

[`double_selection`](@ref)`(y, d, X; method=:partialling)` estimates the
prespecified coefficient ``\alpha_0`` in original outcome and treatment units:

```math
y_i = \mu + \alpha_0 d_i + x_i^\top\beta_0 + \varepsilon_i,
\qquad d_i = m_0(x_i) + v_i.
```

An intercept is always included. Choose the treatment before inspecting
estimated effects; these intervals do not account for choosing a target by its
estimated significance. Write ``g_0(x)=\mathbb E[y\mid x]`` and
``m_0(x)=\mathbb E[d\mid x]``. The partialling score is

```math
\psi_i(\alpha,g,m)
= \{d_i-m(x_i)\}\{y_i-g(x_i)-\alpha[d_i-m(x_i)]\}.
```

At the true nuisance functions its expectation is zero, and its first
derivatives in either nuisance direction vanish. Its Jacobian in
``\alpha`` is ``-\mathbb E[v_i^2]`` and its influence function is
``v_i\varepsilon_i/\mathbb E[v_i^2]``. These properties explain why accurate
nuisance prediction can support inference without exact support recovery.

The asymptotic justification follows the high-dimensional partially linear
framework of [Belloni, Chernozhukov and Hansen (2014)](https://doi.org/10.1093/restud/rdt044).
Working assumptions include iid observations;
``\mathbb E[\varepsilon_i\mid d_i,x_i]=0`` and
``\mathbb E[v_i\mid x_i]=0``; ``\mathbb E[v_i^2]`` and
``\mathbb E[v_i^2\varepsilon_i^2]`` bounded away from zero; bounded fourth
moments of the relevant normalized variables; and sparse eigenvalues bounded
away from zero and infinity on the relevant sparse sets. Both nuisance
functions must have sparse approximations, with approximation errors small
enough for the score remainder to be ``o_p(n^{-1/2})``. A representative
non-cross-fitted sparsity regime is

```math
\frac{s^2\log^2(p\vee n)}{n}\longrightarrow 0.
```

Additional empirical-moment regularity is needed for the particular nuisance
fits; the reference states precise conditions. The actual lasso penalties,
including loadings, must dominate the nuisance
scores with high probability, and optimization error must be negligible for
the nuisance rates and the final score expansion. The function does not test
these assumptions. In particular, selecting a covariance correction or
receiving a convergence flag does not establish valid coverage, and native
`post=false` tuning is not automatically a score-dominating penalty choice.

### Post-Lasso partialling-out

Fit the outcome and treatment separately on the controls, in that order.
Choose residuals ``\tilde u`` and ``\tilde v``, center each, and regress
``\tilde u`` on an intercept and ``\tilde v``. The intercept in the last
regression is retained even when nuisance residual means are numerically zero.
The `refit` keyword chooses the residuals while leaving the fitted nuisance
records intact:

| `post` in the fits | `refit` | Residuals |
|:--|:--|:--|
| either | `nothing` | Native residuals; effective `refit=post` |
| `true` | `true` | Native post-lasso OLS residuals |
| `true` | `false` | `response - lasso_intercept - X*lasso` from the stored final solve |
| `false` | `false` | Native penalized residuals |
| `false` | `true` | OLS residuals on each fit's own selected controls and an intercept |

For non-native OLS refits, the control basis uses the same pinned projection
as the shared final stage: center controls; discard intercept aliases whose
centered norm is at most ``\tau`` times their raw norm; scale remaining columns
to unit norm; then use pivoted QR and retain the first ``r`` directions, where
``r=\#\{j:|R_{jj}|>\tau\}`` and ``\tau=\max(10^{-7},\sqrt{\operatorname{eps}(T)})``.
This rule is distinct from the solve-based rank inside `rlasso` and from R's
`dqrdc2`. Each nuisance uses its own selected set, including when the sets differ.

Precomputed fits provide a controlled comparison at a fixed penalty and
support:

```julia
using HighDimensionalModels, Random

fit_y = rlasso(X, y; post=true, rng=Xoshiro(118))
fit_d = rlasso(X, d; post=true, rng=Xoshiro(119))
ols = double_selection(y, d, X, fit_y, fit_d;
    method=:partialling, refit=true, vcov=:HC1)
penalized = double_selection(y, d, X, fit_y, fit_d;
    method=:partialling, refit=false, vcov=:HC1)
```

Both results retain the same `nuisance_y` and `nuisance_d` objects, including
`selected`, `lasso_λ0` and `lasso_loadings`. Comparing them isolates refitting
at the stored penalty. Running native `post=false` fits is a different
comparison: it also changes the default ``c`` from 1.1 to 0.5 and the sequence
of loading updates. The precomputed interface checks dimensions, element
types, intercepts, matching `post` flags, supports and residual consistency.
These are consistency checks; they do not establish provenance. Calls preserve
both fits and consume no RNG draws.

### Variance and diagnostics

Let ``v`` and ``w`` be the centered final-stage treatment and outcome residuals,
``A=v^\top v``, ``\hat\alpha=v^\top w/A``, and ``e=w-\hat\alpha v``. The
estimated influence values are ``\hat\phi_i=nv_i e_i/A`` and final leverages
are ``h_i=h_{W,i}+v_i^2/A``. For partialling, ``h_{W,i}=1/n`` and ``k=2``.
Thus its HC corrections count only the final intercept and treatment;
they do not subtract either nuisance's selected-control count or rank.
For the shared double-selection variance formulas, ``k=\operatorname{rank}(W)+1``
and ``s`` is its nominal selected-control count, including aliases.

| `vcov` | Estimated variance of ``\hat\alpha`` |
|:--|:--|
| `:homoskedastic` | ``(e^\top e)/(n-k)/A`` |
| `:HC0` | ``\sum_i v_i^2e_i^2/A^2=\sum_i\hat\phi_i^2/n^2`` |
| `:HC1` | ``\mathrm{HC0}\,n/(n-k)`` |
| `:HC2` | ``\sum_i v_i^2e_i^2/(1-h_i)/A^2`` |
| `:HC3` | ``\sum_i v_i^2e_i^2/(1-h_i)^2/A^2`` |
| `:hdm`, partialling (default) | Homoskedastic variance with ``k=2`` |
| `:hdm`, double selection | ``\mathrm{HC0}\,n/(n-s-1)`` |

The `:hdm` conventions match hdm 0.3.2, including its classical final-stage
variance for partialling. For general heteroskedasticity use HC0--HC3; the
classical partialling variance needs an additional homoskedasticity restriction
for its usual inferential interpretation. Intervals and two-sided p-values use
the normal
distribution. `confint(result; level=...)` can change the interval level.

On a common nuisance support with the same OLS projection, partialling gives
the same coefficient, regression residuals and HC0 variance as double selection.
This includes well-conditioned common supports; different numerical rank rules
can matter on nearly aliased designs. With ``r`` centered
control directions, the double-selection homoskedastic and HC1 variances are
``(n-2)/(n-r-2)`` times the partialling variances. HC2, HC3 and the two `:hdm`
conventions generally differ because they use different leverage or variance
corrections.

The result records `post`, effective `refit`, `vcov_type`, `selected_y`,
`selected_d`, their union `selected`, `nuisance_y` and `nuisance_d`. For
partialling, `rank=1` and `dof_residual=n-2` refer to the final regression.
`target_variation` is ``v^\top v/\sum_i(d_i-\bar d)^2``; small values indicate
little remaining treatment variation. Passing the numerical threshold does
not establish strong identification. `max_leverage` and `influence` describe
the final regression. `show` flags nuisance non-convergence without making it
fatal; inspect both `converged` and `solver_converged` on the saved fits.

### Failure behavior

Partialling rejects nonempty `forced`, although native hdm silently ignores
I3 for partialling. Use an intercept throughout; `intercept=false`, observation
weights and offsets are unsupported. With effective `refit=true`, each nuisance
must satisfy ``|S|+1<n`` even if selected columns are aliased. Double selection
accepts only `refit=nothing`, since its final OLS is part of that estimator.

Inputs must be finite, with ``n\ge3``, ``p\ge1``, ``0<\texttt{level}<1``, and
nonconstant outcome and treatment. The shared final stage requires ``k<n``;
hdm double selection also requires ``n-s-1\ge1``. It rejects
``\|v\|\le\tau\|d-\bar d\|``, ``\|e\|\le\tau\|y-\bar y\|`` and, for HC2/HC3,
``\max_i h_i>1-\tau``. Numerical and option errors raise `ArgumentError` with
the quantity and its value; dimension errors raise `DimensionMismatch`.

### Differences from native hdm

hdm zeroes raw coefficients below the absolute threshold ``10^{-6}``, which is
not invariant to control units. Julia retains such nonzero coordinates. In
`test/data/hdm/nearthreshold`, scaling a control by ``10^6`` puts its nuisance
coefficients below that cutoff: native hdm drops it and the effect estimate
moves from 0.5 to 0.8. The Julia estimate remains 0.5. The discrepancy is
intentional in that test. The default Julia partialling results otherwise
retain the native hdm variance convention and nuisance residual path; non-native
`refit` modes and robust `vcov` choices are explicit alternatives.

### Double selection

<!-- Reserved for U5: double-selection algorithm, forced controls and examples. -->

## Observation weights and offsets

[`lasso`](@ref), [`sqrt_lasso`](@ref), [`scaled_lasso`](@ref) and
[`LassoPath`](@ref) take observation `weights` ``w_i \ge 0``, with a positive
sum, and an `offset` ``o``. The lasso then minimizes

```math
\frac{1}{2 \sum_i w_i} \sum_{i=1}^n w_i (y_i - o_i - \beta_0 - x_i^\top \beta)^2 + \lambda \sum_{j=1}^p \omega_j |\beta_j|.
```

Dividing by ``\sum_i w_i`` follows glmnet: ``\lambda`` means the same with and
without weights, unit weights give the unweighted fit, integer weights the fit
of replicated rows, and zero weights leave observations out. The intercept,
the loadings of `standardize_x` and ``\hat\sigma`` are weighted as well. The
square-root lasso takes the weighted norm ``\sqrt{\sum_i w_i r_i^2}`` of the
residuals ``r``, and the scaled lasso estimates
``\hat\sigma = \sqrt{\sum_i w_i r_i^2 / \sum_i w_i}``.

## The data

`X` can be any matrix and `y` any vector of real numbers. A `Matrix`, or a view
of a range of its rows or of some of its columns, is read in place. A
`SparseMatrixCSC` stays sparse; coordinate descent reads its columns entry by
entry, so for now it is slower than a dense matrix. Any other matrix, such as a
transpose, is copied once. The computations are in the floating point type of
`X` and `y`, so `Float32` data give `Float32` estimates.
