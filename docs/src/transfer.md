# Joint-sparse transfer multitask learning

[`transfer_multitask`](@ref) and [`debias`](@ref) fit TMTL(Fused) and
TMTL(Debiased) of
[Lim and Kolar (2026)](https://arxiv.org/abs/2609.30879), which borrow
strength from source domains to estimate a row-sparse coefficient matrix in a
target domain. [`TransferMultitaskPath`](@ref) computes warm-started paths and
[`CVTransferMultitaskPath`](@ref) tunes both stages on target data only.

## The model

Domain ``\ell=0`` is the target and ``\ell=1,\dots,L`` are the sources. Domain
``\ell`` has ``n_\ell`` observations of the same ``K`` responses on the same
``p`` predictors, with one design ``X^\ell`` for all responses or one design
``X^{\ell k}`` per response (Appendix B of the paper). Sizes may differ across
domains; the responses of a domain share its rows. With ``N=\sum_{\ell=0}^L
n_\ell`` over all domains, the target included, TMTL(Fused) solves

```math
\min_{B^0,\dots,B^L,\ \beta_0}\;
\frac{1}{2NK}\sum_{\ell=0}^{L}\sum_{k=1}^{K}
\|y^{\ell k}-\beta_0^{\ell k}-X^{\ell k}B^\ell_{\cdot k}\|^2
+\lambda_0\|B^0\|_{2,1}
+\lambda_0\sum_{\ell=1}^{L}a_\ell\|B^\ell-B^0\|_{2,1},
```

where ``\|A\|_{2,1}=\sum_j\|A_{j\cdot}\|_2`` sums the row norms. The target
block is row sparse, the contrasts ``B^\ell-B^0`` are row sparse, and the
source blocks need not be. Each observation has the same weight in the loss.
The intercepts ``\beta_0^{\ell k}`` are unpenalized, one per domain and
response, so domains may have different means; with `intercept = false` there
are none.

The contrast weights are `contrast_weights = :sqrt`, ``a_\ell=8\sqrt{n_\ell/N}``
(the default), `:linear`, ``a_\ell=8n_\ell/N``, or a vector of positive values.
A larger ``a_\ell`` penalizes the contrast with source ``\ell`` more and so
borrows more from it. The designs are not standardized: the contrasts compare
coefficients across domains, so the predictors need a common scale in every
domain, which per-domain standardization would change. Centering within each
domain and response only removes intercepts.

The reported estimate is the sample-size weighted average of all fitted
blocks,

```math
\widehat W^F=\sum_{\ell=0}^{L}\frac{n_\ell}{N}\widehat B^\ell,
```

not ``\widehat B^0``. ``\widehat W^F`` is generally not row sparse, since the
source blocks are not. Target predictions use the intercepts
``\bar y^0-\bar x^{0\prime}W`` of whichever slope matrix ``W`` is reported.

## The correction

TMTL(Debiased) corrects the fused estimate on the target data,

```math
\widehat D=\arg\min_{D,\ d_0}\;
\frac{1}{2n_0K}\sum_{k=1}^{K}\|y^{0k}-d_0^k-X^{0k}(\widehat W^F_{\cdot k}+D_{\cdot k})\|^2
+\widetilde\lambda\|D\|_{2,1},\qquad
\widehat W^D=\widehat W^F+\widehat D.
```

It is a group lasso of the target residuals ``Y^0-X^0\widehat W^F`` on the
unstandardized target design. For a shared design this is
`multi_lasso(X⁰, Y⁰ - X⁰Ŵᶠ, K * λ̃; standardize_x = false)`; note the factor
``K``, since `multi_lasso` divides its loss by ``2n``, not ``2nK``. The
correction uses the same target sample as the fused fit, so [`debias`](@ref)
takes the target data again: `debias(fit, (X⁰, Y⁰), λ̃)`. The correction is
zero when

```math
\widetilde\lambda\ge\widetilde\lambda_{\max}
=\max_j\Bigl\|\tfrac{1}{n_0K}\bigl[\widetilde X^{0k\prime}_{\cdot j}
\bigl(\widetilde y^{0k}-\widetilde X^{0k}\widehat W^F_{\cdot k}\bigr)\bigr]_{k}\Bigr\|_2,
```

with tildes for centered columns. "Debiased" refers to the aggregation bias of
``\widehat W^F`` in point estimation, which the correction reduces; it is not
an inferential (desparsified) estimator, and the fits give no confidence
intervals or tests. The correction is computed by the row solver of
[`multitask_lasso`](@ref) on unstandardized columns, which reports its own
relative duality gap, so the fused and correction diagnostics are separate
fields of the result.

## Coordinate descent

The default solver works in the coordinates of the paper's Appendix C.1,

```math
\theta^0=B^0,\qquad\theta^\ell=B^\ell-B^0\quad(\ell\ge1),\qquad
B^\ell=\theta^0+\theta^\ell,\qquad
\widehat W^F=\theta^0+\sum_{\ell\ge1}\frac{n_\ell}{N}\theta^\ell,
```

where the penalty ``\lambda_0\|\theta^0\|_{2,1}+\sum_\ell\lambda_0a_\ell\|\theta^\ell\|_{2,1}``
separates over components and rows. Column ``k`` of ``\theta^0`` enters
response ``k`` of every domain, and column ``k`` of ``\theta^\ell`` only that
of domain ``\ell``. Block coordinate descent updates one row of one component
at a time. With centered columns ``\widetilde x^{\ell k}_j``, residuals
``r^{\ell k}`` and scores
``S^\ell_{jk}=\langle\widetilde x^{\ell k}_j,r^{\ell k}\rangle/(NK)``, the
gradient of row ``j`` of ``\theta^0`` is ``-\sum_\ell S^\ell_{j\cdot}`` and that
of ``\theta^\ell`` is ``-S^\ell_{j\cdot}``. The coordinate curvatures are
``h^0_{jk}=\sum_\ell\|\widetilde x^{\ell k}_j\|^2/(NK)`` and
``h^\ell_{jk}=\|\widetilde x^{\ell k}_j\|^2/(NK)``. The row step is a group
soft threshold at the curvature ``H=\max_kh_{jk}``, exact for a shared design
and a majorization with task-specific designs. An update changes the residuals
of every domain the component enters.

A column that is constant in one domain does not force its coefficients to
zero: a row of ``\theta^0`` is determined by the domains where the column
varies. Only coefficients that enter no loss stay at zero. The passes, the
change rule and its relative scale follow the row solver of
[`multitask_lasso`](@ref), with [`CDOptions`](@ref); continuation is used when
`warm_start = false`.

### Certificates and λmax

The fused fit is certified by one global relative duality gap. Let

```math
M=\max\Bigl\{\max_j\Bigl\|\sum_{\ell=0}^{L}S^\ell_{j\cdot}\Bigr\|_2,\;
\max_{\ell\ge1,\,j}\frac{\|S^\ell_{j\cdot}\|_2}{a_\ell}\Bigr\}.
```

The dual point is ``u=t\,r/(NK)`` with ``t=\min(1,\lambda_0/M)``, and its value
``D(u)=\sum\langle\widetilde y,u\rangle-\tfrac{NK}{2}\sum\|u\|^2``. The gap is
the whole primal objective minus ``D(u)``, divided by the objective, with the
guards of the row solver; it is unavailable (`NaN`) at ``\lambda_0=0`` and at a
zero objective. `objective` and `gap` are recomputed from the returned blocks
and intercepts. A positive `gaptol` in the options requires the gap; paths
require the path target.

At zero coefficients, ``M`` is ``\lambda_{\max}``, the smallest level at which
every block is zero, not only ``\widehat W^F``. The automatic grid of
[`TransferMultitaskPath`](@ref) starts at
``\lambda_{\max}(1+\sqrt{\mathrm{eps}})``. Screening applies the sequential
strong rule to each component: row ``j`` of component ``c`` is kept when it is
active or ``\|g_{cj}\|_2\ge w_c(2\lambda_k-\lambda_{k-1})``, with ``w_0=1`` and
``w_\ell=a_\ell``. After convergence every excluded row is checked against
``\|g_{cj}\|_2\le w_c\lambda_k`` and violations are added back
(`kkt_checks` counts the rounds).

## ADMM

`solver = :admm` is the paper's two-block ADMM (Appendix A), kept as an
optional backend for fidelity to the paper and as an independent check of
coordinate descent. The copies ``\Gamma^{0\ell}=B^0`` (``\ell=0,\dots,L``) and
``\Gamma^\ell=B^\ell`` (``\ell\ge1``) carry the penalties; each constraint has
its own penalty ``\rho`` and scaled dual. Each iteration

- updates ``B^0`` and each ``B^\ell`` by a ridge-type least-squares solve; the
  thin SVD of each centered design is cached, so every penalty value and
  ``p>n_\ell`` reuse it;
- soft-thresholds the rows of ``B^0+U^0`` for ``\Gamma^{00}``, and updates
  each pair ``(\Gamma^{0\ell},\Gamma^\ell)`` exactly by eq. (37): with
  ``\mu_\ell=\rho_{0\ell}\rho_\ell/(\rho_{0\ell}+\rho_\ell)`` and
  ``r=S_{\lambda_0a_\ell/\mu_\ell}(q^\ell-q^{0\ell})``,
  ``\Gamma^{0\ell}=(\rho_{0\ell}q^{0\ell}+\rho_\ell q^\ell-\rho_\ell r)/(\rho_{0\ell}+\rho_\ell)``
  and ``\Gamma^\ell=\Gamma^{0\ell}+r``;
- updates the scaled duals and computes the primal residual ``B-\Gamma`` and
  dual residual ``\rho(\Gamma_{\mathrm{old}}-\Gamma)`` of each constraint.

It stops when every constraint has primal residual at most
``\sqrt{pK}\,\epsilon_{\mathrm{abs}}+\epsilon_{\mathrm{rel}}\max(\|B\|,\|\Gamma\|)``
and dual residual at most
``\sqrt{pK}\,\epsilon_{\mathrm{abs}}+\epsilon_{\mathrm{rel}}\rho\|U\|``.
During the first `adapt` iterations the penalties are balanced: ``\rho`` is
multiplied by `increase` (up to `rho_max`) when the primal residual exceeds
`balance` times the dual residual, and divided by `decrease` in the opposite
case; the scaled dual is multiplied by ``\rho_{\mathrm{old}}/\rho_{\mathrm{new}}``.
The penalties are then fixed, so the standard two-block convergence argument
applies. The settings are given as a named tuple, for example
`admm = (; abstol = 1e-10, reltol = 1e-10, max_iter = 20_000)`; the defaults
are listed in [`transfer_multitask`](@ref). The result reports the last
residuals and the final penalties. ADMM returns ``\theta^0=\Gamma^{00}`` and the
contrasts ``\Gamma^\ell-\Gamma^{0\ell}``, so zero rows are exact; the objective
and the gap are computed at that point as for coordinate descent.

## Paths and cross-validation

[`TransferMultitaskPath`](@ref) solves a decreasing sequence of ``\lambda_0``
with fixed contrast weights, warm starting each level. `coef(path)` and
`predict(path, X)` give a result per level.

[`CVTransferMultitaskPath`](@ref) tunes the weight family, ``\lambda_0`` and
then ``\widetilde\lambda`` by held-out target prediction:

- Folds are over target rows, the same for every response, family, level and
  correction level. Every source row is in every training fit.
- For each fold and family, the fused path is fitted on the target training
  rows and all sources at the full-data grid of that family. Sizes, ``N``, the
  `:sqrt` and `:linear` weights and the centering are recomputed from that
  training data, and the strong rule restarts.
- A fold's score is its target sum of squared errors divided by its held-out
  rows times ``K``; `cvm` averages the folds weighted by their sizes. The
  family and ``\lambda_0`` minimize `cvm` among candidates converged on the full
  data and in every fold.
- ``\widetilde\lambda`` is then chosen with the same folds. Each fold corrects
  its own fused fit at the selected family and ``\lambda_0`` on its training
  residuals; the full-data fused fit is never used inside a fold.
- The selected settings are fitted on all the data: `fused` is the full-data
  fused fit and `solution` adds the full-data correction.

The usual limits apply. The grids come from the full data, and the folds are
reused for the sequential selection, so `cvm` and `cvm_correction` are not
unbiased estimates of the selected fit's error; independent test data are
needed for that. The selection compares target prediction only, and nothing
guarantees that transfer improves on a target-only fit: when the sources
differ more than the target sample can detect, borrowing can hurt (negative
transfer).

## Results

A [`TransferMultitaskSolution`](@ref) holds the reported target slopes `W` and
intercepts `intercept`, the fused estimate `fused`, the correction `correction`
(or `nothing`), the domain blocks `B`, the contrasts and the domain intercepts.
`coef(sol)` gives the target coefficients with the intercepts in the first row;
`predict(sol, X⁰new)` gives target predictions; with `domain = ℓ` both use
``\widehat B^\ell`` and its intercepts instead (`domain = 0` gives
``\widehat B^0``, not ``W``). Task-specific designs are passed to `predict` as a
vector of ``K`` matrices.

## Not included

Three extensions of the paper are not implemented: the target-certified
projection of Theorem 4 and Appendix E, which needs explicit calibration
inputs; inference for these estimators; and responses missing within a
domain. See `examples/17_transfer_multitask.jl` for one draw comparing
TMTL(Fused), TMTL(Debiased), a target-only fit and a pooled fit with domain
intercepts.
