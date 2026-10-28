# Convex reference solutions for the dirty multitask model

`oracle.py` draws small multitask data sets with NumPy and solves the two-component
problem

```
minimize over A, S, β₀   Σₜ cₜ ‖yₜ - β₀ₜ - Xₜ(aₜ + sₜ)‖² / (2nₜ)
                          + Σⱼ ωⱼ (λ_shared ‖(sₜⱼ Aₜⱼ)ₜ‖ + λ_specific Σₜ |sₜⱼ Sₜⱼ|)
```

with CVXPY and the interior-point solver Clarabel, with `A` and `S` as separate
variables (not the single-`B` reduction the package uses) and both fixed at zero where
a column is constant within a task. `‖·‖` is the row ℓ∞ norm (`shared_norm = linf`)
or the row ℓ2 norm (`l2`). The column scales `sₜⱼ`, the task weights `cₜ` and the
objective follow their definitions in the documentation; the script does not use
HighDimensionalModels. Every level reached Clarabel's optimal status; `level<k>.csv`
records the tolerance used (1e-10, with 1e-9 as a fallback, since the split into `A`
and `S` is not unique).

Each case directory holds the designs (`X.csv` for a shared design, or `X1.csv`, …),
the responses `y1.csv`, …, `penalty_factors.csv`, `settings.csv` (`tasks`, `shared`,
`shared_norm`, `intercept`, `standardize_x`, `task_weights`, `levels`) and, for each
level `k`, `level<k>.csv` (`lambda_shared`, `lambda_specific`, the reference
`objective`, the solver's iterations and tolerance), `A<k>.csv`, `S<k>.csv`,
`B<k>.csv = A + S` and `intercept<k>.csv`. Only `B`, the intercepts, the fitted values
and the objective are identified in general; the components are one optimal split.

Cases: a shared design (`shared-linf`, ratios 2 and 1.5), task-specific designs with
weights by observations, nonuniform factors and a column constant in one task
(`tasks-linf`), the two ℓ∞ regime endpoints without standardization
(`tasks-linf-endpoints`: ratio 1, row ℓ1/ℓ∞; ratio 3 = q, the entrywise lasso), the ℓ2
variant (`shared-l2`), and the ℓ2 variant without intercept and with a zero penalty
factor (`tasks-l2-zero-factor`).

Regenerate with `python -I oracle.py <output directory>` and copy the output here.
