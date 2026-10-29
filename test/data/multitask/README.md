# Convex reference solutions for the multitask penalties

`oracle.py` draws small multitask data sets with NumPy and solves

```
minimize  Σₜ cₜ ‖yₜ - β₀ₜ - Xₜbₜ‖² / (2nₜ) + λ Σⱼ ωⱼ Pⱼ(s₁ⱼb₁ⱼ, …, s_qⱼb_qⱼ)
```

with CVXPY and the interior-point solver Clarabel (gap and feasibility tolerances
1e-12), for the row ℓ1/ℓ∞ penalty (`Pⱼ = ‖·‖_∞`) and the multitask sparse group
(`Pⱼ = α‖·‖₁ + (1 - α)‖·‖₂`). It does not use HighDimensionalModels: the column
scales `sₜⱼ`, the task weights `cₜ`, λmax and the objective are computed from their
definitions in the documentation. `versions.csv` records the software versions.

Each case directory holds

- `X.csv` (a shared design) or `X1.csv`, `X2.csv`, … (one design per task), and
  `y1.csv`, `y2.csv`, … (the responses);
- `settings.csv`: the number of tasks, whether the design is shared, `row_norm`
  (`linf` or `l2`), `alpha`, `intercept`, `standardize_x`, `task_weights` (`equal`
  or `observations`), the reference `lambda_max` and the number of levels;
- `penalty_factors.csv`: the `ωⱼ`;
- for each level `k`: `level<k>.csv` (`lambda`, the reference `objective` and the
  solver's iterations), `B<k>.csv` (`p × q`) and `intercept<k>.csv`.

The tests compare the objective, the coefficients and the predictions of
`multitask_lasso` and `multi_lasso` with these solutions, and the λmax of the
automatic grids with `lambda_max`. Regenerate with

```
python -I oracle.py <output directory>
```

and copy the output here.
