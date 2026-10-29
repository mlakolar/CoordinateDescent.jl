# Convex reference solutions for nuclear-plus-sparsity regression

`oracle.py` draws small data sets with NumPy and solves

```
minimize  ‖Y - 1β₀' - XB‖²_F / (2n) + λ_rank ‖B‖_* + λ_sparse P(B)
```

with CVXPY and the interior-point solver Clarabel (the nuclear norm through its
semidefinite cone; gap and feasibility tolerances 1e-11), with the intercepts
optimized explicitly. `P(B) = Σⱼ wⱼ Σₜ |Bⱼₜ|` for `sparsity = entry` and
`Σⱼ wⱼ ‖B[j, :]‖₂` for `sparsity = row`. It does not use HighDimensionalModels.
`versions.csv` records the software versions; every level reached Clarabel's
optimal status (the script raises otherwise).

Each case directory holds `X.csv`, `Y.csv`, `weights.csv`, `settings.csv`
(`sparsity`, `intercept`, the marginal thresholds `lambda_rank_max` and
`lambda_sparse_max`, the number of levels) and, for each level `k`, `level<k>.csv`
(`lambda_rank`, `lambda_sparse`, the reference `objective`, the solver's
iterations), `B<k>.csv` and `intercept<k>.csv`. Level 1 of `entry` lies in the
zero region of the pair although both levels are below their marginal thresholds.
The `entry-wide` case has more predictors than observations, so only its
objective and fitted values are identified.

Regenerate with `python -I oracle.py <output directory>` and copy the output here.
