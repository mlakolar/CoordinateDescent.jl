# Convex reference solutions for low-rank-plus-sparse regression

`oracle.py` draws small data sets with NumPy and solves

```
minimize over L, S, β₀   ‖Y - 1β₀' - X(L + S)‖²_F / (2n) + λ_rank ‖L‖_* + λ_sparse P(S)
```

with CVXPY and the interior-point solver Clarabel (the nuclear norm through its
semidefinite cone), with `L` and `S` as separate variables and the intercepts optimized
explicitly. `P(S) = Σⱼ wⱼ Σₜ |Sⱼₜ|` for `sparsity = entry` and `Σⱼ wⱼ ‖S[j, :]‖₂` for
`sparsity = row`. It does not use HighDimensionalModels. Every level reached Clarabel's
optimal status at tolerance 1e-10 (recorded in `level<k>.csv`).

Each case directory holds `X.csv`, `Y.csv`, `weights.csv`, `settings.csv` (`sparsity`,
`intercept`, `design`, the thresholds at the zero pair `lambda_rank_reference` and
`lambda_sparse_reference`, the number of levels) and, for each level `k`,
`level<k>.csv`, `L<k>.csv`, `S<k>.csv`, `B<k>.csv = L + S` and `intercept<k>.csv`.
Only the fitted values and the objective are identified in general, and `B` on
full-rank designs; the components are one optimal split. Cases: `entry` (three
pairs), `row` (nonuniform weights), `entry-noint`, `entry-duplicated` (two equal
columns, so `B` is not identified) and `row-wide` (`p > n`).

Regenerate with `python -I oracle.py <output directory>` and copy the output here.
