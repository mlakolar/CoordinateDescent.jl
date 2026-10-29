# Reference solutions for joint-sparse transfer multitask learning

`oracle.py` (CVXPY with Clarabel, independent of the package) writes one directory per
case. Domain `0` is the target and domains `1..L` the sources; `oracle.py`'s docstring
gives the fused objective, the correction objective, `lambda_max` and the contrast
weights.

- `settings.csv`: `sources` (L), `responses` (K), `predictors` (p), `nobs` (target
  first, `;`-separated), `task_designs`, `intercept`, `contrast_weights` (`sqrt`,
  `linear` or `vector`), `lambda_max`, `levels`, and `identified` (false when the
  target design has a constant column, so the blocks need not be unique).
- `contrast_weights.csv`: the values `a_1..a_L` used.
- Data: `X<l>.csv` (`n_l × p`), or `X<l>_<k>.csv` per response with task-specific
  designs; `Y<l>.csv` (`n_l × K`).
- Level `i`: `level<i>.csv` (`lambda0`, the fused `objective`, Clarabel `iterations`
  and `tolerance`, then `lambda_tilde`, `lambda_tilde_max`, `correction_objective`,
  `correction_iterations`, `correction_tolerance`); `B<l>_<i>.csv` (`p × K`) and
  `intercept<l>_<i>.csv` (`1 × K`) per domain; `W<i>.csv`, the fused estimate
  `Σ (n_l/N) B_l` of the reference solution; `D<i>.csv` and
  `correction_intercept<i>.csv`, the correction computed from that `W` at
  `lambda_tilde = 0.3 lambda_tilde_max`.

Each problem is solved at tolerance 1e-12, or 1e-11 where Clarabel cannot certify
1e-12; the tolerance used is recorded. `versions.csv` lists the software versions.
