# Fixed-matrix construction diagnostics

These are developmental numerical checks. They do not estimate statistical FDR/power or replicated runtime.

| Source | Setting | Attempts | Returned | Feasible | Max full-joint gradient ratio |
|---|---|---:|---:|---:|---:|
| adapter | default | 16 | 16 | 16 | 0.0 |
| adapter | package-default | 32 | 2 | 2 | 0.0 |
| adapter | provisional | 32 | 32 | 32 | 4.811635394167137e-09 |
| adapter | robust | 32 | 32 | 32 | 4.811640317901907e-09 |
| adapter | tighter | 32 | 32 | 32 | 4.8606786881963486e-11 |
| direct | package-default | 32 | 32 | 32 | 4.743476278287382e-05 |
| direct | provisional | 32 | 32 | 32 | 4.811633760736488e-09 |
| direct | robust | 32 | 32 | 32 | 4.811635717490998e-09 |
| direct | tighter | 32 | 32 | 32 | 4.85954573019237e-11 |
| CVXPY-CLARABEL | oracle | 20 | 20 | 20 | None |

The full-joint gradient ratio divides each component by the sum of the corresponding positive inverse-matrix terms (at least one); it differs from the production max-term normalization. Original feasibility, not a backend perturbed matrix, is assessed. Every raw row, failed adapter error, oracle status and objective gap is retained. The conic oracle is a numerical comparator, not an exact proof.

| Matrix | Strategy | Adapter/oracle relative objective difference | Oracle status |
|---|---|---:|---|
| 1 | mvr | -5.64835e-10 | optimal |
| 1 | maxent | -3.09351e-10 | optimal |
| 2 | mvr | -7.25465e-11 | optimal |
| 2 | maxent | -3.96882e-11 | optimal |
| 3 | mvr | -7.5444e-11 | optimal |
| 3 | maxent | -1.92227e-10 | optimal |
| 4 | mvr | -3.9632e-11 | optimal |
| 4 | maxent | -2.17127e-11 | optimal |
| 5 | mvr | -3.93643e-11 | optimal |
| 5 | maxent | -1.44081e-10 | optimal |
| 6 | mvr | -1.68644e-10 | optimal |
| 6 | maxent | -6.2327e-10 | optimal |
| 7 | mvr | -1.36054e-11 | optimal |
| 7 | maxent | -6.58528e-11 | optimal |
| 8 | mvr | -4.84432e-13 | optimal |
| 8 | maxent | -1.48127e-10 | optimal |
| 9 | mvr | -2.3981e-13 | optimal |
| 9 | maxent | -1.64533e-10 | optimal |
| 16 | mvr | -3.96304e-11 | optimal |
| 16 | maxent | -2.17127e-11 | optimal |
