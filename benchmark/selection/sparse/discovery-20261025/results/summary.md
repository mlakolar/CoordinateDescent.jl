# Discovery sparse SLOPE results

Medians across seven process replications; each process contributes three timed batch medians.

| Case | Native ms | Baseline ms | SLOPE.jl ms | R ms |
|---|---:|---:|---:|---:|
| dense | 3.225 | 3.450 | 7.778 | 13.750 |
| wide | 6.132 | 6.751 | 14.480 | 19.400 |
| correlated | 4.087 | 4.367 | 9.684 | 15.286 |
| original-sparse | 3.524 | 12.036 | 7.565 | 17.000 |
| sparse-0.1pct | 9.923 | 55.578 | 35.763 | 44.333 |
| sparse-1pct | 10.874 | 50.072 | 19.507 | 29.500 |
| sparse-10pct | 7.519 | 14.081 | 20.782 | 32.500 |
| centered-scaled | 11.529 | 54.669 | 33.096 | 40.333 |
| long-weak-path | 262.675 | 1580.616 | 8034.789 | 6007.000 |
| large-centered | 17.920 | 305.661 | 47.382 | 55.000 |
| storage-canary | 28.636 | omitted | 50.550 | 62.000 |

Validation:
```json
{
  "timing_processes": 28,
  "replications": 7,
  "timing_rows": 903,
  "accuracy_rows": 301,
  "memory_accuracy_rows": 77,
  "memory_max_relative_residual": 4.118860523483203e-08,
  "memory_processes": 77,
  "profiles": 14,
  "max_relative_residual": 6.365903247596005e-08,
  "max_relative_objective_difference": 1.509903313490213e-14,
  "dense_controls_pass_5pct": true,
  "original_sparse_parity_with_SLOPE_jl": true,
  "largest_design_bytes": 637184,
  "largest_workspace_bytes": 1517840,
  "largest_dense_equivalent_bytes": 1600000000
}
```

Fresh-process peak RSS includes the interpreter, imports, JIT and fitting. It is not solely matrix storage. The 1.6 GB dense-equivalent case omits the old densifying backend by design.
