# Automatic tuning follow-up

Automatic policy frozen before development/holdout execution. Fresh holdout:100 datasets per case. Nominal marginal coverage95%, unadjusted typeI5%. Errors/uncertainty remain in machine-readable files.

| Holdout case | Active coverage | Null coverage | Type I | Power | Length active/null | Median automatic seconds |
|---|---:|---:|---:|---:|---:|---:|
| binomial-correlated | 0.850 ± 0.017 | 0.956 ± 0.010 | 0.044 | 0.158 | 1.457/1.479 | 0.185 |
| binomial-moderate | 0.562 ± 0.021 | 0.964 ± 0.009 | 0.036 | 0.856 | 0.587/0.585 | 0.875 |
| gaussian-correlated | 0.730 ± 0.023 | 0.924 ± 0.011 | 0.076 | 0.292 | 0.757/0.768 | 0.230 |
| gaussian-moderate | 0.776 ± 0.022 | 0.964 ± 0.010 | 0.036 | 0.998 | 0.286/0.286 | 0.960 |

± denotes one dataset-level Monte Carlo standard error, not a confidence limit. Targets are five active and five boundary-null coefficients.

| Speed case | Engine | Valid/expected | Median fit seconds | Median process seconds | Max RSS MiB |
|---|---|---:|---:|---:|---:|
| moderate | ours | 10/10 | 0.0347 | 18.255 | 761.6 |
| moderate | hdi | 10/10 | 1.3690 | 3.276 | 262.1 |
| moderate | econml | 10/10 | 0.8375 | 3.245 | 253.8 |
| moderate | oursqp | 10/10 | 0.0242 | 13.391 | 747.6 |
| moderate | sslasso | 10/10 | 8.9060 | 10.813 | 269.3 |
| moderate | SIHR | 10/10 | 5.4020 | 7.560 | 680.5 |
| moderate | oursqp10 | 10/10 | 0.0056 | 18.331 | 783.7 |
| correlated | ours | 10/10 | 0.0602 | 13.991 | 730.8 |
| correlated | hdi | 10/10 | 1.5635 | 3.498 | 258.4 |
| correlated | econml | 10/10 | 1.0789 | 3.421 | 252.4 |
| correlated | oursqp | 10/10 | 0.0465 | 12.989 | 742.6 |
| correlated | sslasso | 10/10 | 179.2665 | 181.138 | 258.4 |
| correlated | SIHR | 10/10 | 5.0420 | 7.238 | 623.1 |
| correlated | oursqp10 | 10/10 | 0.0046 | 14.997 | 757.9 |

Nodewise engines:500 slopes, fixed common penalties, residual-DF noise estimation. QP engines:500 projection rows at common mu, with numerical certificates. SIHR and oursqp10:10-target complete pipelines with different inferential procedures; SIHR retains default variance inflation and filtering. Tuning time is outside fixed-penalty speed timings.

Validation: 800/800 automatic fits converged; 140/140 benchmark fits valid. Full1000-replication run not submitted.

## Interpretation

**Do not launch the full comparison yet.** Automatic prediction CV removes required manual tuning and adapts to correlation, but active-coefficient intervals remain substantially below 95% coverage, including the moderate designs. Numerical convergence does not resolve this calibration failure. The policy was frozen before seeing these results; it has not been retuned against the holdout.

## Bias and noise diagnostics

| Holdout Gaussian case | Estimated-noise coverage | Known-noise coverage | Oracle initializer + known noise coverage | Signed bias / SE | Empirical SD / RMS SE |
|---|---:|---:|---:|---:|---:|
| gaussian-moderate | 0.776 | 0.770 | 0.954 | -1.141 | 1.098 |
| gaussian-correlated | 0.730 | 0.702 | 0.972 | -1.252 | 0.907 |

Oracle calculations use truth and are diagnostics, not feasible competing methods. Noise estimation alone does not explain the observed failures. Prediction-optimal penalization leaves appreciable inferential bias.

## Numerical comparison and current source

```json
{
  "parity": {
    "ours-hdi": {
      "beta_max_abs": 5.107710111973418e-06,
      "se_max_abs": 8.422579338418501e-07,
      "se_max_relative": 3.740473689561874e-06
    },
    "ours-econml": {
      "beta_max_abs": 8.982307897476005e-09,
      "se_max_abs": 0.0012029195237056034,
      "se_max_relative": 0.005420098351107239,
      "se_ratio_spread": 9.719199667301837e-09
    },
    "oursqp-sslasso": {
      "theta_max_abs": 2.495286555159737e-08
    }
  },
  "qp_certificates": {
    "oursqp": {
      "max_abs_gap": 5.798014957036912e-09,
      "max_residual": 0.3525509370883074
    },
    "sslasso": {
      "max_abs_gap": 1.94999572045162e-12,
      "max_residual": 0.352550935282715
    }
  },
  "current_source_bridge": {
    "datasets": 12,
    "max_beta_difference": 2.195466031196247e-14,
    "max_se_difference": 7.049916206369744e-15,
    "max_initial_lambda_difference": 4.163336342344337e-17,
    "max_node_lambda_difference": 8.326672684688674e-17
  }
}
```

EconML divides residual sum of squares by n minus the number of selected slopes; the Julia/hdi adapter subtracts an additional intercept degree of freedom. The nearly constant SE ratio is therefore expected. All QP rows satisfy the declared feasibility and optimality tolerances. The authors reference includes the documented diagonal fast-path correction.

Measured source is commit f7b587790f46b6ea4c05f082971d582db0ddb435 with the original pinned solver dependencies. After merging master, inference CV explicitly retains gaptol=0 unless the user overrides it. The current-source bridge uses ProximalBase b6c949f and ProximalSolvers 73a1c8e; all 115 inference/golden/tuning checks passed, and three independent datasets per case were compared with the saved holdout fits.

## Runtime and resource interpretation

The Julia fit kernels are faster under equivalent fixed tuning. Fresh Julia process startup/JIT dominates these small problems: measured median complete process times are roughly 13–18 seconds versus roughly 3–4 seconds for hdi/EconML. Persistent workers would change this comparison and were not timed here. Peak Julia process RSS is about 0.8 GiB versus about 0.25 GiB for hdi/EconML; eight workers fit within the 16 GiB allocation.

The exact timing workload extrapolated to 1000 datasets per Gaussian case gives the following conditional estimate. It includes component-only QP tasks and the separate ten-target pipelines; it is not an estimate for the complete four-case defaults-and-matched inference comparison.

```json
{
  "scope": "Exact seven timing tasks per dataset, 1000 datasets per Gaussian case; component and complete-pipeline tasks differ. Excludes native-default competitors and binomial cases; not the full comparison.",
  "warmed_fit_cpu_hours": 58.41792465439377,
  "fresh_process_cpu_hours": 89.69037780089356,
  "ideal_fresh_process_wall_hours_8cpu": 11.211297225111695,
  "planning_wall_hours_8cpu_25pct_margin": 14.014121531389618,
  "cpus": 8,
  "memory_gib": 16,
  "max_process_rss_mib": 783.71484375
}
```

The earlier full-comparison planning budget was about 50 allocated wall-hours at eight CPUs and 16 GiB for the original ten-target task. That remains a provisional reference, not a refreshed validated budget for revised defaults or different target counts. Freeze an inferentially defensible automatic rule and the final comparison protocol before requesting a firm full-study budget. Queue waiting is excluded.

## Validation and recovery

All 800 automatic fits converged and all 140 timing fits produced valid outputs. The first benchmark allocation hit its eight-minute wall limit with two sslasso fits unfinished; only those two were recovered. Successful fits were reused and interrupted logs retained. The 115-check source canary and current-source canary both passed. Full GitHub correctness CI and documentation passed on the preceding merged source; the final explicit CV stopping-rule change is also covered by the current-source Discovery tests. The full 1000-replication comparison was not submitted.
