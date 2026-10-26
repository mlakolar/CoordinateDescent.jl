# Experimental calibrated-loss solver checks

EMPIRICAL. Native RCAL versus a custom ProximalBase ScalarLoss through the existing Julia GLM proximal-Newton solver. This is not an inference coverage experiment and does not promote a logistic default.

```json
{
  "scope": "cv",
  "planned_datasets": 20,
  "planned_policy_rows": 40,
  "rows": 18,
  "missing_dataset_reports": [
    [
      "binomial-moderate",
      1
    ],
    [
      "binomial-moderate",
      2
    ],
    [
      "binomial-moderate",
      3
    ],
    [
      "binomial-moderate",
      4
    ],
    [
      "binomial-moderate",
      5
    ],
    [
      "binomial-largen",
      1
    ],
    [
      "binomial-largen",
      2
    ],
    [
      "binomial-largen",
      3
    ],
    [
      "binomial-largen",
      4
    ],
    [
      "binomial-largen",
      5
    ]
  ],
  "errors": 1,
  "warning_files": 15,
  "all_converged": true,
  "max_julia_kkt": 1.980344583298771e-05,
  "max_reference_kkt": 5.9981189544922e-06,
  "matched_selected_index": 18,
  "reference_index_rows": 18,
  "max_coefficient_error": 6.192126133841569e-05,
  "max_predictor_error": 0.000145261647071509,
  "max_objective_error": 5.806733427426991e-10,
  "max_cv_loss_error": 160.8400875154214
}
```

The fixed-penalty experiment uses four designs with five saved development datasets each and two penalties; weights and offsets are controlled functions of observed predictors. The CV experiment uses actual fitted initial/target regressions and preliminary score offsets, the native five-fold/eight-level criterion, and both minimum/1SE policies. Original normalized weight units are preserved. Errors and warnings remain visible. The Julia solver is checked against the original objective and KKT conditions, not only coefficient differences. Native convergence tolerances allow small differences between numerical minimizers. CV selection differences or native failures must be investigated before substituting the solver.

Source: calibration-loss.jl and corresponding reference/check runners. All numerical work ran on Discovery compute nodes; no core package/default changes.
