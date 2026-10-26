# Experimental calibrated-loss solver checks

EMPIRICAL. Native RCAL versus a custom ProximalBase ScalarLoss through the existing Julia GLM proximal-Newton solver. This is not an inference coverage experiment and does not promote a logistic default.

```json
{
  "scope": "bounded",
  "planned_datasets": 20,
  "planned_policy_rows": 40,
  "rows": 40,
  "missing_dataset_reports": [],
  "errors": 0,
  "warning_files": 0,
  "all_converged": true,
  "max_julia_kkt": 1.980344583298771e-05,
  "max_reference_kkt": 6.06432621059264e-06,
  "matched_selected_index": 38,
  "reference_index_rows": 40,
  "max_coefficient_error": 6.192126133941489e-05,
  "max_predictor_error": 0.00016569472318538013,
  "max_objective_error": 5.806733427426991e-10,
  "max_cv_loss_error": 0.11266138608318954
}
```

The fixed-penalty experiment uses four designs with five saved development datasets each and two penalties; weights and offsets are controlled functions of observed predictors. The CV experiment uses actual fitted initial/target regressions and preliminary score offsets, the native five-fold/eight-level criterion, and both minimum/1SE policies. Original normalized weight units are preserved. Errors and warnings remain visible. The Julia solver is checked against the original objective and KKT conditions, not only coefficient differences. Native convergence tolerances allow small differences between numerical minimizers. CV selection differences or native failures must be investigated before substituting the solver.

Source: calibration-loss.jl and corresponding reference/check runners. All numerical work ran on Discovery compute nodes; no core package/default changes.
