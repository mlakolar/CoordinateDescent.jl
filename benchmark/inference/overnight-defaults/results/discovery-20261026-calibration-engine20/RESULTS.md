# Experimental calibrated-loss solver checks

EMPIRICAL. Native RCAL versus a custom ProximalBase ScalarLoss through the existing Julia GLM proximal-Newton solver. This is not an inference coverage experiment and does not promote a logistic default.

```json
{
  "scope": "fixed",
  "planned_datasets": 20,
  "planned_policy_rows": 40,
  "rows": 37,
  "missing_dataset_reports": [],
  "errors": 3,
  "warning_files": 0,
  "all_converged": true,
  "max_julia_kkt": 1.0190615536889714e-06,
  "max_reference_kkt": 5.77136208860651e-06,
  "parity_rows_passed": 37,
  "max_coefficient_error": 0.0001830651060565902,
  "max_predictor_error": 0.0012400693768532967,
  "max_objective_error": 2.0526300104073414e-09
}
```

The fixed-penalty experiment uses four designs with five saved development datasets each and two penalties; weights and offsets are controlled functions of observed predictors. The CV experiment uses actual fitted initial/target regressions and preliminary score offsets, the native five-fold/eight-level criterion, and both minimum/1SE policies. Original normalized weight units are preserved. Errors and warnings remain visible. The Julia solver is checked against the original objective and KKT conditions, not only coefficient differences. Native convergence tolerances allow small differences between numerical minimizers. CV selection differences or native failures must be investigated before substituting the solver.

Source: calibration-loss.jl and corresponding reference/check runners. All numerical work ran on Discovery compute nodes; no core package/default changes.
