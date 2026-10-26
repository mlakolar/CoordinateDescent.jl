# Calibrated-loss recession diagnostics

Analytical criterion. For binary responses and nonnegative weights, take a direction u = intercept + X v. If u is nonnegative at every positive-response observation and mean(W (1-y) u) + lambda ||v||1 is negative, the penalized calibrated loss tends to minus infinity along that direction. The exponential positive-response terms remain bounded, while the negative-response linear term and penalty have strictly negative net slope. Finite offsets do not change this recession criterion.

EMPIRICAL. Linear programs search for directions with bounded L1 norm. Each reported witness is recomputed in the original predictor coordinates. A small upward intercept adjustment gives positive-response direction slack; the resulting net slope must remain below -1e-8. These are numerical witnesses with saved coefficients, not a formal interval-arithmetic proof. A failed witness search is not proof of a finite minimizer.

```json
{
  "datasets": 20,
  "linear_programs": 1000,
  "successful_LPs": 1000,
  "numerical_recession_witnesses": 196,
  "native_uncertified_fixed_references": 3,
  "uncertified_fixed_references_with_witness": 0
}
```

|Case|Scope|LPs|Numerical negative witnesses|Datasets with witnesses|Most negative slope|
|---|---|---:|---:|---:|---:|
|binomial-benign|actual_cv|240|0|0|None|
|binomial-benign|fixed_reference|10|0|0|None|
|binomial-largen|actual_cv|240|76|5|-0.02415903951833779|
|binomial-largen|fixed_reference|10|0|0|None|
|binomial-moderate|actual_cv|240|120|5|-0.09095736266121807|
|binomial-moderate|fixed_reference|10|0|0|None|
|liu-published|actual_cv|240|0|0|None|
|liu-published|fixed_reference|10|0|0|None|

Actual-CV scope includes the full sample and all five training folds across eight penalties. Fixed-reference scope checks the two penalties in the independent RCAL/Julia engine experiment. Offsets/weights/targets are retained with those studies. A solver should reject an ill-posed level; increasing its iteration count cannot create a finite optimum. This diagnostic does not by itself identify why every slow solve is slow and does not establish inferential coverage. No package default was changed.
