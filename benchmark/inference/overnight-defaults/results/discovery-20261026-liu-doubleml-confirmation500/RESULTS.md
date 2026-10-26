# Liu positive-control DoubleML independent confirmation

EMPIRICAL. The favorable published-regime control n1000,p201 is tested on500fresh confirmation datasets. The native framework, declared learners, nested5x5cross-fitting, two scores and two requested target roles are unchanged. This is not the paper bootstrap-CI replica. Failed other designs remain in the original twelve-case pilot.

|Score|Role|Complete/planned|Coverage ± MCSE (%)|Rejection ± MCSE (%)|Length|
|---|---|---:|---:|---:|---:|
|doubleml_sparse_instrument|active|500/500|95.80 ± 0.90|100.00 ± 0.00|0.3139|
|doubleml_sparse_instrument|null|500/500|96.00 ± 0.88|4.00 ± 0.88|0.2742|
|doubleml_sparse_nuisance_space|active|500/500|95.40 ± 0.94|100.00 ± 0.00|0.3447|
|doubleml_sparse_nuisance_space|null|500/500|96.40 ± 0.83|3.60 ± 0.83|0.2881|

Boundary-aware Wilson Monte Carlo bounds are in summary.json; availability is separate from conditional coverage. No production Julia logistic default is changed. Primary instrument gate applies only to this published regime; nuisance_space is prespecified secondary.

Native numerical checks: SEparity4.16e-17, maximum normal-pvalue error2.22e-16, score residual1.78e-17. Actual workerCPU4.064hours, maximum individual-workerRSS0.239GiB. Warning categories:{'FutureWarning': 150000}. See resource-summary.json and exact-job accounting.

Prospective primary instrument gate:

```json
{
  "valid": 500,
  "planned": 500,
  "active_lower95_mc": 0.9404003019069977,
  "null_upper95_mc": 0.05719346585087601,
  "passed": true
}
```
