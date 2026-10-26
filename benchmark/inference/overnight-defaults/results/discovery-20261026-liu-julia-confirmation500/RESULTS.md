# Julia logistic recipes on the Liu positive-control development data

EMPIRICAL.500matched independent confirmation datasets,n1000,p201,first active and last null role. Native automatic CV recipe and previously frozen restricted-LDPE expert recipe; no new tuning. Same input files as official DoubleML development100. No production logistic default is changed.

|Method|Role|Complete/planned|Coverage ± MCSE (%)|Rejection ± MCSE (%)|Length|
|---|---|---:|---:|---:|---:|
|julia_logistic_default|active|500/500|94.80 ±0.99|100.00 ±0.00|0.2970|
|julia_logistic_default|null|500/500|95.80 ±0.90|4.20 ±0.90|0.2768|
|julia_logistic_restricted_expert|active|500/500|96.60 ±0.81|100.00 ±0.00|0.3449|
|julia_logistic_restricted_expert|null|500/500|94.60 ±1.01|5.40 ±1.01|0.3095|
