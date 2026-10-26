# Julia logistic recipes on the Liu positive-control development data

EMPIRICAL.100matched development datasets,n1000,p201,first active and last null role. Native automatic CV recipe and previously frozen restricted-LDPE expert recipe; no new tuning. Same input files as official DoubleML development100. No production logistic default is changed.

|Method|Role|Complete/planned|Coverage ± MCSE (%)|Rejection ± MCSE (%)|Length|
|---|---|---:|---:|---:|---:|
|julia_logistic_default|active|100/100|92.00 ±2.73|100.00 ±0.00|0.2969|
|julia_logistic_default|null|100/100|94.00 ±2.39|6.00 ±2.39|0.2762|
|julia_logistic_restricted_expert|active|100/100|94.00 ±2.39|100.00 ±0.00|0.3445|
|julia_logistic_restricted_expert|null|100/100|99.00 ±1.00|1.00 ±1.00|0.3085|
