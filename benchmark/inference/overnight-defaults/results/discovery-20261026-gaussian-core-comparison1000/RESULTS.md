# Current Gaussian policy versus native alternatives:1000datasets per core design

EMPIRICAL. Same3000response datasets and ten primary statistical targets per dataset. Julia uses the current canonical-target Gaussian HC3 policy. Native hdi/EconML/sslasso compute all slopes internally;SIHR/hdm request ten targets. Their native statistical recipes differ. This is not matched-penalty engine timing.

|Design|Method|Finite/strict valid per1000|Active coverage ± MCSE (%)|TypeI ± MCSE (%)|Power (%)|Active length|
|---|---|---:|---:|---:|---:|---:|
|gaussian-moderate|Julia_current|1000/1000|96.18 ±0.29|2.68 ±0.23|95.84|0.5213|
|gaussian-moderate|hdi|1000/1000|78.40 ±0.67|4.30 ±0.29|99.62|0.3248|
|gaussian-moderate|EconML|1000/1|70.42 ±0.76|3.36 ±0.26|99.82|0.2865|
|gaussian-moderate|sslasso|1000/0|76.32 ±0.70|4.80 ±0.31|99.98|0.3218|
|gaussian-moderate|SIHR|1000/1000|71.84 ±0.77|2.72 ±0.23|99.82|0.2926|
|gaussian-moderate|hdm|1000/1000|81.30 ±0.63|5.44 ±0.33|99.26|0.3400|
|gaussian-correlated|Julia_current|1000/1000|96.48 ±0.27|4.24 ±0.30|15.26|1.9648|
|gaussian-correlated|hdi|1000/1000|81.30 ±0.59|7.40 ±0.36|31.82|0.8275|
|gaussian-correlated|EconML|1000/727|74.50 ±0.72|6.56 ±0.35|30.38|0.7624|
|gaussian-correlated|sslasso|1000/0|4.86 ±0.29|14.44 ±0.61|25.30|0.4751|
|gaussian-correlated|SIHR|1000/1000|78.32 ±0.76|3.66 ±0.27|20.48|0.8386|
|gaussian-correlated|hdm|1000/1000|94.32 ±0.37|6.48 ±0.39|34.02|1.2647|
|gaussian-benign|Julia_current|1000/1000|94.44 ±0.35|4.16 ±0.29|100.00|0.2189|
|gaussian-benign|hdi|1000/1000|91.18 ±0.44|5.02 ±0.30|100.00|0.1921|
|gaussian-benign|EconML|1000/1000|91.52 ±0.43|5.20 ±0.31|100.00|0.1837|
|gaussian-benign|sslasso|1000/1000|93.74 ±0.37|4.74 ±0.31|100.00|0.2096|
|gaussian-benign|SIHR|1000/1000|92.88 ±0.39|4.96 ±0.32|100.00|0.1935|
|gaussian-benign|hdm|1000/1000|93.90 ±0.36|5.26 ±0.32|100.00|0.1904|

Native coverage is conditional on finite returned intervals, including warned/uncertified fits. Strict availability separately rejects exposed convergence warnings and requires native sslassoQPcertificates. Warnings can concern unselectedCVlevels; strict rejection is conservative,not a final-model failure theorem. Native outputs/formulas were not repaired. Native sslasso adds a bias allowance to intervals; its native pvalues need not be dual to those expanded intervals. A failed certificate cannot be ignored because coverage looks favorable. All target-role/dataset records remain retained.

The first analyzer had a planned-fit denominator typo from an earlier seven-task timing protocol. Native-only five methods x3designs x1000 =15000,all present. The original checks are retained beside corrected checks. No method was dropped.

Actual native workerCPU 135.67hours and maxindividualworkerRSS 0.776GiB. Current-source own refreshCPU 2.29hours and maxworkerRSS 0.913GiB. Warm all-slope20dataset timings and initialCV costs are reported separately;ten-target own times are not all-slope speedups.

This completes the native comparison in three core Gaussian designs. Seven stress-design native arms and refreshed matched-statistical-recipe campaigns are not complete. The ten-design disjoint500validation remains the current automatic-policy calibration evidence. No logistic/QPproductiondefault is promoted.

## Native pipeline timing and memory

Native own tuning included. Julia all-slope20dataset timing is reported separately because its ten-target1000arm has a different target count. Per-process peakRSS,not totaljobmemory.

|Design|Native method|Inferred slopes|Mean fit ± datasetMCSE (s)|Peak workerRSS (GiB)|
|---|---|---:|---:|---:|
|gaussian-moderate|hdi|500|192.548 ±0.389|0.427|
|gaussian-moderate|EconML|500|57.695 ±0.113|0.259|
|gaussian-moderate|sslasso|500|48.023 ±0.092|0.364|
|gaussian-moderate|SIHR|10|20.761 ±0.063|0.753|
|gaussian-moderate|hdm|10|3.245 ±0.015|0.260|
|gaussian-correlated|hdi|500|54.211 ±0.096|0.411|
|gaussian-correlated|EconML|500|22.349 ±0.035|0.258|
|gaussian-correlated|sslasso|500|18.407 ±0.028|0.268|
|gaussian-correlated|SIHR|10|11.807 ±0.031|0.701|
|gaussian-correlated|hdm|10|9.379 ±0.090|0.248|
|gaussian-benign|hdi|100|7.686 ±0.014|0.280|
|gaussian-benign|EconML|100|0.876 ±0.001|0.249|
|gaussian-benign|sslasso|100|0.147 ±0.001|0.220|
|gaussian-benign|SIHR|10|16.742 ±0.027|0.776|
|gaussian-benign|hdm|10|0.448 ±0.001|0.248|

Paired dataset-level differences andMCSE for coverage,rejection and length are retained in paired-differences.json. No independent-target approximation is used. Native convergence/certificate caveats apply to these conditional differences.
