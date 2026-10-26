# Partialled adaptive QP with full Gaussian score and HC3

EMPIRICAL. Development screen:100datasets/case, fourcandidate recipes. This is a new experimental partialled-score estimator,not original JM/sslasso or a changed productionQPdefault.

|Design|Active coverage ± datasetMCSE (%)|TypeI ± MCSE (%)|Power (%)|Active length|
|---|---:|---:|---:|---:|
|gaussian-benign|95.60 ±1.09|4.40 ±0.97|100.00|0.2084|
|gaussian-correlated|97.60 ±0.77|0.80 ±0.39|10.20|1.8435|
|gaussian-heavy|95.60 ±0.83|4.00 ±0.90|98.40|0.3467|
|gaussian-hetero|96.80 ±0.79|3.80 ±0.84|100.00|0.3489|
|gaussian-hetero-target|94.40 ±1.07|4.80 ±0.95|100.00|0.3694|
|gaussian-independent|94.20 ±1.04|3.40 ±0.76|99.80|0.3810|
|gaussian-largen|95.80 ±0.91|4.80 ±0.90|100.00|0.1934|
|gaussian-moderate|94.80 ±1.01|3.40 ±0.81|98.40|0.4281|
|gaussian-random|95.60 ±0.97|4.00 ±0.90|99.80|0.4126|
|gaussian-weak|96.60 ±0.86|2.40 ±0.65|45.00|0.3995|

All requested fits complete and valid. Search-warning receipts:8000;these are failed deeper search levels,not selecteduncertifiedsolutions. Primaryunadjustedroleflags:59. SelectedQPcertificates apply to the residualized design; no proof of original full-design JMconstraints or uniform coverage.

ConsumedCPU1.011hours; maxindividualworkerRSS1.226GiB. Root moment residual/finite-difference derivative/directGaussian solution checks are enforced in the runner. No truth-basedvarianceinflation. Conventionalone-step/refit failures remain in their separate tables. No1000repQPcampaign has been launched. See strict summaries/roles,source and receipts.
