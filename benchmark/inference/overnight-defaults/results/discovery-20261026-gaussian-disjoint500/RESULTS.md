# Disjoint Gaussian validation after target-stream correction

EMPIRICAL. Frozen statistical tuning policy, canonical physical-target random streams, ten prespecified Gaussian designs, 500 datasets/design. Base1000000000 is disjoint from the audited historical seed blocks. Source/protocol3eefe3b was pushed before generation. No tuning against this validation.

|Design|Active coverage ± dataset MCSE (%)|Type I ± MCSE (%)|Power ± MCSE (%)|Active interval length ± MCSE|Warm10-target fit (s)|
|---|---:|---:|---:|---:|---:|
|gaussian-benign|94.80 ± 0.45|4.60 ± 0.43|100.00 ± 0.00|0.2193 ± 0.0004|0.055|
|gaussian-correlated|96.84 ± 0.39|3.76 ± 0.42|13.80 ± 0.79|1.9649 ± 0.0103|2.005|
|gaussian-heavy|96.40 ± 0.38|2.72 ± 0.32|99.40 ± 0.16|0.3947 ± 0.0037|8.029|
|gaussian-hetero|96.16 ± 0.38|3.56 ± 0.38|99.84 ± 0.08|0.4006 ± 0.0011|8.154|
|gaussian-hetero-target|96.24 ± 0.39|3.00 ± 0.34|99.76 ± 0.10|0.4236 ± 0.0014|8.557|
|gaussian-independent|95.96 ± 0.39|2.76 ± 0.33|98.64 ± 0.23|0.4673 ± 0.0014|3.896|
|gaussian-largen|95.64 ± 0.42|5.68 ± 0.48|100.00 ± 0.00|0.2336 ± 0.0003|2.631|
|gaussian-moderate|97.12 ± 0.36|2.08 ± 0.30|96.60 ± 0.38|0.5193 ± 0.0018|4.168|
|gaussian-random|95.56 ± 0.40|2.68 ± 0.34|97.60 ± 0.30|0.5029 ± 0.0015|4.424|
|gaussian-weak|95.88 ± 0.38|3.32 ± 0.37|31.48 ± 0.89|0.4940 ± 0.0014|4.287|

Pooled engineering gates passed:10/10. Unadjusted role flags:10. See per-slot.json and role-review.json. Tolerances are active lower95%MC bound>=92.5%,null upper95%MC bound<=7.5%,>=99%complete valid fits. These are not proof of uniform95%coverage. Correlated and weak-signal power and interval length remain part of the assessment.

All5000datasets and50000target records are finite,converged and unique; zero numerical pvalue/interval discrepancies beyond1e-12. Consumed workerCPU:7.38hours; maximum individual workerRSS:0.891GiB. See accounting for job13455264. Fit timing includes initial CV and projections; first fit compilation and full process startup are separately scoped in fit-time-summary.json and resource receipts. These ten-target times are not all-slope competitor speedups.

Earlier seed-overlap and target-stream defects remain documented with their saved outputs. Historical numerical results were not deleted. The new confirmation addresses disjoint data streams and canonical target requests without changing tuning. Logistic/QP inference remains experimental. Marginal normal intervals only; joint/bootstrap calibration was not evaluated.
