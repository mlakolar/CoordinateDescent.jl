# Automatic Gaussian all-slope timing on20matched datasets per design

EMPIRICAL. The frozen production HC3/RLDPE policy now infers every slope:500moderate,500correlated,100benign. This matches hdi/EconML/sslasso target counts. Their native statistical recipes still differ. SIHR/hdm remain ten-target pipelines. Fixed-penalty matched-engine measurements are a separate experiment.

|Design|Targets|Valid/planned|Warmed automatic fit mean ± MCSE (s)|Separate initial CV mean (s)|Process median (s)|Max worker RSS (GiB)|
|---|---:|---:|---:|---:|---:|---:|
|gaussian-moderate|500|20/20|229.072 ± 2.461|0.353|266.306|0.896|
|gaussian-correlated|500|20/20|98.657 ± 0.769|0.182|125.709|0.875|
|gaussian-benign|100|20/20|1.031 ± 0.012|0.027|26.814|0.882|

```json
{
  "planned_datasets": 60,
  "valid_datasets": 60,
  "expected_target_rows": 22000,
  "target_rows": 22000,
  "unique_target_rows": 22000,
  "errors": 0,
  "primary_target_pairs": 600,
  "max_primary_target_list_difference": 0.009658625141836352,
  "projection_failures": 0,
  "max_noise_ratio": 1.4999998299030743
}
```

The process timer includes imports, small independent warmup, full automatic fit and a separate CV/initializer audit. It is not a pure single-fit cold-start measurement. Fit timers include each policy own tuning. Memory is a per-worker child-process peak. Statistical summaries are exploratory20-dataset checks, with dataset-cluster MCSE;22,000coefficients do not mean22,000independent replications. Primary ten-target estimates are compared with the saved ten-target calls; the old implementation generated projection seeds by requested position and vector length. That reproducibility defect was fixed in e13b40f by drawing physical-predictor seeds. Full ordered all-target output is unchanged, and a 60-dataset bridge passed before new disjoint validation. The old ten-target differences remain recorded here. No policy was retuned. Zero empirical MCSE at a boundary does not imply certainty.
