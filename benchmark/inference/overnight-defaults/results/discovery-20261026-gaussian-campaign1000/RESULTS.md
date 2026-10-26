# Gaussian production-policy arm: 1,000 fresh datasets per design

EMPIRICAL. Frozen Gaussian HC3 policy; 10 designs, 10,000 response datasets and 100,000 marginal coefficient intervals. All fits were complete, finite and converged. This is the own-policy arm; the expanded competitor comparison is not complete. The policy was frozen before the independent 500-per-design confirmation and was not retuned for this campaign.

|Design|Active coverage ± MCSE (%)|Null type I ± MCSE (%)|Power ± MCSE (%)|Active interval length ± MCSE|Warm fit mean (s)|
|---|---:|---:|---:|---:|---:|
|gaussian-benign|94.44 ± 0.35|4.16 ± 0.29|100.00 ± 0.00|0.2189 ± 0.0003|0.057|
|gaussian-correlated|96.48 ± 0.27|4.24 ± 0.30|15.26 ± 0.57|1.9648 ± 0.0072|2.212|
|gaussian-heavy|96.10 ± 0.28|3.52 ± 0.27|98.90 ± 0.18|0.3947 ± 0.0029|8.465|
|gaussian-hetero|96.62 ± 0.26|3.74 ± 0.28|99.80 ± 0.06|0.4021 ± 0.0008|8.284|
|gaussian-hetero-target|96.64 ± 0.27|3.66 ± 0.28|99.66 ± 0.08|0.4246 ± 0.0010|8.480|
|gaussian-independent|96.44 ± 0.26|2.64 ± 0.22|99.04 ± 0.14|0.4664 ± 0.0010|4.580|
|gaussian-largen|95.44 ± 0.30|4.24 ± 0.29|100.00 ± 0.00|0.2336 ± 0.0002|3.098|
|gaussian-moderate|96.18 ± 0.29|2.68 ± 0.23|95.84 ± 0.30|0.5213 ± 0.0012|4.635|
|gaussian-random|96.90 ± 0.25|2.98 ± 0.25|97.86 ± 0.21|0.5046 ± 0.0011|4.603|
|gaussian-weak|95.84 ± 0.28|4.02 ± 0.29|32.82 ± 0.65|0.4924 ± 0.0010|4.817|

All 10/10 pooled engineering calibration gates passed. There are 1 unadjusted target-role gate flags; see role-review.json and per-slot.json. These are finite-design experiments, not proof of uniform nominal coverage.

Job 13450404: 1,426 seconds elapsed; 64 CPUs and 128 GiB allocated. Consumed worker CPU 14.12 CPU-hours. Maximum observed worker RSS 0.882 GiB; median 0.839 GiB. These are per-process peaks, not total simultaneous job memory.

Monte Carlo errors use independent datasets as clusters, pooling five active or five null roles within each dataset. Per-role results and complete/planned counts are retained. Intervals are marginal normal intervals; simultaneous and bootstrap coverage were not evaluated. Correlation and weak signals must be assessed jointly with power and interval length. Refitting is not part of this frozen Gaussian policy because it worsened several development designs. Logistic and QP policies remain experimental.

Recreation: source gaussian-campaign-v1, package-hc3-v1/env-hc3-v1, campaign namespace 92300000, ten prespecified cases; campaign-gaussian.jl with eight shards/case, reps1–1000. analyze-gaussian-campaign-v2.py campaign 1000 then this script on a Discovery compute node. Input Float64-byte SHA256 pairs are retained in diagnostics.csv for exact regeneration in competitor arms.
