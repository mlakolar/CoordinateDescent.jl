# Shared-data knockoff construction evidence

16,128 independent inputs across 16 scenarios, 1,008 per scenario; three paired strategies per input. Pilot shard zero is excluded from main inference. Known population covariance is used in scenarios 1–13; scenarios 14–16 deliberately misspecify it. All procedures share X, y, innovations, lambda=.04 and knockoff+q=.1. The lasso uses cyclic order, tol=1e-9, max 100000; independent original-design accuracy threshold is 1e-7. No per-dataset tuning.

| Scenario | Strategy | Valid / total | FDR | MC SE | Power | MC SE | Mean false | Mean true |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iid-alt | equicorrelated | 1008/1008 | 0.0855 | 0.0026 | 0.8665 | 0.0051 | 1.943 | 17.330 |
| iid-alt | mvr | 1008/1008 | 0.0855 | 0.0026 | 0.8665 | 0.0051 | 1.943 | 17.330 |
| iid-alt | maxent | 1008/1008 | 0.0855 | 0.0026 | 0.8665 | 0.0051 | 1.943 | 17.330 |
| iid-same | equicorrelated | 1008/1008 | 0.0840 | 0.0027 | 0.8701 | 0.0048 | 1.925 | 17.403 |
| iid-same | mvr | 1008/1008 | 0.0840 | 0.0027 | 0.8701 | 0.0048 | 1.925 | 17.403 |
| iid-same | maxent | 1008/1008 | 0.0840 | 0.0027 | 0.8701 | 0.0048 | 1.925 | 17.403 |
| ar04-alt | equicorrelated | 1008/1008 | 0.0022 | 0.0007 | 0.0064 | 0.0018 | 0.030 | 0.128 |
| ar04-alt | mvr | 1008/1008 | 0.0692 | 0.0032 | 0.3990 | 0.0103 | 1.225 | 7.980 |
| ar04-alt | maxent | 1008/1008 | 0.0664 | 0.0031 | 0.3601 | 0.0103 | 1.117 | 7.201 |
| ar04-same | equicorrelated | 1008/1008 | 0.0851 | 0.0026 | 0.9164 | 0.0032 | 1.966 | 18.327 |
| ar04-same | mvr | 1008/1008 | 0.0780 | 0.0026 | 0.8653 | 0.0041 | 1.737 | 17.307 |
| ar04-same | maxent | 1008/1008 | 0.0795 | 0.0025 | 0.8790 | 0.0037 | 1.777 | 17.579 |
| ar08-alt | equicorrelated | 1008/1008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| ar08-alt | mvr | 1008/1008 | 0.0042 | 0.0015 | 0.0025 | 0.0009 | 0.045 | 0.050 |
| ar08-alt | maxent | 1008/1008 | 0.0046 | 0.0017 | 0.0018 | 0.0007 | 0.053 | 0.037 |
| ar08-same | equicorrelated | 1008/1008 | 0.0509 | 0.0025 | 0.6154 | 0.0087 | 1.020 | 12.309 |
| ar08-same | mvr | 1008/1008 | 0.0335 | 0.0021 | 0.4398 | 0.0106 | 0.634 | 8.797 |
| ar08-same | maxent | 1008/1008 | 0.0382 | 0.0022 | 0.4960 | 0.0101 | 0.727 | 9.921 |
| weak-ar04-alt | equicorrelated | 1008/1008 | 0.0022 | 0.0010 | 0.0014 | 0.0007 | 0.023 | 0.029 |
| weak-ar04-alt | mvr | 1008/1008 | 0.0288 | 0.0031 | 0.0436 | 0.0043 | 0.373 | 0.872 |
| weak-ar04-alt | maxent | 1008/1008 | 0.0255 | 0.0029 | 0.0378 | 0.0040 | 0.324 | 0.757 |
| weak-ar04-same | equicorrelated | 1008/1008 | 0.0684 | 0.0029 | 0.5200 | 0.0094 | 1.262 | 10.400 |
| weak-ar04-same | mvr | 1008/1008 | 0.0597 | 0.0028 | 0.3956 | 0.0100 | 1.006 | 7.912 |
| weak-ar04-same | maxent | 1008/1008 | 0.0642 | 0.0029 | 0.4224 | 0.0099 | 1.113 | 8.447 |
| wide-ar04-alt | equicorrelated | 1008/1008 | 0.0007 | 0.0005 | 0.0007 | 0.0005 | 0.008 | 0.014 |
| wide-ar04-alt | mvr | 1008/1008 | 0.0140 | 0.0023 | 0.0151 | 0.0024 | 0.171 | 0.302 |
| wide-ar04-alt | maxent | 1008/1008 | 0.0151 | 0.0025 | 0.0147 | 0.0023 | 0.177 | 0.295 |
| wide-ar04-same | equicorrelated | 1008/1008 | 0.0828 | 0.0029 | 0.7666 | 0.0060 | 1.764 | 15.332 |
| wide-ar04-same | mvr | 1008/1008 | 0.0803 | 0.0029 | 0.6865 | 0.0074 | 1.614 | 13.730 |
| wide-ar04-same | maxent | 1008/1008 | 0.0813 | 0.0029 | 0.7066 | 0.0070 | 1.640 | 14.133 |
| null-iid | equicorrelated | 1008/1008 | 0.0030 | 0.0017 | nan | nan | 0.031 | 0.000 |
| null-iid | mvr | 1008/1008 | 0.0030 | 0.0017 | nan | nan | 0.031 | 0.000 |
| null-iid | maxent | 1008/1008 | 0.0030 | 0.0017 | nan | nan | 0.031 | 0.000 |
| null-ar04 | equicorrelated | 1008/1008 | 0.0020 | 0.0014 | nan | nan | 0.023 | 0.000 |
| null-ar04 | mvr | 1008/1008 | 0.0010 | 0.0010 | nan | nan | 0.010 | 0.000 |
| null-ar04 | maxent | 1008/1008 | 0.0000 | 0.0000 | nan | nan | 0.000 | 0.000 |
| null-ar08 | equicorrelated | 1008/1008 | 0.0010 | 0.0010 | nan | nan | 0.013 | 0.000 |
| null-ar08 | mvr | 1008/1008 | 0.0020 | 0.0014 | nan | nan | 0.022 | 0.000 |
| null-ar08 | maxent | 1008/1008 | 0.0020 | 0.0014 | nan | nan | 0.020 | 0.000 |
| misspecified-iid | equicorrelated | 1008/1008 | 0.0049 | 0.0012 | 0.0158 | 0.0030 | 0.075 | 0.315 |
| misspecified-iid | mvr | 1008/1008 | 0.0866 | 0.0029 | 0.8625 | 0.0054 | 2.029 | 17.249 |
| misspecified-iid | maxent | 1008/1008 | 0.0881 | 0.0029 | 0.8428 | 0.0059 | 2.033 | 16.856 |
| misspecified-ar04 | equicorrelated | 1008/1008 | 0.0576 | 0.0028 | 0.3722 | 0.0102 | 0.934 | 7.444 |
| misspecified-ar04 | mvr | 1008/1008 | 0.0576 | 0.0028 | 0.3722 | 0.0102 | 0.934 | 7.444 |
| misspecified-ar04 | maxent | 1008/1008 | 0.0576 | 0.0028 | 0.3722 | 0.0102 | 0.934 | 7.444 |
| misspecified-ar08 | equicorrelated | 1008/1008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| misspecified-ar08 | mvr | 1008/1008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |
| misspecified-ar08 | maxent | 1008/1008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 |

Global-null power is undefined. Null FDR intervals use Wilson binomial intervals for nonempty selection; other intervals use dataset-level Monte Carlo SEs. Failed inputs are retained, with conservative FDP failure bounds in monte-carlo.csv. Paired differences are computed only on paired valid inputs with their denominator explicit. Separate worker means and every certificate remain available. The estimands are specific to these laws; misspecification, same-sign/alternating differences, severe-correlation and wide failures must remain in conclusions.

Native independent certificates: 48384/48384; maximum residual 2.77591e-09. Native internal stopping flags: 0; fit errors: 0; failed public-filter witnesses: 0. Matched R fits on the first three inputs per shard/scenario: 1008/1008 valid; selected-set disagreements: 0. R inputs are a subset of the primary cohort, not extra independent datasets. Native statistical fitting reuses the fixed population diagonal within a shard; construction cost is measured separately in the runtime study.

| Scenario | Contrast | Metric | Paired difference | 95% MC interval |
|---|---|---|---:|---|
| iid-alt | mvr - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| iid-alt | mvr - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| iid-alt | maxent - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| iid-alt | maxent - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| iid-alt | mvr - maxent | fdp | 0.0000 | [0.0000, 0.0000] |
| iid-alt | mvr - maxent | power | 0.0000 | [0.0000, 0.0000] |
| iid-same | mvr - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| iid-same | mvr - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| iid-same | maxent - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| iid-same | maxent - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| iid-same | mvr - maxent | fdp | 0.0000 | [0.0000, 0.0000] |
| iid-same | mvr - maxent | power | 0.0000 | [0.0000, 0.0000] |
| ar04-alt | mvr - equicorrelated | fdp | 0.0669 | [0.0605, 0.0733] |
| ar04-alt | mvr - equicorrelated | power | 0.3926 | [0.3725, 0.4127] |
| ar04-alt | maxent - equicorrelated | fdp | 0.0642 | [0.0580, 0.0704] |
| ar04-alt | maxent - equicorrelated | power | 0.3537 | [0.3336, 0.3737] |
| ar04-alt | mvr - maxent | fdp | 0.0027 | [-0.0006, 0.0060] |
| ar04-alt | mvr - maxent | power | 0.0389 | [0.0286, 0.0493] |
| ar04-same | mvr - equicorrelated | fdp | -0.0071 | [-0.0123, -0.0020] |
| ar04-same | mvr - equicorrelated | power | -0.0510 | [-0.0589, -0.0432] |
| ar04-same | maxent - equicorrelated | fdp | -0.0056 | [-0.0106, -0.0007] |
| ar04-same | maxent - equicorrelated | power | -0.0374 | [-0.0444, -0.0304] |
| ar04-same | mvr - maxent | fdp | -0.0015 | [-0.0035, 0.0005] |
| ar04-same | mvr - maxent | power | -0.0136 | [-0.0168, -0.0105] |
| ar08-alt | mvr - equicorrelated | fdp | 0.0042 | [0.0013, 0.0072] |
| ar08-alt | mvr - equicorrelated | power | 0.0025 | [0.0008, 0.0042] |
| ar08-alt | maxent - equicorrelated | fdp | 0.0046 | [0.0013, 0.0080] |
| ar08-alt | maxent - equicorrelated | power | 0.0018 | [0.0005, 0.0032] |
| ar08-alt | mvr - maxent | fdp | -0.0004 | [-0.0036, 0.0028] |
| ar08-alt | mvr - maxent | power | 0.0006 | [-0.0009, 0.0022] |
| ar08-same | mvr - equicorrelated | fdp | -0.0174 | [-0.0220, -0.0128] |
| ar08-same | mvr - equicorrelated | power | -0.1756 | [-0.1934, -0.1578] |
| ar08-same | maxent - equicorrelated | fdp | -0.0126 | [-0.0169, -0.0084] |
| ar08-same | maxent - equicorrelated | power | -0.1194 | [-0.1358, -0.1030] |
| ar08-same | mvr - maxent | fdp | -0.0047 | [-0.0071, -0.0024] |
| ar08-same | mvr - maxent | power | -0.0562 | [-0.0665, -0.0459] |
| weak-ar04-alt | mvr - equicorrelated | fdp | 0.0266 | [0.0203, 0.0330] |
| weak-ar04-alt | mvr - equicorrelated | power | 0.0422 | [0.0337, 0.0506] |
| weak-ar04-alt | maxent - equicorrelated | fdp | 0.0233 | [0.0174, 0.0291] |
| weak-ar04-alt | maxent - equicorrelated | power | 0.0364 | [0.0286, 0.0443] |
| weak-ar04-alt | mvr - maxent | fdp | 0.0034 | [-0.0013, 0.0081] |
| weak-ar04-alt | mvr - maxent | power | 0.0058 | [0.0003, 0.0112] |
| weak-ar04-same | mvr - equicorrelated | fdp | -0.0087 | [-0.0148, -0.0026] |
| weak-ar04-same | mvr - equicorrelated | power | -0.1244 | [-0.1446, -0.1042] |
| weak-ar04-same | maxent - equicorrelated | fdp | -0.0042 | [-0.0102, 0.0018] |
| weak-ar04-same | maxent - equicorrelated | power | -0.0976 | [-0.1174, -0.0778] |
| weak-ar04-same | mvr - maxent | fdp | -0.0045 | [-0.0073, -0.0017] |
| weak-ar04-same | mvr - maxent | power | -0.0268 | [-0.0355, -0.0180] |
| wide-ar04-alt | mvr - equicorrelated | fdp | 0.0133 | [0.0088, 0.0178] |
| wide-ar04-alt | mvr - equicorrelated | power | 0.0144 | [0.0097, 0.0191] |
| wide-ar04-alt | maxent - equicorrelated | fdp | 0.0144 | [0.0094, 0.0193] |
| wide-ar04-alt | maxent - equicorrelated | power | 0.0140 | [0.0094, 0.0186] |
| wide-ar04-alt | mvr - maxent | fdp | -0.0011 | [-0.0046, 0.0025] |
| wide-ar04-alt | mvr - maxent | power | 0.0003 | [-0.0025, 0.0032] |
| wide-ar04-same | mvr - equicorrelated | fdp | -0.0025 | [-0.0091, 0.0041] |
| wide-ar04-same | mvr - equicorrelated | power | -0.0801 | [-0.0949, -0.0653] |
| wide-ar04-same | maxent - equicorrelated | fdp | -0.0015 | [-0.0079, 0.0049] |
| wide-ar04-same | maxent - equicorrelated | power | -0.0600 | [-0.0739, -0.0461] |
| wide-ar04-same | mvr - maxent | fdp | -0.0010 | [-0.0040, 0.0020] |
| wide-ar04-same | mvr - maxent | power | -0.0201 | [-0.0266, -0.0136] |
| null-iid | mvr - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| null-iid | mvr - equicorrelated | power | nan | [nan, nan] |
| null-iid | maxent - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| null-iid | maxent - equicorrelated | power | nan | [nan, nan] |
| null-iid | mvr - maxent | fdp | 0.0000 | [0.0000, 0.0000] |
| null-iid | mvr - maxent | power | nan | [nan, nan] |
| null-ar04 | mvr - equicorrelated | fdp | -0.0010 | [-0.0044, 0.0024] |
| null-ar04 | mvr - equicorrelated | power | nan | [nan, nan] |
| null-ar04 | maxent - equicorrelated | fdp | -0.0020 | [-0.0047, 0.0008] |
| null-ar04 | maxent - equicorrelated | power | nan | [nan, nan] |
| null-ar04 | mvr - maxent | fdp | 0.0010 | [-0.0010, 0.0029] |
| null-ar04 | mvr - maxent | power | nan | [nan, nan] |
| null-ar08 | mvr - equicorrelated | fdp | 0.0010 | [-0.0024, 0.0044] |
| null-ar08 | mvr - equicorrelated | power | nan | [nan, nan] |
| null-ar08 | maxent - equicorrelated | fdp | 0.0010 | [-0.0024, 0.0044] |
| null-ar08 | maxent - equicorrelated | power | nan | [nan, nan] |
| null-ar08 | mvr - maxent | fdp | 0.0000 | [-0.0028, 0.0028] |
| null-ar08 | mvr - maxent | power | nan | [nan, nan] |
| misspecified-iid | mvr - equicorrelated | fdp | 0.0817 | [0.0760, 0.0874] |
| misspecified-iid | mvr - equicorrelated | power | 0.8467 | [0.8349, 0.8585] |
| misspecified-iid | maxent - equicorrelated | fdp | 0.0832 | [0.0775, 0.0890] |
| misspecified-iid | maxent - equicorrelated | power | 0.8270 | [0.8145, 0.8396] |
| misspecified-iid | mvr - maxent | fdp | -0.0015 | [-0.0041, 0.0010] |
| misspecified-iid | mvr - maxent | power | 0.0196 | [0.0144, 0.0249] |
| misspecified-ar04 | mvr - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar04 | mvr - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar04 | maxent - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar04 | maxent - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar04 | mvr - maxent | fdp | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar04 | mvr - maxent | power | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar08 | mvr - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar08 | mvr - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar08 | maxent - equicorrelated | fdp | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar08 | maxent - equicorrelated | power | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar08 | mvr - maxent | fdp | 0.0000 | [0.0000, 0.0000] |
| misspecified-ar08 | mvr - maxent | power | 0.0000 | [0.0000, 0.0000] |
