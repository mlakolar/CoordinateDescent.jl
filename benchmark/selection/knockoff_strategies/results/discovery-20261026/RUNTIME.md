# Knockoff construction runtime evidence

7 independent process replications per case/backend, three fresh shared inputs per replication, three warmed rounds per input. The 105 distinct input seeds are shared across procedures. Ratios first take input medians over rounds, then paired medians over three inputs within each process replication, then the median of seven process ratios. Intervals resample the seven process clusters 10,000 times; individual rounds are not independent observations. Actual source/CPU/Julia/thread records and every timing round remain available.

## Existing equicorrelated API

| Case | Payload | Candidate / baseline | 95% process interval | Process range |
|---|---|---:|---|---|
| iid n=600 p=200 | construct | 0.951 | [0.929, 1.007] | 0.927–1.151 |
| iid n=600 p=200 | construct-rng | 0.950 | [0.932, 0.993] | 0.887–1.042 |
| iid n=600 p=200 | diagonal | 3.374 | [3.231, 4.406] | 3.208–4.945 |
| iid n=600 p=200 | filter | 0.960 | [0.949, 0.985] | 0.906–1.115 |
| iid n=600 p=200 | sampling | 0.965 | [0.932, 1.003] | 0.892–1.024 |
| AR(.4) n=600 p=200 | construct | 0.989 | [0.962, 1.105] | 0.947–1.153 |
| AR(.4) n=600 p=200 | construct-rng | 1.034 | [0.969, 1.048] | 0.943–1.064 |
| AR(.4) n=600 p=200 | diagonal | 5.498 | [4.542, 5.840] | 4.228–6.935 |
| AR(.4) n=600 p=200 | filter | 1.064 | [0.972, 1.068] | 0.962–1.080 |
| AR(.4) n=600 p=200 | sampling | 1.073 | [0.994, 1.113] | 0.960–1.123 |
| AR(.8) n=400 p=400 | construct | 0.946 | [0.928, 0.998] | 0.913–1.053 |
| AR(.8) n=400 p=400 | construct-rng | 0.945 | [0.893, 0.997] | 0.883–1.066 |
| AR(.8) n=400 p=400 | diagonal | 5.469 | [5.398, 5.780] | 4.765–6.024 |
| AR(.8) n=400 p=400 | filter | 0.970 | [0.956, 0.982] | 0.911–0.990 |
| AR(.8) n=400 p=400 | sampling | 0.960 | [0.898, 0.986] | 0.874–0.995 |
| wide AR(.4) n=200 p=600 | construct | 0.928 | [0.909, 0.956] | 0.888–0.959 |
| wide AR(.4) n=200 p=600 | construct-rng | 0.926 | [0.874, 0.954] | 0.807–0.990 |
| wide AR(.4) n=200 p=600 | diagonal | 5.203 | [5.124, 5.412] | 4.903–5.554 |
| wide AR(.4) n=200 p=600 | filter | 0.939 | [0.888, 0.965] | 0.880–0.978 |
| wide AR(.4) n=200 p=600 | sampling | 0.914 | [0.868, 0.922] | 0.861–0.961 |
| fixed AR(.4) n=600 p=200 | construct | 0.988 | [0.958, 1.061] | 0.938–1.104 |
| fixed AR(.4) n=600 p=200 | construct-unitnorm | 0.978 | [0.914, 1.048] | 0.907–1.092 |
| fixed AR(.4) n=600 p=200 | diagonal | 4.990 | [4.785, 6.557] | 4.288–6.568 |
| fixed AR(.4) n=600 p=200 | filter | 0.981 | [0.917, 1.020] | 0.916–1.031 |
| fixed AR(.4) n=600 p=200 | sampling | 0.992 | [0.935, 1.082] | 0.907–1.086 |

Ratios above one remain in the evidence. The diagonal row compares the NEW validated factory with the unchanged legacy vector selector, rather than measuring a regression to that selector. The primary baseline/candidate launch had a variant-phase confound; consult ORDER-SENSITIVITY.md before interpreting small before/after speed differences. Low-level constructors retain the same numerical arithmetic but now obtain owned construction metadata internally; full filters expose that metadata. This feature adds choices and validation, not a new faster optimizer. Complete legacy result fields, constructor matrices and input bytes are equal in all 315 assertions. The independent full joint checks and profiling are outside timers.

## Cost of choosing another strategy

| Case | Ratio | Median | 95% process interval |
|---|---|---:|---|
| iid n=600 p=200 | mvr/equicorrelated: diagonal | 1.176 | [1.154, 1.181] |
| iid n=600 p=200 | mvr/equicorrelated: construct | 1.096 | [1.083, 1.102] |
| iid n=600 p=200 | mvr/equicorrelated: filter | 1.078 | [1.071, 1.094] |
| iid n=600 p=200 | maxent/equicorrelated: diagonal | 1.555 | [1.534, 1.566] |
| iid n=600 p=200 | maxent/equicorrelated: construct | 1.354 | [1.339, 1.360] |
| iid n=600 p=200 | maxent/equicorrelated: filter | 1.301 | [1.289, 1.304] |
| AR(.4) n=600 p=200 | mvr/equicorrelated: diagonal | 7.230 | [7.053, 7.719] |
| AR(.4) n=600 p=200 | mvr/equicorrelated: construct | 5.954 | [5.682, 6.015] |
| AR(.4) n=600 p=200 | mvr/equicorrelated: filter | 4.747 | [4.433, 4.829] |
| AR(.4) n=600 p=200 | maxent/equicorrelated: diagonal | 5.250 | [5.040, 5.422] |
| AR(.4) n=600 p=200 | maxent/equicorrelated: construct | 4.231 | [4.154, 4.273] |
| AR(.4) n=600 p=200 | maxent/equicorrelated: filter | 3.513 | [3.181, 3.606] |
| AR(.8) n=400 p=400 | mvr/equicorrelated: diagonal | 10.190 | [10.122, 10.505] |
| AR(.8) n=400 p=400 | mvr/equicorrelated: construct | 8.840 | [8.659, 9.166] |
| AR(.8) n=400 p=400 | mvr/equicorrelated: filter | 7.889 | [7.863, 7.969] |
| AR(.8) n=400 p=400 | maxent/equicorrelated: diagonal | 7.928 | [7.789, 7.990] |
| AR(.8) n=400 p=400 | maxent/equicorrelated: construct | 6.865 | [6.744, 6.994] |
| AR(.8) n=400 p=400 | maxent/equicorrelated: filter | 6.157 | [6.104, 6.203] |
| wide AR(.4) n=200 p=600 | mvr/equicorrelated: diagonal | 10.702 | [10.376, 10.869] |
| wide AR(.4) n=200 p=600 | mvr/equicorrelated: construct | 10.048 | [9.986, 10.153] |
| wide AR(.4) n=200 p=600 | mvr/equicorrelated: filter | 9.280 | [8.896, 9.408] |
| wide AR(.4) n=200 p=600 | maxent/equicorrelated: diagonal | 7.327 | [7.167, 7.476] |
| wide AR(.4) n=200 p=600 | maxent/equicorrelated: construct | 6.930 | [6.653, 6.957] |
| wide AR(.4) n=200 p=600 | maxent/equicorrelated: filter | 6.245 | [6.215, 6.482] |
| fixed AR(.4) n=600 p=200 | mvr/equicorrelated: diagonal | 13.899 | [13.301, 14.544] |
| fixed AR(.4) n=600 p=200 | mvr/equicorrelated: construct | 7.407 | [6.818, 7.713] |
| fixed AR(.4) n=600 p=200 | mvr/equicorrelated: filter | 6.784 | [6.480, 7.030] |
| fixed AR(.4) n=600 p=200 | maxent/equicorrelated: diagonal | 26.862 | [24.540, 27.661] |
| fixed AR(.4) n=600 p=200 | maxent/equicorrelated: construct | 13.100 | [11.993, 14.904] |
| fixed AR(.4) n=600 p=200 | maxent/equicorrelated: filter | 11.788 | [11.079, 12.356] |

Sampling is the public supplied-diagonal constructor, including its required feasibility/factorization work; it is not a cached matrix multiply. Raw optimizer measurements omit the adapter certificate/factory work; their outputs are independently certified outside timers. The validated-factory/raw-optimizer ratios in paired-ratios.csv measure that additional work and do not claim an algorithmic speed advantage. No covariance cache or Cholesky fast path is introduced.

## Applicable public R constructors

| Case | R / candidate | 95% process interval |
|---|---:|---|
| iid n=600 p=200 | 1.620 | [1.573, 1.669] |
| AR(.4) n=600 p=200 | 5.198 | [4.879, 5.525] |
| AR(.8) n=400 p=400 | 25.855 | [24.725, 26.279] |
| wide AR(.4) n=200 p=600 | 5.840 | [5.665, 5.924] |
| fixed AR(.4) n=600 p=200 | 3.413 | [3.333, 3.544] |

R Gaussian and native construct-rng both include RNG setup/draw; language RNG algorithms differ. Both receive identical X/covariance inputs. R fixed and native construct-unitnorm receive identical binary unit-norm designs. R fixed uses the unshrunk SVD boundary policy, native safety 1e-8; R Gaussian default regularization also differs slightly. Their actual root/joint/Gram identities are checked, and installed R source is retained. No R MVR/maxent constructor with the same objective is available in this package; their independent objective comparison is the conic oracle study. Matched R glmnet fits use the separate exact saved augmented matrices from the statistical cohort.

## Correctness, profiles and memory

All 1260 runtime full-filter certificates pass 1e-7 (max 2.38469e-09); all 630 timed raw optimizer certificates pass 1e-6 (max 7.88598e-09). All 28 R root/joint witnesses pass (max root error 2.12256e-12), and every timed R fixed result passes both Gram identities with its actual equicorrelated diagonal. There are 2268 nonempty CPU and 2268 nonempty heap profile reports, separate from timing, each with a positive direct allocation witness. Profiles are observational and do not turn inclusive stack counts into causal time percentages.

| Case | Backend | Median maximum RSS MiB | Range MiB |
|---|---|---:|---|
| iid n=600 p=200 | baseline | 945.7 | 941.4–975.3 |
| iid n=600 p=200 | candidate | 1050.8 | 1022.0–1084.8 |
| iid n=600 p=200 | R | 197.0 | 196.7–200.5 |
| AR(.4) n=600 p=200 | baseline | 939.9 | 921.1–969.7 |
| AR(.4) n=600 p=200 | candidate | 1021.3 | 1014.6–1040.2 |
| AR(.4) n=600 p=200 | R | 196.9 | 196.2–198.9 |
| AR(.8) n=400 p=400 | baseline | 985.9 | 955.3–1010.1 |
| AR(.8) n=400 p=400 | candidate | 1083.3 | 1071.6–1091.4 |
| AR(.8) n=400 p=400 | R | 213.7 | 213.5–215.0 |
| wide AR(.4) n=200 p=600 | baseline | 1012.9 | 1002.1–1029.4 |
| wide AR(.4) n=200 p=600 | candidate | 1121.4 | 1081.6–1137.8 |
| wide AR(.4) n=200 p=600 | R | 236.1 | 236.0–237.9 |
| fixed AR(.4) n=600 p=200 | baseline | 955.0 | 937.3–995.0 |
| fixed AR(.4) n=600 p=200 | candidate | 1041.8 | 1010.7–1082.0 |
| fixed AR(.4) n=600 p=200 | R | 197.4 | 196.1–199.1 |

Maximum RSS includes imports/JIT, warmup, timers, certificates and separate profiles, and records the largest worker/descendant resident size, not a sum of simultaneous child processes or fitted-state memory. Julia baseline and candidate both import the same pinned Knockoffs dependency for this controlled study; R has a different runtime/import/observer footprint. Heap bytes cover Julia-managed allocation and exclude foreign allocations. No cross-language fitted-state memory advantage is claimed.

