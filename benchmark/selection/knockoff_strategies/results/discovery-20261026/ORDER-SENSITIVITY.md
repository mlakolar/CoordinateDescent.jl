# Alternating-order sensitivity

alternating-order sensitivity prompted by primary variant-phase confounding; primary results preserved. The same 105 primary inputs are reused in 70 fresh processes; this is not another independent data cohort. Baseline/candidate order alternates by(case+rep)parity, both run serially on the same allocated CPU per pair. Only equicorrelation is measured. Three rounds per input/three inputs per process, seven independent process pairs, bootstrap10000 seed=106999001. Primary data and all primary timing rows are retained separately. No profiling is repeated.

| Case | Payload | Candidate / baseline | 95% process interval | Process range |
|---|---|---:|---|---|
| 1 | construct | 1.070 | [0.887, 1.120] | 0.862–1.124 |
| 1 | construct-rng | 1.100 | [0.891, 1.130] | 0.883–1.138 |
| 1 | diagonal | 4.724 | [3.053, 4.883] | 2.864–4.943 |
| 1 | filter | 1.039 | [0.885, 1.057] | 0.882–1.093 |
| 1 | sampling | 1.074 | [0.891, 1.151] | 0.866–1.157 |
| 2 | construct | 0.988 | [0.978, 1.003] | 0.880–1.006 |
| 2 | construct-rng | 1.011 | [1.001, 1.017] | 0.889–1.034 |
| 2 | diagonal | 4.526 | [4.265, 4.637] | 3.916–4.821 |
| 2 | filter | 0.980 | [0.929, 0.998] | 0.908–1.004 |
| 2 | sampling | 1.002 | [0.953, 1.015] | 0.942–1.033 |
| 3 | construct | 1.012 | [0.985, 1.058] | 0.981–1.088 |
| 3 | construct-rng | 1.015 | [0.991, 1.033] | 0.985–1.040 |
| 3 | diagonal | 5.462 | [5.246, 5.492] | 5.199–5.765 |
| 3 | filter | 1.019 | [0.995, 1.050] | 0.994–1.064 |
| 3 | sampling | 0.994 | [0.980, 1.009] | 0.977–1.024 |
| 4 | construct | 1.005 | [0.995, 1.028] | 0.984–1.050 |
| 4 | construct-rng | 0.999 | [0.992, 1.015] | 0.976–1.023 |
| 4 | diagonal | 5.703 | [5.600, 5.799] | 5.226–5.863 |
| 4 | filter | 0.992 | [0.989, 1.000] | 0.946–1.025 |
| 4 | sampling | 0.996 | [0.994, 1.007] | 0.978–1.024 |
| 5 | construct | 1.010 | [0.992, 1.014] | 0.984–1.019 |
| 5 | construct-unitnorm | 1.002 | [0.994, 1.019] | 0.984–1.036 |
| 5 | diagonal | 5.733 | [4.805, 6.046] | 4.799–6.085 |
| 5 | filter | 1.002 | [0.979, 1.004] | 0.967–1.009 |
| 5 | sampling | 1.001 | [0.994, 1.010] | 0.982–1.029 |

The diagonal row compares the new validated factory with the unchanged legacy vector selector; its additional work is not an old-selector regression. The other rows compare the same public interfaces. Small differences and adverse intervals remain visible. No pooling with primary timings or selection of favorable rounds.

