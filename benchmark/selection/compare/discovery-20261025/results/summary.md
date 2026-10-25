# Direct comparison on Discovery

7 independent process replications; 140 shared datasets per scenario. Timings exclude startup/JIT and accuracy checks.

| Case | Engine | Median ms | Replication min–max ms |
|---|---|---:|---:|
| cpss | CD | 22.527 | 20.907–24.641 |
| cpss | CD-threaded | 15.995 | 13.979–16.401 |
| cpss | GLMNet.jl | 143.508 | 139.977–152.152 |
| cpss | R-stabs | 122.000 | 116.000–129.000 |
| fixed | CD | 15.094 | 14.179–21.123 |
| fixed | Knockoffs.jl | 40.348 | 38.456–43.115 |
| fixed | R-knockoff | 55.000 | 51.000–60.000 |
| gaussian-1 | CD | 12.267 | 9.671–14.411 |
| gaussian-1 | Knockoffs.jl | 5.859 | 5.638–7.190 |
| gaussian-1 | R-knockoff | 32.000 | 30.000–35.000 |
| gaussian-1-maxent | Knockoffs.jl | 22.871 | 21.160–25.948 |
| gaussian-1-mvr | Knockoffs.jl | 30.597 | 28.465–35.525 |
| gaussian-2 | CD | 369.305 | 362.301–386.068 |
| gaussian-2 | Knockoffs.jl | 158.908 | 156.194–165.819 |
| gaussian-2 | R-knockoff | 1945.000 | 1938.000–2140.000 |
| joint-lasso | CD | 1.545 | 1.429–1.735 |
| joint-lasso | GLMNet.jl | 27.585 | 26.660–29.791 |
| joint-lasso | R-glmnet | 8.000 | 7.500–8.500 |
| slope-1 | CD | 4.589 | 4.039–4.852 |
| slope-1 | R-SLOPE-auto | 17.000 | 15.000–17.000 |
| slope-1 | SLOPE.jl | 8.159 | 7.505–8.811 |
| slope-2 | CD | 6.815 | 6.317–7.562 |
| slope-2 | R-SLOPE-auto | 22.000 | 21.000–24.000 |
| slope-2 | SLOPE.jl | 15.095 | 14.284–15.531 |
| slope-3 | CD | 4.982 | 4.730–5.812 |
| slope-3 | R-SLOPE-auto | 17.000 | 17.000–19.000 |
| slope-3 | SLOPE.jl | 10.281 | 9.610–10.735 |
| slope-4 | CD | 13.627 | 11.984–14.428 |
| slope-4 | R-SLOPE-auto | 19.500 | 18.500–20.000 |
| slope-4 | SLOPE.jl | 8.383 | 7.860–8.598 |

CD/alternative ratios greater than one mean CD is slower.

| Case | Alternative | Median paired CD/alternative | Min–max |
|---|---|---:|---:|
| cpss | CD-threaded | 1.46 | 1.27–1.71 |
| cpss | GLMNet.jl | 0.15 | 0.15–0.17 |
| cpss | R-stabs | 0.18 | 0.17–0.19 |
| fixed | Knockoffs.jl | 0.38 | 0.36–0.51 |
| fixed | R-knockoff | 0.30 | 0.27–0.36 |
| gaussian-1 | Knockoffs.jl | 2.02 | 1.66–2.25 |
| gaussian-1 | R-knockoff | 0.39 | 0.31–0.44 |
| gaussian-2 | Knockoffs.jl | 2.34 | 2.28–2.40 |
| gaussian-2 | R-knockoff | 0.19 | 0.18–0.19 |
| joint-lasso | GLMNet.jl | 0.06 | 0.05–0.06 |
| joint-lasso | R-glmnet | 0.20 | 0.17–0.22 |
| slope-1 | R-SLOPE-auto | 0.29 | 0.24–0.30 |
| slope-1 | SLOPE.jl | 0.57 | 0.50–0.61 |
| slope-2 | R-SLOPE-auto | 0.32 | 0.26–0.33 |
| slope-2 | SLOPE.jl | 0.47 | 0.43–0.49 |
| slope-3 | R-SLOPE-auto | 0.29 | 0.26–0.31 |
| slope-3 | SLOPE.jl | 0.50 | 0.46–0.57 |
| slope-4 | R-SLOPE-auto | 0.68 | 0.62–0.74 |
| slope-4 | SLOPE.jl | 1.59 | 1.41–1.83 |

## Common accuracy certificate

| Case | Engine | Maximum relative prox residual | Maximum coefficient difference vs CD | Maximum relative objective difference |
|---|---|---:|---:|---:|
| joint-lasso | CD | 3.31e-10 | 0 | 0 |
| joint-lasso | GLMNet.jl | 4.18e-08 | 1.07e-07 | 9.66e-15 |
| joint-lasso | R-glmnet | 4.18e-08 | 1.07e-07 | 9.66e-15 |
| slope-1 | CD | 1.62e-08 | 0 | 0 |
| slope-1 | R-auto | 3.46e-08 | 4.28e-08 | 1.22e-15 |
| slope-1 | SLOPE.jl | 1.7e-08 | 2.09e-08 | 4.06e-16 |
| slope-2 | CD | 3.37e-08 | 0 | 0 |
| slope-2 | R-auto | 5.85e-08 | 6.79e-08 | 6.01e-15 |
| slope-2 | SLOPE.jl | 3.98e-08 | 5.56e-08 | 1.71e-15 |
| slope-3 | CD | 1.99e-08 | 0 | 0 |
| slope-3 | R-auto | 4.38e-08 | 4.01e-07 | 7.09e-15 |
| slope-3 | SLOPE.jl | 7.36e-08 | 4.69e-07 | 1.58e-14 |
| slope-4 | CD | 1.95e-08 | 0 | 0 |
| slope-4 | R-auto | 3.73e-08 | 6.46e-08 | 8.88e-16 |
| slope-4 | SLOPE.jl | 1.37e-08 | 1.95e-08 | 3.55e-16 |

## Constructor covariance check

Empirical check: 20,000 independent conditional draws at X=0,p=30,AR(.8), repeated in each process. Implied covariance is recovered using the shared standard-normal innovations.

| Engine | Empirical relative error, median [range] | Maximum implied covariance error |
|---|---:|---:|
| CD | 0.0323 [0.0314, 0.0348] | 1.1189455284439876e-15 |
| Knockoffs.jl | 0.2538 [0.2531, 0.2556] | 0.2508018617799106 |
| R-knockoff | 0.0321 [0.0308, 0.0343] | 9.98331047769038e-16 |

## Shared-design simulations

Known Gaussian population covariance; q=.1 for SLOPE/knockoff+. CPSS targets its documented conditional PFER bound, not FDR=.1. Native constructor pipelines use language-specific random draws. Refer to constructor checks before interpreting their error-control results.

| Scenario | Method | Mean FDP (MC SE) | Power (MC SE) | Mean false selections |
|---|---|---:|---:|---:|
| iid | cpss-cap10-CD | 0.0000 (0.0000) | 0.1514 (0.0051) | 0.000 |
| iid | cpss-cap10-R | 0.0000 (0.0000) | 0.1514 (0.0051) | 0.000 |
| iid | cpss-cap20-CD | 0.0010 (0.0007) | 0.5825 (0.0073) | 0.014 |
| iid | cpss-cap20-R | 0.0010 (0.0007) | 0.5825 (0.0073) | 0.014 |
| iid | cpss-path10-R | 0.0000 (0.0000) | 0.1046 (0.0046) | 0.000 |
| iid | fx-CD | 0.0806 (0.0070) | 0.8425 (0.0110) | 1.750 |
| iid | fx-GLMNet.jl | 0.0806 (0.0070) | 0.8425 (0.0110) | 1.750 |
| iid | fx-R | 0.0806 (0.0070) | 0.8425 (0.0110) | 1.750 |
| iid | mx-equi-CD | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-equi-GLMNet.jl | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-equi-R | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-maxent-CD | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-maxent-GLMNet.jl | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-maxent-R | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-mvr-CD | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-mvr-GLMNet.jl | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-mvr-R | 0.0905 (0.0068) | 0.8871 (0.0138) | 2.057 |
| iid | mx-native-equi-Knockoffs.jl | 0.0768 (0.0069) | 0.8729 (0.0147) | 1.757 |
| iid | mx-native-equi-R | 0.0719 (0.0062) | 0.8739 (0.0141) | 1.607 |
| iid | mx-native-mvr-Knockoffs.jl | 0.0768 (0.0069) | 0.8729 (0.0147) | 1.757 |
| iid | slope-CD | 0.1846 (0.0062) | 0.9539 (0.0043) | 4.507 |
| iid | slope-R | 0.1846 (0.0062) | 0.9539 (0.0043) | 4.507 |
| iid | slope-SLOPE.jl | 0.1846 (0.0062) | 0.9539 (0.0043) | 4.507 |
| AR(.4) | cpss-cap10-CD | 0.0107 (0.0080) | 0.0689 (0.0044) | 0.014 |
| AR(.4) | cpss-cap10-R | 0.0107 (0.0080) | 0.0689 (0.0044) | 0.014 |
| AR(.4) | cpss-cap20-CD | 0.0257 (0.0067) | 0.2314 (0.0074) | 0.107 |
| AR(.4) | cpss-cap20-R | 0.0257 (0.0067) | 0.2314 (0.0074) | 0.107 |
| AR(.4) | cpss-path10-R | 0.0095 (0.0075) | 0.0368 (0.0031) | 0.014 |
| AR(.4) | fx-CD | 0.0574 (0.0077) | 0.4511 (0.0293) | 1.114 |
| AR(.4) | fx-GLMNet.jl | 0.0574 (0.0077) | 0.4511 (0.0293) | 1.114 |
| AR(.4) | fx-R | 0.0574 (0.0077) | 0.4511 (0.0293) | 1.114 |
| AR(.4) | mx-equi-CD | 0.0014 (0.0014) | 0.0029 (0.0029) | 0.014 |
| AR(.4) | mx-equi-GLMNet.jl | 0.0014 (0.0014) | 0.0029 (0.0029) | 0.014 |
| AR(.4) | mx-equi-R | 0.0014 (0.0014) | 0.0029 (0.0029) | 0.014 |
| AR(.4) | mx-maxent-CD | 0.0712 (0.0082) | 0.4146 (0.0288) | 1.286 |
| AR(.4) | mx-maxent-GLMNet.jl | 0.0712 (0.0082) | 0.4146 (0.0288) | 1.286 |
| AR(.4) | mx-maxent-R | 0.0712 (0.0082) | 0.4146 (0.0288) | 1.286 |
| AR(.4) | mx-mvr-CD | 0.0724 (0.0079) | 0.4443 (0.0284) | 1.307 |
| AR(.4) | mx-mvr-GLMNet.jl | 0.0724 (0.0079) | 0.4443 (0.0284) | 1.307 |
| AR(.4) | mx-mvr-R | 0.0724 (0.0079) | 0.4443 (0.0284) | 1.307 |
| AR(.4) | mx-native-equi-Knockoffs.jl | 0.0063 (0.0028) | 0.0211 (0.0086) | 0.071 |
| AR(.4) | mx-native-equi-R | 0.0014 (0.0014) | 0.0029 (0.0029) | 0.014 |
| AR(.4) | mx-native-mvr-Knockoffs.jl | 0.0725 (0.0087) | 0.4196 (0.0288) | 1.336 |
| AR(.4) | slope-CD | 0.2552 (0.0126) | 0.3282 (0.0115) | 2.664 |
| AR(.4) | slope-R | 0.2552 (0.0126) | 0.3282 (0.0115) | 2.664 |
| AR(.4) | slope-SLOPE.jl | 0.2552 (0.0126) | 0.3282 (0.0115) | 2.664 |
| AR(.8) | cpss-cap10-CD | 0.0238 (0.0114) | 0.0193 (0.0024) | 0.036 |
| AR(.8) | cpss-cap10-R | 0.0238 (0.0114) | 0.0193 (0.0024) | 0.036 |
| AR(.8) | cpss-cap20-CD | 0.1845 (0.0285) | 0.0443 (0.0034) | 0.300 |
| AR(.8) | cpss-cap20-R | 0.1845 (0.0285) | 0.0443 (0.0034) | 0.300 |
| AR(.8) | cpss-path10-R | 0.0571 (0.0190) | 0.0157 (0.0020) | 0.064 |
| AR(.8) | fx-CD | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| AR(.8) | fx-GLMNet.jl | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| AR(.8) | fx-R | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| AR(.8) | mx-equi-CD | 0.0045 (0.0045) | 0.0014 (0.0014) | 0.050 |
| AR(.8) | mx-equi-GLMNet.jl | 0.0045 (0.0045) | 0.0014 (0.0014) | 0.050 |
| AR(.8) | mx-equi-R | 0.0045 (0.0045) | 0.0014 (0.0014) | 0.050 |
| AR(.8) | mx-maxent-CD | 0.0048 (0.0048) | 0.0014 (0.0014) | 0.057 |
| AR(.8) | mx-maxent-GLMNet.jl | 0.0048 (0.0048) | 0.0014 (0.0014) | 0.057 |
| AR(.8) | mx-maxent-R | 0.0048 (0.0048) | 0.0014 (0.0014) | 0.057 |
| AR(.8) | mx-mvr-CD | 0.0075 (0.0054) | 0.0036 (0.0026) | 0.079 |
| AR(.8) | mx-mvr-GLMNet.jl | 0.0075 (0.0054) | 0.0036 (0.0026) | 0.079 |
| AR(.8) | mx-mvr-R | 0.0075 (0.0054) | 0.0036 (0.0026) | 0.079 |
| AR(.8) | mx-native-equi-Knockoffs.jl | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| AR(.8) | mx-native-equi-R | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| AR(.8) | mx-native-mvr-Knockoffs.jl | 0.0043 (0.0043) | 0.0014 (0.0014) | 0.043 |
| AR(.8) | slope-CD | 0.1646 (0.0258) | 0.0425 (0.0049) | 0.636 |
| AR(.8) | slope-R | 0.1646 (0.0258) | 0.0425 (0.0049) | 0.636 |
| AR(.8) | slope-SLOPE.jl | 0.1646 (0.0258) | 0.0425 (0.0049) | 0.636 |
| weak AR(.4) | cpss-cap10-CD | 0.0107 (0.0080) | 0.0318 (0.0033) | 0.014 |
| weak AR(.4) | cpss-cap10-R | 0.0107 (0.0080) | 0.0318 (0.0033) | 0.014 |
| weak AR(.4) | cpss-cap20-CD | 0.0773 (0.0177) | 0.0929 (0.0054) | 0.171 |
| weak AR(.4) | cpss-cap20-R | 0.0773 (0.0177) | 0.0929 (0.0054) | 0.171 |
| weak AR(.4) | cpss-path10-R | 0.0321 (0.0145) | 0.0218 (0.0029) | 0.036 |
| weak AR(.4) | fx-CD | 0.0325 (0.0081) | 0.0561 (0.0127) | 0.379 |
| weak AR(.4) | fx-GLMNet.jl | 0.0325 (0.0081) | 0.0561 (0.0127) | 0.379 |
| weak AR(.4) | fx-R | 0.0325 (0.0081) | 0.0561 (0.0127) | 0.379 |
| weak AR(.4) | mx-equi-CD | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| weak AR(.4) | mx-equi-GLMNet.jl | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| weak AR(.4) | mx-equi-R | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| weak AR(.4) | mx-maxent-CD | 0.0170 (0.0058) | 0.0329 (0.0102) | 0.214 |
| weak AR(.4) | mx-maxent-GLMNet.jl | 0.0170 (0.0058) | 0.0329 (0.0102) | 0.214 |
| weak AR(.4) | mx-maxent-R | 0.0170 (0.0058) | 0.0329 (0.0102) | 0.214 |
| weak AR(.4) | mx-mvr-CD | 0.0234 (0.0065) | 0.0450 (0.0116) | 0.279 |
| weak AR(.4) | mx-mvr-GLMNet.jl | 0.0234 (0.0065) | 0.0450 (0.0116) | 0.279 |
| weak AR(.4) | mx-mvr-R | 0.0234 (0.0065) | 0.0450 (0.0116) | 0.279 |
| weak AR(.4) | mx-native-equi-Knockoffs.jl | 0.0019 (0.0019) | 0.0029 (0.0029) | 0.021 |
| weak AR(.4) | mx-native-equi-R | 0.0026 (0.0026) | 0.0032 (0.0032) | 0.036 |
| weak AR(.4) | mx-native-mvr-Knockoffs.jl | 0.0254 (0.0075) | 0.0421 (0.0110) | 0.286 |
| weak AR(.4) | slope-CD | 0.1716 (0.0220) | 0.0811 (0.0069) | 0.671 |
| weak AR(.4) | slope-R | 0.1716 (0.0220) | 0.0811 (0.0069) | 0.671 |
| weak AR(.4) | slope-SLOPE.jl | 0.1716 (0.0220) | 0.0811 (0.0069) | 0.671 |
| null AR(.4) | cpss-cap10-CD | 0.0786 (0.0228) | nan (nan) | 0.079 |
| null AR(.4) | cpss-cap10-R | 0.0786 (0.0228) | nan (nan) | 0.079 |
| null AR(.4) | cpss-cap20-CD | 0.4357 (0.0421) | nan (nan) | 0.586 |
| null AR(.4) | cpss-cap20-R | 0.4357 (0.0421) | nan (nan) | 0.586 |
| null AR(.4) | cpss-path10-R | 0.0500 (0.0185) | nan (nan) | 0.050 |
| null AR(.4) | fx-CD | 0.0000 (0.0000) | nan (nan) | 0.000 |
| null AR(.4) | fx-GLMNet.jl | 0.0000 (0.0000) | nan (nan) | 0.000 |
| null AR(.4) | fx-R | 0.0000 (0.0000) | nan (nan) | 0.000 |
| null AR(.4) | mx-equi-CD | 0.0071 (0.0071) | nan (nan) | 0.071 |
| null AR(.4) | mx-equi-GLMNet.jl | 0.0071 (0.0071) | nan (nan) | 0.071 |
| null AR(.4) | mx-equi-R | 0.0071 (0.0071) | nan (nan) | 0.071 |
| null AR(.4) | mx-maxent-CD | 0.0071 (0.0071) | nan (nan) | 0.086 |
| null AR(.4) | mx-maxent-GLMNet.jl | 0.0071 (0.0071) | nan (nan) | 0.086 |
| null AR(.4) | mx-maxent-R | 0.0071 (0.0071) | nan (nan) | 0.086 |
| null AR(.4) | mx-mvr-CD | 0.0071 (0.0071) | nan (nan) | 0.071 |
| null AR(.4) | mx-mvr-GLMNet.jl | 0.0071 (0.0071) | nan (nan) | 0.071 |
| null AR(.4) | mx-mvr-R | 0.0071 (0.0071) | nan (nan) | 0.071 |
| null AR(.4) | mx-native-equi-Knockoffs.jl | 0.0000 (0.0000) | nan (nan) | 0.000 |
| null AR(.4) | mx-native-equi-R | 0.0000 (0.0000) | nan (nan) | 0.000 |
| null AR(.4) | mx-native-mvr-Knockoffs.jl | 0.0000 (0.0000) | nan (nan) | 0.000 |
| null AR(.4) | slope-CD | 0.0714 (0.0218) | nan (nan) | 0.100 |
| null AR(.4) | slope-R | 0.0714 (0.0218) | nan (nan) | 0.100 |
| null AR(.4) | slope-SLOPE.jl | 0.0714 (0.0218) | nan (nan) | 0.100 |
| wide AR(.4) | cpss-cap10-CD | 0.0071 (0.0071) | 0.0043 (0.0012) | 0.007 |
| wide AR(.4) | cpss-cap10-R | 0.0071 (0.0071) | 0.0043 (0.0012) | 0.007 |
| wide AR(.4) | cpss-cap20-CD | 0.0071 (0.0071) | 0.0111 (0.0018) | 0.007 |
| wide AR(.4) | cpss-cap20-R | 0.0071 (0.0071) | 0.0111 (0.0018) | 0.007 |
| wide AR(.4) | cpss-path10-R | 0.0071 (0.0071) | 0.0082 (0.0017) | 0.007 |
| wide AR(.4) | mx-equi-CD | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| wide AR(.4) | mx-equi-GLMNet.jl | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| wide AR(.4) | mx-equi-R | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| wide AR(.4) | mx-maxent-CD | 0.0075 (0.0053) | 0.0039 (0.0028) | 0.086 |
| wide AR(.4) | mx-maxent-GLMNet.jl | 0.0075 (0.0053) | 0.0039 (0.0028) | 0.086 |
| wide AR(.4) | mx-maxent-R | 0.0075 (0.0053) | 0.0039 (0.0028) | 0.086 |
| wide AR(.4) | mx-mvr-CD | 0.0200 (0.0085) | 0.0125 (0.0053) | 0.214 |
| wide AR(.4) | mx-mvr-GLMNet.jl | 0.0200 (0.0085) | 0.0125 (0.0053) | 0.214 |
| wide AR(.4) | mx-mvr-R | 0.0200 (0.0085) | 0.0125 (0.0053) | 0.214 |
| wide AR(.4) | mx-native-equi-Knockoffs.jl | 0.0000 (0.0000) | 0.0000 (0.0000) | 0.000 |
| wide AR(.4) | mx-native-equi-R | 0.0043 (0.0043) | 0.0014 (0.0014) | 0.043 |
| wide AR(.4) | mx-native-mvr-Knockoffs.jl | 0.0129 (0.0065) | 0.0093 (0.0047) | 0.164 |
| wide AR(.4) | slope-CD | 0.6288 (0.0132) | 0.2471 (0.0090) | 9.793 |
| wide AR(.4) | slope-R | 0.6288 (0.0132) | 0.2471 (0.0090) | 9.793 |
| wide AR(.4) | slope-SLOPE.jl | 0.6288 (0.0132) | 0.2471 (0.0090) | 9.793 |

## Matched implementation agreement

| Scenario | Reference | Alternative | Mismatched selection sets / datasets | Maximum set difference |
|---|---|---|---:|---:|
| iid | slope-CD | slope-SLOPE.jl | 0/140 | 0 |
| iid | slope-CD | slope-R | 0/140 | 0 |
| iid | cpss-cap10-CD | cpss-cap10-R | 0/140 | 0 |
| iid | cpss-cap20-CD | cpss-cap20-R | 0/140 | 0 |
| iid | mx-equi-CD | mx-equi-R | 0/140 | 0 |
| iid | mx-equi-CD | mx-equi-GLMNet.jl | 0/140 | 0 |
| iid | mx-mvr-CD | mx-mvr-R | 0/140 | 0 |
| iid | mx-mvr-CD | mx-mvr-GLMNet.jl | 0/140 | 0 |
| iid | mx-maxent-CD | mx-maxent-R | 0/140 | 0 |
| iid | mx-maxent-CD | mx-maxent-GLMNet.jl | 0/140 | 0 |
| iid | fx-CD | fx-R | 0/140 | 0 |
| iid | fx-CD | fx-GLMNet.jl | 0/140 | 0 |
| AR(.4) | slope-CD | slope-SLOPE.jl | 0/140 | 0 |
| AR(.4) | slope-CD | slope-R | 0/140 | 0 |
| AR(.4) | cpss-cap10-CD | cpss-cap10-R | 0/140 | 0 |
| AR(.4) | cpss-cap20-CD | cpss-cap20-R | 0/140 | 0 |
| AR(.4) | mx-equi-CD | mx-equi-R | 0/140 | 0 |
| AR(.4) | mx-equi-CD | mx-equi-GLMNet.jl | 0/140 | 0 |
| AR(.4) | mx-mvr-CD | mx-mvr-R | 0/140 | 0 |
| AR(.4) | mx-mvr-CD | mx-mvr-GLMNet.jl | 0/140 | 0 |
| AR(.4) | mx-maxent-CD | mx-maxent-R | 0/140 | 0 |
| AR(.4) | mx-maxent-CD | mx-maxent-GLMNet.jl | 0/140 | 0 |
| AR(.4) | fx-CD | fx-R | 0/140 | 0 |
| AR(.4) | fx-CD | fx-GLMNet.jl | 0/140 | 0 |
| AR(.8) | slope-CD | slope-SLOPE.jl | 0/140 | 0 |
| AR(.8) | slope-CD | slope-R | 0/140 | 0 |
| AR(.8) | cpss-cap10-CD | cpss-cap10-R | 0/140 | 0 |
| AR(.8) | cpss-cap20-CD | cpss-cap20-R | 0/140 | 0 |
| AR(.8) | mx-equi-CD | mx-equi-R | 0/140 | 0 |
| AR(.8) | mx-equi-CD | mx-equi-GLMNet.jl | 0/140 | 0 |
| AR(.8) | mx-mvr-CD | mx-mvr-R | 0/140 | 0 |
| AR(.8) | mx-mvr-CD | mx-mvr-GLMNet.jl | 0/140 | 0 |
| AR(.8) | mx-maxent-CD | mx-maxent-R | 0/140 | 0 |
| AR(.8) | mx-maxent-CD | mx-maxent-GLMNet.jl | 0/140 | 0 |
| AR(.8) | fx-CD | fx-R | 0/140 | 0 |
| AR(.8) | fx-CD | fx-GLMNet.jl | 0/140 | 0 |
| weak AR(.4) | slope-CD | slope-SLOPE.jl | 0/140 | 0 |
| weak AR(.4) | slope-CD | slope-R | 0/140 | 0 |
| weak AR(.4) | cpss-cap10-CD | cpss-cap10-R | 0/140 | 0 |
| weak AR(.4) | cpss-cap20-CD | cpss-cap20-R | 0/140 | 0 |
| weak AR(.4) | mx-equi-CD | mx-equi-R | 0/140 | 0 |
| weak AR(.4) | mx-equi-CD | mx-equi-GLMNet.jl | 0/140 | 0 |
| weak AR(.4) | mx-mvr-CD | mx-mvr-R | 0/140 | 0 |
| weak AR(.4) | mx-mvr-CD | mx-mvr-GLMNet.jl | 0/140 | 0 |
| weak AR(.4) | mx-maxent-CD | mx-maxent-R | 0/140 | 0 |
| weak AR(.4) | mx-maxent-CD | mx-maxent-GLMNet.jl | 0/140 | 0 |
| weak AR(.4) | fx-CD | fx-R | 0/140 | 0 |
| weak AR(.4) | fx-CD | fx-GLMNet.jl | 0/140 | 0 |
| null AR(.4) | slope-CD | slope-SLOPE.jl | 0/140 | 0 |
| null AR(.4) | slope-CD | slope-R | 0/140 | 0 |
| null AR(.4) | cpss-cap10-CD | cpss-cap10-R | 0/140 | 0 |
| null AR(.4) | cpss-cap20-CD | cpss-cap20-R | 0/140 | 0 |
| null AR(.4) | mx-equi-CD | mx-equi-R | 0/140 | 0 |
| null AR(.4) | mx-equi-CD | mx-equi-GLMNet.jl | 0/140 | 0 |
| null AR(.4) | mx-mvr-CD | mx-mvr-R | 0/140 | 0 |
| null AR(.4) | mx-mvr-CD | mx-mvr-GLMNet.jl | 0/140 | 0 |
| null AR(.4) | mx-maxent-CD | mx-maxent-R | 0/140 | 0 |
| null AR(.4) | mx-maxent-CD | mx-maxent-GLMNet.jl | 0/140 | 0 |
| null AR(.4) | fx-CD | fx-R | 0/140 | 0 |
| null AR(.4) | fx-CD | fx-GLMNet.jl | 0/140 | 0 |
| wide AR(.4) | slope-CD | slope-SLOPE.jl | 0/140 | 0 |
| wide AR(.4) | slope-CD | slope-R | 0/140 | 0 |
| wide AR(.4) | cpss-cap10-CD | cpss-cap10-R | 0/140 | 0 |
| wide AR(.4) | cpss-cap20-CD | cpss-cap20-R | 0/140 | 0 |
| wide AR(.4) | mx-equi-CD | mx-equi-R | 0/140 | 0 |
| wide AR(.4) | mx-equi-CD | mx-equi-GLMNet.jl | 0/140 | 0 |
| wide AR(.4) | mx-mvr-CD | mx-mvr-R | 0/140 | 0 |
| wide AR(.4) | mx-mvr-CD | mx-mvr-GLMNet.jl | 0/140 | 0 |
| wide AR(.4) | mx-maxent-CD | mx-maxent-R | 0/140 | 0 |
| wide AR(.4) | mx-maxent-CD | mx-maxent-GLMNet.jl | 0/140 | 0 |

## Paired power differences from construction choices

Shared data, Gaussian innovations, conditional sampler, lasso solver and lambda. Only the diagonal construction differs.

| Scenario | Choice | Power difference (MC SE) |
|---|---|---:|
| iid | mvr minus equi | 0.0000 (0.0000) |
| iid | maxent minus equi | 0.0000 (0.0000) |
| AR(.4) | mvr minus equi | 0.4414 (0.0282) |
| AR(.4) | maxent minus equi | 0.4118 (0.0286) |
| AR(.8) | mvr minus equi | 0.0021 (0.0030) |
| AR(.8) | maxent minus equi | 0.0000 (0.0020) |
| weak AR(.4) | mvr minus equi | 0.0450 (0.0116) |
| weak AR(.4) | maxent minus equi | 0.0329 (0.0102) |
| wide AR(.4) | mvr minus equi | 0.0125 (0.0053) |
| wide AR(.4) | maxent minus equi | 0.0039 (0.0028) |
