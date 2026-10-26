# SLOPE completion tables from frozen cohorts

All numerical generation, fitting, independent checks, profiling and aggregation ran under Discovery Slurm.
Native statistical fits:64512/64512 independent original-design residual certificates pass on16128 independent inputs. Five strict1e-9 stopping flags are retained; each still passes the declared1e-7 independent criterion. 450 external statistical certificates fail at their frozen controls and remain excluded from valid-backend rankings. All94500 runtime/default certificates pass. No failed result is relabeled as certified.

FDR is meanFDP; power is the fraction of20 true coefficients selected (undefined at global null). Pointwise95% Student-t Monte Carlo intervals are shown. Complete data include HoeffdingFDP bounds and exact-null binomial intervals. Every scenario has1008 inputs; main and stress cohorts remain separate.

| Scenario | BH FDR [CI] | Gaussian FDR [CI] | BH power [CI] | Gaussian power [CI] | Paired FDR change [CI] | Paired power change [CI] |
|---|---|---|---|---|---|---|
| iid-alt (main) | 0.1818 [0.1767, 0.1869] | 0.0733 [0.0698, 0.0768] | 0.9464 [0.9429, 0.9499] | 0.8570 [0.8519, 0.8622] | -0.1085 [-0.1127, -0.1042] | -0.0893 [-0.0932, -0.0854] |
| iid-same (main) | 0.1820 [0.1770, 0.1871] | 0.0758 [0.0721, 0.0795] | 0.9459 [0.9425, 0.9493] | 0.8530 [0.8476, 0.8584] | -0.1063 [-0.1105, -0.1021] | -0.0929 [-0.0971, -0.0886] |
| ar04-alt (main) | 0.2700 [0.2597, 0.2802] | 0.2010 [0.1900, 0.2120] | 0.3226 [0.3140, 0.3312] | 0.2229 [0.2167, 0.2290] | -0.0690 [-0.0766, -0.0613] | -0.0998 [-0.1053, -0.0942] |
| ar04-same (main) | 0.1149 [0.1108, 0.1190] | 0.0339 [0.0315, 0.0363] | 0.9985 [0.9979, 0.9990] | 0.9866 [0.9850, 0.9882] | -0.0810 [-0.0845, -0.0774] | -0.0119 [-0.0134, -0.0104] |
| ar08-alt (main) | 0.1832 [0.1631, 0.2033] | 0.1663 [0.1464, 0.1863] | 0.0429 [0.0394, 0.0463] | 0.0358 [0.0330, 0.0386] | -0.0169 [-0.0237, -0.0101] | -0.0070 [-0.0086, -0.0055] |
| ar08-same (main) | 0.0734 [0.0696, 0.0771] | 0.0254 [0.0231, 0.0276] | 0.9983 [0.9977, 0.9989] | 0.9774 [0.9753, 0.9795] | -0.0480 [-0.0510, -0.0449] | -0.0209 [-0.0229, -0.0189] |
| weak-ar04-alt (main) | 0.1917 [0.1749, 0.2085] | 0.1706 [0.1534, 0.1879] | 0.0799 [0.0750, 0.0849] | 0.0625 [0.0587, 0.0664] | -0.0211 [-0.0286, -0.0136] | -0.0174 [-0.0198, -0.0149] |
| weak-ar04-same (main) | 0.1112 [0.1068, 0.1156] | 0.0324 [0.0299, 0.0349] | 0.9548 [0.9519, 0.9576] | 0.8813 [0.8770, 0.8857] | -0.0788 [-0.0825, -0.0750] | -0.0734 [-0.0769, -0.0699] |
| wide-ar04-alt (main) | 0.6235 [0.6142, 0.6328] | 0.4889 [0.4746, 0.5032] | 0.2650 [0.2581, 0.2718] | 0.1368 [0.1321, 0.1415] | -0.1346 [-0.1466, -0.1227] | -0.1282 [-0.1335, -0.1229] |
| wide-ar04-same (main) | 0.2352 [0.2296, 0.2409] | 0.0678 [0.0643, 0.0713] | 0.9875 [0.9858, 0.9891] | 0.9399 [0.9367, 0.9431] | -0.1674 [-0.1723, -0.1626] | -0.0476 [-0.0505, -0.0446] |
| null-ar04 (main) | 0.0863 [0.0689, 0.1037] | 0.0843 [0.0671, 0.1015] | undefined [undefined, undefined] | undefined [undefined, undefined] | -0.0020 [-0.0047, 0.0008] | undefined [undefined, undefined] |
| orth-alt (main) | 0.0810 [0.0772, 0.0847] | 0.0136 [0.0119, 0.0152] | 0.9868 [0.9853, 0.9884] | 0.9454 [0.9423, 0.9486] | -0.0674 [-0.0709, -0.0639] | -0.0414 [-0.0442, -0.0386] |
| orth-same (main) | 0.0797 [0.0760, 0.0834] | 0.0123 [0.0108, 0.0138] | 0.9847 [0.9830, 0.9864] | 0.9406 [0.9373, 0.9439] | -0.0674 [-0.0708, -0.0639] | -0.0441 [-0.0470, -0.0413] |
| estimated-iid (stress) | 0.0388 [0.0355, 0.0421] | 0.0156 [0.0130, 0.0181] | 0.6718 [0.6644, 0.6792] | 0.4732 [0.4669, 0.4795] | -0.0233 [-0.0263, -0.0203] | -0.1986 [-0.2049, -0.1923] |
| estimated-ar04 (stress) | 0.0750 [0.0639, 0.0862] | 0.0577 [0.0470, 0.0685] | 0.0890 [0.0841, 0.0939] | 0.0702 [0.0663, 0.0741] | -0.0173 [-0.0228, -0.0118] | -0.0188 [-0.0215, -0.0161] |
| block-alt (stress) | 0.2172 [0.1965, 0.2378] | 0.1989 [0.1783, 0.2196] | 0.0466 [0.0428, 0.0505] | 0.0377 [0.0347, 0.0408] | -0.0182 [-0.0249, -0.0115] | -0.0089 [-0.0107, -0.0071] |

## Complete public procedure runtime

Comparator/native public geometric ratios, with pointwise process-paired95% intervals. Each case uses21 inputs over7 fresh process replications, three warmed rounds/input. Complete fit/path results are timed; preprocessing/profile/certificate placement is documented in PROTOCOL. These are current-procedure comparisons, not a before/after speed claim for the new calibration API.

| Case | Weights | Payload | SLOPE.jl/native [CI] | R/native [CI] | Prepared/native [CI] | Default/native [CI] | Unscreened/native [CI] |
|---|---|---|---|---|---|---|---|
| dense | bh | fit | 1.0384 [0.9047, 1.1920] | 1.2585 [1.0855, 1.4590] | 0.8475 [0.7949, 0.9037] | 0.8962 [0.8050, 0.9979] | not applicable |
| dense | bh | path | 1.7740 [1.4370, 2.1902] | 3.0386 [2.4658, 3.7444] | 0.9098 [0.8726, 0.9486] | 0.6742 [0.5644, 0.8054] | 3.6884 [3.0714, 4.4293] |
| dense | paper | fit | 0.9702 [0.8820, 1.0672] | 1.3131 [1.2225, 1.4103] | 0.8791 [0.8400, 0.9201] | 1.0434 [0.9459, 1.1511] | not applicable |
| dense | paper | path | 1.5417 [1.2447, 1.9096] | 2.5649 [2.0813, 3.1607] | 0.9005 [0.8815, 0.9199] | 0.5487 [0.4523, 0.6657] | 3.2946 [2.9763, 3.6469] |
| dense | julia | fit | 0.9146 [0.8445, 0.9906] | 1.2657 [1.1925, 1.3433] | 0.8622 [0.8108, 0.9168] | 0.9517 [0.8968, 1.0100] | not applicable |
| dense | julia | path | 1.7260 [1.4838, 2.0077] | 2.8939 [2.4573, 3.4080] | 0.9333 [0.8856, 0.9836] | 0.6165 [0.5319, 0.7145] | 3.6337 [3.3652, 3.9235] |
| wide | bh | fit | 0.5446 [0.5001, 0.5929] | 0.7205 [0.6671, 0.7782] | 0.9380 [0.9069, 0.9703] | 0.9616 [0.8973, 1.0304] | not applicable |
| wide | bh | path | 1.7391 [1.5171, 1.9937] | 2.1987 [1.9946, 2.4237] | 0.9782 [0.9375, 1.0208] | 0.6506 [0.5907, 0.7165] | 5.0347 [4.5724, 5.5437] |
| wide | paper | fit | 0.3604 [0.3359, 0.3866] | 0.6273 [0.5932, 0.6634] | 0.9049 [0.8659, 0.9456] | 0.9354 [0.8702, 1.0055] | not applicable |
| wide | paper | path | 1.6369 [1.4341, 1.8684] | 2.3798 [2.0477, 2.7657] | 0.9417 [0.9056, 0.9792] | 0.6567 [0.5488, 0.7858] | 5.7537 [4.9580, 6.6771] |
| wide | julia | fit | 0.3791 [0.3453, 0.4162] | 0.6525 [0.5912, 0.7202] | 0.9209 [0.8689, 0.9760] | 0.9616 [0.8743, 1.0576] | not applicable |
| wide | julia | path | 1.5396 [1.3537, 1.7510] | 2.2412 [1.9053, 2.6363] | 0.9485 [0.9123, 0.9860] | 0.6270 [0.5314, 0.7397] | 5.4264 [4.7132, 6.2476] |
| correlated | bh | fit | 1.0399 [0.9540, 1.1335] | 0.8082 [0.7311, 0.8935] | 0.9261 [0.8985, 0.9545] | 0.9524 [0.8868, 1.0230] | not applicable |
| correlated | bh | path | 1.6107 [1.4245, 1.8212] | 2.3524 [2.0349, 2.7195] | 0.9507 [0.9103, 0.9928] | 0.6425 [0.5773, 0.7152] | 6.2268 [5.5447, 6.9927] |
| correlated | paper | fit | 0.6991 [0.6441, 0.7588] | 0.6388 [0.5775, 0.7066] | 0.9719 [0.9232, 1.0230] | 0.9973 [0.9610, 1.0349] | not applicable |
| correlated | paper | path | 1.7236 [1.4578, 2.0378] | 2.0653 [1.6521, 2.5819] | 0.9787 [0.9594, 0.9984] | 0.4978 [0.4242, 0.5843] | 6.2205 [5.4319, 7.1235] |
| correlated | julia | fit | 0.7101 [0.6569, 0.7676] | 0.6155 [0.5569, 0.6803] | 0.9732 [0.9370, 1.0109] | 0.9685 [0.9363, 1.0018] | not applicable |
| correlated | julia | path | 1.9080 [1.7410, 2.0911] | 2.2806 [1.9593, 2.6545] | 0.9801 [0.9509, 1.0102] | 0.5505 [0.5085, 0.5959] | 6.7374 [6.3151, 7.1878] |
| sparse | bh | fit | 0.6078 [0.5726, 0.6451] | 1.2404 [1.1748, 1.3097] | 0.9795 [0.9636, 0.9957] | 0.9931 [0.9765, 1.0099] | not applicable |
| sparse | bh | path | 2.0563 [1.8872, 2.2406] | 3.7084 [3.5123, 3.9155] | 0.9986 [0.9772, 1.0205] | 0.8389 [0.8127, 0.8659] | 4.4598 [4.3316, 4.5919] |
| sparse | paper | fit | 0.4909 [0.4674, 0.5155] | 1.2521 [1.1874, 1.3203] | 0.9895 [0.9805, 0.9986] | 1.0053 [0.9867, 1.0242] | not applicable |
| sparse | paper | path | 1.7162 [1.5312, 1.9237] | 3.7578 [3.3517, 4.2130] | 0.9926 [0.9852, 0.9999] | 0.8355 [0.7411, 0.9419] | 4.6210 [4.2209, 5.0589] |
| sparse | julia | fit | 0.4874 [0.4699, 0.5055] | 1.2323 [1.1878, 1.2786] | 0.9711 [0.9583, 0.9840] | 0.9946 [0.9638, 1.0264] | not applicable |
| sparse | julia | path | 1.6953 [1.5207, 1.8900] | 3.7074 [3.3403, 4.1150] | 0.9933 [0.9728, 1.0144] | 0.8214 [0.7389, 0.9131] | 4.5783 [4.2463, 4.9362] |

## Scope and retained limits

No empirical FDR<=q acceptance rule applies outside the known-noise orthogonal theorem. Gaussian adjustment, supplied std(y), correlated/misspecified covariance and selected path levels do not inherit that guarantee. The native Gaussian rule has a declared first-nondecrease tail and differs from the pinned Julia convention. Statistical primary fits use native1e-9 and external1e-8; runtime uses native1e-8 and external1e-10, all against the same independent1e-7 criterion. Production defaults remain BH, tol1e-7/max10000 and screening=true. Whole-worker RSS includes imports/JIT/profiling, so it is not fitted-state memory. Pilots, failed launchers, external accuracy failures and native strict-stopping flags remain in the evidence. Within each cohort inputs are independent. Runtime seed numbers overlap some statistical scenario seeds with different dimensions/data laws; the cohorts are not pooled into a combined independent sample.
