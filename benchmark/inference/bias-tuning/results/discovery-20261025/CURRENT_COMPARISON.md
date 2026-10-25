# Current defaults on the paired competitor pilot

EMPIRICAL. Twenty datasets per original design. Current Julia default fits replace the historical default rows on exactly the same saved inputs; matched Julia rows and competitor rows reuse the previously validated fits. The frozen restricted-LDPE candidate is labeled experimental. These legacy pilot inputs are separate from the new 200-dataset-per-design validation; this refresh does not choose or retune the candidate.

The requested task has ten targets. Julia and SIHR compute those targets, while hdi/EconML/sslasso compute all 500 slopes internally. Full-pipeline timing here is not an equal-work solver benchmark; use the separately matched 500-slope/row benchmark for that comparison. New Julia memory is measured for combined worker processes and is not imputed to individual policy rows. Old per-process competitor memory remains in the records.

| Case | Engine | Tuning | Valid/20 | Active coverage ± MCSE | Null coverage | Type I | Power | Length active/null | Median warm seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| binomial-correlated | SIHR | default | 20/20 | 90.0% ± 3.4 pp | 99.0% | 1.0% | 8.0% | 1.575/1.580 | 12.213 |
| binomial-correlated | SIHR | matched | 20/20 | 29.0% ± 5.1 pp | 94.0% | 6.0% | 7.0% | 0.976/0.999 | 4.546 |
| binomial-correlated | hdi | default | 20/20 | 87.0% ± 3.6 pp | 97.0% | 3.0% | 15.0% | 1.456/1.474 | 55.064 |
| binomial-correlated | hdi | matched | 0/20 | unsupported/failed | — | — | — | — | — |
| binomial-correlated | ours | default | 20/20 | 85.0% ± 4.1 pp | 98.0% | 2.0% | 16.0% | 1.442/1.457 | 0.183 |
| binomial-correlated | ours | matched | 20/20 | 27.0% ± 4.4 pp | 94.0% | 6.0% | 5.0% | 1.006/1.007 | 0.006 |
| binomial-correlated | oursrldpe | experimental | 20/20 | 98.0% ± 1.4 pp | 94.0% | 6.0% | 8.0% | 3.506/3.637 | 1.197 |
| binomial-moderate | SIHR | default | 20/20 | 61.0% ± 5.5 pp | 99.0% | 1.0% | 77.0% | 0.642/0.627 | 21.541 |
| binomial-moderate | SIHR | matched | 20/20 | 58.0% ± 5.4 pp | 97.0% | 3.0% | 70.0% | 0.604/0.587 | 5.158 |
| binomial-moderate | hdi | default | 20/20 | 60.0% ± 4.8 pp | 99.0% | 1.0% | 78.0% | 0.600/0.596 | 208.173 |
| binomial-moderate | hdi | matched | 0/20 | unsupported/failed | — | — | — | — | — |
| binomial-moderate | ours | default | 20/20 | 58.0% ± 5.0 pp | 99.0% | 1.0% | 77.0% | 0.586/0.584 | 0.950 |
| binomial-moderate | ours | matched | 20/20 | 67.0% ± 5.3 pp | 98.0% | 2.0% | 70.0% | 0.703/0.694 | 0.009 |
| binomial-moderate | oursrldpe | experimental | 20/20 | 92.0% ± 3.0 pp | 94.0% | 6.0% | 62.0% | 0.946/0.948 | 2.598 |
| gaussian-correlated | SIHR | default | 20/20 | 69.0% ± 6.7 pp | 92.0% | 8.0% | 19.0% | 0.793/0.795 | 13.538 |
| gaussian-correlated | SIHR | matched | 20/20 | 7.0% ± 2.2 pp | 89.0% | 11.0% | 22.0% | 0.395/0.406 | 4.851 |
| gaussian-correlated | econml | default | 20/20 | 66.0% ± 6.3 pp | 92.0% | 8.0% | 24.0% | 0.753/0.759 | 24.492 |
| gaussian-correlated | econml | matched | 20/20 | 12.0% ± 3.0 pp | 88.0% | 12.0% | 21.0% | 0.438/0.436 | 0.825 |
| gaussian-correlated | hdi | default | 20/20 | 76.0% ± 4.9 pp | 91.0% | 9.0% | 27.0% | 0.822/0.833 | 57.666 |
| gaussian-correlated | hdi | matched | 20/20 | 12.0% ± 3.0 pp | 88.0% | 12.0% | 21.0% | 0.440/0.439 | 0.944 |
| gaussian-correlated | ours | default | 20/20 | 67.0% ± 6.7 pp | 91.0% | 9.0% | 23.0% | 0.741/0.749 | 0.240 |
| gaussian-correlated | ours | matched | 20/20 | 12.0% ± 3.0 pp | 88.0% | 12.0% | 21.0% | 0.440/0.439 | 0.004 |
| gaussian-correlated | oursqp | default | 20/20 | 8.0% ± 3.0 pp | 90.0% | 10.0% | 23.0% | 0.400/0.411 | 0.144 |
| gaussian-correlated | oursqp | matched | 20/20 | 7.0% ± 2.2 pp | 88.0% | 12.0% | 22.0% | 0.395/0.406 | 0.005 |
| gaussian-correlated | oursrldpe | experimental | 20/20 | 92.0% ± 3.0 pp | 92.0% | 8.0% | 11.0% | 1.827/1.896 | 1.314 |
| gaussian-correlated | sslasso | default | 20/20 | 5.0% ± 2.0 pp | 98.0% | 16.0% | 21.0% | 0.461/0.467 | 20.477 |
| gaussian-correlated | sslasso | matched | 20/20 | 22.0% ± 5.4 pp | 94.0% | 12.0% | 28.0% | 0.553/0.565 | 99.842 |
| gaussian-moderate | SIHR | default | 20/20 | 73.0% ± 5.5 pp | 98.0% | 2.0% | 100.0% | 0.283/0.282 | 22.228 |
| gaussian-moderate | SIHR | matched | 20/20 | 35.0% ± 5.0 pp | 95.0% | 5.0% | 100.0% | 0.215/0.215 | 5.213 |
| gaussian-moderate | econml | default | 20/20 | 73.0% ± 4.7 pp | 96.0% | 4.0% | 100.0% | 0.286/0.288 | 62.615 |
| gaussian-moderate | econml | matched | 20/20 | 70.0% ± 5.1 pp | 95.0% | 5.0% | 100.0% | 0.285/0.285 | 0.766 |
| gaussian-moderate | hdi | default | 20/20 | 82.0% ± 4.1 pp | 93.0% | 7.0% | 100.0% | 0.321/0.322 | 207.726 |
| gaussian-moderate | hdi | matched | 20/20 | 70.0% ± 5.1 pp | 95.0% | 5.0% | 100.0% | 0.286/0.286 | 1.249 |
| gaussian-moderate | ours | default | 20/20 | 74.0% ± 5.0 pp | 96.0% | 4.0% | 100.0% | 0.290/0.290 | 1.035 |
| gaussian-moderate | ours | matched | 20/20 | 70.0% ± 5.1 pp | 95.0% | 5.0% | 100.0% | 0.286/0.286 | 0.006 |
| gaussian-moderate | oursqp | default | 20/20 | 36.0% ± 5.2 pp | 95.0% | 5.0% | 99.0% | 0.213/0.213 | 0.252 |
| gaussian-moderate | oursqp | matched | 20/20 | 35.0% ± 5.0 pp | 95.0% | 5.0% | 100.0% | 0.214/0.214 | 0.006 |
| gaussian-moderate | oursrldpe | experimental | 20/20 | 94.0% ± 2.9 pp | 95.0% | 5.0% | 97.0% | 0.467/0.469 | 2.604 |
| gaussian-moderate | sslasso | default | 20/20 | 79.0% ± 4.9 pp | 98.0% | 5.0% | 100.0% | 0.326/0.327 | 52.502 |
| gaussian-moderate | sslasso | matched | 20/20 | 65.0% ± 5.6 pp | 99.0% | 6.0% | 100.0% | 0.324/0.326 | 6.168 |

Current Julia defaults now select initial penalties automatically for both response families; the historical required-GLM-lambda note applies only to the superseded pilot rows. Matched rows keep the explicit common penalties. Expert restricted LDPE is not a competitor-matched projection rule.

Native hdi binomial matched preprocessing remains unsupported. EconML and sslasso are Gaussian-only. SIHR can adjust supplied constraints and retains a different target pipeline. sslasso intervals include bias padding and need not be dual to the returned normal p-values. See the original pilot report for warnings and reference-code corrections.

Current-default refresh: 200 fits; 200 successful. Failures, if any, remain in the records; no numerical default is silently tightened for this comparison.
