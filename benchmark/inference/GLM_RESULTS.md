# High-dimensional logistic validation

Discovery: 20 response replications in each of 12 fixed Toeplitz n=100,p=500 designs (rho=0.9). Signal sizes 3/15, first/random support, fixed uniform coefficients on [0,c], c=1/2/4. Initial logistic penalty 0.05; Hessian-weighted nodewise penalty 0.1 on standardized columns. Null decorrelated score: nuisance penalty 0.05, projection penalty 0.03. All settings and seeds were fixed before the run.

| s | support | c | valid fits / attempted | active coverage | null coverage | active length | valid scores | null rejection rate | errors |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | first | 1 | 20 / 20 | 0.867 | 0.988 | 1.733 | 20 | 0.000 | 0 |
| 3 | first | 2 | 20 / 20 | 0.933 | 0.991 | 2.121 | 20 | 0.000 | 0 |
| 3 | first | 4 | 20 / 20 | 0.800 | 0.994 | 2.529 | 20 | 0.000 | 0 |
| 3 | random | 1 | 20 / 20 | 1.000 | 0.983 | 1.600 | 20 | 0.000 | 0 |
| 3 | random | 2 | 20 / 20 | 1.000 | 0.990 | 1.796 | 20 | 0.000 | 0 |
| 3 | random | 4 | 20 / 20 | 0.700 | 0.995 | 2.060 | 20 | 0.000 | 0 |
| 15 | first | 1 | 20 / 20 | 0.980 | 0.998 | 2.321 | 20 | 0.000 | 0 |
| 15 | first | 2 | 20 / 20 | 0.743 | 0.999 | 2.382 | 20 | 0.000 | 0 |
| 15 | first | 4 | 20 / 20 | 0.280 | 0.999 | 2.420 | 20 | 0.000 | 0 |
| 15 | random | 1 | 20 / 20 | 0.880 | 0.989 | 1.556 | 20 | 0.000 | 0 |
| 15 | random | 2 | 20 / 20 | 0.403 | 0.993 | 1.620 | 20 | 0.000 | 0 |
| 15 | random | 4 | 20 / 20 | 0.127 | 0.995 | 1.655 | 20 | 0.000 | 0 |

Coverage is conditional on numerically converged fits; any failed fits remain in the raw CSVs and denominators above. Score rejection uses only converged score fits, with its denominator displayed. Twenty replications give coarse Monte Carlo resolution and do not establish reliable nominal calibration. These fixed penalties differ from the paper’s CV tuning. In particular, strong signals and weak effective information can make the asymptotic approximation inaccurate.


All 240 debiasing fits and all 240 score fits converged, with no exceptions.
Active-variable coverage was as low as 0.127 in the strong, s=15 random-support
setting. The fixed null score rejected in 0/20 replications in each scenario;
this coarse conservative result does not establish uniform type-I calibration
or power. Gaussian and logistic finite-sample limitations therefore remain
explicit in the implementation report.

Validated job: 13423281. The first driver attempt (13423249) failed before
fitting because an unexported loss type was not module-qualified. Its 240
error records and diagnostic summary are retained in the run directory; they
are excluded from scientific estimates. Only the driver namespace changed,
and the retry retained every design, penalty and seed.
