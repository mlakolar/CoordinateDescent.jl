# Discovery optimization comparison

Five independent shared inputs per case; three timing rounds per input. Entries below are the median of the five within-input timing medians. All workers use one CPU and one BLAS thread. MATLAB is excluded.

The main table uses fixed rank 3 or fixed penalty 0.1 times the shared objective-specific lambda-max. A timing is eligible for an accuracy-matched comparison only when all three rounds passed and all five problem replications are present. Nonconvex stationary points can have different objectives; consult paired-ratios.csv before interpreting speed ratios.

| Case | Family | Solver | Median ms | Replication range ms | Certified inputs |
|---|---|---|---:|---:|---:|
| deficient | component | cd_component | 0.6586 | 0.5476–0.7703 | 5/5 |
| deficient | component | component_fista | 1.342 | 1.176–1.496 | 5/5 |
| deficient | component | r_component | 17.61 | 17.14–22.73 | 5/5 |
| deficient | nuclear | cd_nuclear | 32.93 | 29.38–33.44 | 5/5 |
| deficient | nuclear | pa_fista | 76.55 | 62.47–80.62 | 5/5 |
| deficient | nuclear | pa_panoc | 22.44 | 22.3–42.22 | 5/5 |
| deficient | nuclear | ps_admm | 50.61 | 49.57–50.98 | 5/5 |
| deficient | nuclear | ps_fista | 36.58 | 29.63–43.92 | 5/5 |
| deficient | nuclear | py_admm | 65.03 | 57.27–65.69 | 5/5 |
| deficient | nuclear | py_fista | 60.25 | 51.88–61.21 | 5/5 |
| deficient | rank | cd_rank | 3.331 | 3.234–3.418 | 5/5 |
| deficient | rank | r_rrr | 7.517 | 7.111–7.843 | 5/5 |
| deficient | rank | r_rrr_qr | 10.75 | 10.6–11.46 | 5/5 |
| deficient | rank | scipy_rrr | 2.051 | 2.03–2.202 | 5/5 |
| deficient | sparse | block_fista | 1.479 | 1.405–1.512 | 5/5 |
| deficient | sparse | cd_sparse | 3.778 | 3.558–3.913 | 5/5 |
| deficient | sparse | r_srrr | 19.88 | 17.72–20.62 | 5/5 |
| ill | component | cd_component | 2.182 | 1.531–2.704 | 5/5 |
| ill | component | component_fista | 7.85 | 6.596–10.17 | 5/5 |
| ill | component | r_component | 36.76 | 32.83–43.26 | 5/5 |
| ill | nuclear | cd_nuclear | 154.2 | 147.1–186.1 | 5/5 |
| ill | nuclear | pa_fista | 532.3 | 420.2–652.4 | 5/5 |
| ill | nuclear | pa_panoc | 96.25 | 84.47–102.4 | 5/5 |
| ill | nuclear | pa_pg | 455.4 | 342.8–519.5 | 5/5 |
| ill | nuclear | ps_admm | 317.2 | 240.6–357.9 | 5/5 |
| ill | nuclear | ps_fista | 153.4 | 136.2–163.1 | 5/5 |
| ill | nuclear | py_admm | 397.4 | 300.2–449.7 | 5/5 |
| ill | nuclear | py_fista | 441 | 315.6–459.8 | 5/5 |
| ill | rank | cd_rank | 7.748 | 7.462–7.802 | 5/5 |
| ill | rank | r_rrr | 12.58 | 11.8–13.8 | 0/5 |
| ill | rank | r_rrr_qr | 20.1 | 18.78–22.59 | 5/5 |
| ill | rank | scipy_rrr | 5.121 | 4.719–5.409 | 5/5 |
| ill | sparse | block_fista | 16.43 | 11.6–25.05 | 5/5 |
| ill | sparse | cd_sparse | 41.25 | 24.89–67.88 | 5/5 |
| ill | sparse | r_srrr | 165 | 118.8–318.2 | 5/5 |
| medium | component | cd_component | 0.4177 | 0.4026–0.4352 | 5/5 |
| medium | component | component_fista | 0.8094 | 0.7658–0.8139 | 5/5 |
| medium | component | r_component | 14.85 | 14.33–15.44 | 5/5 |
| medium | nuclear | cd_nuclear | 25.79 | 24.06–26.38 | 5/5 |
| medium | nuclear | pa_fista | 62.06 | 52.53–67.93 | 5/5 |
| medium | nuclear | pa_panoc | 16.94 | 16.2–17 | 5/5 |
| medium | nuclear | pa_pg | 46.16 | 45.7–49.92 | 5/5 |
| medium | nuclear | ps_admm | 41.55 | 39.44–43.38 | 5/5 |
| medium | nuclear | ps_fista | 28.41 | 25.11–35.96 | 5/5 |
| medium | nuclear | py_admm | 49.45 | 46.72–52 | 5/5 |
| medium | nuclear | py_fista | 52.68 | 47.8–56.82 | 5/5 |
| medium | rank | cd_rank | 2.375 | 2.283–2.431 | 5/5 |
| medium | rank | pa_rank | 166.5 | 16.52–3699 | 5/5 |
| medium | rank | r_rrr | 6.412 | 6.126–6.884 | 5/5 |
| medium | rank | r_rrr_qr | 9.974 | 9.72–10.86 | 5/5 |
| medium | rank | scipy_rrr | 1.63 | 1.459–1.643 | 5/5 |
| medium | sparse | block_fista | 0.8498 | 0.8334–0.9091 | 5/5 |
| medium | sparse | cd_sparse | 1.412 | 1.37–1.536 | 5/5 |
| medium | sparse | r_srrr | 7.606 | 7.108–8.04 | 5/5 |
| responses | component | cd_component | 2.951 | 2.915–4.509 | 5/5 |
| responses | component | component_fista | 3.662 | 3.582–6.172 | 5/5 |
| responses | component | r_component | 1142 | 999.3–1437 | 5/5 |
| responses | nuclear | cd_nuclear | 146.9 | 129.1–157.9 | 5/5 |
| responses | nuclear | pa_fista | 464.9 | 460.6–482.2 | 5/5 |
| responses | nuclear | pa_panoc | 151 | 135–311.1 | 5/5 |
| responses | nuclear | ps_admm | 214.9 | 204.3–221.8 | 5/5 |
| responses | nuclear | ps_fista | 224.3 | 200–287.5 | 5/5 |
| responses | nuclear | py_admm | 290.7 | 272.9–324.4 | 5/5 |
| responses | nuclear | py_fista | 304.6 | 277–314.8 | 5/5 |
| responses | rank | cd_rank | 8.424 | 8.362–10.5 | 5/5 |
| responses | rank | r_rrr | 75.02 | 66.38–95.73 | 5/5 |
| responses | rank | r_rrr_qr | 113.7 | 107–135.7 | 5/5 |
| responses | rank | scipy_rrr | 4.604 | 4.444–4.665 | 5/5 |
| responses | sparse | block_fista | 3.293 | 3.275–3.784 | 5/5 |
| responses | sparse | cd_sparse | 8.743 | 8.128–10.05 | 5/5 |
| responses | sparse | r_srrr | 34.64 | 28.28–36.22 | 5/5 |
| small | component | cd_component | 0.1822 | 0.1733–0.2049 | 5/5 |
| small | component | component_fista | 0.4735 | 0.4348–0.4976 | 5/5 |
| small | component | r_component | 2.46 | 2.23–2.788 | 5/5 |
| small | nuclear | cd_nuclear | 2.04 | 2.031–2.129 | 5/5 |
| small | nuclear | pa_fista | 4.565 | 4.095–4.654 | 5/5 |
| small | nuclear | pa_panoc | 1.632 | 1.574–2.59 | 5/5 |
| small | nuclear | ps_admm | 3.341 | 3.227–3.474 | 5/5 |
| small | nuclear | ps_fista | 2.493 | 2.291–2.89 | 5/5 |
| small | nuclear | py_admm | 8.109 | 8.068–9.091 | 5/5 |
| small | nuclear | py_fista | 7.601 | 6.867–7.839 | 5/5 |
| small | rank | cd_rank | 1.292 | 1.257–1.311 | 5/5 |
| small | rank | r_rrr | 2.689 | 2.394–3.005 | 5/5 |
| small | rank | r_rrr_qr | 3.485 | 3.102–4.089 | 5/5 |
| small | rank | scipy_rrr | 1.016 | 0.9925–1.062 | 5/5 |
| small | sparse | block_fista | 0.5926 | 0.5847–0.6647 | 5/5 |
| small | sparse | cd_sparse | 0.5444 | 0.5133–0.5995 | 5/5 |
| small | sparse | r_srrr | 6.168 | 5.664–6.502 | 5/5 |
| wide | component | cd_component | 5.587 | 4.93–5.906 | 5/5 |
| wide | component | component_fista | 43.48 | 34.73–45.96 | 5/5 |
| wide | component | r_component | 20.55 | 19.96–22.5 | 5/5 |
| wide | nuclear | cd_nuclear | 386.6 | 349.7–409.4 | 5/5 |
| wide | nuclear | pa_fista | 949.9 | 869.8–1113 | 5/5 |
| wide | nuclear | pa_panoc | 240.7 | 236.8–327.6 | 5/5 |
| wide | nuclear | ps_admm | 668.4 | 621.8–714 | 5/5 |
| wide | nuclear | ps_fista | 314.2 | 268.2–343.7 | 5/5 |
| wide | nuclear | py_admm | 804.5 | 726.3–859.7 | 5/5 |
| wide | nuclear | py_fista | 848.9 | 815.4–966.6 | 5/5 |
| wide | rank | cd_rank | 20.48 | 20.3–21.85 | 5/5 |
| wide | rank | pa_rank | 1.36e+04 | 754.9–3.001e+04 | 5/5 |
| wide | rank | r_rrr | 58.94 | 57.26–60.53 | 5/5 |
| wide | rank | r_rrr_qr | 31.24 | 23.42–34.13 | 5/5 |
| wide | rank | scipy_rrr | 13.77 | 13.51–14.6 | 5/5 |
| wide | sparse | block_fista | 34.7 | 33.23–38.75 | 5/5 |
| wide | sparse | cd_sparse | 22.06 | 19.49–26.07 | 5/5 |
| wide | sparse | r_srrr | 30.31 | 28.03–32.3 | 5/5 |

## Coverage

- Timing observations: 5250.
- Separate profile reports: 255.
- Fresh-process memory observations: 170.
- Main accuracy thresholds: relative primal-dual gap and normalized proximal-gradient residual <=1e-6 for convex nuclear regression; objective/rank checks for fixed rank; actual-factor block residuals <=1e-6 for sparse fits.
- Native stopping tolerances are calibrated to meet these checks. Generic Python/JuliaFirstOrder methods check a proximal residual every10 iterations; those monitor costs are included in fit time. Final independent certificates are outside the fit timer.
- Sparse API factors are retained. For the original Julia API, its frozen outer loop is replayed outside the fit timer to recover factors and verify coefficient equality; this prevents false certificate failures from arbitrary SVD null-space completions.
- Peak RSS is total process memory, including interpreter/packages and warm-up; the separate memory table also gives the post-warm-up baseline. Ordinary worker cumulative RSS is not attributed to individual solvers.
- Python cProfile attributes native SVD/BLAS elapsed time to its Python call boundary; Rprof identifies .Call boundaries. Their percentages are not directly comparable to Julia sampling percentages.
- Raw inputs and fitted coefficients remain in the run directory for recreation and independent checking. Tables, profiles, exact versions, and solver source are retained with this report.
