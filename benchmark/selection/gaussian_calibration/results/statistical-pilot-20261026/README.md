# Disjoint statistical pilot; not the evaluation cohort

Pilot13455149 generated32 inputs (two/scenario,16 scenarios) using disjoint
107000000-family seeds and completed all48 fresh backend processes. Native
BH/paper/Julia/automatic fits, SLOPE.jl three common-weight fits and R three
common-weight plus native-R fits give352 independently certified fits. Saved-fit
recheck13455171 completed in3 seconds, passing every full relative proximal
residual/intercept certificate at1e-7 and192 common-weight comparisons.
Automatic and explicit native Gaussian coefficients are bitwise identical.

The first checker failure is retained: native R weights differ from accurate
SciPy normal quantiles by roughly1e-9 relatively because the pinned R package
uses a rational approximation. The corrected independent checker reproduces
that pinned quantile formula at the original2e-12 sequence tolerance, checks
its difference from SciPy within5e-9, and keeps every fit's1e-7 certificate.
No native/Julia/R fit was repeated for that recheck. The three common supplied
sequences keep the original2e-12 independent weight checks.

`stat-pilot-receipts-v1.tar.gz` contains every per-input certificate, common
backend comparison, worker status/source/thread/input-hash metadata and the
first checker failure. Raw shared input/coefficient binaries remain under
Discovery issue107/stat-pilot-v1. Source snapshots stat-v1.tar and
stat-check-v2.tar remain in the run directory; the corrected harness is in
this branch. Prototype correctness and comparator-preparation receipts live
beside this directory. Numerical generation/fits/validation ran under Slurm.

These data size/test the harness and are EXCLUDED from the declared1008-per-
scenario main evaluation. They are not statistical improvement/default evidence.
Original comparator status CSV true flags mean successful API return, not a
reported convergence certificate; the independent residuals establish accuracy.
The main adapters record unexposed convergence flags as unavailable.

Receipt archive SHA256: `5694963daeb95b358b2000329b3ca20cc35467700e0134ff43971985798d4154`.
